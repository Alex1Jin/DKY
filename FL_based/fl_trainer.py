import argparse
import pandas as pd
import torch
import torch.nn as nn   # torch.nn是专门为神经网络设置的模块化接口
import torch.nn.functional as F  # 对于激活函数和池化层，由于没有可学习参数，一般使用该模块完成
import torch.optim as optim  # 用于优化参数
from torchvision import datasets, transforms  # 定义了datasets的数据格式及常用的数据转换方式

import pdb

from model import lenet
from utils import *

def get_results_filename(dataset="mnist", model_arch="vgg9", epoch=10):
    filename = "{}_{}_epoch_{}".format(dataset, model_arch, epoch)
    filename += "_acc_results.csv"

    return filename

def calc_norm_diff(gs_model, vanilla_model, epoch, fl_round, mode="normal"):
    norm_diff = 0
    for p_index, p in enumerate(gs_model.parameters()):
        norm_diff += torch.norm(list(gs_model.parameters())[p_index] - list(vanilla_model.parameters())[p_index]) ** 2
    norm_diff = torch.sqrt(norm_diff).item()
    if mode == "bad":
        #pdb.set_trace()
        logger.info("===> ND `|w_bad-w_g|` in local epoch: {} | FL round: {} |, is {}".format(epoch, fl_round, norm_diff))
    elif mode == "normal":
        logger.info("===> ND `|w_normal-w_g|` in local epoch: {} | FL round: {} |, is {}".format(epoch, fl_round, norm_diff))
    elif mode == "avg":
        logger.info("===> ND `|w_avg-w_g|` in local epoch: {} | FL round: {} |, is {}".format(epoch, fl_round, norm_diff))

    return norm_diff

####### 每轮直接加权加和平均所选用户的model作为global model
def fed_avg_aggregator(net_list, net_freq, device, model="lenet"):
    if model == "lenet":
        net_avg = lenet.LeNet().to(device)
    whole_aggregator = []

    for p_index, p in enumerate(net_list[0].parameters()):
        # initial
        params_aggregator = torch.zeros(p.size()).to(device)
        for net_index, net in enumerate(net_list):
            # we assume the adv model always comes to the beginning
            params_aggregator = params_aggregator + net_freq[net_index] * list(net.parameters())[p_index].data
        whole_aggregator.append(params_aggregator)

    for param_index, p in enumerate(net_avg.parameters()):
        p.data = whole_aggregator[param_index]
    return net_avg


def train(log_interval, model, device, train_loader, optimizer, epoch, criterion):

# 其中，model指建立好的网络模型；device指模型在哪个设备上运行，CPU还是GPU；train_loader是指数据集；
# optimizer用于优化；epoch指整个数据集训练的次数

    model.train() # 将模型设置为训练状态，对应的有model.eval()只有前传

    for batch_idx, (data, target) in enumerate(train_loader): # batch_idx为索引，（data, target）为值
        data, target = data.to(device), target.to(device) # data为图像，target为label
        optimizer.zero_grad() # 每次迭代把梯度的初值设为0
        output = model(data) # 模型的输出
        # loss = F.nll_loss(output, target) # 利用输出和label计算损失精度
        loss = criterion(output, target)
        loss.backward() # 反向传播，用来计算梯度
        optimizer.step() #更新参数
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, criterion):
    model.eval() # 只有前传，没有反向求梯度调参
    test_loss = 0
    correct = 0
    with torch.no_grad(): # 代表with下面的代码块，不参与反向传播，不更新参数
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset) #损失的平均值

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

class FLTrainer:
    def __init__(self, *args, **kwargs):
        self.hyper_params = None

    def run(self, client_model, *args, **kwargs):
        raise NotImplementedError()

class FederatedLearningTrainer(FLTrainer):
    def __init__(self, arguments=None, *args, **kwargs):
        self.vanilla_model = arguments['vanilla_model']
        self.net_avg = arguments['net_avg']
        self.net_dataidx_map = arguments['net_dataidx_map']
        self.num_nets = arguments['num_nets']
        self.part_nets_per_round = arguments['part_nets_per_round']
        self.fl_round = arguments['fl_round']
        self.local_training_period = arguments['local_training_period']
        self.lr = arguments['lr']
        self.args_gamma = arguments['args_gamma']
        self.test_loader = arguments['test_loader']
        self.batch_size = arguments['batch_size']
        self.test_batch_size = arguments['test_batch_size']
        self.log_interval = arguments['log_interval']
        self.device = arguments['device']
        self.dataset = arguments["dataset"]
        self.model = arguments["model"]
        self.criterion = nn.CrossEntropyLoss()
        self.partition_strategy = arguments["partition_strategy"]

    def run(self):
        main_task_acc = []
        fl_iter_list = []
        wg_norm_list = []
        # let's conduct multi-round training
        for flr in range(1, self.fl_round + 1):
            g_user_indices = []

            selected_node_indices = np.random.choice(self.num_nets, size=self.part_nets_per_round, replace=False)
            num_data_points = [len(self.net_dataidx_map[i]) for i in selected_node_indices]
            total_num_dps_per_round = sum(num_data_points)

            net_freq = [num_data_points[i] / total_num_dps_per_round for i in range(self.part_nets_per_round)]   #某节点样本量占总样本量的比例
            logger.info("Net freq: {}, FL round: {} without adversary".format(net_freq, flr))

            # we need to reconstruct the net list at the beginning
            net_list = [copy.deepcopy(self.net_avg) for _ in range(self.part_nets_per_round)]
            logger.info("################## Starting fl round: {}".format(flr))

            # start the FL process
            for net_idx, net in enumerate(net_list):
                global_user_idx = selected_node_indices[net_idx]
                dataidxs = self.net_dataidx_map[global_user_idx]

                train_dl_local, _ = get_dataloader(self.dataset, './data', self.batch_size,
                                                       self.test_batch_size, dataidxs)  # also get the data loader

                g_user_indices.append(global_user_idx)

                logger.info(
                    "@@@@@@@@ Working on client: {}, which is Global user: {}".format(net_idx, global_user_idx))

                optimizer = optim.SGD(net.parameters(), lr=self.lr, momentum=0.9)

                for param_group in optimizer.param_groups:
                    logger.info("Effective lr in FL round: {} is {}".format(flr, param_group['lr']))

                for e in range(1, self.local_training_period + 1):
                    train(self.log_interval ,net, self.device, train_dl_local, optimizer, e, criterion=self.criterion)
                calc_norm_diff(gs_model=net, vanilla_model=self.net_avg, epoch=e, fl_round=flr, mode="normal")

            model_original = list(self.net_avg.parameters())
            v0 = torch.nn.utils.parameters_to_vector(model_original)
            wg_norm_list.append(torch.norm(v0).item())

            # server端整合模型
            self.net_avg = fed_avg_aggregator(net_list, net_freq, device=self.device, model=self.model)

            v = torch.nn.utils.parameters_to_vector(self.net_avg.parameters())
            logger.info("############ Averaged Model : Norm {}".format(torch.norm(v)))

            calc_norm_diff(gs_model=self.net_avg, vanilla_model=self.vanilla_model, epoch=0, fl_round=flr, mode="avg")
            self.vanilla_model = self.net_avg

            logger.info("Measuring the accuracy of the averaged global model, FL round: {} ...".format(flr))

            overall_acc = test(self.net_avg, self.device, self.test_loader, criterion=self.criterion)

            fl_iter_list.append(flr)
            main_task_acc.append(overall_acc)

        df = pd.DataFrame({'fl_iter': fl_iter_list,
                           'main_task_acc': main_task_acc,
                           'wg_norm': wg_norm_list,
                           })

        results_filename = get_results_filename(dataset=self.dataset, model_arch=self.model, epoch=self.fl_round)

        df.to_csv(results_filename, index=False)
        logger.info("Wrote accuracy results to: {}".format(results_filename))