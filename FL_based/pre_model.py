import argparse
import torch
import torch.nn as nn   # torch.nn是专门为神经网络设置的模块化接口
import torch.nn.functional as F  # 对于激活函数和池化层，由于没有可学习参数，一般使用该模块完成
import torch.optim as optim  # 用于优化参数
from torchvision import datasets, transforms  # 定义了datasets的数据格式及常用的数据转换方式

from model import lenet

def train(args, model, device, train_loader, optimizer, epoch, criterion):

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
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader, criterion):
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

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--dataset', type=str, default='MNIST',
                        help='dataset to use during the training process')
    parser.add_argument('--model', type=str, default='LeNet',
                        help='model to use during the training process')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device to set, can take the value of: cuda or cuda:x')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if use_cuda else "cpu")
    kwargs = {'num_workers': 30, 'pin_memory': False} if use_cuda else {}

    # prepare dataset
    if args.dataset == "MNIST":
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, download=True,
                           transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

    else:
        pass


    # prepare model
    if args.model == "LeNet":
        model = lenet.LeNet().to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        pass

    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, criterion)  # train
        test(args, model, device, test_loader, criterion)  # test

        if (args.save_model):
            torch.save(model.state_dict(), "mnist_cnn.pt")

if __name__ == '__main__':
    main()