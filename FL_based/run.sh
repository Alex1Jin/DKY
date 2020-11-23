python parameter.py \
--dataset mnist \
--model lenet \
--fl_round 10 \
--num_nets 10 \
--part_nets_per_round 10 \
--lr 0.01 \
--batch_size 128 \
--local_train_period 2 \
--device=cuda:0