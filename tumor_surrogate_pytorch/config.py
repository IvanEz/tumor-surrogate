import argparse

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

train_arg = add_argument_group('Training')
train_arg.add_argument('--max_epoch', type=int, default=30)
train_arg.add_argument('--train_batch_size', type=int, default=12)
train_arg.add_argument('--val_batch_size', type=int, default=12)
train_arg.add_argument('--lr_max', type=float, default=0.0001)
train_arg.add_argument('--lr_min', type=float, default=0.0000025)
train_arg.add_argument('--optimizer', type=str, default='adam')
train_arg.add_argument('--beta1', type=float, default=0.9)
train_arg.add_argument('--beta2', type=float, default=0.999)
train_arg.add_argument('--weight_decay', type=float, default=4e-20)
train_arg.add_argument('--save_path', type=str, default='./tumor_surrogate_pytorch/saved_model/model')
train_arg.add_argument('--gpu_id', type=str, default='6')

train_arg = add_argument_group('Data')
train_arg.add_argument('--data_path', type=str, default='/mnt/Drive2/ivan/data/')




def get_config():
    config, unparsed = parser.parse_known_args()

    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id  # "0, 1" for multiple

    return config, unparsed