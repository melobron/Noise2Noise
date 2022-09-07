import argparse
from train import TrainN2N

# Arguments
parser = argparse.ArgumentParser(description='Train N2N SEM')

parser.add_argument('--exp_detail', default='Train N2N SEM', type=str)
parser.add_argument('--gpu_num', default=0, type=int)
parser.add_argument('--seed', default=100, type=int)

# Training parameters
parser.add_argument('--n_epochs', default=500, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--decay_epoch', default=300, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--input_noise', default='gauss_25', type=str)
parser.add_argument('--target_noise', default='gauss_50', type=str)

# Transformations
parser.add_argument('--crop', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--normalize', type=bool, default=True)
parser.add_argument('--mean', type=float, default=0.4050)  # ImageNet Gray: 0.4050
parser.add_argument('--std', type=float, default=0.2927)  # ImageNet Gray: 0.2927

args = parser.parse_args()

# Train Ne2Ne
train_N2N = TrainN2N(args=args)
train_N2N.train()
