import argparse
import yaml
import os
import datetime
import torch
from datasets import get_dataset
from utils import set_debug

def get_args():
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10/cifar100/stl10/imagenet')
    parser.add_argument('--config', '-c', type=str, default='./configs/imagenet.yaml', help="Yaml config file. Don't forget to 'pip install pyyaml' first")
    parser.add_argument('--output_dir', type=str, default='../mae_out/')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_size', type=int, default=2, help='Used along with the --debug command, \
                        run the whole program with only a few sample and find debugs or best batch size')
    parser.add_argument('--epochs', type=int, default=200)
    
    args = parser.parse_args()
    # move configs to arguments
    with open(args.config, 'r') as f:
        for key, value in yaml.load(f, Loader=yaml.FullLoader).items():
            vars(args)[key] = value
    # config data path in command 
    if args.data_path is not None: 
        args.dataset.path = args.data_path
    # add timestamp to output folder. use abspath
    args.output_dir = os.path.join(args.output_dir, 
        *args.config_file.rstrip('.yaml').split('/')[1:])+\
        '-'+datetime.now().strftime('%m%d%H%M%S')
    # create our output folder
    os.makedirs(args.output_dir, exist_ok=False)
    print(f'Outputs will be saved to {os.path.abspath(args.output_dir)}') # useful when scp output folder
    with open(os.path.join(args.output_dir, 'configs.yaml'), 'w') as file: # copy our configurations to output
        yaml.dump(args.__dict__, file, default_flow_style=False)
    
    return args
        

def main(args):
    print(f'Found {torch.cuda.device_count()} gpu(s)')
    train_set = get_dataset(
        data_dir=args.data_dir,
        split='train',
        **args.dataset
    )
    test_set = get_dataset(
        data_dir=args.data_dir,
        split='test',
        **args.dataset
    )
    args.scheduler['lr'] = args.scheduler['lr'] * args.train_loader['batch_size'] / 256
    
    args, train_set, test_set = set_debug(args, train_set, test_set)
    
    train_loader = torch.utils.data.dataloader.DataLoader(
        dataset=train_set,
        num_workers=args.num_workers,
        **args.train_loader
    )
    test_loader = torch.utils.data.dataloader.DataLoader(
        dataset=test_set,
        num_workers=args.num_workers,
        **args.test_loader
    )

























if __name__ == '__main__':
    main(get_args())




