import argparse
import yaml
import os
from datetime import datetime
import torch
from datasets import get_dataset
from utils import set_debug, get_optimizer, get_scheduler
from mae import get_model

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_path', type=str, default=None, help='The folder that has all the datasets.')
    parser.add_argument('--config', '-c', type=str, default='./configs/imagenet.yaml', help="Yaml config file. Don't forget to 'pip install pyyaml' first")
    parser.add_argument('--output_dir', type=str, default='../mae_out/')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--subset_size', type=int, default=2, help='Used along with the --debug command, \
                        run the whole program with only a few samples and find debugs, or the best batch size')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--resume', type=str, default=None, help='path of checkpoint')
    args = parser.parse_args()
    
    # move configs to arguments
    with open(args.config, 'r') as f:
        for key, value in yaml.load(f, Loader=yaml.FullLoader).items():
            vars(args)[key] = value
    
    # config data path in command 
    if args.data_path is not None: 
        args.dataset.root = args.data_path
    
    # add timestamp to output folder.
    args.output_dir = os.path.join(os.path.abspath(args.output_dir), 
        *args.config.rstrip('.yaml').split('/')[1:])+\
        '-'+datetime.now().strftime('%m%d%H%M%S')
    
    # create our output folder
    os.makedirs(args.output_dir, exist_ok=False)
    print(f'Outputs will be saved to {args.output_dir}')

    return args
        

def main(args):
    print(f'Found {torch.cuda.device_count()} gpu(s)')
    train_set = get_dataset(
        split='train',
        image_size=args.image_size,
        **args.dataset
    )
    test_set = get_dataset(
        split='test',
        image_size=args.image_size,
        **args.dataset
    )
    # args.scheduler['lr'] = args.scheduler['lr'] * args.train_loader['batch_size'] / 256
    
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

    # breakpoint()

    model = get_model(image_size=args.image_size, **args.model).to(args.device)
    model = torch.nn.parallel.DataParallel(model)
    
    optimizer = get_optimizer(model, **args.optimizer)
    scheduler = get_scheduler(**args.scheduler)
    # optimizer = torch.optim.AdamW([{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': 5e-2}],
    #                                 lr=0)
    # lr_scheduler = LR_Scheduler(warmup_epochs=warmup_epochs, base_lr=lr)
    
    scheduler.set_optimizer(optimizer)
    
    criterion = torch.nn.CrossEntropyLoss()

    args.param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Param num: {args.param_num}')
    
    if args.resume is not None:
        state_dict = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(state_dict)

    # max_acc = -1
    # max_acc_ep = 0
    model_to_save = None
    training_stat = []
    test_stat = []
    for epoch in range(args.epochs):

        # epoch_data = run_epoch(model, optimizer, scheduler, train_loader, args.device, epoch, args.epochs, criterion)
        model.train()
        # iter_pbar = tqdm(, desc=f'Epoch {epoch}/{num_epochs}', disable=disable_tqdm, ncols=0)
        epoch_list = []
        train_loss = []
        for idx, (images, labels) in enumerate(train_loader):
            lr = scheduler.step(epoch, args.epochs, idx, len(train_loader))
            model.zero_grad()

            images, labels = images.to(args.device), labels.to(args.device)

            out = model(images)

            out['loss'].mean().backward()
            train_loss.append(out['loss'])
            optimizer.step()
            
            data_dict = {'lr':lr, **{key:value.item() for key, value in out.items() if value.ndim == 0}}
            # iter_pbar.set_postfix(data_dict)
            # print(data_dict)
            epoch_list.append(data_dict)
        avg_loss = sum(train_loss)/len(train_loss)
        print(f'Epoch {epoch} Train loss {avg_loss}')
        training_stat.append(epoch_list)
        
    # print(f'Max acc = ', max_acc)
    
    model_to_save = model
    model_to_save = model_to_save.module if hasattr(model_to_save, "module") else model_to_save
    torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, f'ep{epoch}_loss{avg_loss:.2f}.pth'))
    
    
    with open(os.path.join(args.output_dir, 'training_stats.yaml'), 'w') as file:        
        yaml.dump(training_stat, file, default_flow_style=False)
    with open(os.path.join(args.output_dir, 'test_stat.yaml'), 'w') as file:        
        yaml.dump(test_stat, file, default_flow_style=False)

    # copy our configurations to output
    with open(os.path.join(args.output_dir, 'configs.yaml'), 'w') as file: 
        yaml.dump(args.__dict__, file, default_flow_style=False)

    print(f'Output has been saved to {args.output_dir}')






















if __name__ == '__main__':
    main(get_args())




