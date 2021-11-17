import torch

def use_subset(dataset, subset_size):
    dataset = torch.utils.data.Subset(dataset, range(subset_size))
    try:
        dataset.classes, dataset.targets = dataset.dataset.classes, dataset.dataset.targets
    except AttributeError:
        print('Missing attributes in dataset')
    return dataset

def set_debug(args, *datasets):
    
    if args.debug:
        
        # loader_config = dict(
        #     batch_size=debug_size,
        # )
        # args.train_loader.update(loader_config)
        # args.test_loader.update(loader_config)
    
        # train_set = use_subset(train_set, args.train_loader['batch_size'])
        # test_set = use_subset(test_set, args.test_loader['batch_size'])
        args.train_loader['batch_size'] = args.subset_size
        args.test_loader['batch_size'] = args.subset_size
        subsets = []
        for dataset in datasets:
            subsets.append(use_subset(dataset, args.subset_size))
        args.epochs = 1
        args.eval_epochs = 1
        args.num_workers = 0
    return args, *subsets


class LR_Scheduler():
    def __init__(self, warmup_epochs, base_lr, warmup_lr=0, final_lr=0, optimizer=None):
        
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        
        if optimizer is not None: self.set_optimizer(optimizer)
        
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr
            

    def step(self, current_epoch, total_epochs, current_iter, iters_per_epoch):
        lr = self.base_lr
        total_iters = total_epochs * iters_per_epoch
        warmup_iters = self.warmup_epochs * iters_per_epoch
        current_iter = current_epoch * iters_per_epoch + current_iter
        if current_iter < warmup_iters:
            lr = current_iter * (self.base_lr / warmup_iters)
        else:
            lr *= 0.5 * (1. + math.cos(math.pi * (current_iter-warmup_iters) / (total_iters-warmup_iters)))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
def get_scheduler(name='cosine', lr=None, warmup_epochs=0):
  
    if name == 'cosine':
        lr_scheduler = LR_Scheduler(warmup_epochs=warmup_epochs, base_lr=lr)
        
    return lr_scheduler
def get_optimizer(model, name, **optimizer_kwargs): # lr_decay='cosine', warmup_epochs=0, 
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """
    skip = {"classtoken", "pos"}
    decay, no_decay = [], []
    for param_name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or param_name.endswith(".bias") or param_name in skip:
            no_decay.append(param)
        else:
            decay.append(param)
    
    if name == 'AdamW':
        optimizer = torch.optim.AdamW([{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': 5e-2}],
                                    lr=0)
    else:
        raise NotImplementedError
        
    return optimizer
    
