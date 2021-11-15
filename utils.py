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
        for i, dataset in enumerate(datasets):
            datasets[i] = use_subset(dataset, args.debug_size)
        args.epochs = 1
        args.eval_epochs = 1
    return args, *datasets
