import torchvision
import torchvision.transforms as T
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
# filter the annoying warning of torchvision.datasets.ImageNet that tells you downloading is not working

mean_std = dict(
    imagenet=dict(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    cifar10=dict(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    ),
    cifar100=dict(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761]
    )
)

def get_dataset(name, root, split, image_size, download=False, folder_name=None):

    transform = T.Compose([
        T.RandomResizedCrop(image_size) if split=='train' else T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(**mean_std[name])
    ])

    if name == 'cifar10': # make sure cifar-10-batches-py is in "root"
        assert os.path.exists(os.path.join(root, 'cifar-10-batches-py'))
        dataset = torchvision.datasets.CIFAR10(root, train=split=='train', transform=transform, download=download)
        
    elif name == 'cifar100': # make sure cifar-100-python is in 'root'
        assert os.path.exists(os.path.join(root, 'cifar-100-python'))
        dataset = torchvision.datasets.CIFAR100(root, train=split=='train', transform=transform, download=download)
        
    elif name == 'imagenet': 
        dataset = torchvision.datasets.ImageNet(
            os.path.join(root, 'ImageNet' if not folder_name else folder_name), 
            split=split if split=='train' else 'val', transform=None, download=download)

    else:
        raise NotImplementedError

    return dataset
























