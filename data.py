from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class DatasetLoader:
    def __init__(self, dataset_name, batch_size=128, test_batch_size=1024):
        self.dataset_name = dataset_name
        self.root_dir = './data'
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        ])
        self.test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        ])
        

    def load_data(self):
        if self.dataset_name.lower() == 'cifar10':
            trainset = datasets.CIFAR10(root=self.root_dir, train=True, download=True, transform=self.train_transform)
            testset = datasets.CIFAR10(root=self.root_dir, train=False, download=True, transform=self.test_transform)
        
        # Add more datasets here if needed
        else:
            return None, None

        trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        testloader = DataLoader(testset, batch_size=self.test_batch_size, shuffle=False)
        return trainloader, testloader