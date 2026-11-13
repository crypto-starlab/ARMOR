from torchvision import transforms, datasets
from torch.utils.data import Dataset
import torch
import os
from pathlib import Path
from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet34, resnet18

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, num_classes=None):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.num_classes = num_classes

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # These datasets return integer labels; DataLoader converts the batch
        # to int64 once, without constructing a tensor for every sample.
        return image, label

    
class LeNet5(nn.Module):
   def __init__(self):
       super().__init__()
       self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
       self.conv2 = nn.Conv2d(6, 16, 5)
       self.fc1 = nn.Linear(16*5*5, 120)
       self.fc2 = nn.Linear(120, 84)
       self.fc3 = nn.Linear(84, 10)
       
   def forward(self, x):
       x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
       x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
       x = x.view(-1, self.num_flat_features(x))
       x = F.relu(self.fc1(x))
       x = F.relu(self.fc2(x))
       x = self.fc3(x)
       return x
       
   def num_flat_features(self, x):
       size = x.size()[1:]
       num_features = 1
       for s in size:
           num_features *= s
       return num_features


def load_data(args):

    if args.dataset in ["MNIST", "FMNIST"]:
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    elif args.dataset in ["CIFAR10"]:
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    data_path = str(Path(__file__).resolve().parent.parent / 'data')
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    dataset_val = [] 

    if args.dataset == "CIFAR10":
        dataset_train = datasets.CIFAR10(data_path, train=True, download=True, transform=trans)
        dataset_test = datasets.CIFAR10(data_path, train=False, download=True, transform=trans)
    elif args.dataset == "MNIST":
        dataset_train = datasets.MNIST(data_path, train=True, download=True, transform=trans)
        dataset_test = datasets.MNIST(data_path, train=False, download=True, transform=trans)
    elif args.dataset == "FMNIST":
        dataset_train = datasets.FashionMNIST(data_path, train=True, download=True, transform=trans)
        dataset_test = datasets.FashionMNIST(data_path, train=False, download=True, transform=trans)
    
    return dataset_train, dataset_test, dataset_val


def evaluate_weight(net, dataset_test, device, batch_size=32):
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    
    net.eval()
    net.to(device)
    correct, total, test_loss = 0, 0, 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"Accuracy = {acc*100:.2f}% | Loss = {test_loss/len(test_loader):.4f}")
        
    return acc

def init_model(args):
    if args.dataset in ["CIFAR10", "MNIST", "FMNIST"]:
        args.num_classes = 10
    if args.dataset == "MNIST":
        net = LeNet5().to(args.device) 
    elif args.dataset == "FMNIST":
        net = resnet18(num_classes=args.num_classes).to(args.device)
        net.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) 
    elif args.dataset == "CIFAR10":
        net = resnet34(num_classes=args.num_classes).to(args.device)
    net.train()
    return net
