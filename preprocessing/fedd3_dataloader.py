import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import os


def load_data(name, root='./data', download=True, save_pre_data=True):

    data_dict = ['MNIST', 'EMNIST', 'FashionMNIST', 'CelebA', 'CIFAR10', 'QMNIST', 'SVHN', 'CIFAR100']
    assert name in data_dict, "The dataset is not present"

    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)

    file_dir = root+'/prepared/'
    test_data_file = file_dir + name + '_test.pt'
    train_data_file = file_dir + name + '_train.pt'
    test_targets_file = file_dir + name + '_test_label.pt'
    train_targets_file = file_dir + name + '_train_label.pt'
    
    if not os.path.exists(file_dir):
        os.makedirs(file_dir, exist_ok=True)

    all_file_there = os.path.exists(train_data_file) and os.path.exists(test_data_file) and os.path.exists(train_targets_file) and os.path.exists(test_targets_file)

    if save_pre_data or not all_file_there:

        if name == 'MNIST':
            transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
            trainset = torchvision.datasets.MNIST(root=root, train=True, download=download, transform=transform)
            testset = torchvision.datasets.MNIST(root=root, train=False, download=download, transform=transform)
        
        elif name == 'EMNIST':
            # byclass, bymerge, balanced, letters, digits, mnist
            transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
            trainset = torchvision.datasets.EMNIST(root=root, train=True, split= 'letters', download=download, transform=transform)
            testset = torchvision.datasets.EMNIST(root=root, train=False, split= 'letters', download=download, transform=transform)

        elif name == 'FashionMNIST':
            transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
            trainset = torchvision.datasets.FashionMNIST(root=root, train=True, download=download, transform=transform)
            testset = torchvision.datasets.FashionMNIST(root=root, train=False, download=download, transform=transform)

        elif name == 'CelebA':
            # Could not loaded possibly for google drive break downs, try again at week days
            target_transform = transforms.Compose([transforms.ToTensor()])
            transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            trainset = torchvision.datasets.CelebA(root=root, split='train', target_type=list, download=download, transform=transform, target_transform=target_transform)
            testset = torchvision.datasets.CelebA(root=root, split='test', target_type=list, download=download, transform=transform, target_transform=target_transform)

        elif name == 'CIFAR10':
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
            trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=download, transform=transform)
            testset = torchvision.datasets.CIFAR10(root=root, train=False, download=download, transform=transform)
            trainset.targets = torch.Tensor(trainset.targets)
            testset.targets = torch.Tensor(testset.targets)

        elif name == 'CIFAR100':
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            trainset = torchvision.datasets.CIFAR100(root=root, train=True, transform=transform, download=True)
            testset = torchvision.datasets.CIFAR100(root=root, train=False, transform=transform, download=True)
            trainset.targets = torch.Tensor(trainset.targets)
            testset.targets = torch.Tensor(testset.targets)

        elif name == 'QMNIST':
            transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
            trainset = torchvision.datasets.QMNIST(root=root, what='train', compat=True, download=download, transform=transform)
            testset = torchvision.datasets.QMNIST(root=root, what='test', compat=True, download=download, transform=transform)

        elif name == 'SVHN':

            transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))])
            trainset = torchvision.datasets.SVHN(root=root, split='train', download=download, transform=transform)
            testset = torchvision.datasets.SVHN(root=root, split='test', download=download, transform=transform)
            trainset.targets = torch.Tensor(trainset.labels)
            testset.targets = torch.Tensor(testset.labels)

        end = len(trainset)-1
        copy_burden = 50
        for i,x in tqdm(enumerate(trainset)):
            if i == end:
                temp = torch.cat((temp, torch.unsqueeze(x[0],0)))
                train_data = torch.cat((train_data, temp))
                train_targets = trainset.targets

            elif i%copy_burden != 0:
                temp = torch.cat((temp, torch.unsqueeze(x[0],0)))
            
            elif i/copy_burden != 0 and i/copy_burden != 1:
                train_data = torch.cat((train_data, temp))
                temp = torch.unsqueeze(x[0], 0)

            elif i/copy_burden == 0:
                temp = torch.unsqueeze(x[0], 0)

            else:
                train_data = temp
                temp = torch.unsqueeze(x[0], 0)

        end = len(testset)-1
        for i,x in tqdm(enumerate(testset)):
            if i == end:
                temp = torch.cat((temp, torch.unsqueeze(x[0],0)))
                test_data = torch.cat((test_data, temp))
                test_targets = testset.targets

            elif i%copy_burden != 0:
                temp = torch.cat((temp, torch.unsqueeze(x[0],0)))
            
            elif i/copy_burden != 0 and i/copy_burden != 1:
                test_data = torch.cat((test_data, temp))
                temp = torch.unsqueeze(x[0], 0)

            elif i/copy_burden == 0:
                temp = torch.unsqueeze(x[0], 0)

            else:
                test_data = temp
                temp = torch.unsqueeze(x[0], 0)

        torch.save(train_data, train_data_file)
        torch.save(test_data, test_data_file)
        torch.save(train_targets, train_targets_file)
        torch.save(test_targets, test_targets_file)

    elif all_file_there and not save_pre_data:
        train_data = torch.load(train_data_file)
        test_data = torch.load(test_data_file)
        train_targets = torch.load(train_targets_file)
        test_targets = torch.load(test_targets_file)

    # Set len_classes
    len_classes_dict = {
        'MNIST': 10,
        'EMNIST': 27, # ByClass: 62. ByMerge: 814,255 47.Digits: 280,000 10.Letters: 145,600 26.MNIST: 70,000 10.
        'FashionMNIST': 10,
        'CelebA': 0,
        'CIFAR10': 10,
        'CIFAR100': 100,
        'QMNIST': 10,
        'SVHN': 10
    }

    len_classes = len_classes_dict[name]
    
    return len_classes, train_data, train_targets, test_data, test_targets


def divide_data(num_client=1, num_local_class=10, dataset_name='emnist', i_seed=0):
    torch.manual_seed(i_seed)

    num_classes, train_data, train_targets, test_data, test_targets = load_data(dataset_name, download=True, save_pre_data=False)

    # import pdb; pdb.set_trace()
    if num_local_class == -1:
        num_local_class = num_classes
    assert 0 < num_local_class <= num_classes, "number of local class should smaller than global number of class"

    trainset_config = {'users': [],
                       'user_data': {},
                       'num_samples': []}
    config_division = {}  # Count of the classes for division
    config_class = {}  # Configuration of class distribution in clients
    config_data = {}  # Configuration of data indexes for each class : Config_data[cls] = [0, []] | pointer and indexes

    for i in range(num_client):
        config_class['f_{0:05d}'.format(i)] = []
        for j in range(num_local_class):
            cls = (i+j) % num_classes
            if cls not in config_division:
                config_division[cls] = 1
                config_data[cls] = [0, []]

            else:
                config_division[cls] += 1
            config_class['f_{0:05d}'.format(i)].append(cls)

    for cls in config_division.keys():
        indexes = torch.nonzero(train_targets == cls)
        num_datapoint = indexes.shape[0]
        indexes = indexes[torch.randperm(num_datapoint)]
        num_partition = num_datapoint // config_division[cls]
        for i_partition in range(config_division[cls]):
            if i_partition == config_division[cls] - 1:
                config_data[cls][1].append(indexes[i_partition * num_partition:])
            else:
                config_data[cls][1].append(indexes[i_partition * num_partition: (i_partition + 1) * num_partition])

    for user in tqdm(config_class.keys()):
        user_data_indexes = torch.tensor([])
        for cls in config_class[user]:
            user_data_index = config_data[cls][1][config_data[cls][0]]
            user_data_indexes = torch.cat((user_data_indexes, user_data_index))
            config_data[cls][0] += 1
            # print(len(user_data_indexes))
        user_data = train_data[user_data_indexes.tolist()]
        user_targets = train_targets[user_data_indexes.tolist()]
        trainset_config['users'].append(user)
        trainset_config['user_data'][user] = {'x': user_data, 'y': user_targets}
        trainset_config['num_samples'] = len(user_data)
        # print(trainset_config['user_data'][user].shape)

    test_iid_data = {'x': None, 'y': None}
    test_iid_data['x'] = test_data
    test_iid_data['y'] = test_targets

    return trainset_config, test_iid_data


if __name__ == "__main__":
    # 'MNIST', 'EMNIST', 'FashionMNIST', 'CelebA', 'CIFAR10', 'QMNIST', 'SVHN'
    data_dict = ['MNIST', 'EMNIST', 'FashionMNIST', 'CIFAR10', 'QMNIST', 'SVHN']

    divide_data(num_client=20, num_local_class=-1, dataset_name='CIFAR10', i_seed=0)
