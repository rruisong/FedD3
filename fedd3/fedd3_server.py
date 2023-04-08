import os
import json
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
from json import JSONEncoder
from tqdm import tqdm
from utils.models import *
from copy import deepcopy

from postprocessing.recorder import Recorder

json_types = (list, dict, str, int, float, bool, type(None))


class PythonObjectEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, json_types):
            return super().default(self, obj)
        return {'_python_object': pickle.dumps(obj).decode('latin-1')}


def as_python_object(dct):
    if '_python_object' in dct:
        return pickle.loads(dct['_python_object'].encode('latin-1'))
    return dct


class FedServer(object):
    def __init__(self, epoch, batch_size, lr, momentum, num_workers, dataset_id='mnist', server_id='server', model_name="LeNet", i_seed=0, test_on_gpu=True):
        """
        Server in the federated learning for FedD3
        :param epoch: Number of total training epochs in the server
        :param batch_size: Batch size for the training in the server
        :param lr: Learning rate for the training in the server
        :param momentum: Learning momentum for the training in the server
        :param num_workers: Number of the workers for the training in the server
        :param dataset_id: Dataset name for the application scenario
        :param server_id: Id of the server
        :param model_name: Machine learning model name for the application scenario
        :param i_seed: Index of the seed used in the experiment
        :param test_on_gpu: Flag: 1: Run testing on GPU after every epoch, otherwise 0.
        """
        data_dict = ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN', 'CIFAR100']
        assert dataset_id in data_dict, "The dataset is not present"

        self.test_on_gpu = test_on_gpu

        # Server Properties
        self._id = server_id
        self._dataset_id = dataset_id
        self._model_name = model_name
        self._i_seed = i_seed

        # Training related parameters
        self._epoch = epoch
        self._batch_size = batch_size
        self._lr = lr
        self._momentum = momentum
        self._num_workers = num_workers

        # Global test dataset
        self._test_data = None

        # Global distilled dataset
        self._distill_data = None

        # Following private parameters are defined by dataset.
        self.model = None
        self._image_dim = -1
        self._image_channel = -1

        if self._dataset_id == 'MNIST':
            self._num_class = 10
            self._image_dim = 28
            self._image_channel = 1

        if self._dataset_id == 'FashionMNIST':
            self._num_class = 10
            self._image_dim = 28
            self._image_channel = 1

        elif self._dataset_id == 'SVHN':
            self._num_class = 10
            self._image_dim = 32
            self._image_channel = 3

        elif self._dataset_id == 'EMNIST':
            self._num_class = 27
            self._image_dim = 28
            self._image_channel = 1

        elif self._dataset_id == 'CIFAR10':
            self._num_class = 10
            self._image_dim = 32
            self._image_channel = 3

        elif self._dataset_id == 'CIFAR100':
            self._num_class = 100
            self._image_dim = 32
            self._image_channel = 3

        if self._model_name == "ResNet18":
            self.model = generate_resnet(num_classes=self._num_class, in_channels=self._image_channel ,model_name=model_name)
        elif self._model_name == "ResNet50":
            self.model = generate_resnet(num_classes=self._num_class, in_channels=self._image_channel ,model_name=model_name)
        elif self._model_name == "ResNet152":
            self.model = generate_resnet(num_classes=self._num_class, in_channels=self._image_channel, model_name=model_name)
        elif self._model_name == "LeNet":
            self.model = LeNet(self._num_class, self._image_channel)
        elif self._model_name == "VGG11":
            self.model = generate_vgg(self._num_class, self._image_channel, model_name=model_name)
        elif self._model_name == "VGG11_bn":
            self.model = generate_vgg(self._num_class, self._image_channel, model_name=model_name)
        elif self._model_name == "AlexCifarNet":
            self.model = AlexCifarNet()
        elif self._model_name == "CNN":
            self.model = CNN(self._num_class, self._image_channel)
        else:
            print('Model is not supported')

        # Number of model parameter
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.param_len = sum([np.prod(p.size()) for p in model_parameters])
        print('Number of model parameters of %s :' % model_name, ' %d ' % self.param_len)
        # Recording results
        self.recorder = Recorder()
        # Run on the GPU
        gpu = 0
        self._device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")

    def load_test(self, data):
        """
        Server loads the test dataset.
        :param data: Dataset for testing.
        """
        self._test_data = {}
        self._test_data = deepcopy(data)

    def load_distill(self, data):
        """
        Server loads the decentralized distilled dataset.
        :param data: Dataset for training.
        """
        self._distill_data = {}
        self._distill_data = deepcopy(data)

    def train(self, exp_dir, res_root='results', i_seed=0):
        """
        Server trains models on the decentralized distilled datasets from networks
        :param exp_dir: Experiment directory name
        :param res_root: Result directory root for saving the result files
        :param i_seed: Index of the used seed
        :return: Loss in the training.
        """
        torch.manual_seed(i_seed)
        np.random.seed(i_seed)
        state_dict_list = []

        # Create the train and test loader
        with torch.no_grad():

            train_x = self._distill_data['x'].type(torch.FloatTensor).squeeze()
            if len(train_x.shape) == 3:
                train_x = train_x.unsqueeze(1)
            train_y = self._distill_data['y'].type(torch.LongTensor).squeeze()

            test_x = self._test_data['x'].type(torch.FloatTensor).squeeze()
            if len(test_x.shape) == 3:
                test_x = test_x.unsqueeze(1)
            test_y = self._test_data['y'].type(torch.FloatTensor).squeeze()

            train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=self._batch_size, shuffle=True)
            test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=self._batch_size, shuffle=True)

            self.model.to(self._device)
            # optimizer = torch.optim.SGD(self.model.parameters(), lr=self._lr, momentum=self._momentum)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr, weight_decay=1e-4)
            loss_func = nn.CrossEntropyLoss()

        # Train process
        pbar = tqdm(range(self._epoch))
        for epoch in pbar:
            for step, (x, y) in enumerate(train_loader):

                with torch.no_grad():
                    b_x = x.to(self._device)  # Tensor on GPU
                    b_y = y.to(self._device)  # Tensor on GPU

                with torch.enable_grad():
                    self.model.train()
                    output = self.model(b_x)
                    loss = loss_func(output, b_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # Recording the train loss during the training
            self.recorder.res['server']['train_loss'].append(loss.data.cpu().numpy())

            # Test process
            if self.test_on_gpu:
                accuracy_collector = 0
                for step, (x, y) in enumerate(test_loader):
                    with torch.no_grad():
                        b_x = x.to(self._device)  # Tensor on GPU
                        b_y = y.to(self._device)  # Tensor on GPU
                        test_output = self.model(b_x)
                        pred_y = torch.max(test_output, 1)[1].to(self._device).data.squeeze()
                        accuracy_collector = accuracy_collector + sum(pred_y == b_y)
                accuracy = accuracy_collector / self._test_data['y'].size(0)
                self.recorder.res['server']['iid_accuracy'].append(accuracy.cpu().numpy())

                pbar.set_description('Epoch: %d' % epoch +
                                     '| Train loss: %.4f ' % loss.data.cpu().numpy() +
                                     '| Accuracy: %.4f' % accuracy +
                                     '| Max Acc: %.4f' % np.max(np.array(self.recorder.res['server']['iid_accuracy'])))
            else:
                pbar.set_description('Epoch: %d', epoch + '| Train loss: %.4f ' % loss.data.cpu().numpy())
            state_dict_list.append(self.model.state_dict())

        # Save the results
        if not os.path.exists(res_root):
            os.makedirs(res_root)

        with open(os.path.join(res_root, exp_dir), "w") as jsfile:
            json.dump(self.recorder.res, jsfile, cls=PythonObjectEncoder)

        return loss.data.cpu().numpy()
