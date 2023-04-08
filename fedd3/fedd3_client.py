from copy import deepcopy
import random
import torch
import os
import numpy as np

from utils.dd_kip import distill_kip_unit
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture


class FedClient(object):

    def __init__(self, client_id, dataset_id='MNIST'):
        """
        Client in the federated learning for FedD3
        :param client_id: Id of the client
        :param dataset_id: Dataset name for the application scenario
        """
        # Metadata
        self._id = client_id
        self._dataset_id = dataset_id

        # Following private parameters are defined by dataset.
        self._image_length = -1
        self._image_width = -1
        self._image_channel = -1

        if self._dataset_id == 'MNIST':
            self._image_length = 28
            self._image_width = 28
            self._image_channel = 1

        elif self._dataset_id == 'FashionMNIST':
            self._image_length = 28
            self._image_width = 28
            self._image_channel = 1

        elif self._dataset_id == 'CIFAR10':
            self._image_length = 32
            self._image_width = 32
            self._image_channel = 3

        elif self._dataset_id == 'CIFAR100':
            self._image_length = 32
            self._image_width = 32
            self._image_channel = 3

        elif self._dataset_id == 'SVHN':
            self._image_length = 32
            self._image_width = 32
            self._image_channel = 3
        else:
            print('unexpected dataset!')
            exit(0)

        # Local dataset
        self._train_data = None
        self._test_data = None

        # Local distilled dataset
        self._distill_data = {'x': [], 'y': []}

    def load_train(self, data):
        """
        Client loads the decentralized dataset, it can be Non-IID across clients.
        :param data: Local dataset for training.
        """
        self._train_data = {}
        self._train_data = deepcopy(data)

    def load_test(self, data):
        """
        Client loads the test dataset.
        :param data: Dataset for testing.
        """
        self._test_data = {}
        self._test_data = deepcopy(data)

    def kip_distill(self, k,
                    num_train_steps=2000,
                    seed=0,
                    lr=4e-3,
                    threshold=0.995,
                    target_sample_size=5000):
        """
        The client run the FedD3 with KIP-based instance.
        More details on KIP in the paper: https://arxiv.org/abs/2011.00050
        :param k: Number of the local distilled images, this need to be integral times of number of local classes
        :param num_train_steps: Number of the decentralized distillation steps
        :param seed: Index of the used seed
        :param lr: Learning rate of decentralized dataset distillation
        :param threshold: Accuracy threshold for decentralized dataset distillation, when it is exceeded, the distillation breaks out.
        :param target_sample_size: Batch size for decentralized dataset distillation
        :return: Distilled images from decentralized dataset in this client
        """
        res = []
        print("Client %s " % self._id +
              "starts distilling %d " % len(self._train_data['y']) +
              "data points into %s data points" % k)

        params = distill_kip_unit(
            np.array(self._train_data['x'].squeeze(1).permute(0, 2, 3, 1)),
            np.array(self._train_data['y'].squeeze()), num_dd_epoch=num_train_steps, seed=seed, k=k, lr=lr,
            threshold=threshold,
            target_sample_size=target_sample_size, kernel_model='FC', depth=4, width=1024)

        for data, data_label in zip(params['x'], params['y']):
            data = np.asarray(data).tolist()
            label = data_label.argmax(0)
            label = np.asarray(label).tolist()
            k_data_point = [label, data, k]
            res.append(k_data_point)
            self._distill_data['y'].append(k_data_point[0])
            self._distill_data['x'].append(k_data_point[1])

        self._distill_data['x'] = torch.tensor(self._distill_data['x']).permute(0, 3, 1, 2)
        self._distill_data['y'] = torch.tensor(self._distill_data['y'])

        return res

    @property
    def all_select(self):
        """
        The client uploads all of the original dataset
        :return: All of the original images
        """
        return self._train_data

    def herding_gmm(self, k, num_local_class, i_seed):
        """
        The client run the FedD3 with coreset-based instance.
        :param k: Number of the local distilled images, this need to be integral times of number of local classes
        :param num_local_class: Number of the local classes
        :param i_seed: Index of the used seed
        :return: Distilled images from decentralized dataset in this client
        """
        torch.manual_seed(i_seed)
        random.seed(i_seed)
        np.random.seed(i_seed)
        res = []
        self._train_data['y'] = self._train_data['y'].squeeze()
        self._train_data['x'] = self._train_data['x'].squeeze()
        num_datapoint = int(k / num_local_class)
        cls_set = set()
        for cls in self._train_data['y']:
            cls_set.add(cls.item())

        for cls in cls_set:
            sub_data = []
            indexes = torch.nonzero(self._train_data['y'] == cls)
            indexes = indexes[torch.randperm(indexes.shape[0])]
            for index in indexes:
                sub_data.append(self._train_data['x'][index].numpy().reshape(-1).tolist())

            gm = GaussianMixture(n_components=int(k / num_local_class), random_state=0).fit(sub_data)
            for x_data in gm.means_:
                k_data_point = [cls, np.array(x_data).reshape(1, 28, 28), k]
                res.append(k_data_point)

        return res

    def dbscan(self, k, num_local_class, i_seed):
        """
        The client run the FedD3 with DBSCAN-based instance.
        :param k: Number of the local distilled images, this need to be integral times of number of local classes
        :param num_local_class: Number of the local classes
        :param i_seed: Index of the used seed
        :return: Distilled images from decentralized dataset in this client
        """
        torch.manual_seed(i_seed)
        random.seed(i_seed)
        np.random.seed(i_seed)
        res = []
        self._train_data['y'] = self._train_data['y'].squeeze()
        self._train_data['x'] = self._train_data['x'].squeeze()
        num_datapoint = int(k / num_local_class)
        cls_set = set()
        for cls in self._train_data['y']:
            cls_set.add(cls.item())

        for cls in cls_set:
            sub_data = []
            indexes = torch.nonzero(self._train_data['y'] == cls)
            indexes = indexes[torch.randperm(indexes.shape[0])]
            for index in indexes:
                sub_data.append(self._train_data['x'][index].numpy().reshape(-1).tolist())
            db = DBSCAN(eps=40.5, min_samples=2).fit(sub_data)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_

            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            print(n_clusters_)
            n_noise_ = list(labels).count(-1)
            unique_labels = set(labels)
            cluster_centers_ = []
            for k in unique_labels:
                # discard the unclustered points
                if k == -1:
                    continue
                class_member_mask = (labels == k)
                cluster = np.array(sub_data)[class_member_mask & core_samples_mask]
                cluster_centers_.append(np.mean(cluster, axis=0))

            for x_data in cluster_centers_:
                k_data_point = [cls, np.array(x_data).reshape(3, 32, 32), k]
                res.append(k_data_point)

        return res

    def save_distilled_dataset(self, exp_dir='client_models', res_root='results'):
        """
        The client saves the distilled images in corresponding directory
        :param exp_dir: Experiment directory name
        :param res_root: Result directory root for saving the result files
        """
        agent_name = 'clients'
        model_save_dir = os.path.join(res_root, exp_dir, agent_name)
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        torch.save(self._distill_data, os.path.join(model_save_dir, self._id + '_distilled_img.pt'))
