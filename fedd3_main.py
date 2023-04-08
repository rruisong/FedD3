#!/usr/bin/env python3
import torch
import random
import numpy as np
import argparse
from preprocessing.fedd3_dataloader import divide_data
from fedd3.fedd3_client import FedClient
from fedd3.fedd3_server import FedServer

torch.cuda.empty_cache()


def fed_args():
    """
    Arguments for running FedD3
    :return: Arguments for FedD3
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-nc', '--sys-n_client', type=int, required=True, help='Number of the clients')
    parser.add_argument('-ck', '--sys-n_local_class', type=int, required=True, help='Number of the classes in each client')
    parser.add_argument('-ds', '--sys-dataset', type=str, required=True, help='Dataset name, one of the following four datasets: MNIST, CIFAR-10, FashionMnist, SVHN')
    parser.add_argument('-md', '--sys-model', type=str, required=True, help='Model name')
    parser.add_argument('-is', '--sys-i_seed', type=int, required=True, help='Seed used in experiments')
    parser.add_argument('-rr', '--sys-res_root', type=str, required=True, help='Root directory of the results')

    parser.add_argument('-sne', '--server-n_epoch', type=int, required=True, help='Number of training epochs in the server')
    parser.add_argument('-sbs', '--server-bs', type=int, required=True, help='Batch size in the server')
    parser.add_argument('-slr', '--server-lr', type=float, required=True, help='Learning rate in the server')
    parser.add_argument('-smt', '--server-momentum', type=float, required=True, help='Momentum in the server')
    parser.add_argument('-snw', '--server-n_worker', type=int, required=True, help='Number of workers in the server')

    parser.add_argument('-cis', '--client-instance', type=str, required=True, help='Instance used in clients')
    parser.add_argument('-cnd', '--client-n_dd', type=int, required=True, help='Number of distilled images in clients')
    parser.add_argument('-cil', '--client-instance_lr', type=float, required=True, help='Learning rate in clients')
    parser.add_argument('-cib', '--client-instance_bs', type=int, required=True, help='Batch size in clients')
    parser.add_argument('-cie', '--client-instance_max_n_epoch', type=int, required=True, help='Maximal number of epochs in clients')
    parser.add_argument('-cit', '--client-instance_threshold', type=float, required=True, help='Accuracy threshold for dataset distillation in clients')

    args = parser.parse_args()
    return args


def main():
    """
    Main function for FedD3
    """
    args = fed_args()

    mode_list = ["all_select", "kip_distill", "gmm", "dbscan"]
    assert args.client_instance in mode_list, "The mode is not supported"

    dataset_list = ['MNIST', 'CIFAR10', 'FashionMNIST', 'SVHN', 'CIFAR100']
    assert args.sys_dataset in dataset_list, "The dataset is not supported"

    model_list = ["LeNet", 'AlexCifarNet', 'CNN', 'ResNet18', 'ResNet50', "ResNet152"]
    assert args.sys_model in model_list, "The model is not supported"

    # Number of all distilled data points
    num_images = int(args.client_n_dd * args.sys_n_client)

    # Set the experiment name with hyperparameters
    exp_name = '["%s","%s",%d,%d,%d,%d]' % (
        args.sys_dataset, args.sys_model, num_images, args.client_n_dd, args.sys_n_client, args.sys_n_local_class)

    if args.client_instance == "all_select":
        args.sys_n_client = 1
        args.sys_n_local_class = -1

    torch.manual_seed(args.sys_i_seed)
    random.seed(args.sys_i_seed)
    np.random.seed(args.sys_i_seed)

    client_dict = {}
    distill_dataset = {'x': [], 'y': []}

    # Distribute the dataset into each client with respect to number of local classes
    trainset_config, test_iid_data = divide_data(num_client=args.sys_n_client,
                                                 num_local_class=args.sys_n_local_class,
                                                 dataset_name=args.sys_dataset,
                                                 i_seed=args.sys_i_seed)

    # Initialize each client and distill the local data.
    # (Since it is one-shot, initialization does not have to do separately)
    for client_id in trainset_config['users']:
        client_dict[client_id] = FedClient(client_id, dataset_id=args.sys_dataset)
        client_dict[client_id].load_train(trainset_config['user_data'][client_id])

        ret_data = []
        if args.client_instance == "all_select":
            distill_dataset = client_dict[client_id].all_select
        elif args.client_instance == "gmm":
            ret_data = client_dict[client_id].herding_gmm(k=args.client_n_dd, num_local_class=args.sys_n_local_class, i_seed=args.sys_i_seed)
        elif args.client_instance == "dbscan":
            ret_data = client_dict[client_id].dbscan(k=args.client_n_dd, num_local_class=args.sys_n_local_class, i_seed=args.sys_i_seed)
        elif args.client_instance == "kip_distill":
            ret_data = client_dict[client_id].kip_distill(
                args.client_n_dd,
                num_train_steps=args.client_instance_max_n_epoch,
                seed=args.sys_i_seed,
                lr=args.client_instance_lr,
                threshold=args.client_instance_threshold,
                target_sample_size=args.client_instance_bs)
        for k_data_point in ret_data:
            distill_dataset['y'].append(k_data_point[0])
            distill_dataset['x'].append(k_data_point[1])

    if args.client_instance == "all_select":
        distill_dataset['x'] = torch.tensor(distill_dataset['x'])
        distill_dataset['x'] = distill_dataset['x'].squeeze()
        distill_dataset['y'] = torch.tensor(distill_dataset['y'])
    elif args.client_instance == "gmm":
        distill_dataset['x'] = torch.tensor(distill_dataset['x'])
        distill_dataset['y'] = torch.tensor(distill_dataset['y'])
    elif args.client_instance == "dbscan":
        distill_dataset['x'] = torch.tensor(distill_dataset['x'])
        distill_dataset['y'] = torch.tensor(distill_dataset['y'])
    elif args.client_instance == "kip_distill":
        distill_dataset['x'] = torch.tensor(distill_dataset['x'])
        distill_dataset['x'] = distill_dataset['x'].permute(0, 3, 1, 2)
        distill_dataset['y'] = torch.tensor(distill_dataset['y'])

    # Initialize the server in federated learning
    server = FedServer(epoch=args.server_n_epoch,
                       batch_size=args.server_bs,
                       lr=args.server_lr, momentum=args.server_momentum,
                       num_workers=args.server_n_worker,
                       server_id='server',
                       dataset_id=args.sys_dataset,
                       model_name=args.sys_model,
                       i_seed=args.sys_i_seed,
                       test_on_gpu=True)

    # Server loads non-iid test dataset
    server.load_test(test_iid_data)
    # Server collects the decentralized distilled dataset from clients
    server.load_distill(distill_dataset)
    print('Server gets %d images' % len(distill_dataset['y']))
    print('Server starts experiment with  '
          'i_seed=%d' % args.sys_i_seed,
          '| epoch=%d' % args.server_n_epoch,
          '| batch_size=%d' % args.server_bs,
          '| lr=%.4f' % args.server_lr,
          '| momentum=%.4f' % args.server_momentum,
          '| num_workers=%d' % args.server_n_worker,
          '| dataset_id=%s' % args.sys_dataset,
          '| model_name=%s' % args.sys_model)
    # Server trains models
    server.train(exp_name, args.sys_res_root)


if __name__ == "__main__":
    main()
