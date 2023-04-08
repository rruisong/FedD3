# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import numpy as np
import jax
from tqdm import tqdm
from neural_tangents import stax
from jax.example_libraries import optimizers
from jax.config import config
from jax import numpy as jnp
from jax import scipy as sp

config.update('jax_enable_x64', True)


def one_hot(x,
            num_classes,
            center=True,
            dtype=np.float32):
    assert len(x.shape) == 1
    one_hot_vectors = np.array(x[:, None] == np.arange(num_classes), dtype)
    if center:
        one_hot_vectors = one_hot_vectors - 1. / num_classes
    return one_hot_vectors


def get_normalization_data(arr):
    channel_means = np.mean(arr, axis=(0, 1, 2))
    channel_stds = np.std(arr, axis=(0, 1, 2))
    return channel_means, channel_stds


def normalize(arr, mean, std):
    return (arr - mean) / std


def FullyConnectedNetwork(
        depth,
        width,
        W_std=np.sqrt(2),
        b_std=0.1,
        num_classes=10,
        parameterization='ntk',
        activation='relu'):
    activation_fn = stax.Relu()
    dense = functools.partial(
        stax.Dense, W_std=W_std, b_std=b_std, parameterization=parameterization)

    layers = [stax.Flatten()]
    for _ in range(depth):
        layers += [dense(width), activation_fn]
    layers += [stax.Dense(num_classes, W_std=W_std, b_std=b_std,
                          parameterization=parameterization)]

    return stax.serial(*layers)


def get_kernel_fn(architecture, depth, width, parameterization):
    if architecture == 'FC':
        return FullyConnectedNetwork(depth=depth, width=width, parameterization=parameterization)
    else:
        raise NotImplementedError(f'Unrecognized architecture {architecture}')


def class_balanced_sample(sample_size: int,
                          labels: np.ndarray,
                          *arrays: np.ndarray, **kwargs: int):

    if labels.ndim != 1:
        raise ValueError(f'Labels should be one-dimensional, got shape {labels.shape}')
    n = len(labels)
    if not all([n == len(arr) for arr in arrays[1:]]):
        raise ValueError(
            f'All arrays to be subsampled should have the same length. Got lengths {[len(arr) for arr in arrays]}')
    classes = np.unique(labels)
    n_classes = len(classes)
    n_per_class, remainder = divmod(sample_size, n_classes)

    if remainder != 0:
        raise ValueError(
            f'Number of classes {n_classes} in labels must divide sample size {sample_size}.'
        )
    if kwargs.get('seed') is not None:
        np.random.seed(kwargs['seed'])
    inds = np.concatenate([
        np.random.choice(np.where(labels == c)[0], n_per_class, replace=False)
        for c in classes
    ])

    return (inds, labels[inds].copy()) + tuple(
        [arr[inds].copy() for arr in arrays])


def make_loss_acc_fn(kernel_fn):
    LEARN_LABELS = False

    @jax.jit
    def loss_acc_fn(x_support, y_support, x_target, y_target, reg=1e-6):
        y_support = jax.lax.cond(LEARN_LABELS, lambda y: y, jax.lax.stop_gradient, y_support)
        k_ss = kernel_fn(x_support, x_support)
        k_ts = kernel_fn(x_target, x_support)
        k_ss_reg = (k_ss + jnp.abs(reg) * jnp.trace(k_ss) * jnp.eye(k_ss.shape[0]) / k_ss.shape[0])
        pred = jnp.dot(k_ts, sp.linalg.solve(k_ss_reg, y_support, sym_pos=True))
        mse_loss = 0.5 * jnp.mean((pred - y_target) ** 2)
        acc = jnp.mean(jnp.argmax(pred, axis=1) == jnp.argmax(y_target, axis=1))
        return mse_loss, acc

    return loss_acc_fn


def get_update_functions(init_params, kernel_fn, lr):
    opt_init, opt_update, get_params = optimizers.adam(lr)
    opt_state = opt_init(init_params)

    def make_loss_acc_fn(kernel_fn):
        LEARN_LABELS = False

        @jax.jit
        def loss_acc_fn(x_support, y_support, x_target, y_target, reg=1e-6):
            y_support = jax.lax.cond(LEARN_LABELS, lambda y: y, jax.lax.stop_gradient, y_support)
            k_ss = kernel_fn(x_support, x_support)
            k_ts = kernel_fn(x_target, x_support)

            # H(X)
            k_ss_reg = (k_ss + jnp.abs(reg) * jnp.trace(k_ss) * jnp.eye(k_ss.shape[0]) / k_ss.shape[0])
            pred = jnp.dot(k_ts, sp.linalg.solve(k_ss_reg, y_support, sym_pos=True))
            mse_loss = 0.5 * jnp.mean((pred - y_target) ** 2)

            acc = jnp.mean(jnp.argmax(pred, axis=1) == jnp.argmax(y_target, axis=1))
            return mse_loss, acc

        return loss_acc_fn

    loss_acc_fn = make_loss_acc_fn(kernel_fn)
    value_and_grad = jax.value_and_grad(lambda params, x_target, y_target: loss_acc_fn(params['x'], params['y'], x_target, y_target), has_aux=True)

    @jax.jit
    def update_fn(step, opt_state, params, x_target, y_target):
        (loss, acc), dparams = value_and_grad(params, x_target, y_target)
        return opt_update(step, dparams, opt_state), (loss, acc)

    return opt_state, get_params, update_fn

# The main function of KIP-instance for local dataset distillation.
def distill_kip_unit(x_train_raw,
                     y_train_raw,
                     num_dd_epoch,
                     seed,
                     k,
                     lr,
                     threshold,
                     target_sample_size,
                     kernel_model,
                     depth,
                     width):

    _, _, kernel_fn = get_kernel_fn(kernel_model, depth, width, 'ntk')
    KERNEL_FN = jax.jit(functools.partial(kernel_fn, get='ntk'))

    channel_means, channel_stds = get_normalization_data(x_train_raw)
    x_train = normalize(x_train_raw, channel_means, channel_stds)
    y_train = one_hot(y_train_raw, 10)

    _, _, x_init_raw, y_init = class_balanced_sample(k, y_train_raw, x_train_raw, y_train, seed=seed)
    x_init = normalize(x_init_raw, channel_means, channel_stds)
    params_init = {'x': x_init, 'y': y_init}

    opt_state, get_params, update_fn = get_update_functions(params_init, KERNEL_FN, lr)
    params = get_params(opt_state)

    pbar = tqdm(range(1, num_dd_epoch + 1))
    for i in pbar:
        # full batch gradient descent
        _, _, x_target_batch, y_target_batch = class_balanced_sample(target_sample_size, y_train_raw, x_train, y_train)
        opt_state, aux = update_fn(i, opt_state, params, x_target_batch, y_target_batch)
        train_loss, train_acc = aux
        params = get_params(opt_state)
        pbar.set_description('Step: %d ' % i + '|Train Loss: %.4f ' % train_loss + '|Train Acc: %.4f ' % train_acc)

        if train_acc > threshold:
            print("converge at loop ", i)
            break

    return params
