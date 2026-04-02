import os
import sys
import glob
from types import SimpleNamespace
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch.utils.data import TensorDataset, DataLoader

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from modeling.autoencoder.base import Autoencoder
from training.train_autoencoder import load_autoencoder

from common.models import *
from common.visualize import point_cloud_from_arrays
from common.training import Trainer
from common.utils import get_device
from common.callback import *


def pad(arr, target_len=150):
    """
    arr: numpy array of shape (T, H, W, C)
    returns: padded array (target_len, H, W, C)
    """

    T = arr.shape[0]

    if T >= target_len:
        return arr[:target_len]

    pad_shape = (target_len - T,) + arr.shape[1:]
    pad = np.zeros(pad_shape, dtype=arr.dtype)

    return np.concatenate([arr, pad], axis=0)


def compute_bbox(X, pad=0.5):
    """
    Generic bounding box for high dimensional data
    """

    X = X.detach().cpu().numpy()

    mini = X.min(axis=0) - pad
    maxi = X.max(axis=0) + pad

    return mini, maxi


if __name__ == "__main__":

    #### Commandline ####
    parser = argparse.ArgumentParser(
        prog="Training of a 1-Lipschitz architecture",
        description="This scripts runs the training optimization of a 1-Lipschitz neural network on some precomputed point cloud dataset."
    )

    # dataset parameters
    parser.add_argument("dataset", type=str, default="../dataset-good", help="name of the dataset to train on")
    parser.add_argument("-a", "--autoencoder", type=str, default="../experiments/train_autoencoder_shapes/shapes_navi/2026-02-16_14-04-35/logs/version_0/checkpoints/train_autoencoder_shapes-epoch=59-valid_loss=0.000644.ckpt", help="path to trained autoencoder")
    parser.add_argument("-o", "--output-name", type=str, default="training_data", help="custom output folder name")
    parser.add_argument("--unsigned", action="store_true", help="flag for training an unsigned distance field instead of a signed one")

    # model parameters
    parser.add_argument("-model","--model", choices=["ortho", "sll"], default="sll", help="Which Lipschitz architecture to consider. 'SLL' is the one used in the paper. 'Ortho' is the Bjorck orthonormalization-based architecture of Anil et al. (2019)")
    parser.add_argument("-n-layers", "--n-layers", type=int, default=20, help="number of layers in the network")
    parser.add_argument("-n-hidden", "--n-hidden", type=int, default=128, help="size of the layers")

    # optimization parameters
    parser.add_argument("-ne", "--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument('-bs',"--batch-size", type=int, default=200, help="Train batch size")
    parser.add_argument("-tbs", "--test-batch-size", type = int, default = 5000, help="Test batch size")
    parser.add_argument("-lr", "--learning-rate", type=float, default=5e-4, help="Adam's learning rate")
    parser.add_argument("-lm", "--loss-margin", type=float, default=1e-2, help="margin m in the hKR loss")
    parser.add_argument("-lmbd", "--loss-lambda", type=float, default=100., help="lambda in the hKR loss")
    
    # misc
    parser.add_argument("-cp", "--checkpoint-freq", type=int, default=10, help="Number of epochs between each model save")
    parser.add_argument("-cpu", action="store_true", help="force training on CPU")
    args = parser.parse_args()

    #### Config ####
    config = SimpleNamespace(
        signed = not args.unsigned,
        device = get_device(args.cpu),
        n_epochs = args.epochs,
        checkpoint_freq = args.checkpoint_freq,
        batch_size = args.batch_size,
        test_batch_size = args.test_batch_size,
        loss_margin = args.loss_margin,
        loss_regul = args.loss_lambda,
        optimizer = "adam",
        learning_rate = args.learning_rate,
        output_folder = os.path.join("output", args.output_name if len(args.output_name)>0 else args.dataset)
    )
    os.makedirs(config.output_folder, exist_ok=True)
    print("DEVICE:", config.device)

    print("Loading Autoencoder Model...")

    ckpt_path = args.autoencoder
    autoencoder = load_autoencoder(ckpt_path)
    autoencoder.eval()
    dataset = args.dataset

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    files = sorted(glob.glob(os.path.join(dataset,"*.npz")))
    print(f"Found {len(files)} episodes.")
    num_train = int(len(files)*0.8)
    print("Loading Dataset...")

    imgs = []
    acts = []
    dones = []

    for i,file_path in enumerate(files):
        if i % 100 == 0:
            print("Episode:",i)

        file = np.load(file_path,allow_pickle=True)
        imgs.append(pad(file["images"]))
        acts.append(pad(file["actions"]))
        d = file["dones"]
        d = np.where(d == 0, -1, 1)
        dones.append(pad(d))

        imgs_t = torch.from_numpy(np.stack(imgs)).cuda().float()
        acts_t = torch.from_numpy(np.stack(acts)).cuda().float()

        with torch.no_grad():
            encoded_data = autoencoder.encode(imgs_t,acts_t)

        encoded_data = encoded_data.detach().cpu()

        if i < num_train:
            X_train.append(encoded_data)
            y_train.append(torch.from_numpy(np.array(dones)))
        else:
            X_test.append(encoded_data)
            y_test.append(torch.from_numpy(np.array(dones)))

        imgs = []
        acts = []
        dones = []

    X_train = np.array(X_train)
    X_train = X_train.reshape(
        X_train.shape[0],
        X_train.shape[2],
        -1
    )
    y_train = np.array(y_train)
    y_train = y_train.reshape(y_train.shape[0]*y_train.shape[1],-1)

    X_test = np.array(X_test)
    X_test = X_test.reshape(
        X_test.shape[0]*X_test.shape[2],
        -1
    )
    X_test = torch.from_numpy(X_test).cuda()
    y_test = np.array(y_test)
    y_test = y_test.reshape(-1)
    Y_test = torch.from_numpy(y_test).float().cuda()

    print("X_train:",X_train.shape)
    print("y_train:",y_train.shape)

    X_train_in = []
    X_train_out = []

    for i in range(X_train.shape[0]):
        for j in range(X_train.shape[1]):
            if y_train[i,j]:
                X_train_in.append(X_train[i,j])
            else:
                X_train_out.append(X_train[i,j])

    X_train_in = torch.from_numpy(np.array(X_train_in)).cuda()
    X_train_out = torch.from_numpy(np.array(X_train_out)).cuda()

    loader_in = DataLoader(
        TensorDataset(X_train_in),
        batch_size=config.batch_size,
        shuffle=True
    )

    loader_out = DataLoader(
        TensorDataset(X_train_out),
        batch_size=config.batch_size,
        shuffle=True
    )

    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=config.test_batch_size)
    print(f"Succesfully loaded test set: {X_test.shape}\n")

    DIM = X_train_out.shape[1]

    model = select_model(
        args.model,
        DIM,
        args.n_layers,
        args.n_hidden
    ).to(config.device)
    print("PARAMETERS:",count_parameters(model))

    # ---------------------------------------------------
    # Export point cloud (Torch Geometric)
    # ---------------------------------------------------

    if config.signed:
        pc = point_cloud_from_arrays(
            (X_train_in.detach().cpu(),-1.),
            (X_train_out.detach().cpu(),1.)
        )
    else:
        pc = point_cloud_from_arrays(
            (X_train_out.detach().cpu(), 1.)
        )
    torch.save(pc, os.path.join(config.output_folder,"pc_0.pt"))

    # ---------------------------------------------------
    # Training callbacks
    # ---------------------------------------------------

    callbacks = []
    callbacks.append(
        LoggerCB(os.path.join(config.output_folder,"log.csv"))
    )

    if config.checkpoint_freq>0:
        callbacks.append(CheckpointCB([x for x in range(0, config.n_epochs, config.checkpoint_freq) if x>0]))

    callbacks.append(UpdateHkrRegulCB({1 : 1., 5 : 10., 10: 100., 30: config.loss_regul}))
    # callbacks.append(UpdateHkrRegulCB({1 : config.loss_regul}))
    
    if config.signed:
        trainer = Trainer((loader_in, loader_out), test_loader, config)
        trainer.add_callbacks(*callbacks)
        trainer.train_lip(model)
    else:
        trainer = Trainer((loader_out,), test_loader, config)
        trainer.add_callbacks(*callbacks)
        trainer.train_lip_unsigned(model)

    path = os.path.join("output/","model_kr_loss.pt")

    save_model(model,path)