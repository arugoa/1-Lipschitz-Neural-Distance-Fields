from types import SimpleNamespace
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler


from common.models import *
from common.visualize import point_cloud_from_arrays
from common.training import Trainer
from common.utils import get_device
from common.callback import *

import argparse
import os
import sys
import glob

import gymnasium #as gym
import numpy as np
from sklearn.decomposition import PCA
import cv2
import torch

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
dreamer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dreamerv3-torch'))
sys.path.append(dreamer_dir)
saferl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '/PyHJ'))
sys.path.append(saferl_dir)
print(sys.path)
import models
import tools
import networks
import ruamel.yaml as yaml
import PyHJ.reach_rl_gym_envs as reach_rl_gym_envs

from termcolor import cprint
from datetime import datetime
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--expt_name", type=str, default=None)
    parser.add_argument("--resume_run", type=bool, default=False)

    # dataset parameters
    parser.add_argument("dataset", type=str, default="../dataset-good", help="name of the dataset to train on")
    parser.add_argument("-a", "--autoencoder", type=str, default="../experiments/train_autoencoder_shapes/shapes_navi/2026-02-16_14-04-35/logs/version_0/checkpoints/train_autoencoder_shapes-epoch=59-valid_loss=0.000644.ckpt", help="path to trained autoencoder")
    parser.add_argument("-o", "--output-name", type=str, default="training_data", help="custom output folder name")
    parser.add_argument("--unsigned", action="store_true", help="flag for training an unsigned distance field instead of a signed one")
    parser.add_argument("-p", "--pca-dim", type=int, default=2, help="dimension of pca to take of data")

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
    args_intial = parser.parse_args()


    # environment parameters
    config, remaining = parser.parse_known_args()


    if not config.resume_run:
        curr_time = datetime.now().strftime("%m%d/%H%M%S")
        config.expt_name = (
            f"{curr_time}_{config.expt_name}" if config.expt_name else curr_time
        )
    else:
        assert config.expt_name, "Need to provide experiment name to resume run."

    yml = yaml.YAML(typ="safe", pure=True)
    configs = yml.load(
        #(pathlib.Path(sys.argv[0]).parent / "../configs/config.yaml").read_text()
        (pathlib.Path(sys.argv[0]).parent / "../configs.yaml").read_text()
    )

    name_list = ["defaults", *config.configs] if config.configs else ["defaults"]

    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    final_config = parser.parse_args(remaining)

    final_config.logdir = f"{final_config.logdir+'/PyHJ'}/{config.expt_name}"
    #final_config.time_limit = HORIZONS[final_config.task.split("_")[-1]]

    print("---------------------")
    cprint(f"Experiment name: {config.expt_name}", "red", attrs=["bold"])
    cprint(f"Task: {final_config.task}", "cyan", attrs=["bold"])
    cprint(f"Logging to: {final_config.logdir+'/PyHJ'}", "cyan", attrs=["bold"])
    print("---------------------")
    return final_config, args_intial


def pad(arr, target_len=150):
    """
    arr: numpy array of shape (T, H, W, C)
    returns: padded array (target_len, H, W, C)
    """

    T = arr.shape[0]

    if T >= target_len:
        return arr[:target_len]

    pad_shape = (target_len - T,) + arr.shape[1:]
    last = arr[-1:]
    pad = np.repeat(last, target_len - T, axis=0)

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
    args, args_initial = get_args()
    
    #### Encoder config ####
    config = args

    print("Loading Autoencoder Model...")
    env = gymnasium.make(args.task, params = [config])
    shapes = {k: tuple(v.shape) for k, v in env.observation_space_full.spaces.items()}
    enc = networks.MultiEncoder(shapes, **config.encoder)
    enc.to(config.device)

    ckpt_path = config.rssm_ckpt_path
    checkpoint = torch.load(ckpt_path, weights_only=False)
    state_dict = {k.split("_wm.encoder", 1)[1]:v for k,v in checkpoint['agent_state_dict'].items() if '_wm.encoder' in k}
    print("Loading encoder...")
    enc.load_state_dict(state_dict, strict=False)
    enc.eval()
    
    #### Dataset Config ####
    config = SimpleNamespace(
        signed = not args_initial.unsigned,
        device = get_device(args_initial.cpu),
        n_epochs = args_initial.epochs,
        checkpoint_freq = args_initial.checkpoint_freq,
        batch_size = args_initial.batch_size,
        test_batch_size = args_initial.test_batch_size,
        loss_margin = args_initial.loss_margin,
        loss_regul = args_initial.loss_lambda,
        optimizer = "adam",
        learning_rate = args_initial.learning_rate,
        output_folder = os.path.join("output", args_initial.output_name if len(args_initial.output_name)>0 else args_initial.dataset)
    )
    os.makedirs(config.output_folder, exist_ok=True)
    print("DEVICE:", config.device)

    dataset = args_initial.dataset
    files = sorted(glob.glob(os.path.join(dataset,"*.npz")))
    print(f"Found {len(files)} episodes.")
    num_train = int(len(files)*0.7)
    print("Loading Dataset...")

    # Pass 1: count samples to pre-allocate memmaps
    print("Counting samples...")
    n_in, n_out, n_test = 0, 0, 0
    DIM = None

    for i, file_path in enumerate(files):
        file = np.load(file_path, allow_pickle=True)
        d = file["dones"]
        d = np.where(d == 0, 1, -1)
        d_padded = pad(d)  # shape (150,)
        if DIM is None:
            DIM = 8192  # known from encoder output

        if i < num_train:
            n_in  += int((d_padded == 1).sum())
            n_out += int((d_padded != 1).sum())
        else:
            n_test += 150

    print(f"n_in={n_in}, n_out={n_out}, n_test={n_test}")

    print("Fitting scaler + PCA (streaming)...")
    pca_dim = args_initial.pca_dim
    ipca = IncrementalPCA(n_components=pca_dim, batch_size=1024)
    scaler = StandardScaler()

    for i, file_path in enumerate(files):
        if i % 100 == 0:
            print(f"PCA Fit Episode: {i}")
        
        if i >= num_train:
            break

        file = np.load(file_path, allow_pickle=True)

        imgs_np = pad(file["images"])
        acts_np = pad(file["actions"])

        imgs_t = torch.from_numpy(imgs_np[None]).float().to(config.device) / 255.0
        acts_t = torch.from_numpy(acts_np[None]).float().to(config.device)
        obs = {"image": imgs_t, "actions": acts_t}

        with torch.no_grad():
            encoded = enc(obs)

        enc_np = encoded[0].detach().cpu().float().numpy()  # (150, 8192)

        # --- fit scaler + PCA ---
        scaler.partial_fit(enc_np)
        enc_np_scaled = scaler.transform(enc_np)
        ipca.partial_fit(enc_np_scaled)

        del imgs_t, acts_t, encoded
        torch.cuda.empty_cache()

    print("Finished PCA fitting")
    print("Explained variance ratio:", ipca.explained_variance_ratio_.sum())

    # Update DIM AFTER PCA
    DIM = pca_dim

    # Allocate memmaps (NOW using PCA dim)
    mm_path_in   = os.path.join(config.output_folder, "X_train_in.npy")
    mm_path_out  = os.path.join(config.output_folder, "X_train_out.npy")
    mm_path_test = os.path.join(config.output_folder, "X_test.npy")
    mm_path_yt   = os.path.join(config.output_folder, "y_test.npy")

    mm_in   = np.lib.format.open_memmap(mm_path_in,   mode="w+", dtype="float32", shape=(n_in,  DIM))
    mm_out  = np.lib.format.open_memmap(mm_path_out,  mode="w+", dtype="float32", shape=(n_out, DIM))
    mm_test = np.lib.format.open_memmap(mm_path_test, mode="w+", dtype="float32", shape=(n_test, DIM))
    mm_yt   = np.lib.format.open_memmap(mm_path_yt,   mode="w+", dtype="float32", shape=(n_test,))


    # PASS 2B: Encode → scale → PCA → write
    print("Transforming with PCA and writing memmaps...")
    idx_in, idx_out, idx_test = 0, 0, 0

    for i, file_path in enumerate(files):
        if i % 100 == 0:
            print(f"Episode: {i}")

        file = np.load(file_path, allow_pickle=True)

        imgs_np = pad(file["images"])
        acts_np = pad(file["actions"])
        d = file["dones"]
        d = np.where(d == 0, 1, -1)
        d_padded = pad(d)

        # Encode
        imgs_t = torch.from_numpy(imgs_np[None]).float().to(config.device) / 255.0
        acts_t = torch.from_numpy(acts_np[None]).float().to(config.device)
        obs = {"image": imgs_t, "actions": acts_t}

        with torch.no_grad():
            encoded = enc(obs)

        enc_np = encoded[0].detach().cpu().float().numpy()  # (150, 8192)

        enc_np = scaler.transform(enc_np)
        enc_np = ipca.transform(enc_np)  # (150, pca_dim)

        if i < num_train:
            mask_in = (d_padded == 1)
            mask_out = ~mask_in

            n_in_batch = mask_in.sum()
            n_out_batch = mask_out.sum()

            mm_in[idx_in:idx_in+n_in_batch] = enc_np[mask_in]
            mm_out[idx_out:idx_out+n_out_batch] = enc_np[mask_out]

            idx_in += n_in_batch
            idx_out += n_out_batch
        else:
            chunk = len(d_padded)
            mm_test[idx_test:idx_test + chunk] = enc_np
            mm_yt[idx_test:idx_test + chunk] = d_padded.astype("float32")
            idx_test += chunk

        # cleanup
        del imgs_t, acts_t, encoded
        torch.cuda.empty_cache()

    # Flush to disk
    del mm_in, mm_out, mm_test, mm_yt

    print(f"Final counts → in: {idx_in}, out: {idx_out}, test: {idx_test}")

    # Re-open as read-only and wrap in TensorDataset/DataLoader
    class MemmapDataset(torch.utils.data.Dataset):
        def __init__(self, *paths, device="cpu"):
            self.arrays = [np.load(p, mmap_mode="r") for p in paths]
            self.device = device

        def __len__(self):
            return len(self.arrays[0])

        def __getitem__(self, idx):
            return tuple(
                torch.from_numpy(np.array(a[idx])).to(self.device)
                for a in self.arrays
            )

    loader_in  = DataLoader(MemmapDataset(mm_path_in, device=config.device),  batch_size=config.batch_size, shuffle=True)
    loader_out = DataLoader(MemmapDataset(mm_path_out, device=config.device), batch_size=config.batch_size, shuffle=True)
    test_ds    = MemmapDataset(mm_path_test, mm_path_yt, device=config.device)
    test_loader = DataLoader(test_ds, batch_size=config.test_batch_size)

    print(f"Train in: {idx_in}, Train out: {idx_out}, Test: {idx_test}")
    
    model = select_model(
        args_initial.model,
        DIM,
        args_initial.n_layers,
        args_initial.n_hidden
    ).to(config.device)
    print("PARAMETERS:",count_parameters(model))

    # ---------------------------------------------------
    # Export point cloud (Torch Geometric)
    # ---------------------------------------------------

    if config.signed:
        print("we signed")
        X_pc_in  = torch.from_numpy(np.load(mm_path_in))
        X_pc_out = torch.from_numpy(np.load(mm_path_out))
        pc = point_cloud_from_arrays(
            (X_pc_in,  -1.),
            (X_pc_out,  1.)
        )
        del X_pc_in, X_pc_out
    else:
        X_pc_out = torch.from_numpy(np.load(mm_path_out))
        pc = point_cloud_from_arrays(
            (X_pc_out, 1.)
        )
        del X_pc_out

    torch.save(pc, os.path.join(config.output_folder, "pc_0.pt"))
    del pc

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

    path = os.path.join("output/", f"model_hkr_loss_{pca_dim}.pt")

    save_model(model,path)