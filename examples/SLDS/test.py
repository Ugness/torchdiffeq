#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import numpy as np
from torchdiffeq import odeint, odeint_adjoint
from torchdiffeq import odeint_event

from tqdm import tqdm

from train import SLDS, DynsSolver as EventODENet
from train_ode import DynsSolver as ODENet
from train_ode_fusion import DynsSolver as ODEFusionNet

import matplotlib.pyplot as plt

net_dict = {
    'ode':ODENet,
    'event_ode':EventODENet,
    # 'fusion_net':ODEFusionNet
}
load_dict = {
    'ode':'checkpoints/orig_ode/default',
    'event_ode':'checkpoints/default',
    'fusion_net':'checkpoints/ode_fusion/default',
}

import random
random.seed(1234, version=2)

torch.set_default_dtype(torch.float64)

def loss_func(pred_seq, target_seq, coeff=0.01):
    position_loss = F.mse_loss(pred_seq, target_seq)
    pred_velocity = pred_seq[1:] - pred_seq[:1]
    target_velocity = target_seq[1:] - target_seq[:1]
    velocity_loss = F.mse_loss(pred_velocity, target_velocity)
    return position_loss + coeff * velocity_loss

def test(args):

    Solver = net_dict[args.net]

    model = Solver(train=False, layer_norm=False, adjoint=False)
    model = model.cuda()

    ckpt_dir = os.path.join(load_dict[args.net], '49.pth')
    ckpt = torch.load(ckpt_dir)
    model.load_state_dict(ckpt)

    testset = SLDS('test.npy')

    # batchsize == 1
    test_loader = DataLoader(testset, shuffle=False)

    epoch_loss = 0.
    targets = []
    preds = []
    with torch.no_grad():
        for (pos, w) in test_loader:
            # squeeze unnecessary batch dimension
            x = pos[0].cuda()

            w = w.cuda()

            # Split batch to input, target
            input_state = (x[0], w)
            target_pos = x[1:].double()

            pred_t, pred_pos = model(input_state)
            targets.append(target_pos)
            preds.append(pred_pos)
            # MSE Loss
            loss = F.mse_loss(pred_pos, target_pos)

            epoch_loss += loss.item()
        print(f"MSE Loss of Net {args.net}: {epoch_loss / len(testset)}")

    return targets, preds

def vis(args, targets, preds, n_sample=3):
    '''
    1. Run model and Get x y x y.
    2. plot each balls with x y. (plt. 5x5 map, 0.5 rad ball. albedo~t)
    3. plot the target.
    '''
    if n_sample > 0:
        targets = targets[:n_sample]
        preds = preds[:n_sample]

    fig_dir = os.path.join(load_dict[args.net].replace('checkpoints', 'figures'))
    os.makedirs(fig_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(15, 15))
    for i in range(len(preds)):
        ax.plot(preds[:, 0], preds[:, 1], color='b')
    fig.savefig(os.path.join(fig_dir, f'pred.png'))

    fig, ax = plt.subplots(figsize=(15, 15))
    for i in range(len(preds)):
        ax.plot(targets[:, 0], targets[:, 1], color='b')
    fig.savefig(os.path.join(fig_dir, f'target.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default='default')
    parser.add_argument("--event_lr", default=0.0005, type=float)
    parser.add_argument("--inst_lr", default=0.0001, type=float)
    parser.add_argument("--dt", default=1./30., type=float)
    parser.add_argument("--clip_grad", default=5.0, type=float)
    parser.add_argument("--seq_len", default=25, type=int)
    parser.add_argument("--no_subseq", action='store_true')
    parser.add_argument("--no_adjoint", action='store_true')
    parser.add_argument("--init_weight", action='store_true')
    parser.add_argument("--epochs", default=1000, type=int)

    parser.add_argument("--layer_norm", action='store_true')
    parser.add_argument('--net', default='none')

    args = parser.parse_args()
    for net_type in net_dict.keys():
        args.net = net_type
        targets, preds = test(args)
        vis(args, targets, preds, n_sample=3)