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

from train import BouncingBall, DynsSolver as EventODENet
from train_ode import DynsSolver as ODENet
from train_ode_fusion import DynsSolver as ODEFusionNet

import matplotlib
import matplotlib.pyplot as plt

net_dict = {
    'ode':ODENet,
    'event_ode':EventODENet,
    # 'fusion_net':ODEFusionNet
}
load_dict = {
    'ode':'checkpoints/orig_ode/default/99.pth',
    'event_ode':'checkpoints/default/0.pth',
    # 'fusion_net':'checkpoints/ode_fusion/default',
}

import random
random.seed(1234, version=2)

torch.set_default_dtype(torch.float64)

def test(args):

    seqlens = [25, 50, 100]

    Solver = net_dict[args.net]

    model = Solver(dt=args.dt, seq_len=100, layer_norm=args.layer_norm, adjoint=not args.no_adjoint,
                   init_weight=args.init_weight)
    model = model.cuda()

    ckpt_dir = load_dict[args.net]
    ckpt = torch.load(ckpt_dir)
    model.load_state_dict(ckpt)

    testset = BouncingBall('test.npy', seq_len=100, subseq=False)

    # batchsize == 1
    test_loader = DataLoader(testset, shuffle=False)

    epoch_loss = {x:0 for x in seqlens}
    targets = []
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            # squeeze unnecessary batch dimension
            batch = batch.squeeze(0).cuda()

            # Split batch to input, target
            input_state = batch[0]
            target_pos = batch[1:, :4]

            pred_t, pred_state = model(input_state)
            pred_pos = pred_state[:, :4]

            # for stability
            pred_pos = torch.clamp(pred_pos, min=0, max=5)

            targets.append(target_pos)
            preds.append(pred_pos)

            # MSE Loss
            for seqlen in seqlens:
                target = target_pos[:seqlen-1]
                pred = pred_pos[:seqlen - 1]
                loss = F.mse_loss(pred, target)
                epoch_loss[seqlen] += loss.item()
    for seqlen in seqlens:
        print(f"@{seqlen} MSE Loss of Net {args.net}: {epoch_loss[seqlen] / len(testset)}")

    return targets, preds

def vis(args, targets, preds, n_sample=3):
    '''
    1. Run model and Get x y x y.
    2. plot each balls with x y. (plt. 5x5 map, 0.5 rad ball. albedo~t)
    3. plot the target.
    '''
    if n_sample > 0:
        targets = targets[:3]
        preds = preds[:3]

    fig_dir = os.path.join(load_dict[args.net].replace('checkpoints', 'figures'))
    os.makedirs(fig_dir, exist_ok=True)

    cmap = matplotlib.cm.get_cmap('rainbow')
    for i in range(len(preds)):
        fig, ax = plt.subplots(figsize=(15,15))
        ax.set_xlim([0, 5])
        ax.set_ylim([0, 5])
        for t in range(len(preds[i])):
            circle1 = plt.Circle(5.-preds[i][t][:2], 0.5, color='r', alpha=0.1)
            circle2 = plt.Circle(5.-preds[i][t][2:], 0.5, color='b', alpha=0.1)
            ax.add_patch(circle1)
            ax.add_patch(circle2)
        fig.savefig(os.path.join(fig_dir, f'pred_{i}.png'))

        fig, ax = plt.subplots(figsize=(15, 15))
        ax.set_xlim([0, 5])
        ax.set_ylim([0, 5])
        for t in range(len(preds[i])):
            circle1 = plt.Circle(5. - targets[i][t][:2], 0.5, color='r', alpha=0.1)
            circle2 = plt.Circle(5. - targets[i][t][2:], 0.5, color='b', alpha=0.1)
            ax.add_patch(circle1)
            ax.add_patch(circle2)
        fig.savefig(os.path.join(fig_dir, f'target_{i}.png'))

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
        vis(args, targets, preds, n_sample=1)