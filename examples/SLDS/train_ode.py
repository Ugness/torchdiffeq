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

import random
random.seed(1234, version=2)

torch.set_default_dtype(torch.float64)

# Load .npy files.
    # input: (x, y, x, y, 0, 0, 0, 0)
    # state: (x, y, x, y, vx, vy, vx, vy)
# ODE Module Specify
    # Given initial points --> unroll 100.
    # ODESolveEvent: dynamics:f, event_fn: g
    # Instant update h.
# Train / Val roop.

class SLDS(Dataset):

    def __init__(self, root='train.npy'):
        super().__init__()
        self.data = np.expand_dims(np.load(root), 2)

    def __getitem__(self, item):
        # Get state
        pos = self.data[item]
        w = torch.zeros([3])

        # TODO: compute i
        y = pos[0]
        if y[0, 0] >= 2:
            i = 0
        elif y[0, 0] < 2 and y[0, 1] >= 0:
            i = 1
        elif y[0, 0] < 2 and y[0, 1] < 0:
            i = 2
        w[i] = 1.
        return pos, w # --> [25, 8]

    def __len__(self):
        return len(self.data)

class DynsSolver(nn.Module):

    def __init__(self, train=True, layer_norm=False, adjoint=True):
        super().__init__()
        self.event_fn = EventFn()
        self.dynamics = Dyns()
        self.inst_func = InstFn(hidden_dim=1024, n_hidden=2, layer_norm=layer_norm)
        self.ts = torch.linspace(0., 4., 50).cuda() if train else torch.linspace(0., 12., 50).cuda()

        # TODO: implement nfe counting.
        self.nfe = 0

    # Get Collision times first, then intergrate over ts.
    def get_collision_times(self, state):
        event_times = []
        event_states = []
        t = torch.tensor([0.], dtype=torch.float64, requires_grad=True).cuda()

        while t.item() < self.ts.max().item():
            # Use rk4 instead dopri5
            # if not found an event, Set endpoint to 25.
            try:
                t, state = odeint_event(self.dynamics, state, t, event_fn=self.event_fn, reverse_time=False,
                                        odeint_interface=odeint, method='rk4', options={'step_size': 0.01})
                state = [x[-1] for x in state]
                state = self.inst_func(t, state)
                print(t)
            except RuntimeError:
                t = torch.tensor([float(self.ts.max())], requires_grad=True).cuda()
                state = state
                print('no_event')

            event_times.append(t)
            event_states.append(state)

        return event_times, event_states

    def forward(self, state):
        # event_times, event_states = self.get_collision_times(state)

        y0 = state
        sample_ts = self.ts
        results = odeint(self.dynamics, y0, sample_ts,
                                    method='rk4', options={'step_size': 0.01})
        pos = results[0][1:]

        return None, pos.view(-1, 1, 2)


class Dyns(nn.Module):
    def __init__(self):
        super().__init__()
        # Define 3 cases of dynamics (Eq. 18.)
        # [W(2x2), b(1x2)]
        self.mlp = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

        self.nfe = 0

    def forward(self, t, state):
        # state: (x, w). w controls the dynamics.
        x = state[0]
        w = state[1]
        # Following Eq. in Sec. 4.1. (weighted form of Switching Dynamics)
        dx = self.mlp(x)
        return dx, torch.zeros([3], requires_grad=True).cuda()

class EventFn(nn.Module): # positions --> scalar
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 2),
        )

        self.apply(self.weight_init)


    def forward(self, t, state):
        # Implemented Eq. 19.
        # state: x, w where x = (x1, x2)
        input = state[0]
        output = torch.tanh(self.mlp(input))
        return torch.prod(output)
        # VQVAE Trick Quantize near-0 values
        # quant_idx = (output < 0.1) & (output > -0.1)
        # output_q = output.clone().detach()
        # output_q[quant_idx] = 0.
        # output_q = output + (output_q - output).detach()
        # return torch.prod(output_q)

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, 0, 10)
            torch.nn.init.constant_(m.bias, 0)

class InstFn(nn.Module):
    def __init__(self, hidden_dim=1024, n_hidden=2, layer_norm=False):
        super().__init__()
        mlp = []
        mlp.append(nn.Linear(2, hidden_dim))
        if layer_norm:
            mlp.append(nn.LayerNorm(hidden_dim)),
        mlp.append(nn.ReLU())
        for _ in range(n_hidden):
            mlp.append(nn.Linear(hidden_dim, hidden_dim))
            if layer_norm:
                mlp.append(nn.LayerNorm(hidden_dim)),
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(hidden_dim, 3))
        self.mlp = nn.Sequential(*mlp)

    def forward(self, t, state):
        x = state[0]
        input = state[1]
        output = self.mlp(input)
        output = torch.softmax(output, dim=-1)
        return x, output

def get_grad(m):
    total_norm = 0.
    for p in m.parameters():
        try:
            p_norm = p.grad.data.norm(2)
            total_norm += p_norm.item() ** 2
        except AttributeError:
            pass
    return total_norm ** 0.5

def loss_func(pred_seq, target_seq, coeff=0.01):
    position_loss = F.mse_loss(pred_seq, target_seq)
    pred_velocity = pred_seq[1:] - pred_seq[:1]
    target_velocity = target_seq[1:] - target_seq[:1]
    velocity_loss = F.mse_loss(pred_velocity, target_velocity)
    return position_loss + coeff * velocity_loss

def train(args):
    writer = SummaryWriter(os.path.join('logs/orig_ode', args.name))
    trainset = SLDS('train.npy')

    train_loader = DataLoader(trainset, shuffle=True, num_workers=16)

    model = DynsSolver(layer_norm=args.layer_norm, )
    model = model.cuda()

    # Param_group
    params = [
        {'params': model.parameters(), 'lr': args.lr},
        ]
    optimizer = optim.Adam(params)

    os.makedirs(os.path.join('checkpoints/orig_ode', args.name), exist_ok=True)
    for epoch in range(args.max_epochs):
        epoch_loss = 0
        for i, (pos, w) in tqdm(enumerate(train_loader), total=100):
            # squeeze unnecessary batch dimension
            x = pos[0].cuda()

            # w shape: 1,3
            w = w.cuda()

            # Split batch to input, target
            input_state = (x[0], w)
            target_pos = x[1:].double()

            pred_t, pred_pos = model(input_state)
            # Eq. 20.
            loss = loss_func(pred_pos, target_pos)

            try:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()
            except AssertionError:
                pass

            writer.add_scalar('loss', loss.item(), epoch * len(trainset) + i)
            writer.add_scalar('event_grad', get_grad(model.event_fn), epoch * len(trainset) + i)
            writer.add_scalar('inst_grad', get_grad(model.inst_func), epoch * len(trainset) + i)
            epoch_loss += loss.item()
        writer.add_scalar('epoch_loss', epoch_loss / len(trainset), epoch)
        torch.save(model.state_dict(), os.path.join('checkpoints/orig_ode', args.name, f'{epoch}.pth'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default='default')
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--dt", default=1./30., type=float)
    parser.add_argument("--clip_grad", default=5.0, type=float)
    parser.add_argument("--seq_len", default=25, type=int)
    parser.add_argument("--no_subseq", action='store_true')
    parser.add_argument("--max_epochs", default=1000, type=int)
    parser.add_argument("--layer_norm", action='store_true')
    args = parser.parse_args()
    train(args)
