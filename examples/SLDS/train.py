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

    def __init__(self, root='train.npy', seq_len=25, subseq=True):
        super().__init__()
        pass
        # self.data = np.load(root)
        # self.subseq = subseq
        # self.seq_len = seq_len

    def __getitem__(self, item):
        # Parse 25-length subsequence.
        # Get Velocity. [pos, vel]
        # pos = self.data[item]
        # velocity = pos[1:] - pos[:-1]
        # velocity = np.concatenate([np.zeros([1, 4]), velocity], 0)
        #
        # if self.subseq:
        #     start = random.randint(0, 74)
        # else:
        #     start = 0
        # end = start+self.seq_len
        # state = np.concatenate([pos, velocity], -1)
        # return state[start:end] # --> [25, 8]

        # for SLDS, please return initial point and initial condition. (Please Refer the Dyns).
        return torch.zeros([2]), torch.zeros([3])

    def __len__(self):
        return len(self.data)

class DynsSolver(nn.Module):

    def __init__(self, seq_len=25, layer_norm=False, adjoint=True):
        super().__init__()
        self.event_fn = EventFn(layer_norm=layer_norm)
        self.dynamics = Dyns()
        self.inst_func = InstFn(hidden_dim=1024, n_hidden=2, layer_norm=layer_norm)
        self.seq_len = seq_len

        # TODO: implement nfe counting.
        self.nfe = 0

    # Get Collision times first, then intergrate over ts.
    def get_collision_times(self, state):
        event_times = []
        event_states = []
        t = torch.tensor([0.], dtype=torch.float64, requires_grad=True).cuda()

        while t < float(self.seq_len):
            # Use rk4 instead dopri5
            # if not found an event, Set endpoint to 25.
            try:
                try:
                    t, state = odeint_event(self.dynamics, state, t, event_fn=self.event_fn, reverse_time=False,
                                            atol=1e-8, rtol=1e-8, odeint_interface=odeint, method='dopri5')
                except AssertionError:
                    t, state = odeint_event(self.dynamics, state, t, event_fn=self.event_fn, reverse_time=False,
                                            odeint_interface=odeint, method='rk4', options={'step_size': 0.01})
                state = self.inst_func(t, state[-1, :])
            except RuntimeError:
                t = float(self.seq_len)
                state = state.unsqueeze(0)

            event_times.append(t)
            event_states.append(state)

        return event_times, event_states

    def forward(self, state):
        event_times, event_states = self.get_collision_times(state)
        num_interval = len(event_times)

        last_t = 0.
        total_ts = []
        total_results = []
        for i in range(num_interval):
            # integer between (last_t ~ event_times[i])
            start = int(last_t)+1
            end = min(self.seq_len, int(event_times[i])+1)
            sample_ts = [float(last_t)] + [float(t) for t in range(start, end)]
            sample_ts = torch.tensor(sample_ts, requires_grad=True).cuda()
            total_ts.append(sample_ts[1:].clone().detach())
            if i == 0:
                y0 = state
            else:
                y0 = event_states[i]

            try:
                results = odeint(self.dynamics, y0, sample_ts, method='dopri5')[1:]
            except AssertionError:
                results = odeint(self.dynamics, y0, sample_ts,
                                         method='rk4', options={'step_size': 0.01})[1:]
            # only collect positions.
            total_results += results[0]

            last_t = event_times[i]

        return torch.cat(total_ts), torch.stack(total_results)


class Dyns(nn.Module):
    def __init__(self, dt, gravity=9.8, hidden_dim=256):
        super().__init__()
        # Define 3 cases of dynamics (Eq. 18.)
        # [W(2x2), b(1x2)]
        dyn1 = torch.tensor([[0, 1], [-1, 0], [0, 2]], requires_grad=True)
        dyn2 = torch.tensor([[0, 0], [0, 0], [-1, -1]], requires_grad=True)
        dyn3 = torch.tensor([[0, 0], [0, 0], [1, -1]], requires_grad=True)

        # 3x(2+1)x2
        self.dyn = torch.stack([dyn1, dyn2, dyn3], dim=0)

        self.nfe = 0

    def forward(self, t, state):
        # state: (x, w). w controls the dynamics.
        x = state[0].unsqueeze(0)
        w = state[1]
        # Following Eq. in Sec. 4.1. (weighted form of Switching Dynamics)
        dyn = (w.view(-1, 1, 1) * self.dyn).sum(0)
        weight = dyn[:2]
        bias = dyn[2].unsqueeze(0)

        # addmm(bias, m1, m2) = bias + m1 @ m2.
        # @: matrix multiplication
        dx = torch.addmm(bias, x, weight).squeeze()
        return dx, torch.zeros([3], requires_grad=True)

class EventFn(nn.Module): # positions --> scalar
    def __init__(self, hidden_dim, layer_norm=False):
        super().__init__()
        self.mlp = nn.Linear(2, 2)

    def forward(self, t, state):
        # Implemented Eq. 19.
        # state: x, w where x = (x1, x2)
        input = state[0]
        output = torch.tanh(self.mlp(input))
        return torch.prod(output, dim=-1)

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

    def forward(self, t, state, event_act):
        x = state[0]
        input = state[1]
        output = self.mlp(input)
        output = torch.softmax(output, dim=-1)
        return x, output

def loss_func(pred_seq, target_seq, coeff=0.01):
    position_loss = F.mse_loss(pred_seq, target_seq)
    pred_velocity = pred_seq[1:] - pred_seq[:1]
    target_velocity = target_seq[1:] - target_seq[:1]
    velocity_loss = F.mse_loss(pred_velocity, target_velocity)
    return position_loss + coeff * velocity_loss

def train(args):
    writer = SummaryWriter(os.path.join('logs', args.name))
    trainset = SLDS('train.npy', args.seq_len, not args.no_subseq)

    train_loader = DataLoader(trainset, shuffle=True, num_workers=16)

    model = DynsSolver(seq_len=args.seq_len, layer_norm=args.layer_norm, )
    model = model.cuda()

    # Param_group
    params = [
        {'params': model.event_fn.parameters(), 'lr': args.event_lr},
        {'params': model.inst_func.parameters(), 'lr': args.inst_lr}
        ]
    optimizer = optim.Adam(params)

    os.makedirs(os.path.join('checkpoints', args.name), exist_ok=True)
    for epoch in range(100):
        epoch_loss = 0
        for i, batch in tqdm(enumerate(train_loader), total=1000):
            # squeeze unnecessary batch dimension
            batch = batch.squeeze(0).cuda()

            # Split batch to input, target
            input_state = batch[0]
            target_pos = batch[1:, :4]

            pred_t, pred_state = model(input_state)
            pred_pos = pred_state[:, :2]

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
            epoch_loss += loss.item()
        writer.add_scalar('epoch_loss', epoch_loss / len(trainset))
        torch.save(model.state_dict(), os.path.join('checkpoints', args.name, f'{epoch}.pth'))

'''
TODO:
- tensorboard logging.
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default='default')
    parser.add_argument("--event_lr", default=0.0005, type=float)
    parser.add_argument("--inst_lr", default=0.0001, type=float)
    parser.add_argument("--dt", default=1./30., type=float)
    parser.add_argument("--clip_grad", default=5.0, type=float)
    parser.add_argument("--seq_len", default=25, type=int)
    parser.add_argument("--no_subseq", action='store_true')

    parser.add_argument("--layer_norm", action='store_true')
    args = parser.parse_args()
    train(args)
