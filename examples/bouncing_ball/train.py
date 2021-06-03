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

class BouncingBall(Dataset):

    def __init__(self, root='train.npy', seq_len=25, subseq=True):
        super().__init__()
        self.data = np.load(root)
        self.subseq = subseq
        self.seq_len = seq_len

    def __getitem__(self, item):
        # Parse 25-length subsequence.
        # Get Velocity. [pos, vel]
        pos = self.data[item]
        velocity = pos[1:] - pos[:-1]
        velocity = np.concatenate([np.zeros([1, 4]), velocity], 0)

        if self.subseq:
            start = random.randint(0, 74)
        else:
            start = 0
        end = start+self.seq_len
        state = np.concatenate([pos, velocity], -1)
        return state[start:end] # --> [25, 8]

    def __len__(self):
        return len(self.data)

class DynsSolver(nn.Module):

    def __init__(self, dt=1./30., seq_len=25, layer_norm=False, adjoint=True):
        super().__init__()
        self.event_fn = EventFn(hidden_dim=128, layer_norm=layer_norm)
        self.dynamics = GravityDyns(dt)
        self.inst_func = InstFn(hidden_dim=512, n_hidden=3, layer_norm=layer_norm)
        self.seq_len = seq_len

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
                                            odeint_interface=odeint_adjoint, method='dopri5')
                except AssertionError:
                    t, state = odeint_event(self.dynamics, state, t, event_fn=self.event_fn, reverse_time=False,
                                            odeint_interface=odeint_adjoint, method='rk4', options={'step_size': 0.01})
                act = self.event_fn.mlp(state[-1, :4])
                state = self.inst_func(t, state[-1, :], act)
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
                results = odeint_adjoint(self.dynamics, y0, sample_ts, method='dopri5')[1:]
            except AssertionError:
                results = odeint_adjoint(self.dynamics, y0, sample_ts,
                                         method='rk4', options={'step_size': 0.01})[1:]
            total_results += results

            last_t = event_times[i]

        return torch.cat(total_ts), torch.stack(total_results)


class GravityDyns(nn.Module):
    def __init__(self, dt, gravity=9.8, hidden_dim=256):
        super().__init__()
        gravity = gravity * dt * dt
        self.gravity = torch.tensor([0., gravity, 0., gravity], requires_grad=True).cuda()

    def forward(self, t, state):
        dx = state[4:]

        # dv --> constant for event ODE, MLP for NODE.
        return torch.cat([dx, self.gravity], dim=-1)

class EventFn(nn.Module): # positions --> scalar
    def __init__(self, hidden_dim, layer_norm=False):
        super().__init__()
        # gets t, state(x y x y v v v v)
        if layer_norm:
            self.mlp = nn.Sequential(
                nn.Linear(4, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 8)
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(4, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 8)
            )

    def forward(self, t, state):
        input = state[:4] # The event function took as input the positions of the two balls.
        output = torch.tanh(self.mlp(input))
        return torch.prod(output, dim=-1)

class InstFn(nn.Module): # state + activation --> state
    def __init__(self, hidden_dim=512, n_hidden=3, layer_norm=False):
        super().__init__()
        # gets t, state(x y x y v v v v)
        mlp = []
        mlp.append(nn.Linear(16, hidden_dim))
        if layer_norm:
            mlp.append(nn.LayerNorm(hidden_dim)),
        mlp.append(nn.ReLU())
        for _ in range(n_hidden):
            mlp.append(nn.Linear(hidden_dim, hidden_dim))
            if layer_norm:
                mlp.append(nn.LayerNorm(hidden_dim)),
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(hidden_dim, 8))
        self.mlp = nn.Sequential(*mlp)

    def forward(self, t, state, event_act):
        input = torch.cat((state, event_act), dim=-1)
        output = self.mlp(input)
        output = torch.tanh(output)
        return output

def train(args):
    writer = SummaryWriter(os.path.join('logs', args.name))
    trainset = BouncingBall('train.npy', args.seq_len, not args.no_subseq)
    valset = BouncingBall('val.npy')
    testset = BouncingBall('test.npy')

    # batchsize == 1
    train_loader = DataLoader(trainset, shuffle=True, num_workers=16)
    # val_loader = DataLoader(valset, shuffle=False)
    # test_loader = DataLoader(testset, shuffle=False)

    model = DynsSolver(dt=args.dt, seq_len=args.seq_len, layer_norm=args.layer_norm, )
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
            pred_pos = pred_state[:, :4]
            # for stability
            pred_pos = torch.clamp(pred_pos, min=0, max=5)

            # MSE Loss
            loss = F.mse_loss(pred_pos, target_pos)

            try:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()
            except AssertionError:
                pass

            writer.add_scalar('loss', loss.item(), epoch * len(trainset) + i)
            epoch_loss += loss.item()
        torch.save(model.state_dict(), os.path.join('checkpoints', args.name, f'{epoch}.pth'))
        writer.add_scalar('epoch_loss', epoch_loss/len(trainset))

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
