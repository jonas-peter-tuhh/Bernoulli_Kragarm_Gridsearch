# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 17:20:05 2022
@author: Jonas Peter
"""
import random
import time
import scipy.integrate
import torch
import torch.nn as nn
from torch.autograd import Variable
import scipy as sp
import scipy.integrate as integrate
from scipy.integrate import quad
import scipy.special as special
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splrep, splev
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train = True


class Net(nn.Module):
    def __init__(self, num_layers, layers_size):
        super(Net, self).__init__()
        assert num_layers == len(layers_size)
        self.linears = nn.ModuleList([nn.Linear(1, layers_size[0])])
        self.linears.extend([nn.Linear(layers_size[i-1], layers_size[i])
                            for i in range(1, num_layers)])
        self.linears.append(nn.Linear(layers_size[-1], 1))

    def forward(self, x):  # ,p,px):
        # torch.cat([x,p,px],axis=1) # combined two arrays of 1 columns each to one array of 2 columns
        inputs = x
        for i in range(len(self.linears)-1):
            x = torch.tanh(self.linears[i](x))
        output = self.linears[-1](x)
        return output


##
choice_load = input("Möchtest du ein State_Dict laden? (y/n): ")
if choice_load == 'y':
    train = False
    filename = input("Welches State_Dict möchtest du laden?")
    net = Net()
    net = net.to(device)
    net.load_state_dict(torch.load(
        'C:\\Users\\Administrator\\Desktop\\Uni\\Master\\Masterarbeit\\Timoshenko_Kragarm_5.1_v2\\saved_data\\'+filename))
    net.eval()
##
# Hyperparameter
learning_rate = 0.01


# Definition der Parameter des statischen Ersatzsystems
Lb = float(input('Länge des Kragarms [m]: '))
EI = 21  # float(input('EI des Balkens [10^6 kNcm²]: '))
LFS = 1  # int(input('Anzahl Streckenlasten: '))

Ln = np.zeros(LFS)
Lq = np.zeros(LFS)
s = [None] * LFS
normfactor = 10/((11*Lb**5)/(120*EI))

for i in range(LFS):
    # ODE als Loss-Funktion, Streckenlast
    # float(input('Länge Einspannung bis Anfang der ' + str(i + 1) + '. Streckenlast [m]: '))
    Ln[i] = 0
    # float(input('Länge der ' + str(i + 1) + '. Streckenlast [m]: '))
    Lq[i] = Lb
    # input(str(i + 1) + '. Streckenlast eingeben: ')
    s[i] = str(normfactor)+"*x"


def h(x, j):
    return eval(s[j])


def f(x, net):
    u = net(x)
    u_x = torch.autograd.grad(
        u, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
    u_xx = torch.autograd.grad(
        u_x, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
    u_xxx = torch.autograd.grad(
        u_xx, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
    u_xxxx = torch.autograd.grad(
        u_xxx, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
    ode = 0
    for i in range(LFS):
        ode += u_xxxx + (h(x - Ln[i], i))/EI
    return ode


x = np.linspace(0, Lb, 1000)
pt_x = torch.unsqueeze(Variable(torch.from_numpy(
    x).float(), requires_grad=True).to(device), 1)
qx = np.zeros(1000)
for i in range(LFS):
    qx = qx + (h(torch.unsqueeze(Variable(torch.from_numpy(x).float(), requires_grad=False).to(device),
               1) - Ln[i], i)).cpu().detach().numpy().squeeze() * (x <= (Ln[i] + Lq[i])) * (x >= Ln[i])

Q0 = integrate.cumtrapz(qx, x, initial=0)

qxx = (qx) * x

M0 = integrate.cumtrapz(qxx, x, initial=0)


def gridSearch(num_layers, layers_size):
    start = time.time()
    net = Net(num_layers, layers_size)
    net = net.to(device)
    mse_cost_function = torch.nn.MSELoss()  # Mean squared error
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=200, verbose=True, factor=0.75)
    if train:
        # + net_S(torch.unsqueeze(Variable(torch.from_numpy(x).float(), requires_grad=False).to(device), 1))
        y1 = net(torch.unsqueeze(Variable(torch.from_numpy(
            x).float(), requires_grad=False).to(device), 1))
        fig = plt.figure()
        plt.grid()
        ax1 = fig.add_subplot()
        ax1.set_xlim([0, Lb])
        ax1.set_ylim([-20, 0])
        # ax2.set_
        net_out_plot = y1.cpu().detach().numpy()
        line1, = ax1.plot(x, net_out_plot)
        plt.show(block=False)
        f_anal = (-1/120 * normfactor * pt_x**5 + 1/6 *
                  Q0[-1] * pt_x**3 - M0[-1]/2 * pt_x**2)/EI

    iterations = 1000000
    for epoch in range(iterations):
        optimizer.zero_grad()  # to make the gradients zero
        x_bc = np.linspace(0, Lb, 500)
        pt_x_bc = torch.unsqueeze(Variable(torch.from_numpy(
            x_bc).float(), requires_grad=True).to(device), 1)
        # unsqueeze wegen Kompatibilität
        pt_zero = Variable(torch.from_numpy(np.zeros(1)).float(),
                           requires_grad=False).to(device)

        x_collocation = np.random.uniform(
            low=0.0, high=Lb, size=(250 * int(Lb), 1))
        #x_collocation = np.linspace(0, Lb, 1000*int(Lb))
        all_zeros = np.zeros((250 * int(Lb), 1))

        pt_x_collocation = Variable(torch.from_numpy(
            x_collocation).float(), requires_grad=True).to(device)
        pt_all_zeros = Variable(torch.from_numpy(
            all_zeros).float(), requires_grad=False).to(device)
        # ,pt_px_collocation,pt_p_collocation,net)
        f_out = f(pt_x_collocation, net)

        # Randbedingungen
        net_bc_out = net(pt_x_bc)

        u_x = torch.autograd.grad(net_bc_out, pt_x_bc, create_graph=True,
                                  retain_graph=True, grad_outputs=torch.ones_like(net_bc_out))[0]
        u_xx = torch.autograd.grad(u_x, pt_x_bc, create_graph=True,
                                   retain_graph=True, grad_outputs=torch.ones_like(net_bc_out))[0]
        u_xxx = torch.autograd.grad(u_xx, pt_x_bc, create_graph=True,
                                    retain_graph=True, grad_outputs=torch.ones_like(net_bc_out))[0]

        e1 = u_x[0]
        e2 = net_bc_out[0]
        e3 = u_xxx[0] - Q0[-1]/EI
        e4 = u_xx[0] + M0[-1]/EI
        e5 = u_xxx[-1]
        e6 = u_xx[-1]

        mse_bc = mse_cost_function(e1, pt_zero) + mse_cost_function(e2, pt_zero) + mse_cost_function(
            e5, pt_zero) + mse_cost_function(e6, pt_zero) + mse_cost_function(e3, pt_zero) + mse_cost_function(e4, pt_zero)
        mse_f = mse_cost_function(f_out, pt_all_zeros)
        loss = mse_f + mse_bc

        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        with torch.autograd.no_grad():
            if epoch % 10 == 9:
                print(epoch, "Traning Loss:", loss.data)
                plt.grid()
                net_out_v = net(pt_x)
                #net_out_v = net_out.cpu().detach().numpy()
                err = torch.norm(net_out_v - f_anal, 2)
                print(f'Error = {err}')
                if epoch > 5000:
                    return 'Schlechte Konfiguration'
                if err < 0.1 * Lb:
                    print(
                        f"Die L^2 Norm des Fehlers ist {err}.\nStoppe Lernprozess")
                    end = time.time()
                    return end-start
                line1.set_ydata(net(torch.unsqueeze(Variable(torch.from_numpy(
                    x).float(), requires_grad=False).to(device), 1)).cpu().detach().numpy())
                fig.canvas.draw()
                fig.canvas.flush_events()
    ##


# GridSearch
time_elapsed = []

for num_layers in range(2, 10):
    for _ in range(10):  # Wieviele zufällige layers_size pro num_layers
        layers_size = [np.random.randint(2, 15) for _ in range(num_layers)]
        time_elapsed.append(
            (num_layers, layers_size, gridSearch(num_layers, layers_size)))
        plt.close()

with open(r'timing.txt', 'w') as fp:
    for item in time_elapsed:
        # write each item on a new line
        fp.write(f'{item} \n')
