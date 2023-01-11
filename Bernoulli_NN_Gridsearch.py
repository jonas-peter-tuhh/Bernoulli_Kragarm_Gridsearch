# -*- coding: utf-8 -*-
##Wichtig!
##Wichtig!
##!!!Achtung!!! Helpers.py Datei notwendig!!! siehe Helpers-respository --> |https://github.com/jonas-peter-tuhh/Helpers|
#Helpers.py - Datei muss einen Ordner über entsprechenden PINN abgespeichert werden!
"""
Created on Mon Aug 29 17:20:05 2022
@author: Jonas Peter
"""
import sys
import os
#insert path of parent folder to import helpers
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import torch
import torch.nn as nn
from torch.autograd import Variable
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np
from helpers import *
import warnings
import time
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
        x = torch.unsqueeze(x, 1)
        for i in range(len(self.linears)-1):
            x = torch.tanh(self.linears[i](x))
        output = self.linears[-1](x)
        return output
##
# Hyperparameter
# Hyperparameter

learning_rate = 0.01
mse_cost_function = torch.nn.MSELoss()  # Mean squared error


# Definition der Parameter des statischen Ersatzsystems
Lb = float(input('Länge des Kragarms [m]: '))
EI = 21

#Normierungsfaktor (siehe Kapitel 10.3)
normfactor = 10/((11*Lb**5)/(120*EI))

# ODE als Loss-Funktion, Streckenlast
Ln = 0 #float(input('Länge Einspannung bis Anfang der ' + str(i + 1) + '. Streckenlast [m]: '))
Lq = Lb # float(input('Länge der ' + str(i + 1) + '. Streckenlast [m]: '))
s = str(normfactor)+"*x"#input(str(i + 1) + '. Streckenlast eingeben: ')


def h(x):
    return eval(s)

def f(x, net):
    u = net(x)
    _,_,_,u_xxxx = deriv(u,x,4)
    ode = u_xxxx + (h(x - Ln))/EI
    return ode


x = np.linspace(0, Lb, 1000)
qx = h(x)* (x <= (Ln + Lq)) * (x >= Ln)


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
        optimizer, patience=200, verbose=True, factor=0.8)
    if train:
        y1 = net(myconverter(x, False))
        fig = plt.figure()
        plt.grid()
        ax1 = fig.add_subplot()
        ax1.set_xlim([0, Lb])
        ax1.set_ylim([-10, 0])
        net_out_plot = myconverter(y1)
        line1, = ax1.plot(x, net_out_plot)
        plt.show(block=False)
        pt_x = myconverter(x)
        f_anal = (-1 / 120 * normfactor * x ** 5 + 1 / 6 * Q0[-1] * x ** 3 - M0[-1] / 2 * x ** 2) / EI

    iterations = 1000000
    for epoch in range(iterations):
        optimizer.zero_grad()  # to make the gradients zero
        x_bc = np.linspace(0, Lb, 500)
        pt_x_bc = myconverter(x_bc)

        x_collocation = np.random.uniform(low=0.0, high=Lb, size=(250 * int(Lb), 1))
        all_zeros = np.zeros((250 * int(Lb), 1))

        pt_x_collocation = myconverter(x_collocation)
        pt_all_zeros = myconverter(all_zeros, False)
        f_out = f(pt_x_collocation, net)

        # Randbedingungen (siehe Kapitel 10.5)
        net_bc_out = net(pt_x_bc)
        u_x, u_xx, u_xxx = deriv(net_bc_out, pt_x_bc, 3)
        BC3 = net_bc_out[0]
        BC6 = u_xxx[0] - Q0[-1] / EI
        BC7 = u_xxx[-1]
        BC8 = u_xx[0] + M0[-1] / EI
        BC9 = u_xx[-1]
        BC10 = u_x[0]

        mse_Gamma = errsum(mse_cost_function, BC3, BC6, BC7, BC8, BC9, BC10)
        mse_Omega = mse_cost_function(f_out, pt_all_zeros)
        loss = mse_Gamma + mse_Omega

        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        with torch.autograd.no_grad():
            if epoch % 10 == 9:
                print(epoch, "Traning Loss:", loss.data)
                plt.grid()
                net_out_v = myconverter(net(pt_x))
                err = np.linalg.norm(net_out_v - f_anal, 2)
                print(f'Error = {err}')
                if err < 0.1 * Lb:
                    print(f"Die L^2 Norm des Fehlers ist {err}.\nStoppe Lernprozess")
                    # plt.close(fig)
                    break
                line1.set_ydata(myconverter(net(myconverter(x, False))))
                fig.canvas.draw()
                fig.canvas.flush_events()
    ##

# GridSearch
time_elapsed = []
num_layers= 1
for j in range(50, 100, 50):  # Wieviele zufällige layers_size pro num_layers
        layers_size = [j]
        t = 1
        for _ in range(2):
            t += gridSearch(num_layers, layers_size)
            plt.close()
        time_elapsed.append((num_layers, layers_size, t/2))


with open(r'geordnet10m50N.txt', 'w') as fp:
    for item in time_elapsed:
        # write each item on a new line
        fp.write(f'{item} \n')

# os.system('shutdown /s /t 1')
