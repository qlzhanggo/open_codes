#! /usr/bin/env python

import os
import glob
import re

import numpy as np
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from scipy import interpolate
from scipy import integrate

from runaway import runaway

from matplotlib import rcParams
rcParams.update({'font.size':18})
rcParams.update({'lines.linewidth':2})
rcParams.update({'axes.labelpad':0})
rcParams.update({'axes.labelsize':'large'})

# a function to sort the string array in natual order
def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def dsigmadp(gme, gm):
    nu = (gm - 1.)/(gme-1.)
    x = 1./(nu*(1.-nu))
    temp = np.where(nu<1., np.sqrt(gm**2-1.)/gm * gme**2/(gme-1.)**3/(gme+1.) * (x**2 - 3.*x + ((gme-1.)/gme)**2 *(1.+x)), 0.)
    return temp

# integrated cross-section
def sigma(gm, gmin) :
    return np.where(gm>=(2.*gmin-1.), 1./(gm*gm-1.) * ( 0.5*(gm+1.) - gmin - gm*gm*(1./(gm-gmin) - 1./(gmin-1.)) + (2.*gm-1.)/(gm-1.) * np.log((gmin-1.)/(gm-gmin)) ), 0.)


print('----------------------------------------------------------------------')
print('---- Postprocessing RFP result for runaway avalanche growth rate ----')
print('----------------------------------------------------------------------')

t = []
ne = []
je = []
ke = []

with open('output_data/time.txt', 'r') as f:
    f.readline()
    line = f.readline()
    prevt = -0.5
    while line:
        line = line.strip()
        line = line.strip(";")
        if line:
           data = line.split()

           # parse the time #t and time step #step, then obtain density from the dist file at #step              
           if data[0] == 'step:':
               if (float(data[3]) >= (prevt + 0.2) and float(data[3])<=300.0) :
                   t.append(data[3])
                   prevt = float(data[3])

                   fe = runaway('mesh.m', 'input_params.m', 'output_data/soln_{}.dat'.format(data[1]), mute=True)
                   nj = fe.output_ne(0.5)
                   ne.append(nj[0])
                   je.append(nj[1])
                   ke.append(nj[2])

        line = f.readline()

a1 = np.polyfit(np.asarray(t, dtype=np.float64)[200::], np.log(np.asarray(ne, dtype=np.float64)[200::]),1)
print('growth rate is {0}'.format(a1))

# plot the time history 
net = np.vstack((t, ne, je, ke))
np.save('net.npy', net)

t = t[0::]
ne = np.asarray(ne[0::], dtype=np.float64)
je = np.asarray(je[0::], dtype=np.float64)
ke = np.asarray(ke[0::], dtype=np.float64)

fig1 = plt.figure(1)
plt.semilogy(t, ne-ne[0]+1.e-16, linewidth=2)
#plt.plot(t, ne, linewidth=2)
ax1 = fig1.gca()
ax1.set_xlim([0, 80])
ax1.set_xlabel(r'$t/\tau_c$', color='black')
#plt.title(r'$\log_{10} [ne(t)-ne_0]$')
plt.title(r'$n_e(t)$')
plt.savefig('net.eps')

fig1 = plt.figure(2)
plt.semilogy(t, (-je +1.e-16), '-', linewidth=2)
#plt.plot(t, je, linewidth=2)
ax1 = fig1.gca()
ax1.set_xlim([0, 80])
ax1.set_xlabel(r'$t/\tau_c$', color='black')
#ax1.axis([-xmax, xmax, -25, 0])
plt.title(r'$j(t)]$')
plt.savefig('jet.eps')


fig1 = plt.figure(3)
#ax1  = fig.add_subplot(111)
#plt.plot(t, np.log10(-ke/je), '-', linewidth=2)
plt.plot(t, ke, linewidth=2)
ax1 = fig1.gca()
ax1.set_xlabel(r'$t/\tau_c$', color='black')
ax1.set_xlim([0, 80])
#ax1.axis([-xmax, xmax, -25, 0])
plt.title(r'$k_e(t)$')
plt.savefig('ket.eps')

plt.show()


