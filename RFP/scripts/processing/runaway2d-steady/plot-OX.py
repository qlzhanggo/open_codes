#! /usr/bin/env python

import matplotlib
#matplotlib.use('qt4agg')

import matplotlib.pyplot as plt
import numpy as np
from scipy import special as sp

import argparse

from scipy.interpolate import griddata
from scipy import interpolate
from scipy import integrate

from runaway import runaway
from runaway import zeta1

from matplotlib import rcParams
rcParams.update({'font.size':16})
rcParams.update({'lines.linewidth':2})
rcParams.update({'axes.labelpad':0})
rcParams.update({'axes.labelsize':'large'})
rcParams['figure.figsize'] = 6, 5

from scipy import optimize

def plot_stream(x, y, xip12, vp12, Gp, Gxi, E) :
    print("generating 2D stream lines on momentum-space")

    func1 = interpolate.RectBivariateSpline(xip12, vp12, Gp, kx=1, ky=1)
    func2 = interpolate.RectBivariateSpline(xip12, vp12, Gxi, kx=1, ky=1)

    def find_ox(x) :
        return [func1.ev(x[1], x[0]), func2.ev(x[1], x[0])]

    po = optimize.root(find_ox, [12., -0.9], method='hybr')
    print("PO is: {}".format(po.x))

    px = optimize.root(find_ox, [3., -0.7], method='hybr')
    print("PX is: {}".format(px.x))
                
    fig = plt.figure(figsize=(6,4))
    ax  = fig.add_subplot(111)

    ax.streamplot(x, y, func1(y,x), func2(y,x), density=3, linewidth=2)

    ax.set_xlim([1, 15])
    ax.set_ylim([-1, 0])

    ax.set_xlabel('$p$',color='black',fontsize=18,labelpad=-1)
    ax.set_ylabel(r'$\xi $',fontsize=18,rotation=0)
    plt.title('momentum-space flux(E={0})'.format(E))
    plt.tight_layout()

    fig.savefig("flux.eps")
    return (po.x[0], px.x[0])
    
xmin, xmax, ymin, ymax, nx, ny = (1.0, 25.0, -1.0, 0, 300, 200)
x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)   

m=6 
E = np.linspace(1.77, 1.82, m)
E = np.asarray([1.665])
#E = np.asarray([2.40, 2.42, 2.44, 2.46, 2.48, 2.50, 2.52, 2.54, 2.57, 2.59, 2.61, 2.63, 2.65])
#E = np.asarray([1.745])

po = np.zeros_like(E)
px = np.zeros_like(E)

for i in range(E.size) :
  dir_ = 'E_{:.3f}'.format(E[i])

  fe = runaway('{}/mesh.m'.format(dir_), '{}/input_params.m'.format(dir_), '{}/soln.dat'.format(dir_))

  fe.calFlux(1.e-25)

  po[i],px[i] = plot_stream(x, y, fe.xip12, fe.vp12, fe.Gp, fe.Gxi, E[i])
  
    
E = np.array2string(E, separator=",")
po = np.array2string(po, separator=",")
px = np.array2string(px, separator=",")

print("E = {}".format(E))
print("po = {}".format(po))
print("px = {}".format(px))

plt.show()
    
