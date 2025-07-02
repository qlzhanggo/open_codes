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



class draw_re() :

  def __init__(self, rfp) :
    self.__dist = rfp.dist
    self.__vp12 = rfp.vp12
    self.__xip12 = rfp.xip12
    self.__vte = rfp.param['vte']
    self.__xmin = rfp.xmin
    self.__xmax = rfp.xmax
    self.__rfp = rfp

# plot the 2D contour in p,xi
  def plot_fpxi(self, file=None) :
    print("generating 2D distribution contour")
    fig = plt.figure()
    ax = fig.add_subplot(111)

    levels = np.linspace(-30, -6, 25)
    cmap = plt.cm.get_cmap("jet")
    cmap.set_under("white")
    cmap.set_over("darkred")

    xi0 = np.sqrt(2*self.__rfp.param['asp']*self.__rfp.param['ra']/(1+self.__rfp.param['asp']*self.__rfp.param['ra']))
    ax.plot(self.__vp12, xi0*np.ones(self.__vp12.size), linewidth=2, color='black')
    ax.plot(self.__vp12, -xi0*np.ones(self.__vp12.size), linewidth=2, color='black')

    print(np.max((self.__dist[:,:]*self.__vp12[None,:]))
    fvxi = ax.contourf(self.__vp12, self.__xip12, np.log10(self.__dist[:,:]*self.__vp12[None,:]+1e-40),  levels, cmap=cmap, extend='both')
    cbar = plt.colorbar(fvxi)

    plt.grid()
    ax.set_xlabel(r'p/mc', fontsize=14, color='black')
    ax.set_ylabel(r'$\xi$', fontsize=14)
    #ax.legend(loc="upper right", ncol=1, shadow=False, fancybox=False)
    plt.title('distribution function')
    #plt.show()
    if file is not None:
      plt.savefig(file)

  # plot the 1D pitch-angle distribution
  def plot_fxi(self, file=None) :
    print("generating 1D pitch-angle distributuon plot")
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.plot(self.__xip12, np.log10(self.__dist[:,50] + 1e-30), linewidth=2, color='red', label=r'p=%.2g'%(self.__vp12[50]))
    ax.plot(self.__xip12, np.log10(self.__dist[:,100] + 1e-30), linewidth=2, color='blue', label=r'p=%.2g'%(self.__vp12[100]))
    ax.plot(self.__xip12, np.log10(self.__dist[:,250] + 1e-30), linewidth=2, color='cyan', label=r'p=%.2g'%(self.__vp12[250]))
    ax.plot(self.__xip12, np.log10(self.__dist[:,350] + 1e-30), linewidth=2, color='black', label=r'p=%.2g'%(self.__vp12[350]))

    ax.axis([-1., 0, -18, 0])
    ax.set_xlabel(r'$\xi$', fontsize=18, color='black')
    plt.title(r'$f(\xi)$')
    plt.legend(loc='upper right')
    #plt.show()
    if file is not None :
      plt.savefig(file)

  def plot(self) :
     plt.show()


print('----------------------------------------------------------------------')
print('---- Postprocessing RFP result for runaway electron distributions ----')
print('----------------------------------------------------------------------')

parser = argparse.ArgumentParser(description='read and plot the distribution data, calculate the runaway rate')
parser.add_argument('-step', metavar='step', type=int, nargs='?', default=0,
                   help='the time step to process')
parser.add_argument('-flux', metavar='flux', type=bool, nargs='?', default=False,
                   help='the flat to calculate phase-space flux')
parser.add_argument('-dir', metavar='dir', type=str, nargs='?', default=None,
                   help='the folder with data file')
parser.add_argument('-outsuffix', metavar='outsuffix', type=str, nargs='?', default='eps',
                   help='suffix of files containing plots')
parser.add_argument('-no-show', dest='show', action='store_false',
                   help='disable showing plots')
parser.add_argument('-petscint64', dest='petscint64', action='store_true',
                   help='flag if Petsc was compiled with 64-bit integers')
parser.set_defaults(show=True)
parser.set_defaults(petscint64=False)

args = parser.parse_args()

if args.dir is None :
  fe = runaway('mesh.m', 'input_params.m', 'output_data/kon_{}.dat'.format(args.step), petscint64=args.petscint64)
else :
  fe = runaway('{}/mesh.m'.format(args.dir), '{}/input_params.m'.format(args.dir), '{0}/output_data/soln_{1}.dat'.format(args.dir,args.step), petscint64=args.petscint64)

fe.output_ne(1.)

draw = draw_re(fe)

draw.plot_fpxi('dist_{}.{}'.format(args.step, args.outsuffix))

if (args.flux==True) :
    draw.plot_stream(1.0, 25.0, -1., -0, 300, 200)

if args.show:
    draw.plot()

