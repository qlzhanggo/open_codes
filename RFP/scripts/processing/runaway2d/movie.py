#! /usr/bin/env python

import matplotlib
matplotlib.use('qt4agg')

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

# plot the 1D energy distribution
  def plot_fp(self, file=None) :
    print("generating 1D energy distribution plot")
    vte = self.__vte
    fmax = 1.0/(vte**3 * np.pi*np.sqrt(np.pi)) * np.exp((1-np.sqrt(1+self.__vp12**2))/(0.5*vte**2))
    fv_avg = 0.5*integrate.simps(self.__dist, self.__xip12, axis=0, even='avg')

    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.plot(self.__vp12, np.log10(self.__dist[0,:] + 1e-25), linewidth=2, color='red', label=r"$\xi=%.2g$"%(self.__xip12[0]))
    ax.plot(self.__vp12, np.log10(fv_avg + 1.e-25), linewidth=2, color='black', label=r"average")
    ax.plot(self.__vp12, np.log10(fmax + 1e-25), '--', linewidth=2)

    #ax.plot(vp12, -14.71*np.ones(vp12.size), '--',linewidth=2)
    ax.axis([self.__xmin, self.__xmax, -18, 0])
    ax.set_xlabel(r'p/mc', fontsize=18, color='black')
    plt.title('fv')
    plt.legend(loc='upper right')
    #plt.show()
    if file is not None:
      plt.savefig(file)


# plot the 2D contour in p,xi
  def plot_fpxi(self, file=None) :
    print("generating 2D distribution contour")
    fig = plt.figure()
    ax = fig.add_subplot(111)

    levels = np.linspace(-20, -7, 25)
    cmap = plt.cm.get_cmap("jet")
    cmap.set_under("white")
    cmap.set_over("darkred")

    xi0 = np.sqrt(2*self.__rfp.param['asp']*self.__rfp.param['ra']/(1+self.__rfp.param['asp']*self.__rfp.param['ra']))
    ax.plot(self.__vp12, xi0*np.ones(self.__vp12.size), linewidth=2, color='black')
    ax.plot(self.__vp12, -xi0*np.ones(self.__vp12.size), linewidth=2, color='black')

    fvxi = ax.contourf(self.__vp12, self.__xip12, np.log10(self.__dist[:,:]+1.05e-22),  levels, cmap=cmap, extend='both')
    cbar = plt.colorbar(fvxi)

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

  # interpolate nonuniform grid data and do streamplot
  def plot_stream(self, xmin, xmax, ymin, ymax, nx, ny, file=None) :
    print("generating 2D stream lines on momentum-space")
    if (self.__rfp.Gp is None or self.__rfp.Gxi is None):
      self.__rfp.calFlux(1.05e-21)

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)

    func1 = interpolate.RectBivariateSpline(self.__xip12, self.__vp12, self.__rfp.Gp, kx=1, ky=1)
    func2 = interpolate.RectBivariateSpline(self.__xip12, self.__vp12, self.__rfp.Gxi, kx=1, ky=1)

    fig = plt.figure()
    ax  = fig.add_subplot(111)

    ax.streamplot(x, y, func1(y,x), func2(y,x), density=3, linewidth=2)
    plt.title('phase space flux(E={0})'.format(fe.param['E']))
    #plt.show()
    if file is not None :
      plt.savefig(file)

  def plot(self) :
     plt.show()


print('----------------------------------------------------------------------')
print('---- Postprocessing RFP result for runaway electron distributions ----')
print('----------------------------------------------------------------------')

parser = argparse.ArgumentParser(description='read and plot the distribution data, calculate the runaway rate')
parser.add_argument('-dir', metavar='dir', type=str, nargs='?', default=None,
                   help='the folder with data file')

args = parser.parse_args()

if args.dir is None :
  fe = runaway('mesh.m', 'input_params.m', 'output_data/soln_{}.dat'.format(args.step))
else :
  fe = runaway('{}/mesh.m'.format(args.dir), '{}/input_params.m'.format(args.dir), '{0}/output_data/soln_{1}.dat'.format(args.dir,args.step))

fe.output_ne(1.)

draw = draw_re(fe)

draw.plot_fpxi()

draw.plot_fp()

#draw.plot_fxi()

if (args.flux==True) :
    draw.plot_stream(1.0, 25.0, -1., -0, 300, 200)
 
draw.plot()


# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, 4), ylim=(-15, 1))
line, = ax.plot([], [], lw=2)

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

# animation function.  This is called sequentially
def animate(i):
    x = np.linspace(0, 2, 1000)
    y = np.sin(2 * np.pi * (x - 0.01 * i))
    line.set_data(x, y)
    return line,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=200, interval=20, blit=True)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
anim.save('fv.mp4', fps=10, extra_args=['-vcodec', 'libx264'])

plt.show()
