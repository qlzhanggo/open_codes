#! /usr/bin/env python

#import matplotlib
#matplotlib.use('qt4agg')

import matplotlib.pyplot as plt
import numpy as np
from scipy import special as sp
from scipy import integrate
import matplotlib.ticker as ticker

import argparse

from scipy.interpolate import griddata
from scipy import interpolate

from matplotlib import rcParams
rcParams.update({'font.size':18})
rcParams.update({'lines.linewidth':2})
rcParams.update({'axes.labelpad':0})
rcParams.update({'axes.labelsize':'large'})


from runaway import runaway

def zeta1(xi, eps) :
    if (eps>0) :
      xic2 = 2.*eps/(1.+eps)
      kappa2 = (1. + (xi*xi/xic2 - 1.)/(1. - xi*xi))
      return np.where(xi*xi>xic2, 2./np.pi * sp.ellipk(1./kappa2), 4.*np.sqrt(kappa2)/np.pi * sp.ellipk(kappa2))
    else :
      return 1.0

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

    N = self.__rfp.N
    hy = np.zeros(N)
    hy[0] = 0.5*(self.__xip12[1]-self.__xip12[0]) + self.__xip12[0]+1.0
    hy[N-1] = 0.5*(self.__xip12[N-1]-self.__xip12[N-2]) + 1.0 - self.__xip12[N-1]
    hy[1:N-1] = 0.5*(self.__xip12[2:N]-self.__xip12[0:N-2])

    if (self.__rfp.eps>0) :
      print('eps = {}'.format(self.__rfp.eps))
      zeta1_ = zeta1(self.__xip12, self.__rfp.eps)
      N1 = self.__rfp.My2
      N2 = 2*self.__rfp.My2-self.__rfp.My1
      fv_avg = np.sum(self.__dist[0:N1,:]*zeta1_[0:N1,None]*hy[0:N1,None], axis=0)+np.sum(self.__dist[N2::,:]*zeta1_[N2::,None]*hy[N2::,None], axis=0)
    else :
      fv_avg = integrate.simps(self.__dist, self.__xip12, axis=0, even='avg')

    fig = plt.figure()
    ax  = fig.add_subplot(111)
    #ax.plot(self.__vp12, np.log10(self.__dist[0,:] + 1e-35), linewidth=2, color='red', label=r"$\xi=%.2g$"%(self.__xip12[0]))
    # ax.plot(self.__vp12, np.log10(fv_avg + 1.e-35), linewidth=2, color='black', label=r"average")
    # ax.plot(self.__vp12, np.log10(fmax + 1e-35), '--', linewidth=2)

    # ax.plot(np.log10(self.__vp12), np.log10(fv_avg + 1.e-35), linewidth=2, color='black', label=r"average")
    # ax.plot(np.log10(self.__vp12), np.log10(fmax + 1e-35), '--', linewidth=2)

    ax.semilogx(self.__vp12, np.log10(fv_avg + 1.e-35), linewidth=2, color='black', label=r"average")
    ax.semilogx(self.__vp12, np.log10(fmax + 1.e-35), '--', linewidth=2)

    #ax.plot(vp12, -14.71*np.ones(vp12.size), '--',linewidth=2)
    # ax.axis([0, 40, -20.02, 0])
    plt.ylim(-20.02, 0)
    ax.set_xlabel(r'$p/mc$', fontsize=14, color='black')
    ax.set_ylabel(r'$\log_{10} f$', fontsize=14, color='black')
    # plt.title('fv')
    plt.legend(loc='upper right')
    plt.tight_layout()
    #plt.show()
    if file is not None:
      plt.savefig(file)

# plot the 2D contour in p,xi
  def plot_fpxi(self, file=None) :
    print("generating 2D distribution contour")
    fig = plt.figure()
    ax = fig.add_subplot(111)

    levels = np.linspace(-22, -8, 25)
    cmap = plt.cm.get_cmap("jet")
    cmap.set_under("white")
    cmap.set_over("darkred")

    xi0 = np.sqrt(2*self.__rfp.param['asp']*self.__rfp.param['ra']/(1+self.__rfp.param['asp']*self.__rfp.param['ra']))
    ax.plot(self.__vp12, xi0*np.ones(self.__vp12.size), linewidth=2, color='black')
    ax.plot(self.__vp12, -xi0*np.ones(self.__vp12.size), linewidth=2, color='black')

    fvxi = ax.contourf(self.__vp12, self.__xip12, np.log10(self.__dist[:,:]+1.05e-22),  levels, cmap=cmap, extend='both')
    #ax.set_xscale("log")
    # plt.xlim(np.log10(5), 40)
    # ax.set_yscale("log")
    cbar = plt.colorbar(fvxi)

    ax.set_xlabel(r'p/mc', fontsize=14, color='black')
    ax.set_ylabel(r'$\xi$', fontsize=14)
    #ax.legend(loc="upper right", ncol=1, shadow=False, fancybox=False)
    plt.title('distribution function')
    plt.tight_layout()
    #plt.show()
    if file is not None:
      plt.savefig(file)

    #---plot mesh to see---
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cmap = plt.cm.get_cmap("jet")
    cmap.set_under("white")
    cmap.set_over("darkred")
    ax.pcolor(self.__vp12, self.__xip12, np.log10(self.__dist[:,:]+1.05e-22), edgecolors='k', linewidths=.15)
    plt.colorbar(fvxi)
    ax.set_xlabel(r'p/mc', fontsize=14, color='black')
    ax.set_ylabel(r'$\xi$', fontsize=14)
    plt.tight_layout()
    #axes = plt.gca()
    #axes.set_xlim([self.xmin,self.xmax])
    #axes.set_ylim([self.ymin,self.ymax])
    plt.title('distribution function and mesh')


  # plot the 1D pitch-angle distribution
  def plot_fxi(self, file=None) :
    print("generating 1D pitch-angle distributuon plot")
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.plot(self.__xip12, np.log10(self.__dist[:,50] + 1e-30), linewidth=2, color='red', label=r'p=%.2g'%(self.__vp12[50]))
    ax.plot(self.__xip12, np.log10(self.__dist[:,100] + 1e-30), linewidth=2, color='blue', label=r'p=%.2g'%(self.__vp12[100]))
    ax.plot(self.__xip12, np.log10(self.__dist[:,200] + 1e-30), linewidth=2, color='black', label=r'p=%.2g'%(self.__vp12[200]))

    ax.axis([-1., 1., -25, 0])
    ax.set_xlabel(r'$\xi$', fontsize=14, color='black')
    plt.title(r'$f(\xi)$')
    plt.legend(loc='upper right')
    #plt.show()
    if file is not None :
      plt.savefig(file)

  # interpolate nonuniform grid data and do streamplot
  def plot_stream(self, xmin, xmax, ymin, ymax, nx, ny, file=None) :
    print("generating 2D stream lines on momentum-space")
    if (self.__rfp.Gp is None or self.__rfp.Gxi is None):
      self.__rfp.calFlux(1.0e-22)

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)

    funcGp = interpolate.RectBivariateSpline(self.__xip12, self.__vp12, self.__rfp.Gp, kx=1, ky=1)
    funcGxi = interpolate.RectBivariateSpline(self.__xip12, self.__vp12, self.__rfp.Gxi, kx=1, ky=1)

    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.streamplot(x, y, funcGp(y,x), funcGxi(y,x), density=5.0, linewidth=2, minlength=0.05)
    # ax.set_xscale("log")
    # ax.axis([xmin, xmax, ymin, ymax])
    plt.title('phase space flux(E={0})'.format(fe.param['E']))
    #plt.show()
    if file is not None :
      plt.savefig(file)

  def plot_Up(self, file=None) :
    print("generating 1D U(p) curve")
    self.__rfp.computeU(1.e-36);
    #np.savetxt('Up.dat', self.__rfp.Up[0:200], fmt='%f')
    #np.savetxt('p.dat', self.__rfp.vp12[0:200], fmt='%f')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(self.__vp12, self.__rfp.Up, linewidth=2)
    ax.plot(self.__vp12, 0.*self.__vp12, '--',linewidth=2)
    ax.axis([2, 8, -0.001, 0.0005])
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
    ax.set_xlabel(r'p/mc', fontsize=14, color='black')
    #ax.set_ylabel(r'$\xi$', fontsize=14)
    plt.title('Up')
    #ax.legend(loc="upper right", ncol=1, shadow=False, fancybox=False)
    if file is not None :
      plt.savefig(file)

  def plot(self) :
     plt.show()


parser = argparse.ArgumentParser(description='read and plot the distribution data, calculate the runaway rate')
parser.add_argument('-step', metavar='step', type=int, nargs='?', default=0,
                   help='the time step to process')
parser.add_argument('-flux', metavar='flux', type=bool, nargs='?', default=False,
                   help='the flag to calculate phase-space flux')
parser.add_argument('-file', metavar='file', type=str, nargs='?', default='soln.dat',
                   help='the data file')
parser.add_argument('-dir', metavar='dir', type=str, nargs='?', default=None,
                   help='the folder with data file')
parser.add_argument('-petscint64', dest='petscint64', action='store_true',
                   help='flag if Petsc was compiled with 64-bit integers')
parser.set_defaults(petscint64=False)

args = parser.parse_args()

if args.dir is None :
  fe = runaway('mesh.m', 'input_params.m', args.file, petscint64=args.petscint64)
else :
  fe = runaway('{}/mesh.m'.format(args.dir), '{}/input_params.m'.format(args.dir), '{0}/{1}'.format(args.dir,args.file), petscint64=args.petscint64)

ne, je, ke = fe.output_ne(1.)
print('runaway kinetic energy is {0}'.format(ke/ne))

draw = draw_re(fe)

if args.dir is None :
  draw.plot_fpxi("dist.eps")
  if (args.flux==True) :
    draw.plot_stream(np.min(fe.vp12),np.max(fe.vp12), -1., 0, 300, 200, "flux.eps")
#  draw.plot_Up("Up.eps")

else :
  draw.plot_fpxi("./{}/dist.eps".format(args.dir))

  if (args.flux==True) :
    draw.plot_stream(np.min(fe.vp12),np.max(fe.vp12), -1., -0, 300, 200, "./{}/flux.eps".format(args.dir))

draw.plot_fp('fp.eps')

##### plot residual
fe = runaway('mesh.m', 'input_params.m', 'residual.dat', petscint64=args.petscint64)

fig = plt.figure()
ax = fig.add_subplot(111)
levels = np.linspace(-22, -10, 25)
cmap = plt.cm.get_cmap("jet")
cmap.set_under("white")
cmap.set_over("darkred")
fvxi = ax.contourf(fe.vp12, fe.xip12, np.log10(fe.dist[:,:]+1.05e-22),  levels, cmap=cmap, extend='both')
cbar = plt.colorbar(fvxi)
ax.set_xlabel(r'p/mc', fontsize=14, color='black')
ax.set_ylabel(r'$\xi$', fontsize=14)
#ax.legend(loc="upper right", ncol=1, shadow=False, fancybox=False)
plt.title('residual')
draw.plot()



# plot the flow speed (exluding diffusive flux)
#fig = plt.figure(1)
#ax  = fig.add_subplot(111)

#ax.plot(fe.vp12, fe.Up, linewidth=2)
#ax.plot(fe.vp12, 0.*fe.vp12, '--',linewidth=2)

#ax.axis([0.3, 15, -2, 1])
#ax.set_xlabel(r'p/mc', fontsize=14, color='black')
#ax.set_ylabel(r'$\xi$', fontsize=14)
#plt.title('Up')
#ax.legend(loc="upper right", ncol=1, shadow=False, fancybox=False)


# plot pitch-angle dependence



# plot the total flux
#fig = plt.figure(4)
#ax  = fig.add_subplot(111)

#ax.plot(fe.vp12, fe.Sr0)
#ax.axis([fe.xmin, fe.xmax, -1e-15, 1e-15])
#ax.set_xlabel(r'p/mc', fontsize=14, color='black')
#ax.set_ylabel(r'$\xi$', fontsize=14)
#plt.title('Sr')
