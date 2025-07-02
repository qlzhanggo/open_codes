#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from scipy import special as sp
from scipy import interpolate
from scipy import integrate

def Lambda_ee(gm, vte, beta_) :
  return (1.0 + beta_ * np.log( np.sqrt(2.*(gm - 1.))/vte))

def Lambda_ei(p, vte, beta_) :
  return (1.0 + beta_ * np.log(2.0*np.sqrt(2.0)*p/vte))

def Uface(Ur, Uu, Ud, xr, xu, xd, x) :
  dd = 2.*(xr - xu)/(xu - xd) + 1.
  return 0.5*(1. - 1./(dd + 1.))*Ud + 0.5*(1. + 1./(dd - 1.))*Uu - (1./(dd+1.)/(dd-1.))*Ur

def zeta1(xi, eps) :
    if (eps>0) :
      kappa2 = (1. + (xi*xi-2.*eps/(1.+eps))/(2.*eps) / (1. - xi*xi))
      return np.where(xi*xi>2.*eps/(1.+eps), 2./np.pi * sp.ellipk(1./kappa2), 4.*np.sqrt(kappa2)/np.pi * sp.ellipk(kappa2))
    else :
      return np.ones_like(xi)

def zeta2(xi, eps):
    if (eps>0) :
      kappa2 = (1. + (xi*xi-2.*eps/(1.+eps))/(2.*eps) / (1. - xi*xi))
      return np.where(xi*xi>2.*eps/(1.+eps), 2./np.pi * sp.ellipe(1./kappa2), 4./np.sqrt(kappa2)/np.pi * (sp.ellipe(kappa2)- (1.-kappa2)*sp.ellipk(kappa2)))
    else :
      return np.ones_like(xi)


class Dwp :
    def __init__(self, E0_, k10_, k20_, dk10_) :
        self.E0 = E0_
        self.k10 = k10_
        self.k20 = k20_
        self.dk10 = dk10_
        self.v_compute = np.vectorize(self.compute)

    def __call__(self, p, xi) :
        # return the distribution function at x
        return self.v_compute(p,xi)

    def compute(self, p, xi) :
        #omegatau=1.e10
        temp = np.zeros(3)
        gm = np.sqrt(p**2+1.)
        #k1r = 1./(self.k20 * gm + p * xi)  # only true when k20 >> k1r
        if (self.k10 < 0) :
          #solving k*|k10| - k10*v*xi + 1/gm = 0
          coef = [gm**2, 0, (self.k20**2*gm**2 - p**2*xi**2), 2.*p*xi, -1.0]
          roots = np.roots(coef)
          for i in range(4):
            if (np.isreal(roots[i])) :
                k1r = roots[i].real
                if (k1r*p*xi>1.) :
                    k = np.sqrt(k1r**2 + self.k20**2)
                    temp_ =  2. * self.E0/np.sqrt(2.*np.pi)/np.fabs((k**2 + k1r**2)/k + p*xi/gm) * np.exp(-0.5*(k1r - self.k10)**2/self.dk10**2)
                    temp[0] += temp_
                    temp[1] += temp_ * (-p/gm/k - xi)
                    temp[2] += temp_ * (p/gm/k +xi)**2
        else :
          #solving k*|k10| - k10*v*xi - 1/gm = 0  Doppler shifted resonance, taking k10 to be positive
          coef = [gm**2, 0, (self.k20**2*gm**2 - p**2*xi**2), -2.*p*xi, -1.0]
          roots = np.roots(coef)
          for i in range(4):
              if (np.isreal(roots[i])) :
                  k1r = roots[i].real
                  if (k1r*p*xi>-1.) :
                      k = np.sqrt(k1r**2 + self.k20**2)
                      temp_ = 2. * self.E0/np.sqrt(2.*np.pi)/np.fabs((k**2 + k1r**2)/k - p*xi/gm) * np.exp(-0.5*(k1r - self.k10)**2/self.dk10**2)
                      temp[0] += temp_
                      temp[1] += temp_ * (p/gm/k - xi)
                      temp[2] += temp_ * (p/gm/k -xi)**2
        return temp

# quasilinear diffusion for tokamak
class Dwp_ba :
  def __init__(self, w0_, k10_, k20_, dk10_, xic_, v, xi) :
    self.w0, self.k10, self.k20, self.dk10 = (w0_, k10_, k20_, dk10_)
    self.xic = xic_
    self.v_normal = np.vectorize(self.normal)
    #self.v_compute = np.vectorize(self.compute)
    self.M, self.N = (v.size, xi.size)
    self.Dpp = np.zeros((self.N, self.M))
    self.Dpxi = self.Dpp
    self.Dxixi = self.Dpp

    #pv, xiv = np.meshgrid(v, xi)
    self.compute(v, xi)

  def compute(self, pvv, xivv) :
    eps_ = self.xic**2/(2.-self.xic**2);

    for i in range(self.M):
      pv = pvv[i]

      for j in range(self.N):
        xiv = xivv[j]

        if (xiv*xiv < 1) :
          kappa = 1. + (xiv**2/self.xic**2 - 1.)/(1. - xiv**2)
        else :
          kappa = 1.e4

        if (kappa>=1.): #passing
          theta = np.linspace(0., np.pi, 50)
          xi =(1. - (1. - eps_*np.cos(theta))/(1. - eps_) * (1.-xiv**2))
          xi = np.where(xi>0, np.sqrt(xi)* np.sign(xiv), 0.)
          if ((np.isnan(xi)).any() ) :
            print(xi)

          tmp = np.where(xi == 0.,  0., ((1.-eps_*np.cos(theta))/(1.-eps_)* xiv/xi))
          p = pv * np.ones_like(xi)
          #print(np.isnan(1./p))
          (Dpp_, Dpxi_, Dxixi_) = self.v_normal(p, xi)
          Dpp = (integrate.simps(Dpp_*tmp, theta) / np.pi )
          Dpxi = (integrate.simps(Dpxi_, theta) / np.pi )
          Dxixi = (integrate.simps(np.where(tmp==0., 0., Dxixi_/tmp), theta) / np.pi )

        else: #trapping
          theta0 = 2.*np.arcsin(kappa)
          theta = np.linspace(0., 0.95*theta0, 50)

          xi =(1. - (1. - eps_*np.cos(theta))/(1. - eps_) * (1.-xiv**2))
          xi = np.where(xi>0, np.sqrt(xi)* np.sign(xiv), 0.)
          if ((np.isnan(xi)).any() ) :
            print("xiv is {0}, xi is {1}".format(xiv, xi))
            exit()

          tmp = np.where(xi == 0.,  0., ((1.-eps_*np.cos(theta))/(1.-eps_)* xiv/xi))
          p = pv * np.ones_like(xi)
          (Dpp_, Dpxi_, Dxixi_) = self.v_normal(p, xi)

          Dpp = (integrate.simps(np.where(np.isinf(tmp), 0., Dpp_*tmp), theta) / np.pi )
          Dpxi = (integrate.simps(Dpxi_, theta) / np.pi )
          Dxixi = (integrate.simps(np.where(tmp==0., 0., Dxixi_/tmp), theta) / np.pi )

          xi = -xi
          tmp = -tmp #((1.-eps_*np.cos(theta))/(1.-eps_)* xiv[j,i]/xi)
          p = pv * np.ones_like(xi)
          (Dpp_, Dpxi_, Dxixi_) = self.v_normal(p, xi)

          Dpp += (integrate.simps(np.where(np.isinf(tmp), 0., -Dpp_*tmp), theta) / np.pi )
          Dpxi += (integrate.simps(-Dpxi_, theta) / np.pi )
          Dxixi += (integrate.simps(np.where(tmp==0., 0., -Dxixi_/tmp), theta) / np.pi )

        self.Dpp[j,i], self.Dpxi[j,i], self.Dxixi[j,i] = (Dpp , Dpxi, Dxixi)


  def normal(self, p, xi) :
    temp  = np.zeros(3)
    gm = np.sqrt(p*p + 1.)
    coef = [gm**2, 0, (self.k20**2*gm**2 - p**2*xi**2), -2.*p*xi, -1.0]
    #print(gm)
    roots = np.roots(coef)
    for i in range(4):
      if (np.isreal(roots[i])) :
        k1r = roots[i].real
        if (k1r*p*xi>-1.) :
          k = np.sqrt(k1r**2 + self.k20**2)
          temp_ = 2. * self.w0/self.dk10/np.sqrt(2.*np.pi)/np.fabs((k**2 + k1r**2)/k - p*xi/gm) * np.exp(-0.5*(k1r - self.k10)**2/self.dk10**2)
          temp[0] += temp_
          temp[1] += temp_ * (p/gm/k - xi)
          temp[2] += temp_ * (p/gm/k -xi)**2
    return (temp[0], temp[1], temp[2])

  def anomalous(self, p, xi) :
    temp  = np.zeros(3)
    gm = np.sqrt(p*p + 1.)
    coef = [gm**2, 0, (self.k20**2*gm**2 - p**2*xi**2), 2.*p*xi, -1.0]
    roots = np.roots(coef)
    for i in range(4):
      if (np.isreal(roots[i])) :
        k1r = roots[i].real
        if (k1r*p*xi>1.) :
          k = np.sqrt(k1r**2 + self.k20**2)
          temp_ = 2. * self.w0/self.dk10/np.sqrt(2.*np.pi)/np.fabs((k**2 + k1r**2)/k + p*xi/gm) * np.exp(-0.5*(k1r - self.k10)**2/self.dk10**2)
          temp[0] += temp_
          temp[1] += temp_ * (-p/gm/k - xi)
          temp[2] += temp_ * (p/gm/k -xi)**2
    return temp


# ---------------------------------------------------------
# this class holds the method to read RFP data and construct distributions
# ---------------------------------------------------------
class runaway:

  def __init__(self, meshfile, paramfile, distfile, mute=False, petscint64=False) :
    self.M, self.N = None, None
    self.xmin, self.xmax = None, None
    self.param = {'E':None, 'sr':0.1, 'Z':1, 'vte':None, 'ra':0.4, 'rho':0.}
    self.vp12 = []
    self.xip12 = []

    self.Up = []
    self.Sr0 = []
    self.Gp = None
    self.Gxi = None

    self.dist=[]

    self.mute = mute
    self.petscint64 = petscint64

    self.readparam(paramfile)
    self.readmesh(meshfile)
    self.readdist(distfile)

  #--------------------
  # function to read the distribution, defined at cell center
  #--------------------
  def readdist(self, distfile) :
    if (self.M is None or self.N is None):
      print("You forgot to read the mesh first!")
      exit()
    if (self.param['E'] is None) :
      print("You forgot to read the parameters first!")
      exit()

    with open(distfile, 'rb') as f:
      if self.petscint64:
        f.seek(16)
      else:
        f.seek(8)
      dist_raw = np.fromfile(f, np.dtype('>d'))

    dist = np.reshape(dist_raw, (self.N, self.M))
    self.dist_raw=dist
    dist /= self.param['ra']
    dist = dist/(self.vp12[None,:])

    if (self.eps > 0) :
    # reconstruct the xi grid to cover [-1:1]
      xitp = np.sqrt(2.*self.eps/(1.+self.eps))
      self.N = self.N + (self.My2 - self.My1-1)
      xi = np.zeros(self.N)
      xi[0:self.My2] = self.xip12[0:self.My2]
      xi[self.My2:2*self.My2-self.My1-1] = -self.xip12[(self.My2-1):(self.My1):-1]
      xi[2*self.My2-self.My1-1::] = self.xip12[self.My2::]
      del self.xip12
      self.xip12 = xi

      # reconstruct the bounce-averaged dist to cover full phase-space
      self.dist = np.zeros((self.N, self.M))
      self.dist[0:self.My2, :] = dist[0:self.My2, :]
      self.dist[self.My2:(2*self.My2-self.My1-1), :] = dist[(self.My2-1):(self.My1):-1, :]
      self.dist[2*self.My2-self.My1-1::, :] = dist[self.My2::, :]

      self.dist = np.where(self.dist[:,:]<0, 0., self.dist[:,:])

      if (not self.mute) :
        print('')
        print("Mesh and distribution reconstructed to map out the full momentum-space: (M={0}, N={1})".format(self.M, self.N))
        print('')

    else :
      self.dist = np.where(dist[:,:]<0, 0., dist[:,:])

  #-------------------------
  # function to read the mesh
  #-------------------------
  def readmesh(self, meshfile) :
    with open(meshfile, 'r') as f:
     for line in f:
         line = line.strip()
         line = line.strip(";")

         if line:
           data = line.split()

           if   data[0] == 'eps':
                self.eps = float(data[2])
           elif data[0] == 'numxpts':
                self.M = int(data[2])
           elif data[0] == 'numypts':
                self.N = int(data[2])
           elif data[0] == 'pmax':
                self.xmax = float(data[2])
           elif data[0] == 'pmin':
                self.xmin = float(data[2])
           elif data[0] == 'p0':
                pc = float(data[2])
           elif data[0] == 'dph':
                dph = float(data[2])
           elif data[0] == 'dpl':
                dpl = float(data[2])
           elif data[0] == 'xi0':
                xic = float(data[2])
           elif data[0] == 'dxih':
                dxih = float(data[2])
           elif data[0] == 'dxil':
                dxil = float(data[2])

    M, N = (int((self.xmax-self.xmin)/dpl), int(2.0/dxil))
    vp1 = np.zeros(M+1)
    xip1 = np.zeros(N+1)
    self.My1, self.My2 = (-1, -1)

    a,b = (0.5*(dph+dpl), 0.5*(dph-dpl))
    pw = 1.0 #min(10.*dpl, 0.1*self.xmax)
    i=0
    temp=self.xmin + (a + b*np.tanh((self.xmin-pc)/pw))
    while (temp < self.xmax):
      vp1[i] = temp;
      temp = vp1[i]+ (a + b*np.tanh((vp1[i]-pc)/pw))
      i += 1
    vp1[i] = temp
    self.M = i+1
    xmax = temp

    xitp = np.sqrt(2.*self.eps/(1.+self.eps))
    a,b = (0.5*(dxih+dxil), 0.5*(dxih-dxil))
    xiw = 0.06 #min(10.*dxil, 0.1)
    j=0
    temp=-1.0 + (a + b*np.tanh((-1.0-xic)/xiw))
    if (xitp>0 and xitp<1) :
       while (temp <= -xitp) :
         xip1[j] = temp
         temp += (a + b*np.tanh((temp - xic)/xiw))
         j+=1

       #if ( xip1[j-1] > -xitp-0.006) :
#	xip1[j-1] = -xitp;
#       else :
#	xip1[j] = -xitp
#	xip1[j-1] = 0.5*(xip1[j] + xip1[j-2])
#	j+=1

       self.My1 = j-1 #index of the trap-boundary point No1
       xitp = -xip1[self.My1]
       self.eps = xitp**2/(2.-xitp**2)
       temp = -xitp + (a + b*np.tanh((temp - xic)/xiw))

       while (temp < 0) :
         xip1[j] = temp
         temp += (a + b*np.tanh((temp - xic)/xiw))
         j+=1

       xip1[j-1] = 0.
       self.My2 = j
       xip1[self.My2] = xitp
       j+=1
       temp = - xip1[self.My1-1]

    while (temp < 1.0) :
      xip1[j] = temp
      temp += (a + b*np.tanh((temp - xic)/xiw))
      j+=1

    self.N = j

    vp1.resize(self.M)
    xip1.resize(self.N)
    self.vp1 = vp1
    self.xip1 = xip1

# resize the cell center mesh arrays
    if (self.My2 > self.My1) :
      self.vp12 = np.zeros(self.M)
      self.xip12 = np.zeros(self.N)

      for i in range(self.M) :
        if (i == 0) :
          self.vp12[i] = 0.5*(self.xmin + self.vp1[0])
        else :
          self.vp12[i] = 0.5*(self.vp1[i] + self.vp1[i-1])

      for j in range(self.N) :
        if (j == 0):
          self.xip12[j] = 0.5*(-1. + self.xip1[0])
        elif (j == self.N-1) :
          self.xip12[j] = 0.5*(self.xip1[j]+1.)
        elif (j < self.My2) :
          self.xip12[j] = 0.5*(self.xip1[j] + self.xip1[j-1])
        else :
          self.xip12[j] = 0.5*(self.xip1[j] + self.xip1[j+1])

    else :
      self.N += 1
      self.vp12 = np.zeros(self.M)
      self.xip12 = np.zeros(self.N)

      for i in range(self.M) :
        if (i == 0) :
          self.vp12[i] = 0.5*(self.xmin + vp1[0])
        else :
          self.vp12[i] = 0.5*(vp1[i] + vp1[i-1])

      self.xip12[0] = 0.5*(xip1[0] - 1.)
      for j in range(1, self.N-1) :
        self.xip12[j] = 0.5*(xip1[j] + xip1[j-1])
      self.xip12[self.N-1] = 0.5*(xip1[self.N-2]+1.)

    if (not self.mute) :
      print('The computation mesh is (M={0}, N={1})'.format(self.M, self.N))
      if self.eps>0 :
        print('This is a tokamak case with eps={}'.format(self.eps))
        print('trap region starts at {0} with xi={1}'.format(self.My1+1, self.xip12[self.My1+1]))
        print('second passing region starts at {0} with xi={1}'.format(self.My2, self.xip12[self.My2]))


  # --------------------------
  # function to read parameter file
  # --------------------------
  def readparam(self, paramfile) :
    with open(paramfile, 'r') as f:
      for line in f:
         line = line.strip()
         line = line.strip(";")

         if line:
           data = line.split()

           if data[0] == 'ra':
             self.param['ra'] = float(data[2])
           elif data[0] == 'E0':
             self.param['E'] = float(data[2])
           elif data[0] == 'sr':
             self.param['sr'] = float(data[2])
           elif data[0] == 'vte':
             self.param['vte'] = float(data[2])
           elif data[0] == 'Z0':
             self.param['Z'] = float(data[2])
           elif data[0] == 'rho':
             self.param['rho'] = float(data[2])
           elif data[0] == 'asp':
             self.param['asp'] = float(data[2])
           elif data[0] == 'ra':
             self.param['ra'] = float(data[2])
           elif data[0] == 'beta' :
             self.param['beta'] = float(data[2])
           elif data[0] == 'w0' :
             self.param['w0'] = float(data[2])
           elif data[0] == 'k10' :
             self.param['k10'] = float(data[2])
           elif data[0] == 'k20' :
             self.param['k20'] = float(data[2])
           elif data[0] == 'dk10' :
             self.param['dk10'] = float(data[2])

    if (not self.mute) :
      print('E0 = {0}, sr = {1}, Z = {2}, vte = {3}, eps = {4}\n'.format(self.param['E'], self.param['sr'], self.param['Z'], self.param['vte'], self.param['asp']*self.param['ra']))

#--------------------
  def output_ne(self, pc):
    hx = np.zeros(self.M)
    hy = np.zeros(self.N)

    hx[0] = self.vp1[0]-self.xmin
    hx[1:self.M] = self.vp1[1:self.M]-self.vp1[0:self.M-1] 

#    id1 = np.where(np.logical_and(self.vp12<=pc, np.roll(self.vp12,-1)>pc))[0]
#    dy = self.xip1[1:(self.N+1)] - self.xip1[0:(self.N)]
#    fv = 0.5 * np.sum(self.dist[:,id1:id2]*dy[:,None], 0)

    xic = np.sqrt(2.*self.eps/(1.+self.eps))

    if (self.My2>self.My1) :
      hy[0] = self.xip1[0] + 1.0
      hy[self.N-1] = 1.0 - self.xip1[-1]
      hy[1:self.My2] = self.xip1[1:self.My2] - self.xip1[0:self.My2-1]
      hy[self.My2:2*self.My2-self.My1-1] = -hy[self.My2-1:self.My1:-1]
      hy[2*self.My2-self.My1-1:self.N-1] = self.xip1[self.My2+1::] - self.xip1[self.My2:-1]

    else :
      hy[0] = self.xip1[0] + 1.0
      hy[self.N-1] = 1.0 - self.xip1[self.N-2]
      hy[1:self.N-1] = self.xip1[1:self.N-1] - self.xip1[0:self.N-2]

    #if (xic>0) :
    #  func = interpolate.RectBivariateSpline(self.xip12, self.vp12, self.dist*self.vp12[None,:]**2*zeta1(self.xip12, self.eps)[:,None], kx=1, ky=1,s=0.1)
    #  ne = 2.*np.pi * integrate.nquad(func, [[-1,0], [pc,self.vp12[self.M-1]]])[0]
    #  ne += 2.*np.pi * integrate.nquad(func, [[xic,1], [pc,self.vp12[self.M-1]]])[0]
    #else :
    #  func = interpolate.RectBivariateSpline(self.xip12, self.vp12, self.dist*self.vp12[None,:]**2, kx=1, ky=1,s=0.1)
    #  ne = 2.*np.pi * integrate.nquad(func, [[-1,1], [pc,self.vp12[self.M-1]]])[0]

    #if (xic>0) :
    #  func = interpolate.RectBivariateSpline(self.xip12, self.vp12, self.dist*zeta1(self.xip12,self.eps)[:,None]*self.vp12[None,:]**3/np.sqrt(1. + self.vp12[None,:]**2)*self.xip12[:,None], kx=1, ky=1,s=0.1)
    #  je = 2.*np.pi * integrate.nquad(func, [[-1,0], [pc, self.vp12[self.M-1]]])[0]
    #  je += 2.*np.pi * integrate.nquad(func, [[xic,1], [pc, self.vp12[self.M-1]]])[0]
    #else :
    #  func = interpolate.RectBivariateSpline(self.xip12, self.vp12, self.dist*self.vp12[None,:]**3/np.sqrt(1. + self.vp12[None,:]**2)*self.xip12[:,None], kx=1, ky=1, s=0.1)
    #  je = 2.*np.pi * integrate.nquad(func, [[-1,1], [pc, self.vp12[self.M-1]]])[0]


    #if (xic>0) :
    #  func = interpolate.RectBivariateSpline(self.xip12, self.vp12, self.dist*self.vp12[None,:]**2*zeta1(self.xip12, self.eps)[:,None]*(np.sqrt(1. + self.vp12[None,:]**2) - 1.), kx=1, ky=1, s=0.1)
    #  ke = 2.*np.pi * integrate.nquad(func, [[-1,0], [pc, self.vp12[self.M-1]]])[0]
    #  ke += 2.*np.pi * integrate.nquad(func, [[xic,1], [pc, self.vp12[self.M-1]]])[0]
    #else :
    #  func = interpolate.RectBivariateSpline(self.xip12, self.vp12, self.dist*self.vp12[None,:]**2*(np.sqrt(1. + self.vp12[None,:]**2) - 1.), kx=1, ky=1, s=0.1)
    #  ke = 2.*np.pi * integrate.nquad(func, [[-1,1], [pc, self.vp12[self.M-1]]])[0]

    mask=np.zeros_like(self.vp12)
    mask[(self.vp12>self.param['vte']*10)]=1

    if (xic>0) :
#      ne = 2.*np.pi * np.sum(self.dist[0:self.My2,M0::]*self.vp12[None,M0::]**2*zeta1(self.xip12[0:self.My2], self.eps)[:,None]*hx[None,M0::]*hy[0:self.My2,None])
#      ne += 2.*np.pi * np.sum(self.dist[self.My2::,M0::]*self.vp12[None,M0::]**2*zeta1(self.xip12[self.My2::], self.eps)[:,None]*hx[None,M0::]*hy[self.My2::,None])
      ne = 2.*np.pi * np.sum(self.dist[::self.My2,:]*mask[None,:]*hx[None,:]*hy[::self.My2,None]*self.vp12[None,:]**2)

      je = 2.*np.pi * np.sum(self.dist[:self.My2,:]*mask[None,:]*self.vp12[None,:]**3/np.sqrt(1. + self.vp12[None,:]**2)*self.xip12[:self.My2,None]*zeta1(self.xip12[:self.My2], self.eps)[:,None]*hx[None,:]*hy[:self.My2,None])
      je += 2.*np.pi * np.sum(self.dist[self.My2::,:]*mask[None,:]*self.vp12[None,:]**3/np.sqrt(1. + self.vp12[None,:]**2)*self.xip12[self.My2::,None]*zeta1(self.xip12[self.My2::], self.eps)[:,None]*hx[None,:]*hy[self.My2::,None])

      ke = 2.*np.pi * np.sum(self.dist[:self.My2,:]*mask[None,:]*self.vp12[None,:]**2*zeta1(self.xip12[:self.My2], self.eps)[:,None]*(np.sqrt(1. + self.vp12[None,:]**2) - 1.)*hx[None,]*hy[:self.My2,None])
      ke += 2.*np.pi * np.sum(self.dist[self.My2::,:]*mask[None,:]*self.vp12[None,:]**2*zeta1(self.xip12[self.My2::], self.eps)[:,None]*(np.sqrt(1. + self.vp12[None,:]**2) - 1.)*hx[None,:]*hy[self.My2::,None])

    else :
      ne = 2.*np.pi * np.sum(self.dist[:,:]*mask[None,:]*hx[None,:]*hy[:,None]*self.vp12[None,:]**2)
      je = 2.*np.pi * np.sum(self.dist[:,:]*mask[None,:]*hx[None,:]*hy[:,None]*self.vp12[None,:]**3/np.sqrt(1. + self.vp12[None,:]**2)*self.xip12[:,None])
      ke = 2.*np.pi * np.sum(self.dist[:,:]*mask[None,:]*hx[None,:]*hy[:,None]*self.vp12[None,:]**2*(np.sqrt(self.vp12[None,:]**2 + 1.) -1.))
    je=-je  #reflect the negative charge of the electron

    if (not self.mute) :
      print('------------------------------------')
      print('runaway density is {0}'.format(ne))
      print('runaway current is {0}'.format(je))
      print('------------------------------------')

    return (ne, je, ke)

#    print('runaway density is {0}'.format(2.*np.pi * np.sum(self.dist*hx[None,:]*hy[:,None] * self.vp12[None,:]**2)))
#    print('runaway current is {0}'.format(2.*np.pi * np.sum(self.dist*hx[None,:]*hy[:,None] * self.vp12[None,:]**3/np.sqrt(1. + self.vp12[None,:])*self.xip12[:,None])))


# function to compute the 2D phase-space flux
  def calFlux(self, min_):
    vte = self.param['vte']
    Z = self.param['Z']
    E = self.param['E']
    sr = self.param['sr']
    rho = self.param['rho']
    xic = np.sqrt(2.*self.eps/(1.+self.eps))

    beta = self.param['beta']
    w0 = self.param['w0']
    k10 = self.param['k10']
    k20 = self.param['k20']
    dk10 = self.param['dk10']

    if (xic == 0 and w0>0) :
      D_ = Dwp(w0, k10, k20, dk10)
    if (xic > 0 and w0>0) :
      D_ = Dwp_ba(w0, k10, k20, dk10, xic, self.vp12, self.xip12)

    self.Gp = np.zeros((self.N, self.M))
    self.Gxi = np.zeros((self.N, self.M))

    for j in range(self.N):
      for i in range(self.M):

        vip12, xip12 = (self.vp12[i], self.xip12[j])

        x = self.vp12[i]/np.sqrt(1+self.vp12[i]**2)/vte
        cf = 0.5/x**2 *(sp.erf(x) - 2.0*x*np.exp(-x**2)/np.sqrt(np.pi)) / (0.5*vte**2) * Lambda_ee(np.sqrt(vip12**2+1.), vte, beta)
        ca = np.sqrt(1.+self.vp12[i]**2)/self.vp12[i] * (0.5*vte**2) * cf
        cb = 0.5*np.sqrt(1+self.vp12[i]**2)/self.vp12[i] *(Z*Lambda_ei(vip12, vte, beta) + (sp.erf(x) + 0.5*vte**4 * x**2 - 0.5/x**2 *(sp.erf(x)-2.0*x*np.exp(-x**2)/np.sqrt(np.pi)))*Lambda_ee(np.sqrt(vip12**2+1.), vte, beta) ) # - 0.5*ca

        if self.dist[j,i]>min_ :

        ##################################
        # calculate the energy flux
        ##################################
          zeta1_ = zeta1(xip12, self.eps)
          if (xip12>=-xic and xip12<=xic and xic>0) :
            E0 = 0
          else :
            E0 = E

          flux = -(E0*xip12 + zeta1_ * sr*vip12*np.sqrt(1+vip12*vip12)*( (1 - self.xip12[j]**2) + rho**4 * vip12**2*(self.xip12[j]**4) )) - cf * zeta1_
          flux *= vip12**2 * self.dist[j,i]

        # energy diffusive flux due to collision (central differencing)
          if i == self.M-1:
            flux = 0.
          elif i == 0 :
            x0 = 2.*self.xmin - self.vp12[0]
            dx1 = self.vp12[0] - x0
            dx2 = self.vp12[1] - self.vp12[0]
            fmax = 1.0/(vte**3 * np.pi*np.sqrt(np.pi)) * np.exp((1-np.sqrt(1+x0**2))/(0.5*vte**2))
            dfdx =  ((self.dist[j,i+1]-self.dist[j,i])/dx2**2 + (self.dist[j,i]- fmax)/dx1**2)/(1./dx1 + 1./dx2)
            flux -= zeta1_ * vip12**2 * ca * dfdx
          else :
            dx1 = self.vp12[i]-self.vp12[i-1]
            dx2 = self.vp12[i+1]-self.vp12[i]
            dfdx = ((self.dist[j,i+1]-self.dist[j,i])/dx2**2 + (self.dist[j,i]-self.dist[j,i-1])/dx1**2)/(1./dx1 + 1./dx2)
            flux -= zeta1_ * vip12**2 * ca * dfdx

            if (w0>0) :
              v = vip12/np.sqrt(1.+vip12**2)

              if (xic == 0) :
                D = D_(vip12, xip12)*(1.-xip12**2)
              else :
                D = (D_.Dpp[j, i], D_.Dpxi[j, i], D_.Dxixi[j,i])
                #D = [x*(1.-xip12**2) for x in D]
                D = list(map(lambda x:x*(1.-xip12**2), D))

              flux -= vip12**2 * D[0] * dfdx

              if (j>0 and j<self.N-1) :
                dy1 = self.xip12[j]-self.xip12[j-1]
                dy2 = self.xip12[j+1]-self.xip12[j]
                dfdy = ((self.dist[j+1,i]-self.dist[j,i])/dy2**2 + (self.dist[j,i]-self.dist[j-1,i])/dy1**2)/(1./dy1 + 1./dy2)
                flux -= vip12*D[1] * dfdy

          self.Gp[j,i] = flux/self.dist[j,i]


        ##################################
        # calculate the pitch-angle flux #
        #################################
          zeta2_ = zeta2(xip12, self.eps)
          if (xip12<= xic and xip12>=-xic and xic>0) :
            E0 = 0
          else :
            E0 = E

          coef = (-E0*vip12 + zeta2_ * sr*vip12**2/np.sqrt(1+vip12**2)*self.xip12[j])

          if (j == self.N-1 ) :
            self.Gxi[j,i] = 0.0
          elif (j == 0) :
            self.Gxi[j,i] = 0.0
          else :
            if j == self.N-1 :
              self.Gxi[j,i] = ( coef )
            elif j == 0 :
              self.Gxi[j,i] = ( coef )
            else:
              dy1 = self.xip12[j]-self.xip12[j-1]
              dy2 = self.xip12[j+1]-self.xip12[j]
              dfdy = ((self.dist[j+1,i]-self.dist[j,i])/dy2**2 + (self.dist[j,i]-self.dist[j-1,i])/dy1**2)/(1./dy1 + 1./dy2)

              self.Gxi[j,i] = (1-self.xip12[j]**2) * ( coef*self.dist[j,i] - zeta2_ * cb *dfdy)

              if (w0>0) :
                v = vip12/np.sqrt(1.+vip12**2)

                if (xic == 0) :
                  D = D_(vip12, xip12)*(1.-xip12**2)
                else :
                  D = (D_.Dpp[j, i], D_.Dpxi[j, i], D_.Dxixi[j,i])
                  D = [x*(1.-xip12**2) for x in D]

                self.Gxi[j,i] -= D[2] * dfdy

                if (i>0 and i<self.M-1) :
                  dx1 = self.vp12[i]-self.vp12[i-1]
                  dx2 = self.vp12[i+1]-self.vp12[i]
                  dfdx = ((self.dist[j,i+1]-self.dist[j,i])/dx2**2 + (self.dist[j,i]-self.dist[j,i-1])/dx1**2)/(1./dx1 + 1./dx2)
                  self.Gxi[j,i] -= vip12*D[1] * dfdx;

            self.Gxi[j,i] /= self.dist[j,i]

  # calculate the runaway energy flux and flow speed Up(exluding the diffusive flux)
  def computeU(self) :
    self.Up = np.zeros(self.M)
    self.Sr0 = np.zeros(self.M)

    fr = np.zeros(self.M)
    Sr = np.zeros(self.N)
    Sr1 = np.zeros(self.N)
    fn = np.zeros(self.N)

    vte = self.param['vte']
    Z = self.param['Z']
    E = self.param['E']
    sr = self.param['sr']
    rho = self.param['rho']

    dj = 0.5

    hy = np.zeros(self.N)
    hy[0] = 0.5*(self.xip12[1]-self.xip12[0]) + self.xip12[0]+1.0
    hy[self.N-1] = 0.5*(self.xip12[self.N-1]-self.xip12[self.N-2]) + 1.0 - self.xip12[self.N-1]
    hy[1:self.N-1] = 0.5*(self.xip12[2:self.N]-self.xip12[0:self.N-2])

    for i in range(self.M):
      x = self.vp12[i]/np.sqrt(1.+self.vp12[i]**2)/vte
      cf = 0.5/x**2 *(sp.erf(x) - 2.0*x*np.exp(-x**2)/np.sqrt(np.pi)) / (0.5*vte**2)
      ca = np.sqrt(1+self.vp12[i]**2)/self.vp12[i] * (0.5*vte**2) * cf

      vip12 = self.vp12[i]

      for j in range(self.N):

        flux = -( E*self.xip12[j] + sr*vip12*np.sqrt(1+vip12**2)*( (1. - self.xip12[j]**2) + rho**4 * vip12**2 * self.xip12[j]**4) )
        flux -= cf #(cf - ca/vip1)

        flux *= self.dist[j,i] #0.5*(self.dist[j,i+1] + self.dist[j,i])
        Sr[j] = flux*hy[j]

	# energy diffusive flux due to collision (central differencing)
        if i == self.M-1:
          flux = 0.
        elif i == 0 :
          x0 = 2.*self.xmin - self.vp12[0]
          dx1 = self.vp12[0] - x0
          dx2 = self.vp12[1] - self.vp12[0]
          fmax = 1.0/(vte**3 * np.pi*np.sqrt(np.pi)) * np.exp((1-np.sqrt(1+x0**2))/(0.5*vte**2))
          dfdx =  ((self.dist[j,i+1]-self.dist[j,i])/dx2**2 + (self.dist[j,i]- fmax)/dx1**2)/(1./dx1 + 1./dx2)
          flux -= ca * dfdx
          Sr1[j] = flux*hy[j]*vip12**2
        else :
          dx1 = self.vp12[i]-self.vp12[i-1]
          dx2 = self.vp12[i+1]-self.vp12[i]
          dfdx = ((self.dist[j,i+1]-self.dist[j,i])/dx2**2 + (self.dist[j,i]-self.dist[j,i-1])/dx1**2)/(1./dx1 + 1./dx2)
          flux -= ca * dfdx
          Sr1[j] = flux*hy[j]*vip12**2

        fn[j] = 2*np.pi * hy[j] * self.dist[j,i]#0.5*(self.dist[j,i] + self.dist[j,i+1])

      self.Up[i] = np.sum(Sr)/(np.sum(fn) + 1.e-28)
      #self.Sr0[i] = np.sum(Sr1)# - 2*np.pi*dy * self.vp1[i]**2 * ( ca*(np.sum(self.dist[:,i+1])-np.sum(self.dist[:,i]))/(self.vp12[i+1]-self.vp12[i]) )
