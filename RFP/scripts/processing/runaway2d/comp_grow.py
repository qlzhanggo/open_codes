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

import argparse

from runaway import runaway, zeta1

# a function to sort the string array in natual order
def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


print('----------------------------------------------------------------------')
print('---- Postprocessing RFP result for runaway avalanche growth rate ----')
print('----------------------------------------------------------------------')

def read_dist(dir='') :
    t = []
    ne = []
#    je = []
#    ke = []

    with open('{}output_data/time.txt'.format(dir+'/'), 'r') as f:
        f.readline()
        line = f.readline()
        prevt = -0.5
        while line:
            line = line.strip()
            line = line.strip(";")
            if line:
                data = line.split()

                if data[0] == 'step:':
                    if (float(data[3]) >= (prevt + 0.5) and float(data[3])<=200.0) :
                        t.append(data[3])
                        prevt = float(data[3])

                        fe = runaway('{0}mesh.m'.format(dir+'/'), '{}input_params.m'.format(dir+'/'), '{0}output_data/soln_{1}.dat'.format(dir+'/',data[1]), mute=True)
                        nj = fe.output_ne(3.5)
                        eps = fe.eps
                        ne.append(nj[0])
 #                       je.append(nj[1])
 #                       ke.append(nj[2])

            line = f.readline()

    M = np.argwhere(np.asarray(t, dtype=np.float64) >= 35)[0][0]
    print(M)
    a1 = np.polyfit(np.asarray(t, dtype=np.float64)[M::], np.log(np.asarray(ne, dtype=np.float64)[M::]),1)
    print('growth rate is {0}'.format(a1))
    return a1[0]


parser = argparse.ArgumentParser(description='Postprocessing RFP result for runaway avalanche growth rate')
parser.add_argument('-dir', metavar='dir', type=str, nargs='?', default=None,
                   help='the folder with data file')

args = parser.parse_args()

if args.dir is not None :
   os.chdir(args.dir)
    
import re
E = []
for entry in os.scandir('.'):
    if entry.is_dir() and ("E_" in entry.path):
       s = re.sub("\./E_", "", entry.path)
       E.append(s)

E = np.sort(np.asarray(E, dtype=float))
#m = 4
#E = np.linspace(1.73, 1.75, m)
print(E)

growth = np.zeros_like(E)
for i in range(E.size) :
   run = "E_{0:.2f}".format(E[i])
   (growth[i]) = read_dist(run)

print(np.array2string(E,separator=","))
#growth_ = ["{0:.3f}".format(member) for member in growth]
print(np.array2string(growth, separator=","))
