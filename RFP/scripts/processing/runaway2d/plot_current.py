import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("output_data/runaway.txt",delimiter=",")
#print(data)
m,n=np.shape(data)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title("Total current vs runaway current")    
ax1.set_ylabel('current value')
ax1.set_xlabel('time')
ax1.plot(data[:,0],np.abs(data[:,1]), c='r', label='J_RE')
if (n>2):
    ax1.plot(data[:,0],np.abs(data[:,2]), c='b', label='J_full')
ax1.set_yscale('log')
leg = ax1.legend()
plt.savefig('runaway.png')

if (n>2):
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.set_title("E field in self consistent model")    
    ax2.set_ylabel('E field')
    ax2.set_xlabel('time')
    ax2.plot(data[:,0],np.abs(data[:,3]), c='r')
    plt.savefig('E_runaway.png')

plt.show()
