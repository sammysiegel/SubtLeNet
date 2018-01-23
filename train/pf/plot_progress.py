import numpy as np 
from collections import namedtuple
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde

import matplotlib as mpl
mpl.use('cairo')
from matplotlib import pyplot as plt
import seaborn


## general layout                                                                                                                      
seaborn.set(style="ticks")
seaborn.set_context("poster")
mpl.rcParams['axes.linewidth'] = 1.25
fig_size = plt.rcParams['figure.figsize']
fig_size[0] = 10
fig_size[1] = 9
plt.rcParams['figure.figsize'] = fig_size


acc = []
loss = []
with open('train_lstm.log') as flog:
    for line in flog.readlines():
        l = line.split(',')
        acc.append(float(l[1].split('=')[1]))
        loss.append(float(l[2].split('=')[1]))

acc = np.array(acc)
loss = np.array(loss)
batches = np.arange(acc.shape[0])

stack_acc = np.vstack([batches, acc])
stack_loss = np.vstack([batches, loss])
density_acc = gaussian_kde(stack_acc)(stack_acc)
density_loss = gaussian_kde(stack_loss)(stack_loss)

plt.clf()

fig, ax1 = plt.subplots()
idx = density_loss.argsort()
ax1.scatter(batches[idx], loss[idx], c=density_loss[idx], cmap='Reds', edgecolor='')
ax1.set_xlabel('Batch update')
ax1.set_ylabel('Loss', color='r')
ax1.tick_params('y', colors='r')

ax2 = ax1.twinx()
idx = density_loss.argsort()
ax2.scatter(batches[idx], acc[idx], c=density_acc[idx], cmap='Blues', edgecolor='')
ax2.set_ylabel('Accuracy', color='b')
ax2.tick_params('y', colors='b')

fig.tight_layout()

OUTPUT = '/home/snarayan/public_html/figs/testplots/test_lstm/log'
plt.savefig(OUTPUT+'.png',bbox_inches='tight',dpi=300)
plt.savefig(OUTPUT+'.pdf',bbox_inches='tight')

def ma(arr, n):
    csum = np.cumsum(arr)
    csum[n:] = csum[n:] - csum[:-n]
    return csum[n-1:]/n


ma_acc = ma(acc, 500)
ma_loss = ma(loss, 500)

diff = batches.shape[0] - ma_acc.shape[0]
diff_lo = diff/2 
diff_hi = diff_lo if  2*diff_lo == diff else diff_lo+1
batches = batches[diff_lo:-diff_hi]

# plt.clf()
# fig, ax1 = plt.subplots()
ax1.plot(batches, ma_loss, 'k-')
ax1.set_xlabel('Batch update')
ax1.set_ylabel('Loss', color='r')
ax1.tick_params('y', colors='r')

# ax2 = ax1.twinx()
ax2.plot(batches, ma_acc, 'k-')
ax2.set_ylabel('Accuracy', color='b')
ax2.tick_params('y', colors='b')

fig.tight_layout()

OUTPUT = '/home/snarayan/public_html/figs/testplots/test_lstm/ma_log'
plt.savefig(OUTPUT+'.png',bbox_inches='tight',dpi=300)
plt.savefig(OUTPUT+'.pdf',bbox_inches='tight')
