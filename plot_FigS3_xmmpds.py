import numpy as np
import pandas as pd
import matplotlib.ticker as mtick

import time
import os
from astropy.io import fits
import pickle
import subprocess

import qpoSearchUtils as helper
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import string

from scipy import stats
from params import workingdir, emin, emax

""" READ THE XMM EVENTS
"""
fname="xmm_source.evts"
gti="xmm.gti"
# emin = 25
# emax = 250
xmmlcsize=1024
xmmpdsfactor = 1
maxfrequency = 16

with fits.open(gti) as hdu:
	hd = hdu['STDGTI'].data
	tstarts = hd['START']
	tstops = hd['STOP']

print(np.sum(tstops-tstarts))
print(tstops-tstarts)
with fits.open(fname) as hdu:
	hd = hdu['EVENTS'].data
	times = hd['TIME']
	chans = hd['PI']

inx = (chans > emin) & (chans < emax)

times = times[inx]

""" EXTRACT THE AVERAGE PDS
"""
lcs_df, pds_df, avg_pds, nspecs, meanrates, err_meanrates, meantimes, clean_tstarts, clean_tstops, totcounts = helper.extract_avg_pds(times, tstarts, tstops, lcsize=xmmlcsize, lcresofpds=1/(2*maxfrequency), lcresforlcplot=1)

freqs = pds_df['freq']
hot = np.array(helper.rebin(freqs, xmmpdsfactor))
pot = np.array(helper.rebin(avg_pds, xmmpdsfactor))
zot = pot/np.sqrt(nspecs*xmmpdsfactor)

outdir = str(workingdir)+"/output/"
if not os.path.isdir(outdir):
    subprocess.run("mkdir "+str(outdir), shell=True)


popt1, pcov1 = curve_fit(helper.constant, hot, pot, p0=[2.0], sigma=zot)

sample_x = np.linspace(np.min(hot), np.max(hot),500)

plt.rc('axes', linewidth=2.)
fig, axs = plt.subplots(1, 1, figsize=[11.25, 7.5])
# axinset = inset_axes(axs, width="50%", height=2., bbox_to_anchor=(10, 2.001, 80, 2.01), bbox_transform=ax.transAxes, borderpad=9)
axs.step(hot, pot, where='mid', linewidth=2.5, color='black',label='Data')
axs.errorbar(hot, pot, yerr=zot, marker='o', fmt='none', elinewidth=2., capsize=2.5, ecolor='lightgray',zorder=0)
axs.plot(sample_x, helper.constant(sample_x, 2.0), 'r--', linewidth=2, label='Poisson Noise Level')
axs.tick_params(axis='both', which='major', labelsize=21)
# axs.set_title('Soft X-ray PDS of AT2018cow at low Frequencies\n', fontsize=24)
axs.set_xlim(np.min(hot), np.max(hot))
# axs.set_ylim(1.9925,2.025)
axs.set_xscale('log')
# axs.xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
axs.set_xlabel('Frequency (Hz)', fontsize=24)
axs.set_ylabel('Leahy Power', fontsize=24)

axs.legend(loc='best',prop={'size': 18})
plt.savefig(str(outdir)+'/Fig_S3.pdf')
plt.close()
print(nspecs)