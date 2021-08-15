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

import csv

from scipy import stats
from params import *

t0 = time.time()

critexposure=5000
file="exposure_vs_signal.dat"

outdir = str(workingdir)+"/output/"
if not os.path.isdir(outdir):
    subprocess.run("mkdir "+str(outdir), shell=True)


forfile = str(outdir)+'/'+str(file)

""" Read the obsid list (It must be an ascii file..)
"""
df = pd.read_csv(obsidlist, delim_whitespace=True, header=None, converters={i: str for i in range(0,1000)})
obsids = df.iloc[:,0]

""" Read all the events and GTI values
"""
cleantimes = np.array([])
tstarts = np.array([])
tstops = np.array([])

for i in obsids:
    fname = str(workingdir) + '/'+str(i)+'-work-clean/ni' + str(i) + '_0mpu7_cl_bary.evt' if bary else str(workingdir) + '/'+str(i)+'-work-clean/ni' + str(i) + '_0mpu7_cl.evt'
    
    with fits.open(fname) as hdu:
    	hd = hdu['EVENTS'].data
    	times = hd['TIME']
    	chans = hd['PI']

    gtifname = str(workingdir) +'/'+str(i)+'-work-clean/'+ str(i) + '_bary.gti' if bary else str(workingdir) +'/'+str(i)+'-work-clean/'+ str(i) + '.gti'
    with fits.open(gtifname) as hdu:
    	gtidata = hdu['STDGTI'].data
    	curr_tstarts = gtidata['START']
    	curr_tstops = gtidata['STOP']

    inx = (chans >= emin) & (chans <= emax)
    
    deltas = (curr_tstops - curr_tstarts)
    index = (deltas > 0)

    clean_tstarts = curr_tstarts[index]
    clean_tstops = curr_tstops[index]

    cleantimes = np.append(cleantimes, times[inx])
    tstarts = np.append(tstarts, clean_tstarts)
    tstops = np.append(tstops, clean_tstops)

""" Plot the figure
"""

lcs_df, pds_df, avg_pds, nspecs, meanrates, err_meanrates, meantimes, clean_tstarts, clean_tstops, totcounts = helper.extract_avg_pds(cleantimes, tstarts, tstops, lcsize=pdslcsize, lcresofpds=1/(2*maxfrequency), lcresforlcplot=1)

freqs = pds_df['freq']
crit_pdsno = int( (nspecs*pdslcsize - critexposure)/pdslcsize)

""" How many PDS are there?
"""
npds = pds_df.shape[1]-1
print("The total number of power spectra averaged were: "+str(npds))

""" +-================================
 FIRST, GET THE SNR VS EXPOSURE (ACCUMULATED FORWARD) DATA
+-====================================
"""
running_snr_for = np.array([])
running_exposure_for = np.array([])
running_delchi_for = np.array([])
# running_times_for = np.array([])

running_centroid_for = np.array([])
running_width_for = np.array([])
running_centroid_for_err = np.array([])
running_width_for_err = np.array([])

running_cr = np.array([])

for i in range(npds):
	pdscount = i+1
	if pdslcsize*pdscount > critexposure:
		hot,pot,zot = helper.rebin_pds(freqs, np.mean(pds_df.iloc[:,1:pdscount+1],axis=1), pdscount, pdsrebinfactor, nu1min=50, nu1max=160, nu2min=300, nu2max=550)
		_,_,curr_p2,curr_p3,_,_,newchisqr, _, oldchisqr, _, curr_snr = helper.model_pds(hot,pot,zot,meanrates[0:i], err_meanrates[0:i], nu0=230, numin=170, numax=300)
		curr_expos = pdscount*pdslcsize

		running_exposure_for = np.append(running_exposure_for, curr_expos)
		running_snr_for = np.append(running_snr_for,curr_snr)
		running_delchi_for = np.append(running_delchi_for, oldchisqr-newchisqr)
		running_centroid_for = np.append(running_centroid_for, curr_p2[0])
		running_centroid_for_err = np.append(running_centroid_for_err, curr_p2[1])
		running_width_for = np.append(running_width_for, curr_p3[0])
		running_width_for_err = np.append(running_width_for_err, curr_p3[1])
		# running_times_for = np.append(running_times_for, meantimes[i-1])
		running_cr = np.append(running_cr, np.mean(meanrates[0:i]))

""" SAVE THESE DATA TO AN ASCII FILE
"""

with open(forfile,'a') as f1:
	writer = csv.writer(f1, delimiter='\t', lineterminator='\n')
	row = ['Exposure', 'delchi','snr','qpo centroid', 'qpo centroid error', 'qpo width', 'qpo width error',"mean rate"]
	writer.writerow(row)
	for i in range(len(running_delchi_for)):
		row = [running_exposure_for[i], round(running_delchi_for[i],1), round(running_snr_for[i],1), round(running_centroid_for[i],1), round(running_centroid_for_err[i],1), round(running_width_for[i],1), round(running_width_for_err[i],1), running_cr[i]]
		writer.writerow(row)


fig, ax2 = plt.subplots(1, 1, figsize=(25, 18), dpi=50)


color='dodgerblue'
ax2.set_ylabel('QPO\'s signal-to-noise ratio', fontsize=48, color=color)
ax2.scatter(running_exposure_for, running_snr_for, color=color, s=250, marker='D', zorder=2,label='SNR')
ax2.tick_params(axis='y', labelcolor=color,labelsize=45,direction="in", length=16, width=2, color=color,pad=10)
ax2.tick_params(axis="x", direction="in", length=16, width=2, color="black")

ax2.tick_params(axis="x", direction="in", length=16, width=2, color="black",pad=10)
ax2.set_xlabel('Exposure (in seconds)', fontsize=48)
ax2.tick_params(axis='x', which='major', labelsize=45)


ax = ax2.twinx()  # instantiate a second axes that shares the same x-axis

color = 'crimson'
ax.set_ylabel('QPO\'s signal-to-noise ratio/count rate', color=color,fontsize=48)  # we already handled the x-label with ax1
ax.scatter(running_exposure_for, running_snr_for/running_cr, color=color, s=250, marker='X',zorder=10,label='SNR/count rate')
ax.tick_params(axis='y', labelcolor=color,labelsize=45,direction="in", length=16, width=2, color=color,pad=10)
ax.tick_params(axis='x', labelcolor='black',labelsize=45,direction="in", length=16, width=2, color='k',pad=10)
ax.legend(loc=(0.65,0.14),prop={'size': 42})
# fig.tight_layout()

# ax2.legend(loc='upper right',prop={'size': 30})
ax2.legend(loc=(0.65,0.26),prop={'size': 42})

# print(np,min(meanrates), np.max(meanrates))
plt.savefig(str(outdir)+"/Fig_S7.pdf")
plt.close(fig)

