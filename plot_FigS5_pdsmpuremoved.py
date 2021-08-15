
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

from params import *
import string

t0 = time.time()



def model_pds(hot,pot,zot,meanrates, err_meanrates):
	popt1, pcov1 = curve_fit(helper.constant, hot, pot, p0=[2.0], sigma=zot)
	popt, pcov = curve_fit(helper.constant_plus_qpo, hot, pot, p0=[2.0, 1.0, 230, 30], sigma=zot,maxfev=5000)
	p0, p1, p2, p3 = popt
	# unpack uncertainties in fitting parameters from diagonal of covariance matrix
	dp0, dp1, dp2, dp3 = [np.sqrt(pcov[j, j]) for j in range(popt.size)]
	# print(p2,dp2,p3,dp3)
	# Fractional RMS of QPO#1:

	numerator = (np.pi * p1 * p3) / 2.0
	denom = np.mean(meanrates)

	rms1 = 100.0 * np.sqrt(numerator / denom)
	mean_meanrate_error = np.sqrt(np.sum(err_meanrates**2)) / np.size(err_meanrates)

	# erms1 = rms1 * np.sqrt(0.5 * np.pi) * 0.5 * np.sqrt((dp1 / p1) ** 2 + (dp3 / p3) ** 2 + (mean_meanrate_error / np.mean(meanrates)) ** 2)
	erms1 = rms1 * 0.5 * np.sqrt((dp1 / p1) ** 2 + (dp3 / p3) ** 2 + (mean_meanrate_error / np.mean(meanrates)) ** 2)
	"""
	Calculate the reduced chi-square
	"""
	resids = pot - helper.constant_plus_qpo(hot, *popt)
	newchisqr = round(((resids / zot) ** 2).sum(), 1)
	newdof = np.size(hot) - 4 - 1

	"""
	Calculate the reduced chi-square
	"""
	resids = pot - helper.constant(hot, *popt1)
	oldchisqr = round(((resids / zot) ** 2).sum(), 1)
	olddof = np.size(hot) - 1 - 1

	sample_x = np.linspace(np.min(hot), np.max(hot),500)

	inx = (hot >= p2 - 1 * p3) & (hot <= p2 + 1 * p3)
	qpopot = pot[inx] - p0
	qpozot = zot[inx]
	qpointegral = np.sum(qpopot)
	qpointegral_error = np.sqrt(np.sum(qpozot ** 2))

	return p0,p1,p2,p3,rms1,erms1, newchisqr, newdof, oldchisqr, olddof

def run_me(excludempu,det_dict,pdsrebinfactor,lcsize=256, lcresofpds=1/2048):
	""" Read the obsid list (It must be an ascii file..)
	"""
	df = pd.read_csv(obsidlist, delim_whitespace=True, header=None, converters={i: str for i in range(0,1000)})
	obsids = df.iloc[:,0]

	cleantimes = np.array([])
	tstarts = np.array([])
	tstops = np.array([])
	cleandetids = np.array([])

	for i in obsids:
	    fname = str(workingdir) + '/'+str(i)+'-work-clean/ni' + str(i) + '_0mpu7_cl_bary.evt' if bary else str(workingdir) + '/'+str(i)+'-work-clean/ni' + str(i) + '_0mpu7_cl.evt'
	    
	    with fits.open(fname) as hdu:
	    	hd = hdu['EVENTS'].data
	    	times = hd['TIME']
	    	chans = hd['PI']
	    	detids = hd['DET_ID']

	    gtifname = str(workingdir) +'/'+str(i)+'-work-clean/'+ str(i) + '_bary.gti' if bary else str(workingdir) +'/'+str(i)+'-work-clean/'+ str(i) + '.gti'
	    with fits.open(gtifname) as hdu:
	    	gtidata = hdu['STDGTI'].data
	    	curr_tstarts = gtidata['START']
	    	curr_tstops = gtidata['STOP']

	    """ FIRST, FILTER BASED ON ENERGY
	    """
	    inx = (chans >= emin) & (chans <= emax)
	    
	    deltas = (curr_tstops - curr_tstarts)
	    index = (deltas > 0)

	    clean_tstarts = curr_tstarts[index]
	    clean_tstops = curr_tstops[index]

	    stimes = times[inx]
	    sdetids = detids[inx]

	    tstarts = np.append(tstarts, clean_tstarts)
	    tstops = np.append(tstops, clean_tstops)

	    """ THEN, FILTER BASED ON DET_ID
	    """
	    inx = [True if i not in det_dict[str(excludempu)] else False for i in sdetids]

	    cleantimes = np.append(cleantimes, stimes[inx])
	    cleandetids = np.append(cleandetids, sdetids[inx])

	lcs_df, pds_df, avg_pds, nspecs, meanrates, err_meanrates, meantimes, clean_tstarts, clean_tstops, totcounts = helper.extract_avg_pds(cleantimes, tstarts, tstops, lcsize=pdslcsize, lcresofpds=1/(2*maxfrequency), lcresforlcplot=1)
	freqs = pds_df['freq']

	hot = np.array(helper.rebin(freqs, pdsrebinfactor))
	pot = np.array(helper.rebin(avg_pds, pdsrebinfactor))
	inx = (hot<200) | (hot>250)
	meanval = np.mean(pot[inx])
	zot = pot/np.sqrt(nspecs*pdsrebinfactor)
	pot = pot*2/meanval
	zot = zot*2/meanval

	p0,p1,p2,p3,rms1,erms1, newchisqr, newdof, oldchisqr, olddof = model_pds(hot, pot, zot, meanrates, err_meanrates)
	

	return hot,pot,zot,p0,p1,p2,p3,rms1, erms1

""" PLOT
"""
outdir = str(workingdir)+"/output/"
if not os.path.isdir(outdir):
    subprocess.run("mkdir "+str(outdir), shell=True)

plt.rc('axes', linewidth=3.)
fig, ax = plt.subplots(3, 3, figsize=[20, 15])
axs = ax.ravel()
i=0
for key in det_dict:
	hot,pot,zot,p0,p1,p2,p3, rms, erms = run_me(key,det_dict=det_dict,pdsrebinfactor=pdsrebinfactor)
	print("-MPU"+str(key)+":",rms,erms)
	sample_x = np.linspace(np.min(hot), np.max(hot),500)
	axs[i].step(hot, pot, where='mid', linewidth=2.5, color='black')
	axs[i].errorbar(hot, pot, yerr=zot, marker='o', fmt='none', elinewidth=2., capsize=2.5, ecolor='lightgray',zorder=0)
	axs[i].plot(hot, helper.constant_plus_qpo(hot, p0,p1,p2,p3), 'r--', linewidth=2, zorder=10)
	axs[i].tick_params(axis='both', which='major', labelsize=21)
	axs[i].set_title('Average PDS Excluding MPU'+str(key), fontsize=24)
	axs[i].set_xlim(35, 0.5*np.max(hot))
	axs[i].set_ylim(1.99,2.02)
	axs[i].set_xscale('log')
	axs[i].xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
	axs[i].set_xlabel('Frequency (Hz)', fontsize=24)
	axs[i].set_ylabel('Leahy Power', fontsize=24)

	i+=1

plt.tight_layout()
axs[8].set_axis_off()
axs[7].set_axis_off()

""" ANNOTATE
"""
axs = axs.flat
for n, ax in enumerate(axs):
	ax.text(-0.15, 1.05, "("+str(string.ascii_lowercase[n])+")", transform=ax.transAxes, size=24, weight='bold')

plt.savefig(str(outdir)+"/Fig_S5.pdf")