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

outdir = str(workingdir)+"/output/movie/"
if not os.path.isdir(outdir):
    subprocess.run("mkdir "+str(outdir), shell=True)

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

meantimes = helper.niMET_to_MJD(meantimes)

meantimes = meantimes - mjdstart

freqs = pds_df['freq']


def model_pds(hot,pot,zot,meanrates, err_meanrates):
    """Fit the PDS with a constant
    """
    popt1, pcov1 = curve_fit(helper.constant, hot, pot, p0=[2.0], sigma=zot)
    """Fit the PDS with a constant + QPO
    """
    popt, pcov = curve_fit(helper.constant_plus_qpo, hot, pot, p0=[2.0, 1.0, 230, 20], sigma=zot, bounds=([1.8,0.0,200,0.0], [2.2, np.inf, 300, np.inf]))
    p0, p1, p2, p3 = popt

    return p0,p1,p2,p3

def plot_and_save(hot,pot,zot,meanrates, err_meanrates, meantimes, lclims, npds, totexposure, workingfolder):
    """ model the given pds
    """
    p0,p1,p2,p3 = model_pds(hot, pot, zot, meanrates, err_meanrates)
    """ Smoothed frequency values
    """
    sample_x = np.linspace(np.min(hot), np.max(hot),1000)
    """ plot the data
    """
    plt.rc('axes', linewidth=2.)
    fig, axs = plt.subplots(2, 1, figsize=[10, 18.])
    # axinset = inset_axes(axs, width="50%", height=2., bbox_to_anchor=(10, 2.001, 80, 2.01), bbox_transform=ax.transAxes, borderpad=9)
    axs[0].step(hot, pot, where='mid', linewidth=2.5, color='black',label='data')
    axs[0].errorbar(hot, pot, yerr=zot, marker='o', fmt='none', elinewidth=2., capsize=2.5, ecolor='lightgray')
    axs[0].plot(hot, helper.constant_plus_qpo(hot, p0,p1,p2,p3), 'r--', label='fit', linewidth=2, zorder=10)
    axs[0].tick_params(axis='both', which='major', labelsize=21)
    axs[0].set_title('Exposure accumulated = '+str(totexposure)+'\n', fontsize=24)
    axs[0].set_xlim(30, 0.6*np.max(hot))
    # axs[0].set_ylim(1.9875,2.0275)
    axs[0].set_ylim(1.99,2.02)
    # axs[0].axvline(225)
    axs[0].axvspan(225-10, 225+10, alpha=0.25, color='red')

    # axs[0].axvline(p2, color='k',linestyle='--')
    axs[0].set_xscale('log')
    axs[0].xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
    axs[0].set_xlabel('Frequency (Hz)', fontsize=24)
    axs[0].set_ylabel('Leahy Power', fontsize=24)
    axs[0].set_xticks([50,100,200,400])

    axs[1].plot(meantimes, meanrates, '-o')
    axs[1].set_xlabel('Time (days since MJD '+str(mjdstart)+')', fontsize=24)
    axs[1].set_ylabel('NICER count rate (0.25-2.5 keV, cps)', fontsize=24)
    axs[1].tick_params(axis='both', which='major', labelsize=21)
    axs[1].set_xlim(lclims[0], lclims[1])
    axs[1].set_ylim(0, 6)

    plt.tight_layout()
    plt.savefig(str(workingfolder)+"/image"+str(npds).zfill(3)+".png")
    plt.close()

"""
This function rebins a given PDS 
"""
def rebin_pds(freqs, pds, nspecs, factor):

    hot = np.array(helper.rebin(freqs, factor))
    pot = np.array(helper.rebin(pds, factor))
    inx = (hot<200) | (hot>250)
    meanval = np.mean(pot[inx])
    zot = pot/np.sqrt(nspecs*factor)
    newpot = pot*2/meanval
    newzot = zot*2/meanval

    return hot, newpot, newzot

for i in range(nspecs):
    pdscount = i+1

    curr_meanrates = meanrates[0:pdscount]
    curr_meantimes = meantimes[0:pdscount]
    curr_err_meanrates = err_meanrates[0:pdscount]

    # curr_hot = np.array(helper.rebin(freqs, pdsrebinfactor))
    # curr_pot = np.array(helper.rebin(np.mean(pds_df.iloc[:,1:pdscount+1],axis=1), pdsrebinfactor))
    # curr_zot = curr_pot/np.sqrt(pdscount*pdsrebinfactor)

    hot,pot,zot = rebin_pds(freqs, np.mean(pds_df.iloc[:,1:pdscount+1],axis=1), pdscount, pdsrebinfactor)

    plot_and_save(hot, pot, zot, curr_meanrates, curr_err_meanrates, curr_meantimes, [3,65], pdscount, pdslcsize*pdscount, outdir)
