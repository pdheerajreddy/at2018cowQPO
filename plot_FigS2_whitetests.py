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
from params import *
import ks

t0 = time.time()

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
hot = np.array(helper.rebin(freqs, pdsrebinfactor))
pot = np.array(helper.rebin(avg_pds, pdsrebinfactor))
zot = pot/np.sqrt(nspecs*pdsrebinfactor)

outdir = str(workingdir)+"/output/"
if not os.path.isdir(outdir):
    subprocess.run("mkdir "+str(outdir), shell=True)

"""
Normlaize the PDS
"""

inx = (hot>=100) & (hot<200) | (hot>250) & (hot <= 600)
meanval = np.mean(pot[inx])
zot = 2*zot/meanval
pot = pot*2/meanval
zot = zot*2/meanval

""" Now only consider power values outside the QPO
"""

inx = (hot>=100) & (hot<=200) | (hot>=250) & (hot <=600)
obs_pows = pot[inx]

err_obs_pows = zot[inx]
fout = hot[inx]
print(len(obs_pows))

""" the maximum value in the pds
"""
inx = (hot>180) & (hot<280)
pmaxval = np.max(pot[inx])

""" Now extract the PDF, PDF, KS and Anderson Darling plots
"""

outfname = "Fig_S2.pdf"
scalingfactor = nspecs*pdsrebinfactor

gp = ks.test_anderson_darling_null(obs_pows,'chi2',2,scalingfactor=scalingfactor)


kd = ks.test_kstest_null(obs_pows,'chi2',2,scalingfactor=scalingfactor)

ks.plot_all(obs_pows,err_obs_pows,'chi2',2,outdir=outdir,fname=outfname, scalingfactor=scalingfactor,pmax=pmaxval)

ksstaerr = ks.ksstat_uncer(obs_pows, err_obs_pows, 'chi2',2,scalingfactor=scalingfactor)
print(ksstaerr)

