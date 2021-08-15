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


""" Now only consider power values outside the QPO
"""

inx = (hot>=100) & (hot<=200) | (hot>=250) & (hot <= 600)
obs_pows = pot[inx]

print(len(obs_pows))
"""
A linear model to fit the data
"""
def line(x,p):
    return p[0]*x + p[1]


""" A function to plot the prob plot given 
"""

def probplotv2(pot, nspecs, rebinfactor, outdir):
	obs_pows = pot*nspecs*rebinfactor
	""" Get the probplot values
	"""
	res = stats.probplot(obs_pows,dist='chi2',sparams=(2*nspecs*rebinfactor,))
	""" bestfit line params
	"""
	bestslope = res[1][0]
	bestintercept = res[1][1]/(nspecs*rebinfactor)
	""" theory and obs values
	"""
	theoryquantiles = [ i/(nspecs*rebinfactor) for i in res[0][0]]
	obsvals = [i/(nspecs*rebinfactor) for i in res[0][1]]
	""" the plot params
	"""
	x = np.linspace(np.min(theoryquantiles), np.max(theoryquantiles), 100, endpoint=True)
	y = line(x,[bestslope,bestintercept])

	plt.rc('axes', linewidth=4.)
	fig, axs = plt.subplots(1, 1, figsize=[23, 18],dpi=50)
	axs.tick_params(axis='both', which='major', length=15, width=4,pad=12)
	axs.tick_params(axis='both', which='minor', length=8, width=4,pad=12)
	axs.tick_params(axis='both', which='major', labelsize=42)

	axs.scatter(theoryquantiles, obsvals, marker='o',s=250)
	axs.plot(x,y,'r-',lw=4)

	# plt.scatter(theoryquantiles,obsvals,marker='o')
	# plt.plot(x,y,'r-',lw=2)
	plt.xlabel('Theoretical quantiles for a $\chi^2$ distribution', fontsize=48)
	plt.ylabel('Observed noise powers (ordered)', fontsize=48)
	# plt.title('$\chi^2$ Probability plot')
	plt.tight_layout()
	plt.savefig(str(outdir)+'/Fig_S1.pdf')


""" Run the probplot function
"""

probplotv2(obs_pows, nspecs, pdsrebinfactor, outdir)
