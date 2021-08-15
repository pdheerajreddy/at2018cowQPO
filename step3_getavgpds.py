import numpy as np
import pandas as pd
import qpoSearchUtils as helper

import time
import os
from astropy.io import fits
import pickle
import subprocess
from params import *
from scipy.optimize import curve_fit
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt

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

"""
Extract the average power density spectrum 
"""

lcs_df, pds_df, avg_pds, nspecs, meanrates, err_meanrates, meantimes, clean_tstarts, clean_tstops, totcounts = helper.extract_avg_pds(cleantimes, tstarts, tstops, lcsize=pdslcsize, lcresofpds=1/(2*maxfrequency), lcresforlcplot=1)

print(nspecs*pdslcsize)
freqs = pds_df['freq']
hot = np.array(helper.rebin(freqs, pdsrebinfactor))
pot = np.array(helper.rebin(avg_pds, pdsrebinfactor))
zot = pot/np.sqrt(nspecs*pdsrebinfactor)

outdir = str(workingdir)+"/output/"
if not os.path.isdir(outdir):
    subprocess.run("mkdir "+str(outdir), shell=True)

""" Get the strongest QPO candidate, it's frequency and delta-chi square value
"""
curr_max_chisqr, curr_maxchi_nu = helper.get_max_delta_chisqr_given_pds(hot,pot,zot)

""" Normalize the mean continuum level to 2 (excluding the QPO candidate bins)
"""
ingex = (hot < curr_maxchi_nu*0.9) | (hot > curr_maxchi_nu*1.1)
meanlevel = np.mean(pot[ingex])
zot = 2*zot/meanlevel
pot = 2*pot/meanlevel

""" Model the strongest QPO candidate
"""
# popt, pcov = curve_fit(helper.constant_plus_qpo, hot, pot, p0=[2.0, 1.0, curr_maxchi_nu, 0.1*curr_maxchi_nu], sigma=zot, bounds=([0.0,-np.inf,0.95*curr_maxchi_nu,0.0], [np.inf, np.inf, 1.05*curr_maxchi_nu, np.inf]),maxfev=5000)
popt, pcov = curve_fit(helper.constant_plus_qpo, hot, pot, p0=[2.0, 1.0, 225, 0.1*curr_maxchi_nu], sigma=zot, bounds=([0.0,-np.inf,0.9*225,0.0], [np.inf, np.inf, 1.05*225, np.inf]),maxfev=5000)
p0, p1, p2, p3 = popt

# unpack uncertainties in fitting parameters from diagonal of covariance matrix
dp0, dp1, dp2, dp3 = [np.sqrt(pcov[j, j]) for j in range(popt.size)]

""" Calculate the SNR (integral above continuum over error on intergral)
"""
sample_x = np.linspace(np.min(hot), np.max(hot),500)

inx = (hot >= p2 - 1 * np.max([p3,hot[1]-hot[0]])) & (hot <= p2 + 1 * np.max([p3,hot[1]-hot[0]]))

qpopot = pot[inx] - p0
qpozot = zot[inx]
qpointegral = np.sum(qpopot)
qpointegral_error = np.sqrt(np.sum(qpozot ** 2))
print(qpointegral, qpointegral_error)
print("QPO SNR:" +str(qpointegral/qpointegral_error))
# Fractional RMS of QPO:
denom = np.mean(meanrates)
rms1 = 100.0 * np.sqrt(qpointegral*(hot[1]-hot[0])/ denom)
mean_meanrate_error = np.sqrt(np.sum(err_meanrates**2)) / np.size(err_meanrates)

print("mean count rate "+str(denom)+"+-"+str(mean_meanrate_error)+" counts/sec")

# erms1 = rms1 * 0.5 * np.sqrt((dp1 / p1) ** 2 + (dp3 / p3) ** 2 + (mean_meanrate_error / np.mean(meanrates)) ** 2)


erms1 = rms1 * 0.5 * np.sqrt( (qpointegral_error/qpointegral)**2 + (mean_meanrate_error / np.mean(meanrates))**2)

print("QPO rms:"+str(rms1)+"+-"+str(erms1))



""" Plot
"""
plt.rc('axes', linewidth=4.)
fig, axs = plt.subplots(1, 1, figsize=[16, 12])
# axinset = inset_axes(axs, width="50%", height=2., bbox_to_anchor=(10, 2.001, 80, 2.01), bbox_transform=ax.transAxes, borderpad=9)
axs.step(hot, pot, where='mid', linewidth=3.5, color='black')
axs.errorbar(hot, pot, yerr=zot, marker='o', fmt='none', elinewidth=2., capsize=2.5, ecolor='gray',zorder=0)
axs.plot(hot, helper.constant_plus_qpo(hot, p0,p1,p2,p3), 'r--', label='fit', linewidth=2, zorder=10)
axs.tick_params(axis='both', which='major', labelsize=30)
axs.set_title('Soft X-ray Power Spectrum of Supernova AT2018cow\n', fontsize=24)
axs.set_xlim(20, 0.6*np.max(hot))
# axs.axvline(p2)
# axs.axvline(2*p2)
# axs.axvline(0.5*p2)

axs.set_ylim(1.9925,2.02)
axs.tick_params(axis='both', which='major', length=10, width=3)
axs.tick_params(axis='both', which='minor', length=5, width=3)

axs.set_xscale('log')
axs.xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
axs.set_xlabel('Frequency (Hz)', fontsize=40)
axs.set_ylabel('Leahy Power', fontsize=40)

plt.tight_layout()
plt.savefig(str(outdir)+"/pds_"+str(nameid)+"."+str(saveformat))
plt.close('all')

print("QPO centroid frequency: "+str(p2)+"+-"+str(dp2))
print("The delchi-square improvement for the strongest QPO candidate is:"+str(curr_max_chisqr)) 
print("The strongest QPO candidate has a centroid frequency of "+str(curr_maxchi_nu))

print("Total time to run: "+str(time.time()-t0)+" secs")