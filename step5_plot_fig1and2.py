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
popt, pcov = curve_fit(helper.constant_plus_qpo, hot, pot, p0=[2.0, 1.0, curr_maxchi_nu, 0.1*curr_maxchi_nu], sigma=zot, bounds=([0.0,-np.inf,0.95*curr_maxchi_nu,0.0], [np.inf, np.inf, 1.05*curr_maxchi_nu, np.inf]),maxfev=5000)
p0, p1, p2, p3 = popt

# unpack uncertainties in fitting parameters from diagonal of covariance matrix
dp0, dp1, dp2, dp3 = [np.sqrt(pcov[j, j]) for j in range(popt.size)]

sample_x = np.linspace(np.min(hot), np.max(hot),500)

print("The QPO centroid is "+str(p2)+"+-"+str(dp2))
plt.rc('axes', linewidth=4.)
fig, axs = plt.subplots(1, 2, figsize=[30, 15],dpi=50)
# axinset = inset_axes(axs, width="50%", height=2., bbox_to_anchor=(10, 2.001, 80, 2.01), bbox_transform=ax.transAxes, borderpad=9)
axs[0].step(hot, pot, where='mid', linewidth=3.5, color='black')
axs[0].errorbar(hot, pot, yerr=zot, marker='o', fmt='none', elinewidth=2., capsize=2.5, ecolor='gray',zorder=0)
axs[0].plot(hot, helper.constant_plus_qpo(hot, p0,p1,p2,p3), 'r--', label='fit', linewidth=3, zorder=10)
axs[0].tick_params(axis='both', which='major', labelsize=45)
# axs[0].set_title('Soft X-ray Power Spectrum of Supernova AT2018cow\n', fontsize=24)
axs[0].set_xlim(30, 0.65*np.max(hot))

axs[0].set_ylim(1.99,2.02)
axs[0].tick_params(axis='both', which='major', length=15, width=4,pad=12)
axs[0].tick_params(axis='both', which='minor', length=8, width=4,pad=12)

axs[0].set_xscale('log')
axs[0].xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
axs[0].set_xlabel('Frequency (Hz)', fontsize=48)
axs[0].set_ylabel('Leahy Power', fontsize=48)
# axs[0].set_title('AT2018cow\'s average NICER power spectrum \n', fontsize=48)
axs[0].set_xticks([50,100,200,400])

""" pkl files
"""

df = pd.read_csv(fappklist, delim_whitespace=True, header=None)
files = df.iloc[:,0]

maxvals = np.array([])
for i in files:
    res = pickle.load(open(i,"rb"))
    maxvals = np.append(maxvals, res['maxchisqrs'])

edf = helper.compute_edf(maxvals)
signi = np.array([1.0 - i for i in edf[1]])


threesigma = 1/371
foursigma = 1/10000

axs[1].tick_params(axis='both', which='major', labelsize=45,direction='out',width=3,pad=12)

axs[1].semilogy(edf[0], signi, drawstyle='steps', label='Monte Carlo simulated')

max_obs = curr_max_chisqr
axs[1].axvline(max_obs,linestyle='--',label='Observed $\Delta\chi^{2}$ of the QPO',color='red')
# axs[1].set_title('Results from Monte Carlo Simulations \n', fontsize=24)
# axs[1].set_title('Statistical Significance of the 225 Hz QPO \n', fontsize=48)
# axs[1].set_xlabel('Maximum '+r'$\Delta\chi^{2}$ including frequencies up to 1024 Hz', fontsize=24)
# axs[1].set_xlabel('Maximum $\Delta\chi^{2}$ of a false QPO in Simulated Noise PDS \n for a search including all frequencies up to 1024 Hz', fontsize=24)
axs[1].set_xlabel('Maximum $\Delta\chi^{2}$ of a false QPO \n in simulated noise PDS', fontsize=48)
# axs[1].set_ylabel('Global significance', fontsize=24)
axs[1].set_ylabel('Global False Alarm Probability', fontsize=48)
axs[1].legend(loc='best', frameon=False)
axs[1].axhline(threesigma, linestyle='-.', color='magenta')
axs[1].text(10, threesigma, '99.73%', fontsize=36, va='center', ha='center', backgroundcolor='w', color='magenta')
axs[1].text(10, foursigma, '99.99%', fontsize=36, va='center', ha='center', backgroundcolor='w', color='black')

axs[1].axhline(foursigma, linestyle='dotted', color='black')
axs[1].set_ylim(6e-5,)
axs[1].set_xlim(np.min(edf[0]),30)
axs[1].legend(loc='upper right', prop={'size': 33})
plt.tight_layout(h_pad=8,pad=5)
axs[1].tick_params(axis='both', which='major', length=15, width=4)
axs[1].tick_params(axis='both', which='minor', length=8, width=4)


""" ANNOTATE
"""
axs = axs.flat
for n, ax in enumerate(axs):
	ax.text(-0.2, 1.0315, "("+str(string.ascii_lowercase[n])+")", transform=ax.transAxes, size=40, weight='bold')

# plt.tight_layout()
plt.savefig(str(outdir)+"/Fig1.pdf")
plt.close('all')


"""
PLot Fig. 2
"""

meantimes = helper.niMET_to_MJD(meantimes)

meantimes = meantimes - mjdstart

"""NOW, READ THE SWIFT LIGHT CURVE
"""
df = pd.read_csv(xrtlc, header=None, delim_whitespace=True)

times = df.iloc[:,6]
inx = np.argsort(times)
rates = df.iloc[:,1]
erates = df.iloc[:,2]

times = times[inx]
rates = rates[inx]
erates = erates[inx]
times = times - mjdstart

"""ONLY CONSIDER TIME BELOW DAY 65
"""
inx = (times < 65) & (times > 6)
times = times[inx]
rates = rates[inx]
erates = erates[inx]

fig, ax2 = plt.subplots(2, 1, figsize=(17, 24), dpi=60,sharex=True,gridspec_kw={'hspace': 0})

color='dodgerblue'
ax2[0].set_ylabel('Swift/XRT count rate', fontsize=48, color=color)
# ax2.scatter(times, rates, color=color, s=200,zorder=0,alpha=0.5)
ax2[0].scatter(times, rates, color=color, s=250, marker='D', zorder=2,label='Neil Gehrels Swift')
ax2[0].errorbar(times, rates,yerr=erates,fmt='D--',capsize=10,linewidth=2,markersize=20, color=color, alpha=0.75,zorder=1)
# ax.errorbar(times_so, fractions,yerr=e_fractions,color='lightskyblue',zorder=0)
ax2[0].tick_params(axis='y', labelcolor=color,labelsize=45,direction="in", length=16, width=2, color=color,pad=10)
ax2[0].tick_params(axis="x", direction="in", length=16, width=2, color="black")

ax2[0].set_xticks([10,20,30,40,50,60])
ax2[0].xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
ax2[0].tick_params(axis="x", direction="in", length=16, width=2, color="black",pad=10)
ax2[0].set_xlabel('Time (in days since MJD '+str(mjdstart)+')', fontsize=48)
ax2[0].tick_params(axis='x', which='major', labelsize=45)


ax = ax2[0].twinx()  # instantiate a second axes that shares the same x-axis

color = 'crimson'
ax.set_ylabel('NICER/XTI count rate', color=color,fontsize=48)  # we already handled the x-label with ax1
# ax.scatter(meantimes, meanrates, color=color, s=200, marker='D',zorder=10)
ax.scatter(meantimes, meanrates, color=color, s=250, marker='X',zorder=10,label='NICER')
ax.errorbar(meantimes, meanrates, yerr=err_meanrates, fmt='X--',capsize=10,linewidth=2,markersize=20, color=color, alpha=0.35,zorder=10)
ax.tick_params(axis='y', labelcolor=color,labelsize=45,direction="in", length=16, width=2, color=color,pad=10)
# ax.set_title("Long-term Evolution of the X-ray corona in TDE AT 2018fyk's",fontsize=48)
# ax.set_xscale('log')
ax.legend(loc=(0.4,0.77),prop={'size': 42},frameon=False)
# ax.set_ylim([-0.2,7])
# ax2.set_ylim([0,0.6])
# fig.tight_layout()

# ax2.legend(loc='upper right',prop={'size': 30})
ax2[0].legend(loc=(0.4,0.84),prop={'size': 42},frameon=False)

""" Plot the rms vs time
"""
lcs_df, pds_df, avg_pds, nspecs, meanrates, err_meanrates, meantimes, clean_tstarts, clean_tstops, totcounts = helper.extract_avg_pds(cleantimes, tstarts, tstops, lcsize=pdslcsize, lcresofpds=1/(2*maxfrequency), lcresforlcplot=1)


meantimes = helper.niMET_to_MJD(meantimes)

meantimes = meantimes - mjdstart

rms_arr = np.array([])
erms_arr = np.array([])
tdays = np.array([])

mjds_array_lower = np.array([])
mjds_array_high = np.array([])

""" First 16.5 ks
"""
pdscountstart=0
pdscountend=64

hot,pot,zot = helper.rebin_pds(freqs, np.mean(pds_df.iloc[:,1+pdscountstart:pdscountend+1],axis=1), pdscountend-pdscountstart, pdsrebinfactor, nu1min=50, nu1max=160, nu2min=300, nu2max=550)
_,_,curr_p2,curr_p3,rms,erms,newchisqr, _, oldchisqr, _, curr_snr = helper.model_pds(hot,pot,zot,meanrates[pdscountstart:pdscountend+1], err_meanrates[pdscountstart:pdscountend+1], nu0=230, numin=170, numax=300)

rms_arr = np.append(rms_arr, rms)
erms_arr = np.append(erms_arr, erms)

curr_meanrates = meanrates[pdscountstart:pdscountend+1]
curr_meantimes = meantimes[pdscountstart:pdscountend+1]

tdays = np.append(tdays, np.sum(curr_meanrates*curr_meantimes)/np.sum(curr_meanrates))

mjds_array_lower = np.append(mjds_array_lower, (np.sum(curr_meanrates*curr_meantimes)/np.sum(curr_meanrates)) - np.min(curr_meantimes))
mjds_array_high = np.append(mjds_array_high, np.max(curr_meantimes) - (np.sum(curr_meanrates*curr_meantimes)/np.sum(curr_meanrates)))

""" second half
"""
pdscountstart=65
pdscountend=105

hot,pot,zot = helper.rebin_pds(freqs, np.mean(pds_df.iloc[:,1+pdscountstart:pdscountend+1],axis=1), pdscountend-pdscountstart, pdsrebinfactor, nu1min=50, nu1max=160, nu2min=300, nu2max=550)
_,_,curr_p2,curr_p3,rms,erms,newchisqr, _, oldchisqr, _, curr_snr = helper.model_pds(hot,pot,zot,meanrates[pdscountstart:pdscountend+1], err_meanrates[pdscountstart:pdscountend+1], nu0=230, numin=170, numax=300)

rms_arr = np.append(rms_arr, rms)
erms_arr = np.append(erms_arr, erms)
curr_meanrates = meanrates[pdscountstart:pdscountend+1]
curr_meantimes = meantimes[pdscountstart:pdscountend+1]

tdays = np.append(tdays, np.sum(curr_meanrates*curr_meantimes)/np.sum(curr_meanrates))

mjds_array_lower = np.append(mjds_array_lower, (np.sum(curr_meanrates*curr_meantimes)/np.sum(curr_meanrates)) - np.min(curr_meantimes))
mjds_array_high = np.append(mjds_array_high, np.max(curr_meantimes) - (np.sum(curr_meanrates*curr_meantimes)/np.sum(curr_meanrates)))

print(np,min(meanrates), np.max(meanrates))

ax2[1].scatter(tdays, rms_arr, s=600,zorder=10,color='black')
ax2[1].scatter(tdays, rms_arr, s=1200, facecolors='none',color='black', zorder=9)
ax2[1].errorbar(tdays, rms_arr,xerr=[ mjds_array_lower,mjds_array_high], yerr=erms_arr,fmt='o',capsize=10,linewidth=2,markersize=5, alpha=0.5,zorder=2,color='black')
ax2[1].tick_params(axis='both', which='major', labelsize=45)
ax2[1].set_ylabel('QPO\'s fractional rms amplitude\n'+str(" ")+r'(Percentage of mean flux)', fontsize=48)
ax2[1].set_xlabel("Time (in days since MJD "+str(mjdstart)+")",fontsize=48)
ax2[1].tick_params(axis="x", direction="in", length=16, width=2, color="black",pad=10)
ax2[1].tick_params(axis='x', which='major', labelsize=45)

""" ANNOTATE
"""
ax2 = ax2.flat
for n, ax in enumerate(ax2):
    ax.text(0.01, 0.95, "("+str(string.ascii_lowercase[n])+")", transform=ax.transAxes, size=40, weight='bold')

print(rms_arr)
print(erms_arr)
print(tdays)
print(mjds_array_high)
print(mjds_array_lower)
plt.tight_layout()
plt.savefig(str(outdir)+"/Fig2.pdf")
plt.close(fig)