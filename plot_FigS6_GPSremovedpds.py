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

from astropy.table import Table

from params import *

t0 = time.time()

pathToftools = str(os.environ['HEADAS']) + '/bin/'
pathToheainit = str(os.environ['HEADAS'])

""" Read the obsid list (It must be an ascii file..)
"""
df = pd.read_csv(obsidlist, delim_whitespace=True, header=None, converters={i: str for i in range(0,1000)})
obsids = df.iloc[:,0]

"""
+++ooo---+++ooo---+++ooo---+++ooo---+++ooo---+++ooo---+++ooo---+++ooo---
STEP 1: Read the NON-bary events and remove those within +- 4ms of integer. 
Save the rest into a fits file and barycenter them
+++ooo---+++ooo---+++ooo---+++ooo---+++ooo---+++ooo---+++ooo---+++ooo---
"""

cleantimes = np.array([])
cleanpis = np.array([])
cleanpifasts = np.array([])
cleandetids = np.array([])

"""
Read and filter times based on emin, emax 
"""
for i in obsids:
    fname = str(workingdir) + '/'+str(i)+'-work-clean/ni' + str(i) + '_0mpu7_cl.evt'
    table = Table.read(fname, memmap=True, hdu='EVENTS')
    times = np.array(table['TIME'])
    pis = np.array(table['PI'])
    pifasts = np.array(table['PI_FAST'])
    detids = np.array(table['DET_ID'])

    """ Get the fractional values of the times
    """
    frac_times = np.modf(times)[0]
    inx = (frac_times > 0.010) & (frac_times < 0.99)

    gootimes = times[inx]
    goopis = pis[inx]
    goopifasts = pifasts[inx]
    goodetids = detids[inx]

    inx = (goopis > emin) & (goopis < emax)
    # cleantimes = np.append(cleantimes, gootimes[inx])
    # cleanpis = np.append(cleanpis, goopis[inx])
    # cleanpifasts = np.append(cleanpifasts, goopifasts[inx])
    # cleandetids = np.append(cleandetids, goodetids[inx])

    cleantimes = gootimes[inx]
    cleanpis = goopis[inx]
    cleanpifasts = goopifasts[inx]
    cleandetids = goodetids[inx]

    """ Now, make a GPS excluded event list
    """

    col1 = fits.Column(name='TIME', format='D', unit='s', array=cleantimes)
    col2 = fits.Column(name='PI', format='I', unit='chan', array=cleanpis)
    col3 = fits.Column(name='PI_FAST', format='I', array=cleanpifasts, null=-32768)
    col4 = fits.Column(name='DET_ID', format='I', array=cleandetids)

    with fits.open(fname) as hdd:
        evts_hdu_header_prime = hdd[0].header
        evts_header = hdd[1].header

    empty_primary = fits.PrimaryHDU(header=evts_hdu_header_prime)
    evts_hdu = fits.BinTableHDU.from_columns([col1, col2, col3, col4], name='EVENTS', header=evts_header)

    hdul = fits.HDUList([empty_primary, evts_hdu])

    outputfile = str(workingdir) + '/'+str(i)+'-work-clean/ni' + str(i) + '_0mpu7_cl_GPS_removed.evt'
    subprocess.run("rm -rf "+str(outputfile), shell=True)
    hdul.writeto(outputfile)

    """Barycorr the GPD_removed events
    """
    pathToLocalpfiles = str(workingdir)+'/'+str(i)+'_pfiles/'
    subprocess.run('rm -rf ' + str(pathToLocalpfiles), shell=True)
    subprocess.run('mkdir ' + str(pathToLocalpfiles), shell=True)

    outfinalfile = str(workingdir) + '/'+str(i)+'-work-clean/ni' + str(i) + '_0mpu7_cl_GPS_removed_bary.evt'
    subprocess.run('. ' + str(pathToheainit) + '/headas-init.sh; unset PFILES; export PFILES="' + str(pathToLocalpfiles) + ';' + str(syspfilesdir) + '"; ' + str(pathToftools) + '/barycorr infile='+str(outputfile)+' outfile='+str(outfinalfile)+' orbitfiles='+str(workingdir)+'/'+str(i)+'/auxil/ni'+str(i)+'.orb.gz ra='+str(ra)+' dec='+str(dec)+ ' refframe=ICRS ephem=JPLEPH.430 clobber=YES', shell=True)
    subprocess.run("rm -rf "+str(outputfile),shell=True)

"""
+++ooo---+++ooo---+++ooo---+++ooo---+++ooo---+++ooo---+++ooo---+++ooo---
Read the barycentered GPS removed events and read the QPOGTIS from clean_evnts_....fits and extract an average PDS
+++ooo---+++ooo---+++ooo---+++ooo---+++ooo---+++ooo---+++ooo---+++ooo---
"""
alltimes = np.array([])
for i in obsids:
    fname = str(workingdir) + '/'+str(i)+'-work-clean/ni' + str(i) + '_0mpu7_cl_GPS_removed_bary.evt'
    table = Table.read(fname, memmap=True, hdu='EVENTS')
    times = np.array(table['TIME'])
    alltimes = np.append(alltimes, times)


"""
Open and read the GTIs
"""
tstarts = np.array([])
tstops = np.array([])


for i in obsids:
    fname = str(workingdir) + '/'+str(i)+'-work-clean/ni' + str(i) + '_0mpu7_cl_bary.evt' if bary else str(workingdir) + '/'+str(i)+'-work-clean/ni' + str(i) + '_0mpu7_cl.evt'
    
    gtifname = str(workingdir) +'/'+str(i)+'-work-clean/'+ str(i) + '_bary.gti' if bary else str(workingdir) +'/'+str(i)+'-work-clean/'+ str(i) + '.gti'
    with fits.open(gtifname) as hdu:
        gtidata = hdu['STDGTI'].data
        curr_tstarts = gtidata['START']
        curr_tstops = gtidata['STOP']

    deltas = (curr_tstops - curr_tstarts)
    index = (deltas > 0)

    clean_tstarts = curr_tstarts[index]
    clean_tstops = curr_tstops[index]

    tstarts = np.append(tstarts, clean_tstarts)
    tstops = np.append(tstops, clean_tstops)


lcs_df, pds_df, avg_pds, nspecs, meanrates, err_meanrates, meantimes, clean_tstarts, clean_tstops, totcounts = helper.extract_avg_pds(alltimes, tstarts, tstops, lcsize=pdslcsize, lcresofpds=1/(2*maxfrequency), lcresforlcplot=1)

print('The total exposure used for PDS: '+str(np.sum(clean_tstops-clean_tstarts)))

freqs = pds_df['freq']

def rebin_pds(freqs, pds, nspecs, factor):

    hot = np.array(helper.rebin(freqs, factor))
    pot = np.array(helper.rebin(pds, factor))
    inx = (hot<200) | (hot>250)
    meanval = np.mean(pot[inx])
    zot = pot/np.sqrt(nspecs*factor)
    pot = pot*2/meanval
    zot = zot*2/meanval

    return hot, pot, zot

def model_pds(hot,pot,zot,meanrates, err_meanrates):
    popt1, pcov1 = curve_fit(helper.constant, hot, pot, p0=[2.0], sigma=zot)
    popt, pcov = curve_fit(helper.constant_plus_qpo, hot, pot, p0=[2.0, 1.0, 230, 30], sigma=zot, maxfev=5000)
    p0, p1, p2, p3 = popt
    # unpack uncertainties in fitting parameters from diagonal of covariance matrix
    dp0, dp1, dp2, dp3 = [np.sqrt(pcov[j, j]) for j in range(popt.size)]
    print(p2,dp2,p3,dp3)
    # Fractional RMS of QPO#1:

    numerator = (np.pi * p1 * p3) / 2.0
    denom = np.mean(meanrates)

    rms1 = 100.0 * np.sqrt(numerator / denom)
    mean_meanrate_error = np.sqrt(np.sum(err_meanrates**2)) / np.size(err_meanrates)

    erms1 = rms1 * np.sqrt(0.5 * np.pi) * 0.5 * np.sqrt((dp1 / p1) ** 2 + (dp3 / p3) ** 2 + (mean_meanrate_error / np.mean(meanrates)) ** 2)
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

hot1,pot1,zot1 = rebin_pds(freqs, avg_pds, nspecs, pdsrebinfactor)
p0,p1,p2,p3,rms1,erms1, newchisqr, newdof, oldchisqr, olddof = model_pds(hot1, pot1, zot1, meanrates, err_meanrates)
sample_x = np.linspace(np.min(hot1), np.max(hot1),500)
print(newchisqr, newdof, oldchisqr, olddof, rms1, erms1)

plt.rc('axes', linewidth=5.)
fig, axs = plt.subplots(1, 1, figsize=[18, 15],dpi=50)
# axinset = inset_axes(axs, width="50%", height=2., bbox_to_anchor=(10, 2.001, 80, 2.01), bbox_transform=ax.transAxes, borderpad=9)
axs.step(hot1, pot1, where='mid', linewidth=4.5, color='black')
axs.errorbar(hot1, pot1, yerr=zot1, marker='o', fmt='none', elinewidth=2., capsize=2.5, ecolor='gray',zorder=0)
axs.plot(hot1, helper.constant_plus_qpo(hot1, p0,p1,p2,p3), 'r--', label='fit', linewidth=3, zorder=10)
axs.tick_params(axis='both', which='major', labelsize=45)
# axs[0].set_title('Soft X-ray Power Spectrum of Supernova AT2018cow\n', fontsize=24)
axs.set_xlim(30, 0.65*np.max(hot1))

axs.set_ylim(1.99,2.02)
axs.tick_params(axis='both', which='major', length=15, width=4,pad=12)
axs.tick_params(axis='both', which='minor', length=8, width=4,pad=12)

axs.set_xscale('log')
axs.xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
axs.set_xlabel('Frequency (Hz)', fontsize=48)
axs.set_ylabel('Leahy Power', fontsize=48)


# plt.rc('axes', linewidth=2.)
# fig, axs = plt.subplots(1, 1, figsize=[11, 9])
# # axinset = inset_axes(axs, width="50%", height=2., bbox_to_anchor=(10, 2.001, 80, 2.01), bbox_transform=ax.transAxes, borderpad=9)
# axs.step(hot1, pot1, where='mid', linewidth=2.5, color='black',label='data')
# axs.errorbar(hot1, pot1, yerr=zot1, marker='o', fmt='none', elinewidth=2., capsize=2.5, ecolor='gray')
# axs.plot(sample_x, helper.constant_plus_qpo(sample_x, p0,p1,p2,p3), 'r--', label='fit', linewidth=2, zorder=10)
# axs.tick_params(axis='both', which='major', labelsize=45)
# axs.set_title('AT2018cow\'s NICER PDS excluding plausible GPS Noise Events\n', fontsize=24)
# axs.set_xlim(np.min(hot1), 0.6*np.max(hot1))
# axs.set_ylim(1.99,2.02)
# axs.set_xscale('log')
# axs.xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
# axs.set_xlabel('Frequency (Hz)', fontsize=24)
# axs.set_ylabel('Leahy Power', fontsize=24)

plt.tight_layout()
outdir = str(workingdir)+"/output/"
if not os.path.isdir(outdir):
    subprocess.run("mkdir "+str(outdir), shell=True)

plt.savefig(str(outdir)+"/Fig_S6.pdf")