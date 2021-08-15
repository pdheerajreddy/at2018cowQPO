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
import string

outdir = str(workingdir)+"/output/"
if not os.path.isdir(outdir):
    subprocess.run("mkdir "+str(outdir), shell=True)

t0 = time.time()

""" Read the obsid list (It must be an ascii file..)
"""
df = pd.read_csv(obsidlist, delim_whitespace=True, header=None, converters={i: str for i in range(0,1000)})
obsids = df.iloc[:,0]

#+++---------------------------------------
#               OVERSHOOTS 
#+++---------------------------------------

ostimes = np.array([])
tstarts = np.array([])
tstops = np.array([])

for i in obsids:
    fname = str(workingdir) + '/'+str(i)+'-work-clean/overshoots/ni' + str(i) + '_0mpu7_ufa_bary.evt' if bary else str(workingdir) + '/'+str(i)+'-work-clean/overshoots/ni' + str(i) + '_0mpu7_ufa.evt'
    
    with fits.open(fname) as hdu:
        hd = hdu['EVENTS'].data
        times = hd['TIME']

    gtifname = str(workingdir) +'/'+str(i)+'-work-clean/'+ str(i) + '_bary.gti' if bary else str(workingdir) +'/'+str(i)+'-work-clean/'+ str(i) + '.gti'
    with fits.open(gtifname) as hdu:
        gtidata = hdu['STDGTI'].data
        curr_tstarts = gtidata['START']
        curr_tstops = gtidata['STOP']

    deltas = (curr_tstops - curr_tstarts)
    index = (deltas > 0)

    clean_tstarts = curr_tstarts[index]
    clean_tstops = curr_tstops[index]

    ostimes = np.append(ostimes, times)
    tstarts = np.append(tstarts, clean_tstarts)
    tstops = np.append(tstops, clean_tstops)


""" EXTRACT THE AVERAGE PDS
"""
_, os_pds_df, os_avg_pds, os_nspecs, os_meanrates, os_err_meanrates, os_meantimes, _,_, os_totcounts = helper.extract_avg_pds(ostimes, tstarts, tstops, lcsize=pdslcsize, lcresofpds=1/(2.0*maxfrequency), lcresforlcplot=1)

print("overshoot mean rate", np.mean(os_meanrates))

os_freqs = os_pds_df['freq']

os_hot = np.array(helper.rebin(os_freqs, pdsrebinfactor))
os_pot = np.array(helper.rebin(os_avg_pds, pdsrebinfactor))
# inx = (hot>50) & (hot<150) | (hot>300) & (hot < 450)
meanval = np.mean(os_pot)
os_zot = os_pot/np.sqrt(os_nspecs*pdsrebinfactor)
os_pot = os_pot*2/meanval
os_zot = os_zot*2/meanval


"""
++++++0000000+++++++---------++++++0000000+++++++---------++++++0000000+++++++---------++++++0000000+++++++---------
-------------------------------------------- 13-17 keV PDS
++++++0000000+++++++---------++++++0000000+++++++---------++++++0000000+++++++---------++++++0000000+++++++---------
"""
hgtimes = np.array([])

eminbk = 1300
emaxbk = 1700

for i in obsids:
    fname = str(workingdir) + '/'+str(i)+'-work-clean/ni' + str(i) + '_0mpu7_cl_bary.evt' if bary else str(workingdir) + '/'+str(i)+'-work-clean/ni' + str(i) + '_0mpu7_cl.evt'
    
    with fits.open(fname) as hdu:
        hd = hdu['EVENTS'].data
        times = hd['TIME']
        chans = hd['PI']

    inx = (chans >= eminbk) & (chans <= emaxbk)
    hgtimes = np.append(hgtimes, times[inx])

_, hg_pds_df, hg_avg_pds, hg_nspecs, hg_meanrates, hg_err_meanrates, hg_meantimes, _,_, hg_totcounts = helper.extract_avg_pds(hgtimes, tstarts, tstops, lcsize=pdslcsize, lcresofpds=1/(2.0*maxfrequency), lcresforlcplot=1)

print("Mean 13-17 keV count rate is "+str(np.mean(hg_meanrates)))

hg_freqs = hg_pds_df['freq']

hg_hot = np.array(helper.rebin(hg_freqs, pdsrebinfactor))
hg_pot = np.array(helper.rebin(hg_avg_pds, pdsrebinfactor))

meanval = np.mean(hg_pot)
hg_zot = hg_pot/np.sqrt(hg_nspecs*pdsrebinfactor)
hg_pot = hg_pot*2/meanval
hg_zot = hg_zot*2/meanval





"""
++++++0000000+++++++---------++++++0000000+++++++---------++++++0000000+++++++---------++++++0000000+++++++---------
0.-0.2 keV PDS
++++++0000000+++++++---------++++++0000000+++++++---------++++++0000000+++++++---------++++++0000000+++++++---------
"""
nztimes = np.array([])

eminnz = 0
emaxnz = 20

for i in obsids:
    fname = str(workingdir) + '/'+str(i)+'-work-clean/ni' + str(i) + '_0mpu7_cl_bary.evt' if bary else str(workingdir) + '/'+str(i)+'-work-clean/ni' + str(i) + '_0mpu7_cl.evt'
    
    with fits.open(fname) as hdu:
        hd = hdu['EVENTS'].data
        times = hd['TIME']
        chans = hd['PI']

    inx = (chans >= eminnz) & (chans <= emaxnz)
    nztimes = np.append(nztimes, times[inx])


_, nz_pds_df, nz_avg_pds, nz_nspecs, nz_meanrates, nz_err_meanrates, nz_meantimes, _,_, nz_totcounts = helper.extract_avg_pds(nztimes, tstarts, tstops, lcsize=pdslcsize, lcresofpds=1/(2.0*maxfrequency), lcresforlcplot=1)

print('0-0.2 keV mean rate is ',np.mean(nz_meanrates))

nz_freqs = nz_pds_df['freq']

nz_hot = np.array(helper.rebin(nz_freqs, pdsrebinfactor))
nz_pot = np.array(helper.rebin(nz_avg_pds, pdsrebinfactor))
# inx = (hg_hot>50) & (hg_hot<150) | (hg_hot>300) & (hg_hot < 450)
meanval = np.mean(nz_pot)
nz_zot = nz_pot/np.sqrt(nz_nspecs*pdsrebinfactor)
nz_pot = nz_pot*2/meanval
nz_zot = nz_zot*2/meanval


"""
++++++0000000+++++++---------++++++0000000+++++++---------++++++0000000+++++++---------++++++0000000+++++++---------
Trump rejected rate
++++++0000000+++++++---------++++++0000000+++++++---------++++++0000000+++++++---------++++++0000000+++++++---------
"""

rejtimes = np.array([])

for i in obsids:
    fname = str(workingdir) + '/'+str(i)+'-work-clean/ni' + str(i) + '_0mpu7_cl_bary.evt' if bary else str(workingdir) + '/'+str(i)+'-work-clean/ni' + str(i) + '_0mpu7_cl.evt'

    with fits.open(fname) as hdul:
        evtdata = hdul['EVENTS'].data
        times = evtdata['TIME']
        pis = evtdata['PI']
        pirat = evtdata['PI_RATIO']

    mask = np.isfinite(pirat)
    goopirats = pirat[mask]
    goopis = pis[mask]
    gootimes = times[mask]

    """ now filter based on pi range
    """
    inx = (goopis > 25) & (goopis < 1000)
    ggpis = goopis[inx]
    ggpirats = goopirats[inx]
    ggtimes = gootimes[inx]

    mask = (ggpirats > (1.1 + (120/ggpis)) )

    rejtimes = np.append(rejtimes, ggtimes[mask])

_, rej_pds_df, rej_avg_pds, rej_nspecs, rej_meanrates, rej_err_meanrates, rej_meantimes, _,_, rej_totcounts = helper.extract_avg_pds(rejtimes, tstarts, tstops, lcsize=pdslcsize, lcresofpds=1/(2.0*maxfrequency), lcresforlcplot=1)

print('0.25-10 keV trumpet-reject mean rate is ',np.mean(rej_meanrates))

rej_freqs = rej_pds_df['freq']

rej_hot = np.array(helper.rebin(rej_freqs, pdsrebinfactor))
rej_pot = np.array(helper.rebin(rej_avg_pds, pdsrebinfactor))
meanval = np.mean(rej_pot)
rej_zot = rej_pot/np.sqrt(rej_nspecs*pdsrebinfactor)
rej_pot = rej_pot*2/meanval
rej_zot = rej_zot*2/meanval

"""plot it
"""

plt.rc('axes', linewidth=2.)
fig, ax = plt.subplots(2, 2, figsize=[16, 15])
axs = ax.ravel()

def plot_me(hot,pot,zot,id, titletext):

    axs[id].step(hot, pot, where='mid', linewidth=2.5, color='black')
    axs[id].errorbar(hot, pot, yerr=zot, marker='o', fmt='none', elinewidth=2., capsize=2.5, ecolor='lightgray',zorder=0)
    # axs.plot(sample_x, utils.constant_plus_qpo(sample_x, p0,p1,p2,p3), 'r--', label='fit', linewidth=2, zorder=10)
    axs[id].tick_params(axis='both', which='major', labelsize=21)
    axs[id].set_title(str(titletext), fontsize=24)
    axs[id].set_xlim(np.min(hg_hot), np.max(hg_hot))
    # axs[1].set_ylim(1.9925,2.025)
    axs[id].axvspan(225-10, 225+10, alpha=0.5, color='red')
    axs[id].set_xscale('log')
    axs[id].xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
    axs[id].set_xlabel('Frequency (Hz)', fontsize=24)
    axs[id].set_ylabel('Leahy Power', fontsize=24)
    axs[id].axhline(2,linestyle='--',color='red',linewidth=2.)

plot_me(os_hot, os_pot, os_zot, 0, 'PDS of Overshoot events \n (Particle background)')
plot_me(hg_hot, hg_pot, hg_zot, 1, 'PDS of 13-17 keV events \n (Particle background)')
plot_me(nz_hot, nz_pot, nz_zot, 2, 'PDS of 0.0-0.2 keV events \n (Optical light leak events)')
plot_me(rej_hot, rej_pot, rej_zot, 3, 'PDS of Trumpet-Rejected Events \n (cosmic X-rays within the FoV)')

# plt.tight_layout()

""" ANNOTATE
"""
axs = axs.flat
for n, ax in enumerate(axs):
    ax.text(-0.15, 1.05, "("+str(string.ascii_lowercase[n])+")", transform=ax.transAxes, size=30, weight='bold')


plt.tight_layout()
plt.savefig(str(outdir)+"/Fig_S4.pdf")
plt.close()

# plt.scatter(hg_meantimes, hg_meanrates)
# plt.show()
