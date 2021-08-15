import numpy as np
import pandas as pd
import matplotlib.ticker as mtick
import getfaps as gf

import time
import os
from astropy.io import fits
import pickle
import subprocess
from params import *

"""
Define the basic params and read in all the event files and GTIs
"""

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

""" --++
Run an initial set of simulations for estimating delta-chi-square threshold	
"""

Nsims = 5000
myobj = gf.faps(cleantimes=cleantimes, tstarts=tstarts, tstops=tstops, workingdir=workingdir, pdslcsize=pdslcsize, maxfrequency=maxfrequency, pdsrebinfactor=pdsrebinfactor, Nsims=Nsims, simsdir="simfaps5")
myobj.run()

print("Total time taken for "+str(Nsims)+" is "+str(time.time()-t0)+" seconds")