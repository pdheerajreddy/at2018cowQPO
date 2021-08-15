import numpy as np
import subprocess
import os
from astropy.io import fits
import pandas as pd
import pickle

fname="all_clean_events_and_gtis.fits"

with fits.open(fname) as hdu:
	hd = hdu['EVENTS'].data
	times = hd['TIME']
	chans = hd['PI']

	gtidata = hdu['GTI'].data
	tstarts = gtidata['START']
	tstops = gtidata['STOP']

print("The barycenter-corrected arrival times of X-ray events are in the array named times")
print("The barycenter-corrected GTI start and end times are in arrays names tstarts and tstops, respectively")