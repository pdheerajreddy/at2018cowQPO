import numpy as np
import subprocess
import os
from astropy.io import fits
import pandas as pd
import pickle
from astropy.table import Table
import qpoSearchUtils as helper


from params import *

""" Read the obsid list (It must be an ascii file..)
"""
df = pd.read_csv(obsidlist, delim_whitespace=True, header=None, converters={i: str for i in range(0,1000)})
obsids = df.iloc[:,0]


cleantimes = np.array([])
cleanpis = np.array([])
cleanpifasts = np.array([])
cleandetids = np.array([])

"""
Read and filter times based on emin, emax 
"""
for i in obsids:
	fname = str(workingdir) + '/'+str(i)+'-work-clean/ni' + str(i) + '_0mpu7_cl_bary.evt' if bary else str(workingdir) + '/'+str(i)+'-work-clean/ni' + str(i) + '_0mpu7_cl.evt'

	table = Table.read(fname, memmap=True, hdu='EVENTS')
	times = np.array(table['TIME'])
	pis = np.array(table['PI'])
	pifasts = np.array(table['PI_FAST'])
	detids = np.array(table['DET_ID'])

	inx = (pis >= emin) & (pis <= emax)
	cleantimes = np.append(cleantimes, times[inx])
	cleanpis = np.append(cleanpis, pis[inx])
	cleanpifasts = np.append(cleanpifasts, pifasts[inx])
	cleandetids = np.append(cleandetids, detids[inx])

"""
Read the barycenter corrected GTIs
"""
barytstarts =np.array([])
barytstops = np.array([])


for i in obsids:

	gtifname = str(workingdir) +'/'+str(i)+'-work-clean/'+ str(i) + '_bary.gti' if bary else str(workingdir) +'/'+str(i)+'-work-clean/'+ str(i) + '.gti'

	with fits.open(gtifname) as hdu:
		gtidata = hdu['STDGTI'].data
		curr_tstarts = gtidata['START']
		curr_tstops = gtidata['STOP']

	deltas = (curr_tstops - curr_tstarts)
	index = (deltas > 0)

	clean_tstarts = curr_tstarts[index]
	clean_tstops = curr_tstops[index]

	barytstarts = np.append(barytstarts, clean_tstarts)
	barytstops = np.append(barytstops, clean_tstops)

inx = np.argsort(barytstarts)
barytstarts = barytstarts[inx]
barytstops = barytstops[inx]

lcs_df, pds_df, avg_pds, nspecs, meanrates, err_meanrates, meantimes, clean_tstarts, clean_tstops, totcounts = helper.extract_avg_pds(cleantimes, barytstarts, barytstops, lcsize=pdslcsize, lcresofpds=1/(2*maxfrequency), lcresforlcplot=1)


barytstarts = clean_tstarts
barytstops = clean_tstops

"""
Filter events outside of the GTIs
"""
goodevt_times = np.array([])
goodevt_pis = np.array([])
goodevt_pifasts = np.array([])
goodevt_detids = np.array([])

for i in range(len(barytstarts)):
    inx = (cleantimes >= barytstarts[i]) & (cleantimes <= barytstops[i])
    goodevt_times = np.append(goodevt_times, cleantimes[inx])
    goodevt_pis = np.append(goodevt_pis, cleanpis[inx])
    goodevt_pifasts = np.append(goodevt_pifasts, cleanpifasts[inx])
    goodevt_detids = np.append(goodevt_detids, cleandetids[inx])

good_table = {}
good_table['TIME'] = goodevt_times
good_table['PI'] = goodevt_pis
good_table['PI_FAST'] = goodevt_pifasts
good_table['DET_ID'] = goodevt_detids

print("The total exposure is"+str(np.sum(barytstops-barytstarts)))

"""
+++ooo---+++ooo---+++ooo---+++ooo---+++ooo---+++ooo---+++ooo---+++ooo---
Save a FITS file that contains 3 extensions:
0. primary (with some minimal info)
1. clean events (those between barygtis and Emin and Emax)
2. GTIs
+++ooo---+++ooo---+++ooo---+++ooo---+++ooo---+++ooo---+++ooo---+++ooo---
"""
"""
Read some keywords from the first cl_bary.evt file. Copy the same keywords into the eventfile
"""
sample_evtfile = str(workingdir)+'/'+str(obsids[0])+'-work-'+str(outdir_id)+'/ni'+str(obsids[0])+'_0mpu7_cl_bary.evt'
with fits.open(sample_evtfile) as hdk:
	head1 = hdk[0].header
	head2 = hdk['EVENTS'].header

"""
0. Primary extension
"""
hdr = fits.Header()
hdr['OBJECT'] = (head1['OBJECT'], head1.comments['OBJECT'])

hdr['TELESCOP'] = (head1['TELESCOP'], head1.comments['TELESCOP'])
hdr['INSTRUME'] = (head1['INSTRUME'], head1.comments['INSTRUME'])
hdr['MJDREFI'] = (head1['MJDREFI'], head1.comments['MJDREFI'])
hdr['MJDREFF'] = (head1['MJDREFF'], head1.comments['MJDREFF'])
hdr['COMMENT'] = "Cleaned events with energy values in 0.25 and 2.5 keV (PI between 25 and 250). All times (event times, the start and the stop times of GTIs) are barycenter corrected."
empty_primary = fits.PrimaryHDU(header=hdr)

"""
1. events data
"""
col1 = fits.Column(name='TIME', format='D', unit='s', array=goodevt_times)
col2 = fits.Column(name='PI', format='I', unit='chan', array=goodevt_pis)
col3 = fits.Column(name='PI_FAST', format='I', array=goodevt_pifasts, null=-32768)
col4 = fits.Column(name='DET_ID', format='I', array=goodevt_detids)

evts_hdu_header = fits.Header()

evts_hdu = fits.BinTableHDU.from_columns([col1, col2, col3, col4], name='EVENTS', header=evts_hdu_header)

evts_hdu.header['MJDREFI'] = (head1['MJDREFI'], head1.comments['MJDREFI'])
evts_hdu.header['MJDREFF'] = (head1['MJDREFF'], head1.comments['MJDREFF'])

evts_hdu.header['TELESCOP'] = (head1['TELESCOP'], head1.comments['TELESCOP'])
evts_hdu.header['INSTRUME'] = (head1['INSTRUME'], head1.comments['INSTRUME'])
evts_hdu.header['OBJECT'] = (head1['OBJECT'], head1.comments['OBJECT'])

evts_hdu.header['TIMEREF'] = (head2['TIMEREF'], head2.comments['TIMEREF'])
evts_hdu.header['RA_OBJ'] = (head2['RA_OBJ'], head2.comments['RA_OBJ'])
evts_hdu.header['DEC_OBJ'] = (head2['DEC_OBJ'], head2.comments['DEC_OBJ'])
evts_hdu.header['EQUINOX'] = (head2['EQUINOX'], head2.comments['EQUINOX'])
evts_hdu.header['RADECSYS'] = (head2['RADECSYS'], head2.comments['RADECSYS'])
evts_hdu.header['TREFPOS'] = (head2['TREFPOS'], head2.comments['TREFPOS'])
evts_hdu.header['TREFDIR'] = (head2['TREFDIR'], head2.comments['TREFDIR'])
evts_hdu.header['PLEPHEM'] = (head2['PLEPHEM'], head2.comments['PLEPHEM'])

"""
2. GTIs
"""
c1 = fits.Column(name='START', format='D', unit='s', array=barytstarts)
c2 = fits.Column(name='STOP', format='D', unit='s', array=barytstops)


gti_hdu_header = fits.Header()

gti_hdu = fits.BinTableHDU.from_columns([c1, c2], name='GTI', header=gti_hdu_header)


gti_hdu.header['MJDREFI'] = (head1['MJDREFI'], head1.comments['MJDREFI'])
gti_hdu.header['MJDREFF'] = (head1['MJDREFF'], head1.comments['MJDREFF'])

gti_hdu.header['TELESCOP'] = (head1['TELESCOP'], head1.comments['TELESCOP'])
gti_hdu.header['INSTRUME'] = (head1['INSTRUME'], head1.comments['INSTRUME'])
gti_hdu.header['OBJECT'] = (head1['OBJECT'], head1.comments['OBJECT'])

gti_hdu.header['TIMEREF'] = (head2['TIMEREF'], head2.comments['TIMEREF'])
gti_hdu.header['RA_OBJ'] = (head2['RA_OBJ'], head2.comments['RA_OBJ'])
gti_hdu.header['DEC_OBJ'] = (head2['DEC_OBJ'], head2.comments['DEC_OBJ'])
gti_hdu.header['EQUINOX'] = (head2['EQUINOX'], head2.comments['EQUINOX'])
gti_hdu.header['RADECSYS'] = (head2['RADECSYS'], head2.comments['RADECSYS'])
gti_hdu.header['TREFPOS'] = (head2['TREFPOS'], head2.comments['TREFPOS'])
gti_hdu.header['TREFDIR'] = (head2['TREFDIR'], head2.comments['TREFDIR'])
gti_hdu.header['PLEPHEM'] = (head2['PLEPHEM'], head2.comments['PLEPHEM'])


"""
Combine all that info into a FITS file
"""
hdul = fits.HDUList([empty_primary, evts_hdu, gti_hdu])

outputfile = str(workingdir)+'/output/all_clean_events_and_gtis.fits'
subprocess.run("rm -rf "+str(outputfile), shell=True)
hdul.writeto(outputfile)
