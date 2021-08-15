#!/home/dheeraj/anaconda3/bin/python3.6
import numpy as np
import pandas as pd
import os
import subprocess
from astropy.io import fits
from multiprocessing import Pool
import multiprocessing
import time
import argparse
from argparse import RawTextHelpFormatter
from colorama import Fore, Back, Style
import csv

class nicerpipe:

	def __init__(self, workingdir, systemPfilesDir, mpulist, pirange, trumpfilt, outdir_id, nimaketime_params = {"nicersaafilt": "YES", "saafilt": "NO", "trackfilt": "YES", "st_valid": "YES", "ang_dist":0.015, "elv":30, "br_earth":40, "cor_range":"-", "underonly_range":"0-200", "overonly_range":"0-1.0", "overonly_expr":'"1.52*COR_SAX**(-0.633)"', "min_fpm":38, "ingtis":"NONE", "expr":"NONE"}, barycen=True):
		self.workingdir = workingdir
		self.systemPfilesDir = systemPfilesDir
		self.mpulist = mpulist
		self.barycen = barycen
		self.pirange = pirange
		self.trumpfilt = trumpfilt
		self.outdir_id = outdir_id

		self.nimaketime_params = nimaketime_params
		"""
		Define some standard paths that you can use later in this method
		"""
		self.pathToftools = str(os.environ['HEADAS']) + '/bin/'
		self.pathToheainit = str(os.environ['HEADAS'])

		assert isinstance(mpulist, str), ('mpulist MUST be a string; e.g., "0-6"')

		self.mapdict = {"overshoots": "EVENT_FLAGS=bxxx01x", "undershoots": "EVENT_FLAGS=bxxx001", "forcedtriggers": "EVENT_FLAGS=bxxx1xx"}

	def create_pfilesdir(self, obsid):
		self.pathToLocalpfiles = str(self.workingdir)+'/'+str(obsid)+'_pfiles/'
		subprocess.run('rm -rf ' + str(self.pathToLocalpfiles), shell=True)
		subprocess.run('mkdir ' + str(self.pathToLocalpfiles), shell=True)
		return True
	"""
	NICER pipeline consists of 5 steps:
	1) nicercal --> apply the calibration to raw nicer data
	2) niprefilter2 --> make an mkf2 file from the existing mkf file
	3) nimaketime --> create a GTI based on values of the different columns in the mkf2 file
	4) nimpumerge --> merge all the unfiltered (but calibrated) files from individual MPUs
	5) nicerclean --> filter the merged event list from the step above using the GTI file and some addtional filters
	Remember EVENT_FLAGS=bxxxx00 means remove overshoots and undershoots
	"""
	def nicercal(self, obsid, filtexpr="EVENT_FLAGS=bxxxx00", clobber="YES"):
		assert isinstance(obsid, str), ("obsid must be given as a string")
		assert clobber in ["YES", "NO"], ("Clobber can only be YES or NO!")
		assert isinstance(filtexpr, str), ("filtexpr MUST be a string starting with EVENT_FLAGS=...")

		subprocess.run('echo aws; . ' + str(self.pathToheainit) + '/headas-init.sh; unset PFILES; export PFILES="' + str(self.pathToLocalpfiles) + ';' + str(self.systemPfilesDir) + '"; ' + str(self.pathToftools) + '/nicercal indir=' + str(self.workingdir)+'/'+str(obsid) + ' outdir=' + str(self.workingdir) + '/'+str(obsid)+'-work-'+str(self.outdir_id)+' filtexpr="' + str(filtexpr) + '" mpulist='+str(self.mpulist)+' clobber='+str(clobber), shell=True)
		return True

	def niprefilter2(self, obsid):
		assert isinstance(obsid, str), ("obsid must be given as a string")
		"""
		Check to see if there is an mkf or an mkf.gz file
		"""
		if os.path.isfile(str(self.workingdir)+'/'+str(obsid)+'/auxil/ni'+str(obsid)+'.mkf'):
			infile=str(self.workingdir)+'/'+str(obsid)+'/auxil/ni'+str(obsid)+'.mkf'
		else:
			infile=str(self.workingdir)+'/'+str(obsid)+'/auxil/ni'+str(obsid)+'.mkf.gz'
		
		subprocess.run('. ' + str(self.pathToheainit) + '/headas-init.sh; unset PFILES; export PFILES="' + str(self.pathToLocalpfiles) + ';' + str(self.systemPfilesDir) + '"; ' + str(self.pathToftools) + '/niprefilter2 indir=' + str(self.workingdir)+'/'+str(obsid) + ' infile='+str(infile) + ' outfile='+str(self.workingdir) + '/'+str(obsid)+'-work-'+str(self.outdir_id)+'/ni'+str(obsid)+'.mkf2 coltypes="base,3c50" clobber=YES', shell=True)
		return True

	def nimaketime(self, obsid):
		assert isinstance(obsid, str), ("obsid must be given as a string")
		subprocess.run('. ' + str(self.pathToheainit) + '/headas-init.sh; unset PFILES; export PFILES="' + str(self.pathToLocalpfiles) + ';' + str(self.systemPfilesDir) + '"; ' + str(self.pathToftools) + '/nimaketime '+str(self.workingdir) + '/'+str(obsid)+'-work-'+str(self.outdir_id)+'/ni'+str(obsid)+'.mkf2'+' '+str(self.workingdir) + '/'+str(obsid)+'-work-'+str(self.outdir_id)+'/'+str(obsid)+'.gti nicersaafilt='+str(self.nimaketime_params["nicersaafilt"])+' saafilt='+str(self.nimaketime_params["saafilt"])+' trackfilt='+str(self.nimaketime_params["trackfilt"])+' st_valid='+str(self.nimaketime_params["st_valid"])+' ang_dist='+str(self.nimaketime_params["ang_dist"])+' elv='+str(self.nimaketime_params["elv"])+' br_earth='+str(self.nimaketime_params["br_earth"])+' cor_range='+str(self.nimaketime_params["cor_range"])+' underonly_range='+str(self.nimaketime_params["underonly_range"])+' overonly_range='+str(self.nimaketime_params["overonly_range"])+' overonly_expr='+str(self.nimaketime_params["overonly_expr"])+' min_fpm='+str(self.nimaketime_params["min_fpm"])+' ingtis='+str(self.nimaketime_params["ingtis"])+' expr='+str(self.nimaketime_params["expr"])+' clobber=YES ', shell=True)
		return True

	def nimpumerge(self, obsid):
		assert isinstance(obsid, str), ("obsid must be given as a string")
		subprocess.run("ls -1 "+str(self.workingdir)+"/"+str(obsid)+"-work-"+str(self.outdir_id)+"/ni*mpu[0-6]_ufa.evt > "+str(self.workingdir)+"/"+str(obsid)+"-work-"+str(self.outdir_id)+"/ufa.lis", shell=True)
		subprocess.run('. ' + str(self.pathToheainit) + '/headas-init.sh; unset PFILES; export PFILES="' + str(self.pathToLocalpfiles) + ';' + str(self.systemPfilesDir) + '"; ' + str(self.pathToftools) + '/nimpumerge infiles=@'+str(self.workingdir)+'/'+str(obsid)+'-work-'+str(self.outdir_id)+'/ufa.lis outfile='+str(self.workingdir)+'/'+str(obsid)+'-work-'+str(self.outdir_id)+'/ni'+str(obsid)+'_0mpu7_ufa.evt mpulist='+str(self.mpulist)+' clobber=YES', shell=True)
		return True

	"""
	EXTRACT THE SOURCE EVENTS
	"""
	def nicerclean_source(self, obsid, filtexpr='EVENT_FLAGS=bx1x000'):
		assert isinstance(obsid, str), ("obsid must be given as a string")
		subprocess.run('. ' + str(self.pathToheainit) + '/headas-init.sh; unset PFILES; export PFILES="' + str(self.pathToLocalpfiles) + ';' + str(self.systemPfilesDir) + '"; ' + str(self.pathToftools) + '/nicerclean infile='+str(self.workingdir)+'/'+str(obsid)+'-work-'+str(self.outdir_id)+'/ni'+str(obsid)+'_0mpu7_ufa.evt'+' outfile='+str(self.workingdir)+'/'+str(obsid)+'-work-'+str(self.outdir_id)+'/ni'+str(obsid)+'_0mpu7_cl.evt gtifile="NONE" trumpetfilt='+str(self.trumpfilt)+' pirange='+str(self.pirange)+' filtexpr='+str(filtexpr)+' fastconst=1.1 fastsig=1200.0 fastquart=0.0', shell=True)

		return True
	"""
	RUN BARYCENTER CORRECTION ON THE CLEAN EVENTLIST, GTI, AND THE MKF2 FILES
	"""
	def barcorr_source(self, obsid, ra, dec):
		assert isinstance(obsid, str), ("obsid must be given as a string")
		subprocess.run('. ' + str(self.pathToheainit) + '/headas-init.sh; unset PFILES; export PFILES="' + str(self.pathToLocalpfiles) + ';' + str(self.systemPfilesDir) + '"; ' + str(self.pathToftools) + '/barycorr infile='+str(self.workingdir)+'/'+str(obsid)+'-work-'+str(self.outdir_id)+'/ni'+str(obsid)+'_0mpu7_cl.evt outfile='+str(self.workingdir)+'/'+str(obsid)+'-work-'+str(self.outdir_id)+'/ni'+str(obsid)+'_0mpu7_cl_bary.evt orbitfiles='+str(self.workingdir)+'/'+str(obsid)+'/auxil/ni'+str(obsid)+'.orb.gz ra='+str(ra)+' dec='+str(dec)+ ' refframe=ICRS ephem=JPLEPH.430 clobber=YES', shell=True)
		subprocess.run('. ' + str(self.pathToheainit) + '/headas-init.sh; unset PFILES; export PFILES="' + str(self.pathToLocalpfiles) + ';' + str(self.systemPfilesDir) + '"; ' + str(self.pathToftools) + '/barycorr infile='+str(self.workingdir)+'/'+str(obsid)+'-work-'+str(self.outdir_id)+'/'+str(obsid)+'.gti outfile='+str(self.workingdir)+'/'+str(obsid)+'-work-'+str(self.outdir_id)+'/'+str(obsid)+'_bary.gti orbitfiles='+str(self.workingdir)+'/'+str(obsid)+'/auxil/ni'+str(obsid)+'.orb.gz ra='+str(ra)+' dec='+str(dec)+ ' refframe=ICRS ephem=JPLEPH.430 clobber=YES', shell=True)
		subprocess.run('. ' + str(self.pathToheainit) + '/headas-init.sh; unset PFILES; export PFILES="' + str(self.pathToLocalpfiles) + ';' + str(self.systemPfilesDir) + '"; ' + str(self.pathToftools) + '/barycorr infile='+str(self.workingdir)+'/'+str(obsid)+'-work-'+str(self.outdir_id)+'/ni'+str(obsid)+'.mkf2 outfile='+str(self.workingdir)+'/'+str(obsid)+'-work-'+str(self.outdir_id)+'/ni'+str(obsid)+'_bary.mkf2 orbitfiles='+str(self.workingdir)+'/'+str(obsid)+'/auxil/ni'+str(obsid)+'.orb.gz ra='+str(ra)+' dec='+str(dec)+ ' refframe=ICRS ephem=JPLEPH.430 clobber=YES', shell=True)
		return True

	"""
	A FUNCTION TO EXTRACT THE UNDERSHOOT, OVERSHOOT AND FORCED TRIGGER EVENT FILES
	"""
	def extract_under_over_events(self, obsid, triggerType, ra, dec):
		"""
		The input trigger type can ONLY be one of the following
		"""
		validinput = {"overshoots", "undershoots", "forcedtriggers"}
		"""
		Raise a valueerror if the input trigger type is not one of the above
		"""
		if triggerType.lower() not in validinput:
		    raise ValueError("The triggerType can only be one of the following values: " + str(validinput))

		pathToObsidWork = str(self.workingdir) + '/' + str(obsid) + '-work-'+str(self.outdir_id)+'/'

		"""
		outputfolder
		"""
		outputfolder = str(pathToObsidWork)+'/'+str(triggerType.lower())

		# ---------++++++++++++++++++------------------+++++++++++++++++++-------------------
		# STEP1: RUN nicercal USING THE RELEVANT EVENT_FLAGS, INSTEAD OF THE DEFAULT VALUE
		# ---------++++++++++++++++++------------------+++++++++++++++++++-------------------
		subprocess.run('. ' + str(self.pathToheainit) + '/headas-init.sh; unset PFILES; export PFILES="' + str(self.pathToLocalpfiles) + ';' + str(self.systemPfilesDir) + '"; ' + str(self.pathToftools) + '/nicercal indir=' + str(self.workingdir)+'/'+str(obsid) + ' outdir=' +str(outputfolder)+' filtexpr="' + str(self.mapdict[str(triggerType.lower())]) + '" mpulist='+str(self.mpulist)+' clobber=YES', shell=True)

		# ---------++++++++++++++++++------------------+++++++++++++++++++-------------------
		# STEP2: Run nimpumerge AND CREATE A COMBINED/FINAL ufa EVENTLIST (NOT: This will be named 0mpu7 even if nmpus < 7)
		# ---------++++++++++++++++++------------------+++++++++++++++++++-------------------
		subprocess.run("ls -1 "+str(outputfolder)+"/ni*mpu[0-6]_ufa.evt > "+str(outputfolder)+"/ufa.lis", shell=True)
		subprocess.run('. ' + str(self.pathToheainit) + '/headas-init.sh; unset PFILES; export PFILES="' + str(self.pathToLocalpfiles) + ';' + str(self.systemPfilesDir) + '"; ' + str(self.pathToftools) + '/nimpumerge infiles=@'+str(outputfolder)+'/ufa.lis outfile='+str(outputfolder)+'/ni'+str(obsid)+'_0mpu7_ufa.evt mpulist='+str(self.mpulist)+' clobber=YES', shell=True)

		""" RUN BARYCENTER CORRECTION
		"""
		if self.barycen:
		    subprocess.run('. ' + str(self.pathToheainit) + '/headas-init.sh; unset PFILES; export PFILES="' + str(self.pathToLocalpfiles) + ';' + str(self.systemPfilesDir) + '"; ' + str(self.pathToftools) + '/barycorr infile='+str(outputfolder)+'/ni'+str(obsid)+'_0mpu7_ufa.evt outfile='+str(outputfolder)+'/ni'+str(obsid)+'_0mpu7_ufa_bary.evt orbitfiles='+str(self.workingdir)+'/'+str(obsid)+'/auxil/ni'+str(obsid)+'.orb.gz ra='+str(ra)+' dec='+str(dec)+ ' refframe=ICRS ephem=JPLEPH.430 clobber=YES', shell=True)

		return True
	"""
	RUN THE PIPELINE FOR ONE OBSID
	"""
	def run(self, coords):
		ra = coords[0]
		dec = coords[1]
		obsid = coords[2]
		print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
		print(str(obsid))
		print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

		self.create_pfilesdir(obsid) and self.nicercal(obsid) and self.niprefilter2(obsid) and	self.nimaketime(obsid) and self.nimpumerge(obsid) and self.nicerclean_source(obsid) and self.extract_under_over_events(obsid, "undershoots", ra, dec) and self.extract_under_over_events(obsid, "overshoots", ra, dec) and self.extract_under_over_events(obsid, "forcedtriggers", ra, dec)

		if self.barycen:
			self.barcorr_source(obsid, ra, dec)
	"""
	RUN THE DATA REDUCTION PIPELINE FOR ALL THE OBSIDS ... USING MULTIPROCESSING
	"""
	def run_parallel_pipe(self, coords):
		"""
		create a pool object and run in parallel
		"""
		mypool = Pool()
		mypool.map(self.run, coords)
		mypool.close()
		mypool.join()

"""
A FUNCTION TO CHECK FOR A STRING IN AN ASCII FILE
"""

def check_for_string(filename, string_to_check):
	with open(filename, 'r') as read_obj:
		for line in read_obj:
			if string_to_check in line:
				return True
"""
A FUNCTION TO CONVERT STRING TO A BOOLEAN
"""
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
	t0 = time.time()
	parser = argparse.ArgumentParser(description=' --------------------------------------------- \n This is a good Python script to reduce NICER data: \n -------------------------------------------- \n  Here is how to run it: \n'+Fore.RED+'niDataRedBasic.py --workdir XXXXXXXXX /home/dheeraj/Work/AT2018cow_May2020/nicer --syspfilesdir \"/usr/local/heasoft/heasoft-6.27.2/x86_64-pc-linux-gnu-libc2.27/syspfiles/\" --obsidlist \"/home/dheeraj/Work/AT2018cow_May2020/nicer/list\" --ra \"244.000927\" --dec \"+22.2680\" --mpulist \"0-6\" --pirange \"0:1800\" --trumpfilt \"YES\" --outdir_id \"clean\"'+Style.RESET_ALL, formatter_class=RawTextHelpFormatter)
	parser.add_argument('--workdir', type=str, required=True, help='Absolute path to the directory where the raw data sits. Also this is where the final reduced data will be stored.')
	parser.add_argument('--syspfilesdir', type=str, required=True, help='Absolute path to the location where system pfiles are located')
	parser.add_argument('--obsidlist', type=str, required=True, help='The name of the ASCII list with all the observation ids')
	parser.add_argument('--ra', type=str, required=True, help='RA of the source')
	parser.add_argument('--dec', type=str, required=True, help='DEC of the source')
	parser.add_argument('--mpulist', type=str, required=True, help='List of MPUs (comma separated)')
	parser.add_argument('--pirange', type=str, required=True, help='PI range to extract')
	parser.add_argument('--trumpfilt', type=str, required=True, help='Filter events outside the Trumpet?')
	parser.add_argument('--outdir_id', type=str, required=True, help='A tag for the output folders')
	parser.add_argument('--passphrase', type=str, required=True, help='The passphrase to decrypt the NICER data')
	parser.add_argument('--fromscratch', type=str2bool, required=True, help='If True then the entire data is processed from scratch. If False, only the obsids without a -work folder are processed')

	"""
	Parse the input arguments
	"""
	args = parser.parse_args()

	workdir = args.workdir
	syspfilesdir = args.syspfilesdir
	obsidlist = args.obsidlist
	ra = args.ra
	dec = args.dec
	mpulist = args.mpulist
	pirange = args.pirange
	trumpfilt = args.trumpfilt
	outdir_id = args.outdir_id
	passphrase = args.passphrase
	fromscratch = args.fromscratch

	df = pd.read_csv(obsidlist, delim_whitespace=True, header=None, converters={i: str for i in range(0,1000)})
	obsids = df.iloc[:,0]
	
	""" Figure out the obsids that are encrypted and run gpg on them
	"""
	gpgfile=str(workdir)+"/listgpg.dat"
	subprocess.run("rm -rf "+str(gpgfile), shell=True)
	subprocess.run("touch "+str(gpgfile), shell=True)

	failed_obsids = np.array([]) #<--- This will contain the obsids where decryption has failed

	with open(gpgfile,'a') as f2:
		writer = csv.writer(f2, delimiter='\t', lineterminator='\n')
		for i in obsids:
			if os.path.isfile(str(workdir)+"/"+str(i)+"/xti/event_cl/ni"+str(i)+"_0mpu7_ufa.evt.gz.gpg") and not os.path.isfile(str(workdir)+"/"+str(i)+"/xti/event_cl/ni"+str(i)+"_0mpu7_ufa.evt.gz"):
			    row = [i]
			    writer.writerow(row)
			    """ Now, decrypt the data
			    """
			    subprocess.run('find '+str(i)+' -name "*.gpg" -print0 | xargs -n1 -0 gpg  --batch --yes --passphrase "'+str(passphrase)+'" --ignore-mdc-error 2> gpg.log', shell=True)
			    if check_for_string("gpg.log", "decryption failed"):
			    	failed_obsids = np.append(failed_obsids, str(i))
	
	""" UPDATE LIST TO REMOVE OBSIDS THAT FAILED GPG-DECRYPTION
	"""
	updatedlist=str(workdir)+"/obsids_clean.list"
	subprocess.run("rm -rf "+str(updatedlist), shell=True)
	subprocess.run("touch "+str(updatedlist), shell=True)

	with open(obsidlist) as oldfile, open(updatedlist, 'w') as newfile:
	    for line in oldfile:
	        if not any(bad_word in line for bad_word in failed_obsids):
	            newfile.write(line)

	""" Run the nicerpipeline now ...
	"""
	df = pd.read_csv(updatedlist, delim_whitespace=True, header=None, converters={i: str for i in range(0,1000)})
	obsids = df.iloc[:,0]

	""" Depending on fromscratch is True or False ...
	If fromscratch is True: then all obsids are reprocessed
	If fromscratch is False: only obsids without obsid-work-"outdir_id" will be processed
	"""
	coords = []
	if fromscratch:
		for i in obsids:
			coords.append((ra, dec, str(i)))
	else:
		for i in obsids:
			condition = not os.path.isdir(str(workdir)+'/'+str(i)+'-work-'+str(outdir_id)) or not os.path.isfile(str(workdir)+'/'+str(i)+'-work-'+str(outdir_id)+'/ni'+str(i)+"_0mpu7_cl.evt") or not os.path.isdir(str(workdir)+'/'+str(i)+'-work-'+str(outdir_id)+'/undershoots') #or not os.path.isdir(str(workdir)+'/'+str(i)+'-work-'+str(outdir_id)+'/overshoots')
			print(condition)
			if condition:
				coords.append((ra, dec, str(i)))

	# print(coords)
	if len(coords) > 0:
		obj = nicerpipe(workdir, syspfilesdir, mpulist, pirange, trumpfilt, outdir_id)
		obj.run_parallel_pipe(coords)
	else:
		print("Nothing to do here ... already up to date")
	print(time.time()-t0)