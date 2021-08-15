import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import sys
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.ticker as mtick
import os
import qpoSearchUtils as helper
import time
from multiprocessing import Pool
import subprocess
import pickle
import multiprocessing

class faps:

	def __init__(self, cleantimes, tstarts, tstops, workingdir, pdslcsize=256, maxfrequency=1024, pdsrebinfactor=2048, Nsims=500, simsdir="simfaps"):

		self.cleantimes = cleantimes
		self.tstarts = tstarts
		self.tstops = tstops
		self.pdslcsize = pdslcsize
		self.maxfrequency = maxfrequency
		self.pdsrebinfactor = pdsrebinfactor
		self.workingdir = workingdir
		self.Nsims = Nsims

		self.simsdir = simsdir

		deltas = self.tstops - self.tstarts

		assert np.max(deltas) > self.pdslcsize, ("The maximum GTI size is "+str(np.max(deltas))+" while you want PDS of "+str(self.pdslcsize)+" ... reduce pdslcsize")

		""" Decide whether or to estimate initial faps vs max.delchi
		If fap_condition is True that means data already exists.
		"""
		if os.path.isfile(str(self.workingdir)+"/"+str(self.simsdir)+"/maxchisqrs_0.pkl"):
			if os.path.getsize(str(self.workingdir)+"/"+str(self.simsdir)+"/maxchisqrs_0.pkl") > 100:
				self.fap_condition = True
			else:
				self.fap_condition = False
				subprocess.run("rm -rf "+str(self.workingdir)+"/"+str(self.simsdir), shell=True)
				subprocess.run("mkdir "+str(self.workingdir)+"/"+str(self.simsdir), shell=True)
		else:
			self.fap_condition = False
			subprocess.run("rm -rf "+str(self.workingdir)+"/"+str(self.simsdir), shell=True)
			subprocess.run("mkdir "+str(self.workingdir)+"/"+str(self.simsdir), shell=True)

	""" --++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++
					STEP 1: COMPUTE FLASE ALARM PROBABILITY VS MAX.DELTA-CHI-SQUARE
				PURPOSE: TO COMPUTE THE MAX.DELTA-CHI-SQUARE CORRESPONDING TO SAY 0.001 FAP
	--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++
	"""

	def estimate_faps(self, input):

		ids = input[0]
		curr_nsims = input[1]

		maxvals = np.array([])
		maxnus = np.array([])

		for i in range(curr_nsims):

			_, pds_df, avg_pds, nspecs,_,_,_,_,_,_ = helper.extract_avg_pds_shuffle(self.cleantimes, self.tstarts, self.tstops, lcsize=self.pdslcsize, lcresofpds=1/(2*self.maxfrequency), lcresforlcplot=1)

			freqs = pds_df['freq']

			hot = np.array(helper.rebin(freqs, self.pdsrebinfactor))
			pot = np.array(helper.rebin(avg_pds, self.pdsrebinfactor))
			zot = pot/np.sqrt(nspecs*self.pdsrebinfactor)

			"""
			Record the maximum del-chi-square and the corresponding frequency of the best-fit
			"""
			curr_max_chisqr, curr_maxchi_nu = helper.get_max_delta_chisqr_given_pds(hot,pot,zot)

			maxvals = np.append(maxvals, curr_max_chisqr)
			maxnus = np.append(maxnus, curr_maxchi_nu)

		oup_dict = {}
		oup_dict['maxchisqrs'] = maxvals
		oup_dict['maxnus'] = maxnus

		filename = str(self.workingdir)+"/"+str(self.simsdir)+"/maxchisqrs_"+str(ids)+".pkl"
		subprocess.run("rm -rf "+str(filename), shell=True)
		pickle.dump(oup_dict, open(str(filename), "wb"))

	def run_faps_parallel(self):
		inputs = []
		result_ids = []

		ncores = multiprocessing.cpu_count() 

		each_nsims = int(self.Nsims/ncores)

		for i in range(ncores):
			inputs.append((i, each_nsims))
				
		mypool = Pool()
		mypool.map(self.estimate_faps, inputs)
		mypool.close()
		mypool.join()

	def run(self):
		if not self.fap_condition:
			self.run_faps_parallel()
		else:
			print("You already have the max-delchi-square values from simualted data in "+str(self.workingdir)+"/"+str(self.simsdir)+"/")

