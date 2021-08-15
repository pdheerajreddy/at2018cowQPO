""" DEFINE ALL YOU PARAMETERS HERE
"""
syspfilesdir="/Users/dheeraj/software/heasoft-6.28/x86_64-apple-darwin20.3.0/syspfiles/" #<--- PATH TO THE SYSTEM-LEVEL PFILES (NICER DATA REDUCTION RELATED)
mpulist="0-6"	#<--- MPU LIST TO BE REDUCED
pirange="0:1800"	#<--- PI RANGE
trumpfilt="NO"		#<--- TRUMPET FILTER THE DATA?
outdir_id="clean"	#<--- TAG FOR THE OUTPUT DIRECTORIES

workingdir="/Users/dheeraj/work/2021/at2018cow/data"	#<--- LOCATION OF THE RAW DATA DOWNLOADED FROM HEASARC USING STEP1
obsidlist = "obsids.list"							#<--- AN ASCII LIST OF OBSERVATION IDS, ONE PER ROW

""" COORDINATES OF THE SOURCE
"""
ra=244.000927
dec=22.2680

bary = True #<--- Barycenter correct the data
emin = 25	#<--- Minimum energy channel
emax = 250	#<--- Maximum energy channel

saveformat="png" #<--- output format for saving plots
nameid="source"  #<--- name of the file to be saved

""" PDS EXTRACTIONS PARAMETERS
"""
pdslcsize=256		#<--- Length of the individual light curve segments
maxfrequency=1024	#<--- Maximum frequency
pdsrebinfactor=2048	#<--- Rebinning factor of the final PDS

fappklist="maxchi.pkllist"	#<--- list of all the pickle files containing the delta-chi-square values

xrtlc="lightcurve_30_250.dat"
mjdstart = 58285.44

det_dict ={"0":[0,1,2,3,4,5,6,7], "1":[10,11,12,13,14,15,16, 17], "2":[20,21,22,23,24,25,26,27],"3":[30,31,32,33,34,35,36,37], "4":[40,41,42,43,44,45,46,47], "5":[50,51,52,53,54,55,56,57],"6":[60,61,62,63,64,65,66,67]}

