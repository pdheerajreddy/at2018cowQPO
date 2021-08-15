# at2018cowQPO
Data and code relevant to 224 Hz soft X-ray QPO from AT2018cow
Following the steps in here will allow you to reproduce the results from Pasham et al. "Evidence for a compact object in the aftermath of the extragalacric transient AT2018cow"

All the results will be saved to a folder named "output" in "workingdir" which you define in params.py

#-----------------------------------------------------
THE FOLLOWING ARE REQUIRED BEFORE YOU CAN RUN THE PROGRAMS:
#-----------------------------------------------------

1. wget: https://www.gnu.org/software/wget/
This is necessary if you want to download data from heasarc using wget. Other ways to download data from heasarc work too.

On a Mac you can install wget as follows:
> /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

> brew install wget

2. Python 3

3. HEASoft: https://heasarc.gsfc.nasa.gov/lheasoft/install.html

4. CALBD must be installed and setup:
https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/install.html

5. Python virtual environment library

# --+++--+++--+++--+++--+++--+++--+++--+++--+++--+++--+++
	INSTALL AND CREATE A VIRTUAL ENVIRONMENT:
# --+++--+++--+++--+++--+++--+++--+++--+++--+++--+++--+++

Create a work folder:
> mkdir /home/astronomer/at2018cow
> cd /home/astronomer/at2018cow
> unzip cow_codes.zip

sudo apt-get install python3-venv (on Ubuntu)

create a virtual environment:

> python3.X -m venv ./venv (X is whatever version of Python 3 you are interested in)

>. venv/bin/activate

Download the zipped file containing all the programs: cow_codes.zip

> pip3 install -r requirements.txt

on a Mac: 
> pip3 install --upgrade -r requirements.txt (--upgrade can be omitted)

# --+++--+++--+++--+++--+++--+++--+++--+++--+++--+++--+++
     A STEP BY STEP GUIDE TO REPRODUCE THE RESULTS
# --+++--+++--+++--+++--+++--+++--+++--+++--+++--+++--+++

Run all the following from within the newly created virtual environment above

..................................................
STEP1: Download the data from the HESARC archive: https://heasarc.gsfc.nasa.gov/cgi-bin/W3Browse/w3browse.pl
..................................................

The coordinates of the source are (ra,dec)=(244.000927, 22.2680)

The bash script to download data can be found in this directory as step1_download.bash

Create a working directory, say, /home/astronomer/at2018cow.

On a terminal:

> cd /home/astronomer/at2018cow
> ./step1_download.bash

You need to create a list of observation ids as follows.

> ls -1 -d [0-9]* > obsids.list (if there is not one already)

..............................
STEP2: Reduce the NICER data.
..............................

You need to edit the parameter file (params.py) as per your system's configuration

"workingfolder" and "syspfilesdir" must be edited as they will most certainly be different of your system.
The other values can be left as default.

Then run step2_reduce.py to extract the cleaned eventlists
> cd /home/astronomer/at2018cow
> python step2_reduce.py		

..................................................
STEP3: Extract the average power density spectrum.
..................................................

> python step3_getavgpds.py 	#<--- replace X with your version of python

..............................................
STEP4: Estimate the flase alarm probabilities.
..............................................

> python step4_getfaps.py

This step can take very long time depending on the number of random realizations (simulations). 
Recommended to skip and use the values already simulated, i.e., those listed in simfapsN where N is from 1..8

> ls -1 $PWD/simfaps[0-9]/* > maxchi.pkllist 

......................................................................................................
STEP5: Plot Figure 1 and 2. The average PDS, false alaram probabilities and the NICER+XRT light curve.
......................................................................................................

> python step5_plotfig1and2.py

.................................
Plot Figure S1. Probability plot.
.................................

> python plot_FigS1_probplot.py

..................................................................
Plot Figure S2. Plots that show the various tests for white noise.
..................................................................

> python3 plot_FigS2_whitetests.py

..............................................................................
Plot Figure S3. XMM PDS to test for white noise at low frequencues (<1e-2 Hz)
..............................................................................

> python plot_FigS3_xmmpds.py

This program uses xmm.gti and xmm_source.evts. These are the good time interval and the event files, respectively. 

................................................................
Plot Figure S5. Extract and plot average PDS with one MPU removed at a time.
................................................................

> python plot_FigS5_pdsmpuremoved.py

.................................................................................
Plot Figure S6. Plot average PDS with the any plausible GPS noise events removed.
.................................................................................

> python plot_FigS6_GPSremovedpds.py

..............................................................
Plot Figure S7. Estimate and plot signal-to-noise vs exposure
..............................................................

> python plot_FigS7_snrvsexpos.py

The plot data is saved as exposure_vs_signal.dat in the output directory
The various columns in that ASCII file are: Exposure, delta chi square improvement over fitting a constants, QPO's signal to noise ratio, qpo centroid,	qpo centroid error, qpo width, qpo width error, mean count rate

...........................................................................
COMBINE ALL EVENTS AND GTIS INTO A SINGLE FITS FILE (EASY FOR READING DATA)
...........................................................................

Finally, all the BARYCENTER CORRECTED events between 0.25 and 2.5 keV and the standard GTIs are combied into a single FITS file named "all_clean_events_and_gtis.fits"

This file can be generated using the python script named "combine_all_events_and_gtis.py"

> python combine_all_events_and_gtis.py

These data can be directly used for pulsar searches

..............................................................
Plot Figure S4. Plot the average PDS of various noise sources.
..............................................................

For this plot, you need to reduce the data again withOUT trumpet filtering (step1 below):	

> python plot_FigS4_step1_pdsofnoises.py

Then extract the PDS of various noises and plot:

> python plot_FigS4_step2_pdsofnoises.py

+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.
+-.+-.+-.+-.+-.+-.+-.+-. IMPORTANT -- IMPORTANT +-.+-.+-.+-.+-.+-.+-.+-.+-.+-.
+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.
IF YOU WANT TO GO BACK TO ALL THE OTHER PLOTS MAKE SURE TO RUN STEP 2 AGAIN 
WITH trumpfilt="YES" in params.py
+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.+-.



