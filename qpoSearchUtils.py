import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from astropy.time import Time

def constant(nu, p0):

    pf = np.zeros(len(nu))
    pf.fill(p0)

    return pf

def constant_plus_qpo(nu, p0, p1, p2, p3):
    # Now, define the Lorentzian for the QPO
    qpo1 = p1 / (1.0 + (2.0 * (nu - p2) / p3) ** 2.0)
    pf = qpo1 + p0

    return pf

def constant_plus_powlaw(nu, p0, p1, p2):
    pf = p0 + p1*nu**(-p2)

    return pf

def extract_avg_pds(cleantimes, tstart, tstop, lcsize, lcresofpds, lcresforlcplot):

    if lcresforlcplot < 1:
        response = input("You set the light curve resolution for plotting to be less than 1-second ...\n do you know what you are doing?")
        if response.lower() == 'no' or response.lower() == 'n':
            raise ValueError("You have set the light curve resolution (for plotting purposes) to be too narrow ...")
        else:
            print("Okay!")

    if lcresforlcplot > lcsize:
        lcresforlcplot = lcsize
    """
    Argsort the tstarts
    """
    inx = np.argsort(tstart)
    tstart = tstart[inx]
    tstop = tstop[inx]

    """
    Define the basic arrays
    """
    nspecs = 0
    deltas = tstop - tstart
    pdslength = int(lcsize / lcresofpds / 2)
    avg_pds = np.zeros(pdslength)

    meanrates = []
    err_meanrates = []
    meantimes = []

    clean_tstarts = np.array([])
    clean_tstops = np.array([])

    """
    Generate the frequency array
    """
    freqs = np.zeros(pdslength)
    dt = 1.0 / lcsize
    for k in range(pdslength):
        freqs[k] = (k + 1) * dt

    """
    Generate the times for the light curve (plotting purposes only)
    """
    lctimes = np.zeros(int(lcsize/lcresforlcplot))
    for k in range(int(lcsize/lcresforlcplot)):
        lctimes[k] = k*lcresforlcplot

    """
    Add the 1st columns to pds_df and lcs_df. This will also serve the purpose of creating two pandas dataframes
    """
    pds_df = pd.DataFrame(data={'freq': freqs})
    lcs_df = pd.DataFrame(data={'times': lctimes})

    """
    Main code begins here ... go into each GTI and divide it into Nsegs segments of length=lcsize
    """
    totcounts = 0
    for i in range(len(deltas)):
        Nsegs = int(deltas[i]/lcsize)
        if Nsegs > 0:

            for k in range(Nsegs):
                curr_start = tstart[i] + k*lcsize
                curr_stop = curr_start + lcsize
                nbins = int((curr_stop - curr_start) / lcresofpds)
                curr_range = (curr_start, curr_stop)

                # First, compute the light curve
                lc = np.histogram(cleantimes, bins=nbins, range=curr_range)[0]

               	condition = np.sum(lc) > 0
                if condition:
                    clean_tstarts = np.append(clean_tstarts, curr_start)
                    clean_tstops = np.append(clean_tstops, curr_stop)

                    # Subtract the mean from the counts curve
                    lcmeansub = lc - np.mean(lc)

                    # Compute the FFT of the mean-subtracted counts curve and power spectrum thereafter
                    curr_fft = np.fft.fft(lcmeansub, axis=-1)
                    curr_pds = np.real(curr_fft * np.conj(curr_fft))
                    curr_pds = curr_pds[1:int((nbins / 2)+1)]

                    # Leahy normalize the power spectrum
                    curr_pds = curr_pds * 2 / np.sum(lc)

                    pds_df['pds#' + str(nspecs)] = curr_pds

                    # Add the current pds to the average pds
                    avg_pds = avg_pds + curr_pds

                    """
                    Now, also compute the light curves for plotting purposes. Bin them with lcresforplot
                    """
                    nbins = int((curr_stop-curr_start)/ lcresforlcplot)
                    lcplot = np.histogram(cleantimes, bins=nbins, range=curr_range)[0]
                    lcplot = lcplot/lcresforlcplot
                    lcs_df['lc#' + str(nspecs)] = lcplot

                    """
                    save the mean count rates
                    """
                    inx = (cleantimes >= curr_start) & (cleantimes < curr_stop)
                    curr_meanrate = np.sum(inx)/(curr_stop-curr_start)
                    curr_err_meanrate = np.sqrt(np.sum(inx))/(curr_stop-curr_start)

                    meanrates = np.append(meanrates, curr_meanrate)
                    err_meanrates = np.append(err_meanrates, curr_err_meanrate)
                    meantimes = np.append(meantimes, np.mean([curr_start, curr_stop]))
                    totcounts += curr_meanrate*lcsize

                    """
                    Very important to update this parameter
                    """
                    nspecs += 1

    avg_pds = avg_pds/nspecs
    return lcs_df, pds_df, avg_pds, nspecs, meanrates, err_meanrates, meantimes, clean_tstarts, clean_tstops, totcounts

"""
Rebin an array
"""

def rebin(array, rebinfac):
    if len(array) % rebinfac == 0:
        Nrebin = int(len(array) / rebinfac)
        newarray = [None]*Nrebin
        for i in range(Nrebin):
            startinx = i*rebinfac
            stopinx = startinx + rebinfac

            newarray[i] = np.mean(array[startinx:stopinx])
        return np.array(newarray)
    else:
        raise TypeError("rebinfac MUST be an integer factor of array size")


"""
Define a function that return Gehrels error bars for low counts
"""

def gehrels_onesigma_errors(x): 

    if isinstance(x, (list, np.ndarray)):

        # print("You are using Gehrels 1986 error bars, input array MUST be integers!!!")
        errors = np.zeros(len(x))
        for i in range(len(x)):
            if x[i] == 0:
                errors[i] = 1.86
            elif x[i] <= 10:
                low_val = x[i]*( 1 - (1/(9*x[i])) - (1/(3*np.sqrt(x[i]))) )**3
                hi_val = x[i] + 1 + np.sqrt(x[i] + 0.75)
                errors[i] = np.mean([ hi_val - x[i], x[i] - low_val])
            else:
                errors[i] = np.sqrt(x[i])
    else:
        if x == 0:
            errors = 1.86
        elif x <= 10:
            low_val = x*(1 - (1/(9*x)) - (1/(3*np.sqrt(x))) )**3
            hi_val = x + 1 + np.sqrt(x + 0.75)
            errors = np.mean([ hi_val - x, x - low_val])
        else:
            errors = np.sqrt(x)
    return errors


""" RESCALE THE PDS GIVEN FREQUENCY LIMTIS
"""
def rebin_pds(freqs, pds, nspecs, factor, nu1min, nu1max, nu2min, nu2max):

    hot = np.array(rebin(freqs, factor))
    pot = np.array(rebin(pds, factor))
    inx = (hot>nu1min) & (hot<nu1max) | (hot>nu2min) & (hot < nu2max)
    meanval = np.mean(pot[inx])
    zot = pot/np.sqrt(nspecs*factor)
    newpot = pot*2/meanval
    newzot = zot*2/meanval

    return hot, newpot, newzot

""" MODEL THE PDS AND RETURN SOME KEY PARAMETERS
"""

def model_pds(hot,pot,zot,meanrates, err_meanrates, nu0, numin, numax):
    """Fit the PDS with a constant
    """
    popt1, pcov1 = curve_fit(constant, hot, pot, p0=[2.0], sigma=zot)
    """Fit the PDS with a constant + QPO
    """
    popt, pcov = curve_fit(constant_plus_qpo, hot, pot, p0=[2.0, 1.0, nu0, 20], sigma=zot, bounds=([0.0,0.0,numin,0.0], [np.inf, np.inf, numax, np.inf]))
    p0, p1, p2, p3 = popt
    """unpack uncertainties in fitting parameters from diagonal of covariance matrix
    """
    dp0, dp1, dp2, dp3 = [np.sqrt(pcov[j, j]) for j in range(popt.size)]
    """ Estimate the integral and its error within +- QPO width
    """
    inx = (hot >= p2 - 1 * np.max([p3,hot[1]-hot[0]])) & (hot <= p2 + 1 * np.max([p3,hot[1]-hot[0]]))

    qpopot = pot[inx] - p0
    qpozot = zot[inx]
    qpointegral = np.sum(qpopot)
    qpointegral_error = np.sqrt(np.sum(qpozot ** 2))
    
    """Fractional RMS of QPO:
    """
    # numerator = (np.pi * p1 * p3) / 2.0
    denom = np.mean(meanrates)

    rms1 = 100.0 * np.sqrt(qpointegral*(hot[1]-hot[0])/ denom)

    # rms1 = 100.0 * np.sqrt(numerator / denom)
    mean_meanrate_error = np.sqrt(np.sum(err_meanrates)) / np.size(err_meanrates)
    erms1 = rms1 * 0.5 * np.sqrt( (qpointegral_error/qpointegral)**2 + (mean_meanrate_error / np.mean(meanrates))**2)

    # erms1 = rms1 * np.sqrt(0.5 * np.pi) * 0.5 * np.sqrt((dp1 / p1) ** 2 + (dp3 / p3) ** 2 + (mean_meanrate_error / np.mean(meanrates)) ** 2)
    # erms1 = rms1 * 0.5 * np.sqrt((dp1 / p1) ** 2 + (dp3 / p3) ** 2 + (mean_meanrate_error / np.mean(meanrates)) ** 2)
    """ Calculate the reduced chi-square for constant + QPO
    """
    resids = pot - constant_plus_qpo(hot, *popt)
    newchisqr = round(((resids / zot) ** 2).sum(), 1)
    newdof = np.size(hot) - 4 - 1
    """ Calculate the reduced chi-square for constant alone
    """
    resids = pot - constant(hot, *popt1)
    oldchisqr = round(((resids / zot) ** 2).sum(), 1)
    olddof = np.size(hot) - 1 - 1
    
    """ If the sum of the qpointegral is > 0, recored it, if not, reject
    """
    if np.sum(qpointegral) > 0:
        snr = qpointegral/qpointegral_error
    else:
        snr=0
    """ return the best-fit parameter value for constant+qpo and rms+- error, chisqr for both the constant and constant+qpo, and snr
    """
    return (p0,dp0),(p1,dp1),(p2,dp2),(p3,dp3),rms1,erms1, newchisqr, newdof, oldchisqr, olddof, snr


"""
Convert nicer MET time to MJD values
"""
def niMET_to_MJD(met_times):

    t=Time(['1998-01-01T00:00:00', '2014-01-01T00:00:00'], format='fits')
    timeoffset_secs = t.unix[1] - t.unix[0]

    chandra_times = met_times + timeoffset_secs - 2 # <--- There is a two second offset between this and xTime on HEASARC so I just manually added this 2

    new_t = Time([chandra_times], format='cxcsec')
    mjdtimes = list(new_t.mjd)

    return np.array(mjdtimes[0])

""" Same as extract_avg_pds but here you randomly shuffle each time series before extract a power spectrum
"""
def extract_avg_pds_shuffle(cleantimes, tstart, tstop, lcsize, lcresofpds, lcresforlcplot):

    if lcresforlcplot < 1:
        response = input("You set the light curve resolution for plotting to be less than 1-second ...\n do you know what you are doing?")
        if response.lower() == 'no' or response.lower() == 'n':
            raise ValueError("You have set the light curve resolution (for plotting purposes) to be too narrow ...")
        else:
            print("Okay!")

    if lcresforlcplot > lcsize:
        lcresforlcplot = lcsize
    """
    Argsort the tstarts
    """
    inx = np.argsort(tstart)
    tstart = tstart[inx]
    tstop = tstop[inx]

    """
    Define the basic arrays
    """
    nspecs = 0
    deltas = tstop - tstart
    pdslength = int(lcsize / lcresofpds / 2)
    avg_pds = np.zeros(pdslength)

    meanrates = []
    err_meanrates = []
    meantimes = []

    clean_tstarts = np.array([])
    clean_tstops = np.array([])

    """
    Generate the frequency array
    """
    freqs = np.zeros(pdslength)
    dt = 1.0 / lcsize
    for k in range(pdslength):
        freqs[k] = (k + 1) * dt

    """
    Generate the times for the light curve (plotting purposes only)
    """
    lctimes = np.zeros(int(lcsize/lcresforlcplot))
    for k in range(int(lcsize/lcresforlcplot)):
        lctimes[k] = k*lcresforlcplot

    """
    Add the 1st columns to pds_df and lcs_df. This will also serve the purpose of creating two pandas dataframes
    """
    pds_df = pd.DataFrame(data={'freq': freqs})
    lcs_df = pd.DataFrame(data={'times': lctimes})

    """
    Main code begins here ... go into each GTI and divide it into Nsegs segments of length=lcsize
    """
    totcounts = 0
    for i in range(len(deltas)):
        Nsegs = int(deltas[i]/lcsize)
        if Nsegs > 0:

            for k in range(Nsegs):
                curr_start = tstart[i] + k*lcsize
                curr_stop = curr_start + lcsize
                nbins = int((curr_stop - curr_start) / lcresofpds)
                curr_range = (curr_start, curr_stop)

                # First, compute the light curve
                lc = np.histogram(cleantimes, bins=nbins, range=curr_range)[0]

                condition = np.sum(lc) > 0
                if condition:
                    np.random.seed()
                    np.random.shuffle(lc)   #<--- This is where you shuffle

                    clean_tstarts = np.append(clean_tstarts, curr_start)
                    clean_tstops = np.append(clean_tstops, curr_stop)

                    # Subtract the mean from the counts curve
                    lcmeansub = lc - np.mean(lc)

                    # Compute the FFT of the mean-subtracted counts curve and power spectrum thereafter
                    curr_fft = np.fft.fft(lcmeansub, axis=-1)
                    curr_pds = np.real(curr_fft * np.conj(curr_fft))
                    curr_pds = curr_pds[1:int((nbins / 2)+1)]

                    # Leahy normalize the power spectrum
                    curr_pds = curr_pds * 2 / np.sum(lc)

                    pds_df['pds#' + str(nspecs)] = curr_pds

                    # Add the current pds to the average pds
                    avg_pds = avg_pds + curr_pds

                    """
                    Now, also compute the light curves for plotting purposes. Bin them with lcresforplot
                    """
                    nbins = int((curr_stop-curr_start)/ lcresforlcplot)
                    lcplot = np.histogram(cleantimes, bins=nbins, range=curr_range)[0]
                    lcplot = lcplot/lcresforlcplot
                    lcs_df['lc#' + str(nspecs)] = lcplot

                    """
                    save the mean count rates
                    """
                    inx = (cleantimes >= curr_start) & (cleantimes < curr_stop)
                    curr_meanrate = np.sum(inx)/(curr_stop-curr_start)
                    curr_err_meanrate = np.sqrt(np.sum(inx))/(curr_stop-curr_start)

                    meanrates = np.append(meanrates, curr_meanrate)
                    err_meanrates = np.append(err_meanrates, curr_err_meanrate)
                    meantimes = np.append(meantimes, np.mean([curr_start, curr_stop]))
                    totcounts += curr_meanrate*lcsize

                    """
                    Very important to update this parameter
                    """
                    nspecs += 1

    avg_pds = avg_pds/nspecs
    return lcs_df, pds_df, avg_pds, nspecs, meanrates, err_meanrates, meantimes, clean_tstarts, clean_tstops, totcounts


""" 
COMPUTE THE EMPIRICAL DISTRIBUTION FUNCTION
"""

def compute_edf(array):
    """
    This function computes the empirical distribution function (EDF) of a given array of data. An EDF is nothing but a
    cumulative distribution function (CDF), but when you are referring to a sample you say EDF and when talking about a
    continuous function you say CDF.

    :param array: An array whose EDF will be computed
    :return: A tuple of two arrays: [0] = sorted_input array and [1] = edf of the input array
    """

    # {STEP1}. Sort the array

    sorted_array = np.sort(array)

    # {STEP2}. Compute the EDF
    n = sorted_array.size
    edf = np.arange(1,n+1)/n
   
    return sorted_array,edf


"""
Given a frequency range the method will compute the delta chi-square value between a constant and a Lorentzian + constant models
"""
def get_delchi_qpo_given_freq(hot, pot, zot, nu0, nu1):
    """
    hot: frequencies
    pot: pds
    zot: error on pds
    nu0: lower bound on the center of the qpo
    nu1: upper bound on the center of the qpo
    """
    popt1, pcov1 = curve_fit(constant, hot, pot, p0=[2.0], sigma=zot,maxfev=5000)

    popt, pcov = curve_fit(constant_plus_qpo, hot, pot, p0=[2.0, 1.0, (nu0+nu1)/2.0, 1], sigma=zot, bounds=([0.0,-np.inf,nu0,0.0], [np.inf, np.inf, nu1, np.inf]),maxfev=5000)

    """
    Calculate the chi-square with constant_plus_qpo model
    """
    resids = pot - constant_plus_qpo(hot, *popt)
    chisqr_with_qpo = round(((resids / zot) ** 2).sum(), 1)

    """
    Calculate the chi-square with just a constant model
    """
    resids = pot - constant(hot, *popt1)
    chisqr_with_constant = round(((resids / zot) ** 2).sum(), 1)

    """
    Delta chi-square. This value should be 0 if it finds negative normalization for the QPO! which does not make sense for any real QPO
    """
    delta_chi_sqr = - chisqr_with_qpo + chisqr_with_constant if popt[1] > 0 else 0

    return delta_chi_sqr,popt[2]

""" 
This method will compute the MAXIMUM delta chi-square of a QPO-like feature within the entire PDS
"""
def get_max_delta_chisqr_given_pds(hot, pot, zot):
    """
    Estimate the delta chi-square array for each (nu_starts, nu_stops) pair and return the maximum value
    """
    nu_starts = hot[:-1]
    nu_stops = hot[1:]

    del_chisqrs = np.array([])
    nus = np.array([])
    for i in range(len(nu_starts)):
        dchisqr, curr_nu = get_delchi_qpo_given_freq(hot, pot, zot, nu_starts[i], nu_stops[i])
        del_chisqrs = np.append(del_chisqrs, dchisqr)
        nus = np.append(nus, curr_nu)
    max_inx = np.argmax(del_chisqrs)

    max_delta_chisqr = del_chisqrs[max_inx]
    max_nu = nus[max_inx]
    
    return max_delta_chisqr, max_nu