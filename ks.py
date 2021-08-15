import scipy.stats
import numpy as np
import time

# starttime = time.time()

# <-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->
# ---------------------------- COMPUTE THE EMPIRICAL DISTRIBUTION FUNCTION OF AN INPUT DATA ARRAY ----------------------
# <-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->

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

# <-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->
# ----------- COMPUTE THE KS-STATISTIC OF AN INPUT DATA ARRAY AGAINST A GIVEN DISTRIBUTION -----------------------------
# <-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->

def kstest(array,distribution_name,*args,scalingfactor=1):
    """
   The Kolmogorov-Smirnov test is defined as follows:

   H0: The NULL hypothesis is that the data is drawn from the given distribution
   Ha: The alternative hypothesis is that the data is NOT drawn from the given distribution

   To test this, a test statistic has been defined as:

   D-stat = max[F(Yi) - i/N, (i+1)/N - F(Yi)] where 0<=i<=N

   F is the distribution that the NULL hypothesis is tested against. The data is an array of numbers given by
   [x0,x1,x2 ..... xN-1] while the ordered data is given as [Y0, Y1 .... YN-1].

   NOTE that D-stat is often given INCORRECTLY as max[F(Yi) - i/N, i/N - F(Yi)].

   Examples of usage:
    (1). kstest(array,'chi2',2) will return the D-statistic of the KS-test by comparing the EDF of array with the CDF of
    a chi-square distribution with 2 degrees of freedom

    (2). kstest(array,'normal',2.0,3.0) will return the D-statistic of the KS-test by comparing the EDF of array with
    the CDF of a normal distribution with a mean of 2 and sigma of 3.0.

   :param array: The input array for which you are testing the NULL hypothesis that it is sampled from the distribution
   named "distribution_name" with a given set of parameters (*args).
   :param distribution_name: The name of the distribution for which you are comparing against under the NULL hypothesis
   :param args: The parameters of the distribution. Only functions listed in scipy.stats.
   :return: The value of the D-statistic.
   :param scalingfactor: This determines how you scale the PDF and CDFs.
    """

    # {STEP1}. Order the input array.

    sorted_arr = np.sort(array)

    # {STEP2}. Use the getattr method to convert the input distribution name to a method to call

    method_to_call = getattr(scipy.stats, distribution_name)

    # {STEP3}. Compute the expected Cumulative distribution function using the method_to_call/distribution function

    mod_args = tuple([i*scalingfactor for i in args])

    expec_cdf = np.array([method_to_call.cdf(sorted_arr*scalingfactor,*mod_args)])
    expec_cdf = expec_cdf[0]

    # {STEP4}. Compute the KS D-statistic which is simply the maximum of Dplus and Dminus

    N = len(array)

    dplus = np.max(np.arange(1.0, N+1)/N - expec_cdf)
    dminus = np.max(expec_cdf-np.arange(0,N)/N)
    dstat = np.max([dplus,dminus])
    return dstat

# <-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->
# -- A GENERIC FUNCTION TO ESTIMATE THE X-VALUE CORRESPONDING TO A Y-VALUE NEATEST TO A GIVEN VALUE  ---
# <-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->

def find_nearest(xvals,yvals,value):
    """
    Find the x- and y-values corresponding to the y-value that is nearest to "value"
    :param xvals: Input x-values
    :param yvals: Corresponding y-values
    :param value: the y-value you are interested in
    :return: a dictionary with exact y-value and corresponding x-value
    """
    yvals = np.array(yvals)
    xvals = np.array(xvals)

    index = np.abs(yvals - value).argmin()
    return {"Exact y-value": yvals[index],
            "x-value": xvals[index]}

# <-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->
# -- ESTIMATE THE SIGNIFICANCE OF THE D-STATISTIC. AT WHAT SIGNIFICANCE LEVEL CAN YOU NOT REJECT THE NULL HYPOTHESIS ---
# <-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->

def kstest_significances(array,distribution_name,*args,scalingfactor=1):
    """
    -------------
    SIGNIFICANCES:
    -------------
    In order to estimate the significance of rejecting or not rejecting the NULL hypothesis at a certain level we can
    use Monte Carlo simulations. The idea is simple: You derive a distribution of D-stat values for a sample of size
    same as the input array. From that distribution of D-stat values you can get the percentage significances by
    evaluating the cumulative distribution function.

    These numbers would enable you to reject or cannot reject the NULL hypothesis at X (e.g., 98%) significance level.

    :param array: same as kstest function above
    :param distribution_name: same as kstest function
    :param args: same as kstest function
    :return: the 90, 95 and 99% significance levels
    """
    N = len(array)

    # {STEP1}. Use the getattr method to convert the input distribution name to a method to call

    method_to_call = getattr(scipy.stats, distribution_name)

    # {STEP2}. Draw a random sample of N values from a distribution named "distribution_name", and extract the value of
    # D-stat between the drawn sample and the distribution. Repeat this process a large number of times (Nsims). This
    # will give you a range of D-stat values. You can derive the significance levels from the distribution of D-stat
    # values.

    mod_args = tuple([i * scalingfactor for i in args])

    dstat_sims = []    # <-- Simulated Dstatistic values
    Nsims = 10000        # <-- Number of Monte Carlo simulations

    for _ in range(Nsims):
        sample = method_to_call.rvs(*mod_args, size=N)/scalingfactor  # <-- Extract a sample from "distribution_name" of
                                                    # EXACTLY the same size as the input array
        dstat_sims.append(kstest(sample,distribution_name,*args,scalingfactor=scalingfactor))   # <-- Compute the D-stat of the drawn sample

    edf_dstat = compute_edf(dstat_sims)
    significance99 = find_nearest(edf_dstat[0], edf_dstat[1], 0.99)['x-value']
    significance95 = find_nearest(edf_dstat[0], edf_dstat[1], 0.95)['x-value']
    significance90 = find_nearest(edf_dstat[0], edf_dstat[1], 0.90)['x-value']

    return {"90%":significance90,
            "95%": significance95,
            "99%":significance99},dstat_sims

# <-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->
# -- TEST NULL HYPOTHESIS USING KSTEST ---
# <-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->

def test_kstest_null(array,distribution_name,*args,scalingfactor=1):

    dstat = kstest(array,distribution_name,*args,scalingfactor=scalingfactor)
    # print(scipy.stats.kstest(array*64,'chi2',args=(128,)))

    significances = kstest_significances(array,distribution_name,*args,scalingfactor=scalingfactor)[0]

    print("The value of the D-statistic of the KS-test is "+str(dstat))
    for k,v in significances.items():
        print("The "+str(k)+" significance value is "+str(v))

    print("Therefore .... \n")
    if any([dstat < v for k,v in significances.items()]):
        bool_index = [k for k,v in significances.items() if v > dstat]
        for k in bool_index:
            print("NULL hypothesis CANNOT be rejected at the "+str(k)+' level')
    else:
        print("The NULL hypothesis CAN be rejected at at least the 99% significance level")
    print("--------------------------------------------------------------------------------------")

# <-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->
# -- A FUNCTION TO ESTIMATE UNCERTAINTY ON THE KS TEST STATISTIC GIVEN 1-SIGMA GAUSSIAN ERROR BARS ON THE INPUT ARRAY --
# <-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->

def ksstat_uncer(array, err_array, distribution_name, *args, scalingfactor=1):

    Nsims = 10000

    dstats = []
    for i in range(Nsims):
        curr_arr = [np.random.normal(2, j) for i,j in zip(array, err_array)]
        curr_arr = [2 * i / np.mean(curr_arr) for i in curr_arr]
        dstats.append(kstest(curr_arr,distribution_name,*args,scalingfactor=scalingfactor))

    onesigma = np.std(dstats)
    edf_astat = compute_edf(dstats)
    significance95 = find_nearest(edf_astat[0], edf_astat[1], 0.95)['x-value']
    print('THe X-value at 95% is'+str(significance95))
    return onesigma


# <-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->
# -------------------------------------- COMPUTE THE ANDERSON DARLING TEST STATISTIC -----------------------------------
# <-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->
import matplotlib.pyplot as plt

def anderson_darling(array,distribution_name,*args,scalingfactor=1):

    # {STEP1}. Sort the input array.

    sorted_arr = np.sort(array)

    # {STEP2}. Use the getattr method to convert the input distribution name to a method to call

    method_to_call = getattr(scipy.stats, distribution_name)

    # {STEP3}. Compute the expected Cumulative distribution function using the method_to_call/distribution function

    mod_args = tuple([i * scalingfactor for i in args])

    expec_cdf = np.array([method_to_call.cdf(sorted_arr*scalingfactor, *mod_args)])
    expec_cdf = expec_cdf[0]

    # # {STEP4}. Compute the Anderson-Darling's statistic which is A = -N-S.

    N = len(array)
    # S = sum(((2 * j + 1) * (math.log(expec_cdf[j]) + math.log(1.0 - expec_cdf[N - 1 - j]))) / N for j in range(N))
    if all([k != 0 and k !=1 for k in expec_cdf]):
        S = np.sum([(((2 * j) + 1) * (np.log(np.abs(expec_cdf[j])) + np.log(np.abs(1.0 - expec_cdf[N - 1 - j])))) / N for j in range(N)])
    else:
        raise ValueError("Some CDF values are 0 or 1. This means log(0) --> Undefined")
    A = - N - S # <-- This is the Anderson-Darling Statistic

    return A

# <-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->
# ------------------------ COMPUTE THE SIGNIFICANCE LEVELS FOR THE ANDERSON DARLING TEST -------------------------------
# <-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->

def anderson_darling_significances(array,distribution_name,*args,scalingfactor=1):
    """
    -------------
    SIGNIFICANCES:
    -------------
    In order to estimate the significance of rejecting or not rejecting the NULL hypothesis at a certain level we can
    use Monte Carlo simulations. The idea is simple: You derive a distribution of D-stat values for a sample of size
    same as the input array. From that distribution of D-stat values you can get the percentage significances by
    evaluating the cumulative distribution function.

    These numbers would enable you to reject or cannot reject the NULL hypothesis at X (e.g., 98%) significance level.

    :param array: same as anderson_darling function
    :param distribution_name: same as anderson_darling function
    :param args: same as anderson_darling function
    :return: the 90, 95 and 99% significance levels
    """
    N = len(array)

    # {STEP1}. Use the getattr method to convert the input distribution name to a method to call

    method_to_call = getattr(scipy.stats, distribution_name)

    # {STEP2}. Draw a random sample of N values from a distribution named "distribution_name", and extract the value of
    # A-stat between the drawn sample and the distribution. Repeat this process a large number of times (Nsims). This
    # will give you a range of A-stat values. You can derive the significance levels from the distribution of A-stat
    # values.

    astat_sims = []  # <-- Simulated Dstatistic values
    Nsims = 10000  # <-- Number of Monte Carlo simulations

    mod_args = tuple([i * scalingfactor for i in args])

    for _ in range(Nsims):
        sample = method_to_call.rvs(*mod_args, size=N)/scalingfactor  # <-- Extract a sample from "distribution_name" of
        # EXACTLY the same size as the input array
        astat_sims.append(anderson_darling(sample, distribution_name, *args,scalingfactor=scalingfactor))  # <-- Compute the D-stat of the drawn sample

    edf_astat = compute_edf(astat_sims)
    significance99 = find_nearest(edf_astat[0], edf_astat[1], 0.99)['x-value']
    significance95 = find_nearest(edf_astat[0], edf_astat[1], 0.95)['x-value']
    significance90 = find_nearest(edf_astat[0], edf_astat[1], 0.90)['x-value']

    return {"90%": significance90,
            "95%": significance95,
            "99%": significance99},astat_sims

# <-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->
# --------------------------- TEST THE NULL HYPOTHESIS USING THE ANDERSON DARLING STATISTIC ----------------------------
# <-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->

def test_anderson_darling_null(array,distribution_name,*args,scalingfactor=1):

    dstat = anderson_darling(array,distribution_name,*args,scalingfactor=scalingfactor)
    significances = anderson_darling_significances(array,distribution_name,*args,scalingfactor=scalingfactor)[0]

    print("The value of the A-statistic of the Anderson Darling-test is "+str(dstat))
    for k,v in significances.items():
        print("The "+str(k)+" significance value is "+str(v))

    print("Therefore .... \n")
    if any([dstat < v for k,v in significances.items()]):
        bool_index = [k for k,v in significances.items() if v > dstat]
        for k in bool_index:
            print("NULL hypothesis CANNOT be rejected at the "+str(k)+' level')
    else:
        print("The NULL hypothesis CAN be rejected at at least the 99% significance level")
    print("--------------------------------------------------------------------------------------")

# <-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->
# -- A FUNCTION TO ESTIMATE UNCERTAINTY ON THE TEST STATISTIC GIVEN 1-SIGMA GAUSSIAN ERROR BARS ON THE INPUT ARRAY --
# <-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->

def anderstat_uncer(array, err_array, distribution_name, *args, scalingfactor=1):

    Nsims = 10000

    dstats = []
    for i in range(Nsims):
        curr_arr = [np.random.normal(2, j) for i,j in zip(array, err_array)]
        cor_curr_arr = [2.0*i/np.mean(curr_arr) for i in curr_arr]
        dstats.append(anderson_darling(cor_curr_arr,distribution_name,*args,scalingfactor=scalingfactor))
        # THe mean value has to be set to 2

    onesigma = np.std(dstats)
    edf_astat = compute_edf(dstats)
    significance95 = find_nearest(edf_astat[0], edf_astat[1], 0.95)['x-value']
    print('THe X-value at 95% is ' + str(significance95))

    return onesigma

# <-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->
# -- PLOT THE EDF AND A MODEL CDF ---
# <-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->--<-->

import matplotlib.pyplot as plt
import matplotlib

def plot_all(array,err_array, distribution_name,*args, outdir, fname, scalingfactor=1,**kwargs):

    method_to_call = getattr(scipy.stats, distribution_name)

    mod_args = tuple([i * scalingfactor for i in args])

    matplotlib.rcParams.update({'font.size': 16,'legend.fontsize': 13})
    fig,ax = plt.subplots(2,2)
    # fig.set_size_inches(6.5, 5.5)
    fig.set_size_inches(20, 12.5)

    # <+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>
    #  ------------------------- PLOT EDF + EXPECTED CDF ------------------------
    # <+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>

    # C.1.1 Compute the EDF of the observed_powers (mean of 2MW)

    sorted_arr,edf = compute_edf(array*scalingfactor)

    # C.1.2 Scale the x and y values by scalefactor

    scaled_arr = sorted_arr/scalingfactor
    scaled_edf = np.array(edf)

    # C.1.3 Plot the empirical distribution function

    ax[0, 0].plot(scaled_arr, scaled_edf, drawstyle='steps', label='Observed CDF (EDF)', fillstyle='full')
    # ax[0, 0].fill_between(scaled_arr, scaled_edf, step='pre', alpha=0.5)

    # C.2.1 Compute the model CDF, a chi-sqr with 2MW dof

    x0 = np.linspace(min(sorted_arr), kwargs['pmax']*scalingfactor,100)
    expec_cdf = np.array([method_to_call.cdf(x0,*mod_args)])[0]

    # C.2.2 Scale the CDF x and y vals

    scaled_x0 = x0/scalingfactor
    scaled_cdf = expec_cdf

    # C.2.3 PLot the CDF

    ax[0, 0].plot(scaled_x0,scaled_cdf,color='red',label='$\chi^2$ CDF')

    ax[0, 0].set_ylabel('Probability')
    ax[0, 0].set_xlabel('Noise Power Value')
    ax[0, 0].set_title('Cumulative Distribution Function of Noise powers\n (Observed vs $\chi^2$-distribution)')
    # ax[0,0].plot([kwargs['pmax'],kwargs['pmax']],[0,1],linestyle='--',label='QPO\'s highest bin value')
    ax[0, 0].legend(loc='best', frameon=False)
    ax[0, 0].text(-0.1, 1.1, '(a)', transform=ax[0, 0].transAxes,
            size=20, weight='bold')
    # <+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>
    #  --------------------------------- PLOT the PDF ---------------------------
    # <+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>

    # P.1.1 Compute a PDF of a chi-sqr distribution with 2MW dof

    x = np.linspace(method_to_call.ppf(0.0001, mod_args), method_to_call.ppf(0.999999999, mod_args), 100)
    expec_pdf = np.array([method_to_call.pdf(x, *mod_args)])[0]

    # P.1.2 Scale the x values by MW. No need to scale y as they are already normalized.

    scaled_x = x/scalingfactor

    # P.1.3 Plot the expected PDF = scaled version of chi-sqr with 2MW dof
    ax[0, 1].plot(scaled_x, expec_pdf, color='red', label='$\chi^2$ PDF')

    # P.2.1 Then compute a histogram of the observed noise powers (mean value of 2MW)

    # obs_pdf, bins = np.histogram(scaled_arr * scalingfactor, bins='auto', density=True)
    obs_pdf, bins = np.histogram(scaled_arr * scalingfactor, bins='auto', density=True)

    # P.2.2 Again, scale the bin values (x-values). No need to scale y as density=True

    scaled_bins = bins[1:]/scalingfactor

    # P.2.3 Plot the scaled observed PDF

    ax[0, 1].plot(scaled_bins, obs_pdf, drawstyle='steps', label='Observed PDF', fillstyle='full')
    ax[0,1].fill_between(scaled_bins,obs_pdf,step='pre',alpha=0.5)
    ax[0,1].set_xlim(1.98,2.02)

    # ax[0,1].plot([kwargs['pmax'],kwargs['pmax']],[0,np.max(obs_pdf)],linestyle='--',label='QPO\'s highest bin value')

    ax[0, 1].set_ylabel('Density')
    ax[0, 1].set_xlabel('Noise Power Value')
    ax[0, 1].set_title('Probability Density Function of Noise powers\n (Observed vs $\chi^2$-distribution)')
    ax[0, 1].legend(loc='best', frameon=False)
    ax[0, 1].text(-0.1, 1.1, '(b)', transform=ax[0, 1].transAxes,
                  size=20, weight='bold')
    # <+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>
    #  ---------------------- PLOT the KS-STAT DISTRIBUTION ---------------------
    # <+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>

    dstat_vals = kstest_significances(array, distribution_name, *args, scalingfactor=scalingfactor)[1]

    dstat_data = kstest(array,distribution_name,*args,scalingfactor=scalingfactor)
    # overplot the 1-sigma uncertainties

    err_dstat_data = ksstat_uncer(array, err_array, distribution_name,*args,scalingfactor=scalingfactor)

    xfills = [dstat_data - err_dstat_data, dstat_data + err_dstat_data, dstat_data + err_dstat_data, dstat_data - err_dstat_data]
    yfills = [0, 0, 1.1 * np.max(np.histogram(dstat_vals, bins='auto')[0]),
              1.1 * np.max(np.histogram(dstat_vals, bins='auto')[0])]
    # ax[1, 0].fill_between(xfills, yfills, color='red', alpha=0.5, label='$\pm$1$\sigma$ uncertainty (observed data)')

    ax[1,0].hist(dstat_vals,bins='auto',edgecolor='black',alpha=0.5)
    ax[1, 0].set_ylabel('Number of Simulations')
    ax[1, 0].set_xlabel('Kolmogorov–Smirnov Test Statistic')
    ax[1, 0].set_title('Test Statistic\'s Distribution Using Simulations')
    ax[1,0].plot([dstat_data,dstat_data],[0,1.1*np.max(np.histogram(dstat_vals,bins='auto')[0])],linestyle='--',lw=1.75,color='red',label='Observed Value')

    # Overplot 1-sigma values, cdfof 0.8413

    edf_dstat_vals = compute_edf(dstat_vals)
    onesigmaplus = find_nearest(edf_dstat_vals[0], edf_dstat_vals[1], 0.8413)['x-value']
    onesigmaminus = find_nearest(edf_dstat_vals[0], edf_dstat_vals[1], 0.1587)['x-value']
    print(find_nearest(edf_dstat_vals[0], edf_dstat_vals[1], 0.95)['x-value'])
    # ax[1, 0].plot([onesigmaplus, onesigmaplus], [0, np.max(np.histogram(dstat_vals, bins='auto')[0])], linestyle=':',color='green',lw=1.2)
    # ax[1, 0].plot([onesigmaminus, onesigmaminus], [0, np.max(np.histogram(dstat_vals, bins='auto')[0])], linestyle=':',color='green',lw=1.2)
    ax[1, 0].plot([np.median(dstat_vals), np.median(dstat_vals)], [0, 1.1*np.max(np.histogram(dstat_vals, bins='auto')[0])],color='magenta',lw=1.75,linestyle='-',label='Distribution\'s Median Value')

    xfills = [onesigmaminus,onesigmaplus,onesigmaplus,onesigmaminus]
    yfills = [0,0,1.1*np.max(np.histogram(dstat_vals, bins='auto')[0]),1.1*np.max(np.histogram(dstat_vals, bins='auto')[0])]
    ax[1,0].fill_between(xfills,yfills,color='green',alpha=0.5,label='$\pm$1$\sigma$ interval (distribution)')
    ax[1, 0].legend(loc='best', frameon=False)
    ax[1, 0].text(-0.1, 1.1, '(c)', transform=ax[1, 0].transAxes,
                  size=20, weight='bold')

    # <+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>
    #  ------- PLOT the ANDERSON DARLING STATISTIC DISTRIBUTION -----------------
    # <+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>__<+_+>

    astat_vals = anderson_darling_significances(array, distribution_name, *args, scalingfactor=scalingfactor)[1]

    ax[1,1].hist(astat_vals,bins='auto',alpha=0.5,edgecolor='black')
    astat_data = anderson_darling(array,distribution_name,*args,scalingfactor=scalingfactor)
    err_astat_data = anderstat_uncer(array, err_array, distribution_name,*args,scalingfactor=scalingfactor)

    xfills = [astat_data - err_astat_data, astat_data + err_astat_data, astat_data + err_astat_data, astat_data - err_astat_data]
    yfills = [0, 0, 1.1 * np.max(np.histogram(astat_vals, bins='auto')[0]),
              1.1 * np.max(np.histogram(astat_vals, bins='auto')[0])]
    # ax[1, 1].fill_between(xfills, yfills, color='red', alpha=0.5, label='$\pm$1$\sigma$ uncertainty (observed data)')


    ax[1,1].plot([astat_data,astat_data],[0,1.1*np.max(np.histogram(astat_vals,bins='auto')[0])],linestyle='--',color='red',lw=1.75,label='Observed Value')
    ax[1, 1].set_ylabel('Number of Simulations')
    ax[1, 1].set_xlabel('Anderson-Darling Test Statistic')
    ax[1, 1].set_title('Test Statistic\'s Distribution Using Simulations')

    # Overplot 1-sigma values, cdf of 0.8413

    edf_astat_vals = compute_edf(astat_vals)
    onesigmaplus = find_nearest(edf_astat_vals[0], edf_astat_vals[1], 0.8413)['x-value']
    onesigmaminus = find_nearest(edf_astat_vals[0], edf_astat_vals[1], 0.1587)['x-value']

    # ax[1, 1].plot([onesigmaplus, onesigmaplus], [0, np.max(np.histogram(astat_vals, bins='auto')[0])], linestyle=':',color='green',lw=1.2)
    # ax[1, 1].plot([onesigmaminus, onesigmaminus], [0, np.max(np.histogram(astat_vals, bins='auto')[0])], linestyle=':',color='green',lw=1.2)
    ax[1, 1].plot([np.median(astat_vals), np.median(astat_vals)], [0, 1.1*np.max(np.histogram(astat_vals, bins='auto')[0])],color='magenta',lw=1.75,label='Distribution\'s Median Value')

    xfills = [onesigmaminus,onesigmaplus,onesigmaplus,onesigmaminus]
    yfills = [0,0,1.1*np.max(np.histogram(astat_vals, bins='auto')[0]),1.1*np.max(np.histogram(astat_vals, bins='auto')[0])]
    ax[1,1].fill_between(xfills,yfills,color='green',alpha=0.5,label='$\pm$1$\sigma$ interval (distribution)')
    ax[1, 1].legend(loc='best', frameon=False)
    ax[1, 1].text(-0.1, 1.1, '(d)', transform=ax[1, 1].transAxes,
                  size=20, weight='bold')

    plt.subplots_adjust(wspace=0.25,hspace=0.285)
    plt.savefig(str(outdir)+'/'+str(fname))
    # plt.tight_layout(w_pad=0.1,h_pad=0.2)
    # plt.show()

    # pvals = compute_edf(dstat_vals)
    # y = np.array([1.0-j for j in pvals[1]])
    # print(1-find_nearest(pvals[0], pvals[1], dstat_data)['x-value'])
    # plt.plot(pvals[0],y)
    # plt.tight_layout()
    # plt.show()