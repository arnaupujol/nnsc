"""
UTILS

This module defines a diversity of useful methods for implementing NNSC.

:Author: Arnau Pujol <arnaupv@gmail.com>

:Version: 1.0
"""

import numpy as np

def XWhite(X):
    """
    This method applies a data whitening through Singular Value Decomposition.

    Parameters:
    -----------
    X: np.ndarray
        Data with properties vs objects with shape (objects, properties)

    Returns:
    --------
    Xw: np.ndarray
        Data from SVD eigenvectors
    P: np.ndarray
        Covariance of properties
    """
    R = np.dot(X.T,X)
    U,S,V = np.linalg.svd(R)
    P = np.dot(np.diag(1./np.sqrt(S)),U.T)
    Xw = np.dot(X,P.T)
    return Xw,P

def select_var(varname, in_X, input_props, galsim_X, galsim_props):
    """
    This method selects input data according to some properties.

    Parameters:
    -----------
    varname: str
        Name of property 1
    in_X: ndarray
        The input data from which to select the properties
    input_props: list of strings
        List of input properties in in_X
    galsim_X: ndarray
        The galsim data from which to select the properties
    galsim_props: list of strings
        List of galsim properties in galsim_X.

    Returns:
    --------
    selected_var: np.array
        Array of selected property
    """
    if varname in input_props:
        ii = input_props.index(varname)
        selected_var = in_X[:,ii]
    elif varname in galsim_props:
        ii = galsim_props.index(varname)
        selected_var = galsim_X[:,ii]
    else:
    	print("ERROR: variable " + varname + " not found in either input_props or galsim_props")
    return selected_var

def get_sersic_frac(frac, has_disk, has_no_disk):
    """
    This method masks the objects to obtain the right fraction of
    galaxies with no disk over the total.

    Parameters:
    -----------
    frac: float in [0.,100.]
        Fraction of Sersic/total that we want to obtain
    has_disk: np.array
        Mask selecting galaxies with disk
    has_no_disk: np.array
        Mask selecting single Sersic galaxies

    Returns:
    --------
    total_mask: boolean np.ndarray
        Final mask with num(Sersic)/total = frac.
    """
    total_mask = has_disk + has_no_disk
    n_disk = np.sum(has_disk)
    n_sersic = np.sum(has_no_disk)
    n_total = np.sum(total_mask)
    original_frac = n_sersic/float(n_total)
    if frac < original_frac:
        prob_bulge = frac*n_disk/(n_sersic - frac*n_sersic)
        rands = np.random.uniform(size=np.sum(has_no_disk))
        bulge_mask_filter  = rands < prob_bulge
        total_mask[has_no_disk] = bulge_mask_filter
    else:
        prob_disk = (n_sersic/frac - n_sersic)/n_disk
        rands = np.random.uniform(size=np.sum(has_disk))
        disk_mask_filter  = rands < prob_disk
        total_mask[has_disk] = disk_mask_filter
    return total_mask

def get_varnames_to_plot(selection):
    """
    Given a data selection, this method returns the list of variables for which we compare
    the estimated bias performance.

    Parameters:
    -----------
    selection: str {'original', 'reduced', 'qbeta', 'ellip', 'sel8'}
        The selected variables are defined from this selection

    Returns:
    --------
    varnames_1d: list of str
        List of variables to analyse in 1d
    varnames, var2names: list of str, list of str
        Two lists of corresponding variables to analyse simultaneously
    vars2dowithdisk, vars2dowithoutdisk: list of ints, list of ints
        Index referring to varnames and var2names to include for galaxies with
        and without disk.
    """
    varnames_1d = ['bulge_n', 'disk_hlr', 'disk_flux', 'gal_sn', \
    'g1_intrinsic', 'g2_intrinsic', 'psf_theta', 'in_beta', 'in_q']
    if selection in ['original', 'reduced']:
        varnames_1d = varnames_1d + ['gFIT_final_e1', 'gFIT_final_e2', \
        'gFIT_out_flux', 'KSB_final_e1', 'KSB_final_e2', 'KSB_out_sn', \
        'KSB_out_beta','KSB_out_q', 'gFIT_SE_GAL_SNR', 'gFIT_SE_GAL_MAG', \
        'gFIT_SE_PSF_FWHM_IMAGE']
        varnames = ['gFIT_final_e1', 'KSB_final_e1', 'g1_intrinsic', \
    	   'gFIT_out_beta', 'KSB_out_beta', 'in_beta', \
    	   'bulge_hlr', 'bulge_flux', \
    	   'KSB_out_sn', 'bulge_hlr']
        var2names = ['gFIT_final_e2', 'KSB_final_e2', 'g2_intrinsic', \
    	    'gFIT_out_q', 'KSB_out_q', 'in_q', \
    	    'disk_hlr', 'disk_flux', \
    	    'gFIT_out_flux', 'bulge_n']
        vars2dowithdisk = [3,5,6,7]
        vars2dowithoutdisk = [3,5,8,9]
    elif selection == 'qbeta':
        varnames_1d = varnames_1d + ['gFIT_out_beta', 'gFIT_out_q', \
        'KSB_out_beta', 'KSB_out_q']
        varnames = ['g1_intrinsic', \
    	   'gFIT_out_beta', 'KSB_out_beta', 'in_beta', \
    	   'bulge_hlr', 'bulge_flux', 'bulge_hlr']
        var2names = ['g2_intrinsic', \
    	    'gFIT_out_q', 'KSB_out_q', 'in_q', \
    	    'disk_hlr', 'disk_flux', 'bulge_n']
        vars2dowithdisk = [1,3,4,5]
        vars2dowithoutdisk = [1,3,6]
    elif selection == 'ellip':
        varnames = ['gFIT_final_e1', 'KSB_final_e1', 'g1_intrinsic', \
        'gFIT_out_beta', 'KSB_out_beta', 'in_beta', 'bulge_hlr', 'bulge_flux', 'bulge_hlr']
        var2names = ['gFIT_final_e2', 'KSB_final_e2', 'g2_intrinsic', \
    	'gFIT_out_q', 'KSB_out_q', 'in_q', 'disk_hlr', 'disk_flux', 'bulge_n']
        vars2dowithdisk = [3,5,6,7]
        vars2dowithoutdisk = [3,5,8]
    elif selection == 'sel8':
        varnames_1d = varnames_1d + ['gFIT_final_e1', 'gFIT_final_e2', \
        'gFIT_out_flux', 'KSB_final_e1', 'KSB_final_e2', 'KSB_out_sn', \
        'KSB_out_beta','KSB_out_q']
        varnames = ['KSB_final_e1', 'g1_intrinsic', \
    	   'in_beta', \
    	   'bulge_hlr', 'bulge_flux', 'bulge_hlr']
        var2names = ['KSB_final_e2', 'g2_intrinsic', \
    	    'in_q', \
    	    'disk_hlr', 'disk_flux', 'bulge_n']
        vars2dowithdisk = [3,4]
        vars2dowithoutdisk = [5,]
    else:
        print("ERROR: invalid value of selection: " + params['selection'])
    vars1dwithdisk = [1, 2]
    vars1dwithoutdisk = [0]
    return varnames_1d, vars1dwithdisk, vars1dwithoutdisk, varnames, var2names, vars2dowithdisk, vars2dowithoutdisk

def get_mean_per_2dbin(var, var2, zvar, numbins = [10,10], jk_num = 20, error_mode = 'jk'):#TODO add other arguments
    """
    This method gets the mean value (and error) of a variable as a function of
    two more.

    Parameters:
    -----------
    var: np.array
        Array of variable in x-axis
    var2: np.array
        Array of variable in y-axis
    zvar: np.array
        Array which mean is calculated
    numbins: [int, int]
        List of len(2) defining the 2d number of bins
    jk_num: int
        Integer defining the number of JK subsamples used for the error
    error_mode: str {'jk', 'std'}
        If 'jk', JK error is calculated. If 'std', error obtained from
        standard deviation (default is 'jk')

    Returns:
    --------
    mean_var: np.ndarray
        2d-array defining the mean var per 2d-bin
    mean_var2: np.ndarray
        2d-array defining the mean var2 per 2d-bin
    mean_val: np.ndarray
        mean values of zvar in each 2d bin
    err_val: np.ndarray
        The corresponding error bars of mean_val
    """
    bin_edges, bin_edges2, mean_var, mean_var2 = get_2dbin_edges(var, var2, numbins=numbins)
    mean_val = np.zeros(numbins)
    err_val = np.zeros(numbins)
    for i in range(len(bin_edges) - 1):
        for j in range(len(bin_edges2[0]) - 1):
            filter_arr = (var > bin_edges[i])*(var <= bin_edges[i + 1])*(var2 > bin_edges2[i][j])*(var2 <= bin_edges2[i][j + 1])
            jk_indeces = get_jk_indeces_1d(zvar[filter_arr], jk_num)
            if error_mode == 'jk':
                sub_mean = [np.mean(zvar[filter_arr][jk_indeces != k]) for k in range(jk_num)]
                mean_val[i][j] = np.mean(sub_mean)
                err_val[i][j] = jack_knife(mean_val[i][j], sub_mean)
            elif error_mode == 'std':
                mean_val[i][j] = np.mean(zvar[filter_arr])
                err_val[i][j] = np.std(zvar[filter_arr])
            else:
                print("WRONG error_model in plot_mean_per_2dbin")
    return mean_var, mean_var2, mean_val, err_val

def get_bin_edges(array, nbins, equal_bins = True):
    """From an array, this method defines the bin edges which define
    the number of bins nbins.

    Parameters:
    -----------
    array: np.array
        Array of values
    nbins: int
        Number of bins
    equal_bins: bool
        If True, each bin has the same number of elements. Otherwise, the bin
        separation is linear (default is True)

    Returns:
    --------
    bin_edges: np.array
        Array of nbins+1 elements
    """
    if equal_bins:
        lims = np.linspace(0,len(array[(array < np.inf)]) - 1, nbins + 1)
        lims = np.array(lims, dtype = int)
        bin_edges = np.sort(array[(array < np.inf)])[lims]
        return np.unique(bin_edges)
    else:
        return np.linspace(min(array), max(array), nbins + 1)

def get_2dbin_edges(var, var2, numbins = [10,10]):
    """
    This method defines the edges of the 2d binning of two arrays of values,
    so that each 2d-bin has the same number of elements.

    Parameters:
    -----------
    var, var2: np.array, np.array
        Array of values form which we bin
    numbins: [int, int]
        List of two elements defining the number of bins in both dimensions

    Returns:
    --------
    bin_edges: np.array
        It defines the values of the edges in the 1st dimension, with
        shape = (numbins[1] + 1)
    bin_edges2: np.ndarray
        It defines the edges values of var2 for each bin in var, with
        shape = (numbins[1], numbins[2] + 1)
    mean_var: np.ndarray
        Mean value of var for each 2d-bin.
    mean_var2: np.ndarray
        Mean value of var2 for each 2d-bin
    """
    #Define edges
    lims = np.linspace(0,len(var) - 1, numbins[0] + 1)
    lims = np.array(lims, dtype = int)
    bin_edges = np.sort(var)[lims]
    lims2 = [np.linspace(0,len(var[(var >= bin_edges[i])*(var <= bin_edges[i + 1])]) - 1, numbins[1] + 1) for i in range(len(bin_edges) - 1)]
    lims2 = np.array(lims2, dtype = int)
    bin_edges2 = np.array([ np.sort(var2[(var >= bin_edges[i])*(var <= bin_edges[i + 1])])[lims2[i]] for i in range(len(bin_edges) - 1) ])
    #Define mean_var
    mean_var = np.array([np.mean(var[(var > bin_edges[i])*(var <= bin_edges[i + 1])*(var2 > bin_edges2[i][j])*(var2 <= bin_edges2[i][j + 1])]) for i in range(len(bin_edges) - 1) for j in range(len(bin_edges2[1]) - 1) ])
    mean_var2 = np.array([np.mean(var2[(var > bin_edges[i])*(var <= bin_edges[i + 1])*(var2 > bin_edges2[i][j])*(var2 <= bin_edges2[i][j + 1])]) for i in range(len(bin_edges) - 1) for j in range(len(bin_edges2[1]) - 1) ])
    mean_var = mean_var.reshape(numbins)
    mean_var2 = mean_var2.reshape(numbins)
    return bin_edges, bin_edges2, mean_var, mean_var2

def jack_knife(var, jk_var):
	"""
	This method gives the Jack-Knife error of var from the jk_var subsamples.
	Parameters:
    -----------
	var: float or np.array
        The mean value of the variable
	jk_var: np.ndarray
        The variable from the subsamples, with shape (jk subsamples, bins)

	 Returns:
     --------
	jk_err: np.array
        The JK error of var
	"""
	if type(var) == np.ndarray:
		jk_dim = jk_var.shape[0]
		err = (jk_dim - 1.)/jk_dim * (jk_var - var)**2.
		jk_err = np.sqrt(np.sum(err, axis = 0))
	else:
		jk_dim = len(jk_var)
		err = (jk_dim - 1.)/jk_dim * (jk_var - var)**2.
		jk_err = np.sqrt(np.sum(err))
	return jk_err

def get_jk_indeces_1d(array, jk_num, rand_order = True):
    """
    This method assigns equally distributed indeces to the elements of an array.

    Parameters:
    -----------
    array: np.array
        Data array
    jk_num: int
        Number of JK subsamples to identify
    rand_order: bool
        If True, the indeces are assigned in a random order (default is True)

    Returns:
    --------
    jk_indeces: np.array
        Array assigning an index (from 0 to jk_num - 1) to each data element
    """
    ratio = len(array)/jk_num + int(len(array)%jk_num > 0)
    jk_indeces = np.arange(len(array), dtype = int)/ratio
    np.random.shuffle(jk_indeces)
    return jk_indeces

def chi_square (var_1, err_1, var_2, err_2, use_err = True):
	"""
	This method calculates and returns the \Chi^2 value of two between two
    variables with their errors.

	Parameters:
    -----------
	var_1, err_1, var_2, err_2: np.arrays
        Arrays of values of the same length.
	use_err: bool
        If False, errors are ignored in the calculation (default is True)

	Returns:
    --------
	chi: float
	"""
	if var_1.shape == err_1.shape and var_1.shape == var_2.shape and var_1.shape == err_2.shape:
		if use_err:
			chi = np.mean((var_1 - var_2)**2. /(err_1**2. + err_2**2))
			return chi
		else:
			chi = np.mean((var_1 - var_2)**2.)
			return chi
	else:
		print("All the variables of Chi_square must have the same shape")

def select_xvar_yvar(varname, var2name, in_X, input_props, galsim_X, galsim_props, Itest):
    """
    This method selects input data according to some properties.

    Parameters:
    -----------
    varname: str
        Name of property 1
    var2name: str
        Name of property 2
    in_X: np.ndarray
        The input data from which to select the properties
    input_props: list of strings
        List of input properties in in_X
    galsim_X: np.ndarray
        The galsim data from which to select the properties
    galsim_props: list of strings
        List of galsim properties in galsim_X
    Itest: np.array
        Indeces of objects selected from the data

    Returns:
    --------
    selected_xvar, selected_yvar: np.array, np.array
        The two selected properties
    """
    if varname in input_props:
        ii = input_props.index(varname)
        selected_xvar = in_X[:,ii][Itest]
    elif varname in galsim_props:
        ii = galsim_props.index(varname)
        selected_xvar = galsim_X[:,ii][Itest]
    else:
    	print("ERROR: variable " + varname + " not found in either input_props or galsim_props")
    if var2name in input_props:
        jj = input_props.index(var2name)
        selected_yvar = in_X[:,jj][Itest]
    elif var2name in galsim_props:
        jj = galsim_props.index(var2name)
        selected_yvar = galsim_X[:,jj][Itest]
    else:
    	print("ERROR: variable " + var2name + " not found in either input_props or galsim_props")
    return selected_xvar, selected_yvar

def apply_disk_filter(var2name, selected_xvar, selected_yvar, Ptest, Pest):
    """
    This method applies a filter to restrict to galaxies with or without disk
    if necessary.

    Parameters:
    -----------
    var2name: str
        Name of property
    selected_xvar: np.array
        Input property in x-axis
    selected_yvar: np.array
        Input property in y-axis
    Ptest: np.ndarray
        True parameters
    Pest: np.ndarray
        Estimated parameters

    Returns:
    --------
    xvar, yvar: np.array, np.array
        Input properties with applied filter
    test_var, est_var: np.array, np.array
        True and estimated parameters with applied filter

    Note:
    -----
    If several bias components are estimated, so if Ptest and Pest have more
    than one component, only the 1st is taken.
    """
    if 'disk' in var2name or 'bulge_n' in var2name:
        if 'disk_flux' in var2name:
            filter_arr = selected_yvar != 0
        elif 'disk_hlr' in var2name:
            filter_arr = selected_yvar != 1
        elif 'bulge_n' in var2name:
            filter_arr = selected_yvar != 4
        if selected_xvar is not None:
            xvar = selected_xvar[filter_arr]
        else:
            xvar = selected_xvar
        yvar = selected_yvar[filter_arr]
        test_var = Ptest[:,0][filter_arr]
        est_var = Pest[:,0][filter_arr]
    else:
        xvar = selected_xvar
        yvar = selected_yvar
        test_var = Ptest[:,0]
        est_var = Pest[:,0]
    return xvar, yvar, test_var, est_var

def yxscales(varname, var2name):
    """
    This method defines the x and y scales of two variables.

    Parameters:
    -----------
    varname: str
        Variable name in x
    var2name: str
        Variable name in y

    Returns:
    --------
    xscale: str {'log', 'linear'}
        X scale of plot
    yscale: str {'log', 'linear'}
        Y scale of plot
    """
    if varname in ['bulge_hlr', 'bulge_flux', 'KSB_out_sn']:
        xscale = 'log'
    else:
        xscale = 'linear'
    if var2name in ['disk_hlr', 'disk_flux', 'gFIT_out_flux']:
        yscale = 'log'
    else:
        yscale = 'linear'
    return xscale, yscale

def xscale(varname):
    """
    This method defines the scale of a variable.

    Parameters:
    -----------
    varname: str
        Variable name

    Returns:
    --------
    xscale: str {'log', 'linear'}
        Scale of plot
    """
    if varname in ['bulge_hlr', 'bulge_flux', 'KSB_out_sn', 'disk_hlr', 'disk_flux', 'gFIT_out_flux']:
        xscale = 'log'
    else:
        xscale = 'linear'
    return xscale

def get_timetext(time_secs):
    """
    This method returns a string describing a time period.

    Parameters:
    -----------
    time_secs: int or float
        Number of seconds in time interval

    Returns:
    --------
    timetext: str
        String describing the time
    """
    if time_secs < 60.:
        timetext = 't='+str(round((time_secs), 2)) + 'sec'
    elif time_secs < 3600.:
        timetext = 't='+str(round((time_secs)/60., 2)) + 'min'
    else:
        timetext = 't='+str(round((time_secs)/3600., 2)) + 'h'
    return timetext
