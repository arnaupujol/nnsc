"""
PLOTTING

This module defines methods to visualize the results.

:Author: Arnau Pujol <arnaupv@gmail.com>

:Version: 1.0.0
"""


import numpy as np
import matplotlib.pyplot as plt
import nnsc.utils as ut

def costs(n_epochs, costs, output_path, save = False, show = True, verbose = True, text = '', costs_v = None, yscale = 'linear', ylim = None):
    """
    This method plots the evolution of the cost function over the epochs.

    Parameters:
    -----------
    n_epochs: int
        Number of epochs
    costs: np.array
        Array with the n_epochs different costs
    output_path: str
        Path to the output plot file
    save: bool
        If True, it saves the plots (default is False)
    show: bool
        If True, it shows the plot (default is True)
    verbose: bool
        If True, some text is shown (default is True)
    text: str
        Text annotated in the plot
    costs_v: np.array
        Array with the n_epochs different validation costs
    yscale: str {'linear','log'}
        Scale for y axis (default is 'linear')
    ylim: [float, float], optional
        It can be a 2 lenght list specifying the y axis limits

    Returns:
    --------
    Plot showing the costs
    """
    plt.plot(range(n_epochs), costs, lw = 3, alpha = .8, label = 'training set')
    if costs_v is not None:
        plt.plot(range(n_epochs), costs_v, lw = 3, alpha = .8, label = 'test set')
        plt.legend(frameon = False)
    plt.xlabel('epoch')
    plt.ylabel('cost')
    plt.yscale(yscale)
    if ylim is not None:
        plt.ylim(ylim)
    plt.annotate(text, (.3,.8), xycoords = 'figure fraction')
    if save:
            plt.savefig(output_path)
            print("costs plot saved in " + output_path)
    if show:
            plt.show()
    else:
            plt.close()


def plot_mean_per_2dbin(var, varname, var2, var2name, zvar, zvarname, numbins = [10,10], jk_num = 20, error_mode = 'jk', show = True, save = False, out_name = '/tmp/plot.pdf', xscale = 'linear', yscale = 'linear', xlim = None, ylim = None, cmap = 'viridis', vmin = None, vmax = None):
    """
    This method plots the mean value (and error) of a variable as a function
    of two more.

    Parameters:
    -----------
    var: np.array
        Array of variable in x-axis
    var_name: str
        Name of x-axis
    var2: np.array
        Array of variable in y-axis
    var2name: str
        Name of y-axis
    zvar: np.array
        Array which mean is calculated
    zvarname: str
        Name of variable zvar, shown in plot title
    numbins: [int, int]
        List of len(2) defining the 2d number of bins
    jk_num: int
        Integer defining the number of JK subsamples used for the error
    error_mode: str {'jk', 'std'}
        If 'jk', JK error is calculated. If 'std', error obtained from standard
        deviation (default is 'jk')
    show: bool
        If True, it shows the plot
    save: bool
        If True, it saves the plots
    out_name: str
        Name of the output plot file
    xscale: str {'linear', 'log'}
        Scale of x-axis (default is 'linear')
    yscale: str {'linear', 'log'}
        Scale of y-axis (default is 'linear')
    xlim: [float, float], optional
        Range of values in x-axis
    ylim: [float, float], optional
        Range of values in y-axis
    cmap: str
        Colour map used for the scatter plots
    vmin, vmax: float, float
        Minimum and maximum values for the colour map

    Returns:
    --------
    Plots showing:
        mean_var: 2d-array defining the mean var per 2d-bin.
        mean_var2: 2d-array defining the mean var2 per 2d-bin.
        mean_val: mean values of zvar in each 2d bin.
        err_val: the corresponding error bars of mean_val.
    """
    mean_var, mean_var2, mean_val, err_val = ut.get_mean_per_2dbin(var, var2, zvar, numbins = numbins, jk_num = jk_num, error_mode = error_mode)
    #plt.figure(0)

    if 'c1' in zvarname or 'c2' in zvarname:
        if vmin is None:
            vmin = -.05
        if vmax is None:
            vmax = .05
    else:
        if vmin is None:
            vmin = -.2
        if vmax is None:
            vmax = .2
    plt.scatter(mean_var, mean_var2, c = mean_val, s = 70*np.min(err_val)/err_val, lw = .3, vmin = vmin, vmax = vmax, cmap = cmap)
    plt.xlabel(varname)
    plt.xscale(xscale)
    plt.ylabel(var2name)
    plt.yscale(yscale)
    plt.title(zvarname)
    plt.colorbar()
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if save:
        plt.savefig(out_name)
        print("Plot saved in " + out_name)
    if show:
        plt.show()
    else:
        plt.close()

def plot_chi2_per_2dbin(var, varname, var2, var2name, zvar, zvar2, titlename, numbins = [10,10], jk_num = 20, error_mode = 'jk', show = True, save = False, out_name = '/tmp/plot', xscale = 'linear', yscale = 'linear', use_err = True, make_plots = True, vmax = None):
    """
    This method plots the error on the estimation vs true values as a function
    of two binned properties.

    Properties:
    -----------
    var: np.array
        Array of variable in x-axis
    var_name: str
        Name of variable in x-axis
    var2: np.array
        Array of variable in y-axis
    var2name: str
        Name of variable in y-axis
    zvar: np.array
        Array which mean is calculated
    zvar2: np.array
        Array which mean is calculated and compared with zvar
    titlename: str
        Name shown in plot title
    numbins: [int, int]
        List of len(2) defining the 2d number of bins
    jk_num: int
        Integer defining the number of JK subsamples used for the error
    error_mode: str {'jk', 'std'}
        If 'jk', JK error is calculated. If 'std', error obtained from standard
        deviation (default is 'jk')
    show: bool
        If True, it shows the plot
    save: bool
        If True, it saves the plots
    out_name: str
        Name of the output plot file
    xscale: str {'linear', 'log'}
        Scale of x-axis (default is 'linear')
    yscale: str {'linear', 'log'}
        Scale of y-axis (default is 'linear')
    use_err: bool
        If True, it uses the error bars to calculate of chi^2 (default is True)
    make_plots: bool
        It specifies if the plots are made (default is True)
    vmax: float
        It specifies the maximum value for the colormap

    Returns:
    --------
    Plots showing:
        mean_var: 2d-array defining the mean var per 2d-bin
        mean_var2: 2d-array defining the mean var2 per 2d-bin
        chi2: chi^2 between zvar and zvar2
        If save, plots are saved in out_name as pdf
        If show, the plots are also shown
    """
    mean_var, mean_var2, mean_val, err_val = ut.get_mean_per_2dbin(var, var2, zvar, numbins = numbins, jk_num = jk_num, error_mode = error_mode)
    mean_var, mean_var2, mean_val2, err_val2 = ut.get_mean_per_2dbin(var, var2, zvar2, numbins = numbins, jk_num = jk_num, error_mode = error_mode)
    chi2 = ut.chi_square(mean_val, err_val, mean_val2, err_val2, use_err)
    print("Chi^2 = " + str(chi2))
    if make_plots:
        if use_err:
            plt.scatter(mean_var, mean_var2, c = (mean_val - mean_val2)**2./(err_val**2. + err_val2**2.), s = 35, lw = .3, vmax = 5)
            if print_err:
                plt.annotate(r'$\chi^2/\nu = $' + str(round(chi2,5)), (.23, .85), xycoords = 'figure fraction')
            plt.title(r'$\chi^2$ of true vs estimated ' + titlename)
        else:
            if vmax is None:
                if 'c1' in titlename or 'c2' in titlename:
                    vmax =.05
                else:
                    vmax = .2
            plt.scatter(mean_var, mean_var2, c = np.abs(mean_val - mean_val2), s = 35, lw = .3, vmax = vmax)
            plt.title(r'$\sigma$ of true vs estimated ' + titlename)
        plt.xlabel(varname)
        plt.xscale(xscale)
        plt.ylabel(var2name)
        plt.yscale(yscale)
        plt.colorbar()
        if save:
            if use_err:
                plt.savefig(out_name + ".pdf")
                print("Plot saved in " + out_name + '.pdf')
            else:
                plt.savefig(out_name + "_noerr.pdf")
                print("Plot saved in " + out_name + '_noerr.pdf')
        if show:
            plt.show()
        else:
            plt.close()

def make_all_plots(output_path, fname, xvar, yvar, test_var, est_var, varname, var2name, dowithdisk, dowithoutdisk, xscale, yscale, numbins, params):
    """
    This method makes and/or saves all the 2d test plots.

    Parameters:
    -----------
    output_path: str
        Output path to save files
    fname: str
        Model name
    xvar: np.array
        Variable for x-axis
    yvar: np.array
        Variable for y-axis
    test_var: np.array
        Test variable to compare
    est_var: np.array
        Estimated variable to compare
    varname: str
        Names of variable in x-axis
    var2name: str
        Names of variable in y-axis
    dowithdisk: bool
        It specifies whether it also makes plots for only galaxies with disk
    dowithoutdisk: bool
        It specifies whether it also makes plots for only galaxies without disk
    xscale: str {'linear', 'log'}
        Scale of x-axis (default is 'linear')
    yscale: str {'linear', 'log'}
        Scale of y-axis (default is 'linear')
    numbins: [int, int]
        List of len(2) defining the 2d number of bins
    params: dict
        Dictionary specifying model and execution parameters

    Returns:
    --------
    out_name_test, out_name_est, out_name_chi2: pdf files
        If params['save'] and params['make_plots'], plots are saved in the files
        If params['make_plots'] and params['show'], the plots are also shown
    """

    out_name_test = output_path + fname + '_' + params['output_data'] +'_vs_' +  varname + '_' + var2name + '_test.pdf'
    out_name_est = output_path + fname + '_' + params['output_data'] +'_vs_' +  varname + '_' + var2name + '_est.pdf'
    out_name_chi2 = output_path + fname + '_' + params['output_data'] +'_vs_' +  varname + '_' + var2name + '_chi2'
    if params['make_plots']:
        plot_mean_per_2dbin(xvar, varname, yvar, var2name, test_var, "true " + params['output_data'], out_name = out_name_test, \
    	    save = params['save'], xscale = xscale, yscale = yscale, numbins= numbins, show = params['show'])
        plot_mean_per_2dbin(xvar, varname, yvar, var2name, est_var, "estimated " + params['output_data'], out_name = out_name_est, \
    	    save = params['save'], xscale = xscale, yscale = yscale, numbins= numbins, show = params['show'])
        plot_chi2_per_2dbin(xvar, varname, yvar, var2name, test_var, est_var, params['output_data'], out_name = out_name_chi2, \
    	    save = params['save'], xscale = xscale, yscale = yscale, numbins= numbins, show = params['show'],\
    	    use_err = params['use_err'], make_plots = params['make_plots'])
    if dowithdisk:
        filter_arr = galsim_X[:,galsim_props.index('disk_flux')][Itest] != 0
        xvar = selected_xvar[filter_arr]
        yvar = selected_yvar[filter_arr]
        test_var = Ptest[:,0][filter_arr]
        est_var = Pest[:,0][filter_arr]
        out_name_test = output_path + fname + '_' + params['output_data'] +'_vs_' +  varname + '_' + var2name + '_withdisk_test.pdf'
        out_name_est = output_path + fname + '_' + params['output_data'] +'_vs_' +  varname + '_' + var2name + '_withdisk_est.pdf'
        out_name_chi2 = output_path + fname + '_' + params['output_data'] +'_vs_' +  varname + '_' + var2name + '_withdisk_chi2'
        if params['make_plots']:
            plot_mean_per_2dbin(xvar, varname, yvar, var2name, test_var, "true " + params['output_data'] + " with disk", out_name = out_name_test, \
    		save = params['save'], xscale = xscale, yscale = yscale, numbins= numbins, show = params['show'])
            plot_mean_per_2dbin(xvar, varname, yvar, var2name, est_var, "estimated " + params['output_data'] + " with disk", out_name = out_name_est, \
    		save = params['save'], xscale = xscale, yscale = yscale, numbins= numbins, show = params['show'])
            plot_chi2_per_2dbin(xvar, varname, yvar, var2name, test_var, est_var, params['output_data'] + " with disk", out_name = out_name_chi2, \
    		save = params['save'], xscale = xscale, yscale = yscale, numbins= numbins, show = params['show'], \
    		use_err = params['use_err'], make_plots = params['make_plots'])
    if dowithoutdisk:
        filter_arr = galsim_X[:,galsim_props.index('disk_flux')][Itest] == 0
        xvar = selected_xvar[filter_arr]
        yvar = selected_yvar[filter_arr]
        test_var = Ptest[:,0][filter_arr]
        est_var = Pest[:,0][filter_arr]
        out_name_test = output_path + fname + '_' + params['output_data'] +'_vs_' +  varname + '_' + var2name + '_withoutdisk_test.pdf'
        out_name_est = output_path + fname + '_' + params['output_data'] +'_vs_' +  varname + '_' + var2name + '_withoutdisk_est.pdf'
        out_name_chi2 = output_path + fname + '_' + params['output_data'] +'_vs_' +  varname + '_' + var2name + '_withoutdisk_chi2'
        if params['make_plots']:
            plot_mean_per_2dbin(xvar, varname, yvar, var2name, test_var, "true " + params['output_data'] + " no disk", out_name = out_name_test, \
    		save = params['save'], xscale = xscale, yscale = yscale, numbins= numbins, show = params['show'])
            plot_mean_per_2dbin(xvar, varname, yvar, var2name, est_var, "estimated " + params['output_data'] + " no disk", out_name = out_name_est, \
    		save = params['save'], xscale = xscale, yscale = yscale, numbins= numbins, show = params['show'])
            plot_chi2_per_2dbin(xvar, varname, yvar, var2name, test_var, est_var, params['output_data'] + " no disk", out_name = out_name_chi2, \
    		save = params['save'], xscale = xscale, yscale = yscale, numbins= numbins, show = params['show'], \
    		use_err = params['use_err'], make_plots = params['make_plots'])

def plot_mean_per_bin(xvar, xname, yvar, yname, nbins, filter_arr = None, jk_num = 50, c  = 'k', show = True, marker = 's', ylims = None, leg_loc = 'upper center', equal_bins = True, linestyle = '-', lw = 3, leg_ncol = 1, error_mode = 'jk', save = False, out_name = '/tmp/plot.pdf', get_out = False, rand_xshift = False, median = False, bin_edges = None, close = True):
    """
    This method plots the mean value of a variable as a function of another one.

    Parameters:
    -----------
    xvar: np.array
        Variable that is binned in the x-axis
    xname: str
        Name of the variable that appears in x label
    yvar: np.array
        Variable which mean is calculated in each x bin
    yname: str
        Name of the variable, shown in the legend
    nbins: int
        It specifies the number of x bins used
    filter_array: bool np.array
        Boolean array which defines a selection of the objects in xvar and yvar
    jk_num: int
        Number of Jack-Knife subsamples used to calculate the error bars
    c: str or list representing colour [R,G,B]
        Colour of the error bars and points used
    show: bool
        If True, it shows the plot (default is True)
    marker: str
        Marker of plotted points
    ylims: None or [float,float]
        It defines the y-axes limits
    leg_loc: str or int
        Position of legend
    equal_bins: bool
        If True, each bin has the same number of elements. Otherwise, bins are
        defined linearly according to the min and max values (default is True)
    linestyle: str
        Linestyle of the plot
    lw: float
        Line width of the plot
    leg_ncol: int
        Number of columns in the legend
    error_mode: str {'jk', 'std'}
        If 'jk', JK error is calculated. If 'std', error obtained from standard
        deviation (default is 'jk')
    save: bool
        If True, it saves the plots (default is False)
    out_name: str
        Name of the output plot file
    get_out: bool
        If True, it returns the mean xvar, mean yvar and its error
        (default is False)
    rand_xshift: Bool
        If True, a small random shift in the x-axis is applied on the plot to
        improve visualization (default is False)
    median: bool
        If True, the median instead of the mean is calculated for each bin
        (default is False)
    bin_edges: np.array or list of floats or None
        If it is a list or array of values, it is used as the bin edges instead
        of obtaining them from nbins (default is None)
    close: bool
        If True, the plot is closed if show is False (default is True)

    Returns:
    --------
    An errorbar plot showing:
        x_plot: mean x for each bin
        y_plot, err_plot: mean y for each bin and its error
    If save, the plot is saved in out_name as pdf
    If show, the plot is shown
    """
    if filter_arr is None:
        filter_arr = np.ones_like(xvar, dtype=bool)
    xvar_f = xvar[filter_arr]
    yvar_f = yvar[filter_arr]
    if bin_edges is None:
        bin_edges = ut.get_bin_edges(xvar_f, nbins, equal_bins = equal_bins)
    nbins = len(bin_edges) - 1
    random_amplitude = (bin_edges[1] - bin_edges[0])/20.
    x_plot = []
    y_plot = []
    err_plot = []
    for i in range(nbins):
        filter_bin = (xvar_f >= bin_edges[i])*(xvar_f < bin_edges[i+1])
        jk_indeces = ut.get_jk_indeces_1d(xvar_f[filter_bin], jk_num)
        np.random.shuffle(jk_indeces)
        if error_mode == 'jk':
            if median:
                sub_mean = [np.median(yvar_f[filter_bin][jk_indeces != j]) for j in range(jk_num)]
            else:
                sub_mean = [np.mean(yvar_f[filter_bin][jk_indeces != j]) for j in range(jk_num)]
            mean = np.mean(sub_mean)
            err = ut.jack_knife(mean, sub_mean)
        elif error_mode == 'std':
            if median:
                mean = np.median(yvar_f[filter_bin])
            else:
                mean = np.mean(yvar_f[filter_bin])
            err = np.std(yvar_f[filter_bin])
        else:
            print("WRONG error_model in plot_mean_per_bin")
        if rand_xshift:
            rand =  np.random.random()*random_amplitude
        else:
            rand = 0.
        if median:
            x_plot.append(np.median(xvar_f[filter_bin]) + rand)
        else:
            x_plot.append(np.mean(xvar_f[filter_bin]) + rand)
        y_plot.append(mean)
        err_plot.append(err)
    plt.errorbar(x_plot, y_plot, err_plot, c = c, marker = marker, markersize = 5, label = yname, linestyle = linestyle, lw = lw)
    if ylims is not None:
        plt.ylim(ylims[0], ylims[1])
    plt.xlabel(xname)
    if save:
        plt.savefig(out_name)
        print("Plot saved in " + out_name)
    if show:
        plt.legend(loc = leg_loc, frameon = False, fontsize = 10, ncol = leg_ncol)
        plt.show()
    elif close:
        plt.close()
    if get_out:
        return x_plot, y_plot, err_plot
