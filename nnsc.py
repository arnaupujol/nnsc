"""
NNSC

This is the main code to run NNSC from the input config specifications using
the method presented in Pujol et al. (2020).

:Author: Arnau Pujol <arnaupv@gmail.com>

:Version: 1.0
"""

import os
import sys
import numpy as np
import scipy.io as sio
import nnsc.deep_reg as deep_reg
import nnsc.utils as ut
from config import paths, names
from nnsc.import_data import get_extra_filter, get_input_data, get_galsim_data, get_m
from nnsc import plotting
from nnsc import draw
from config import param
from time import time

time0 = time()
draw.walle("running bias_ldae.py")

#Defining parameters
params = param.params
param.update(params, sys.argv)

#Defining paths
output_path = paths.output_data
datapath = paths.input_data

#Define whether a filter on the galaxy components is applied
extra_filter = get_extra_filter(params['extra_filter'], path = datapath)

#Loading input parameters
X, input_props = get_input_data(extra_filter = extra_filter, selection = params['selection'], path = datapath) #TODO test
galsim_X, galsim_props = get_galsim_data(extra_filter = extra_filter, path = datapath)

#Loading output parameters for supervised learning
if params['output_data'] == 'm':
	Theta = get_m(extra_filter = extra_filter, path = datapath)
elif params['output_data'] == 'm1':
	Theta = np.array([get_m(extra_filter = extra_filter, path = datapath)[0]])
elif params['output_data'] == 'm2':
	Theta = np.array([get_m(extra_filter = extra_filter, path = datapath)[1]])
elif params['output_data'] == 'c1':
	Theta = np.array([get_m(extra_filter = extra_filter, path = datapath)[2]])
elif params['output_data'] == 'c2':
	Theta = np.array([get_m(extra_filter = extra_filter, path = datapath)[3]])

#This is to preserve the input values after applying the whitening on  X
in_X = np.copy(X)

#Parameter definitions
if params['top_fs'] > 0: #it reduces the input data to the top properties after whitening
    dim = [params['top_fs'],params['dim1'],params['dim2']]
else:
    dim = [X.shape[1],params['dim1'],params['dim2']]
ct = [params['ct'], params['ct'], params['ct']]
if params['dim3'] > 0:
	dim.append(params['dim3'])
	ct.append(params['ct'])
if params['dim4'] > 0:
	dim.append(params['dim4'])
	ct.append(params['ct'])
if params['dim5'] > 0:
	dim.append(params['dim5'])
	ct.append(params['ct'])
dim.append(Theta.shape[0])
ct.append(params['ct'])

#Define  model name
fname=names.fname(params)

#Apply whitening on data
X,Pwhite = ut.XWhite(X)

X = X/np.maximum(1,np.sqrt(np.sum(X*X,axis=0)))
X = X - np.min(X, axis=0)
X = X/np.max(X, axis=0)

if params['top_fs'] > 0: #Selecting the top properties after whitening if it applies
    X = X[:,:params['top_fs']]

if "sfrac" in params['extra_filter']:# It defines the fraction of single Sersic galaxies for the catalogue selection
	disk_flux = ut.select_var('disk_flux', in_X, input_props, galsim_X, galsim_props)
	has_disk = disk_flux > 0.
	has_no_disk = disk_flux == 0.
	sfrac_mask = ut.get_sersic_frac(float(params['extra_filter'][5:])/100., has_disk, has_no_disk)
	K = np.random.permutation(np.arange(len(X))[sfrac_mask])
else:
    K = np.random.permutation(len(X))

Itrain = K[0:params['n_train']]
if params['save']:
	print("saving " + output_path + 'Itrain_' + fname + '.npy')
	np.save(output_path + 'Itrain_' + fname + '.npy', Itrain)
Itest = K[params['n_train']:params['n_train']+params['n_test']]

Xtrain = X[Itrain,:]
Ttrain = Theta[:,Itrain]

Xtest = X[Itest,:]
Ttest = Theta[:,Itest]

#Run training
print("Learn the network")

activation = 'leaky_relu'
if params['version'] == 'deep_reg':#Regression with Leaky ReLU as activation functions
	aep = deep_reg.main_reg(Xtrain, Ttrain.T, dim = dim, ct = ct, n_epochs = params['n_epochs'], batch_size = params['batch_size'], starter_learning_rate = params['learning_rate'], fname = output_path + fname, decay_after = params['lr_decay'], Ytest = Xtest, Ttest = Ttest.T)
elif params['version'] == 'deep_reg_est':#Same as deep_reg but Ttrain, Ttest values centred to 1
	aep = deep_reg.main_reg(Xtrain, Ttrain.T + 1, dim = dim, ct = ct, n_epochs = params['n_epochs'], batch_size = params['batch_size'], starter_learning_rate = params['learning_rate'], fname = output_path + fname, decay_after = params['lr_decay'], Ytest = Xtest, Ttest = Ttest.T + 1)
elif params['version'] == 'deep_regl':#Same as deep_reg_est but linear function on last hidden layer activation functions.
    activation = 'leaky_relu_l'
    aep = deep_reg.main_reg(Xtrain, Ttrain.T + 1, dim = dim, ct = ct, n_epochs = params['n_epochs'], batch_size = params['batch_size'], starter_learning_rate = params['learning_rate'], fname = output_path + fname, decay_after = params['lr_decay'], activation = activation, Ytest = Xtest, Ttest = Ttest.T + 1)
elif params['version'] == 'deep_regh':#Same as deep_reg but tanh as activation functions
    activation = 'tanh'
    aep = deep_reg.main_reg(Xtrain, Ttrain.T, dim = dim, ct = ct, n_epochs = params['n_epochs'], batch_size = params['batch_size'], starter_learning_rate = params['learning_rate'], fname = output_path + fname, decay_after = params['lr_decay'], activation = activation, Ytest = Xtest, Ttest = Ttest.T)
elif params['version'] == 'deep_regrh':#Same as deep_reg but tanh on last hidden layer activation functions
    activation = 'relu_tanh'
    aep = deep_reg.main_reg(Xtrain, Ttrain.T, dim = dim, ct = ct, n_epochs = params['n_epochs'], batch_size = params['batch_size'], starter_learning_rate = params['learning_rate'], fname = output_path + fname, decay_after = params['lr_decay'], activation = activation, Ytest = Xtest, Ttest = Ttest.T)
else:
	print("ERROR: wrong assignment of params[version]: " + params['version'])

#Generate estimated bias for test set
print("Validate the estimation procedure")
R0 = deep_reg.ParamEstModel(fname=output_path + fname,Xtrain=Xtrain,Xtest=Xtest,Theta_test=Ttest.T, version = params['version'], activation = activation)
Pest = R0["Pest"]
Ptest = Ttest.T
E = Pest - Ptest

time1 = time()
draw.walle("bias_ldae.py completed in " + str(round(time1 - time0, 2)) + " seconds")

#Plotting costs
costs = sio.loadmat(output_path + fname)['costs'][0]
if params['make_plots']:
    timetext = ut.get_timetext(time1 - time0)
    costs_v = sio.loadmat(output_path + fname)['costs_v'][0]
    plotting.costs(params['n_epochs'], costs, output_path + 'costs_' + fname + '.pdf', save = params['save'], show = params['show'], text = timetext, costs_v = costs_v, yscale = 'log')

#PLOTTING bias vs properties
varnames_1d, vars1dwithdisk, vars1dwithoutdisk, varnames, var2names, vars2dowithdisk, vars2dowithoutdisk = ut.get_varnames_to_plot(params['selection'])

if not params['save']:
    os.remove(output_path + fname + '.mat')

#Plotting 1d dependencies
for i in range(len(varnames_1d)):
    numbins = 8
    varname = varnames_1d[i]
    #defining yvar
    selected_yvar = ut.select_var(varname, in_X, input_props, galsim_X, galsim_props)[Itest]
    #defining Ptest and Pest and filters
    xvar, yvar, test_var, est_var = ut.apply_disk_filter(varname, None, selected_yvar, Ptest, Pest)
    #Define output name
    out_name = output_path + fname + '_' + params['output_data'] +'_vs_' +  varname + '.pdf'
    if params['make_plots']: #TODO test for all selections
        plotting.plot_mean_per_bin(yvar, varname, test_var, 'True ' + params['output_data'], numbins, c  = 'k', show = False, save = False, out_name = out_name, close = False)
        plotting.plot_mean_per_bin(yvar, varname, est_var, 'Estimated ' + params['output_data'], numbins, c  = 'tab:red', show = params['show'], save = params['save'], out_name = out_name)

#Plotting and calculating chi^2
for i in range(len(varnames)):
    varname = varnames[i]
    var2name = var2names[i]
    print("calculating " + varname + " " + var2name)
    #defining xvar and yvar
    selected_xvar, selected_yvar = ut.select_xvar_yvar(varname, var2name, in_X, input_props, galsim_X, galsim_props, Itest)
    #defining Ptest and Pest and filters
    xvar, yvar, test_var, est_var = ut.apply_disk_filter(var2name, selected_xvar, selected_yvar, Ptest, Pest)
    #defining plotting scales
    xscale, yscale = ut.yxscales(varname, var2name)
    #defining binning
    numbins = [10, 10]
    dowithdisk, dowithoutdisk = False, False
    if i in vars2dowithdisk:
        dowithdisk == True
    if i in vars2dowithoutdisk:
        dowithoutdisk == True
    #making 2d-plots
    plotting.make_all_plots(output_path, fname, xvar, yvar, test_var, est_var, varname, var2name, dowithdisk, dowithoutdisk, xscale, yscale, numbins, params)
