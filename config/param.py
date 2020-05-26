"""Determines the parameters of the codes."""

params = {
'output_data' : 'm', # data that we want to predict
'n_train' : 3000, # number of samples used for training
'n_test' : 3000, # number of samples used for testing
'dim1' : 30, # number of neurons in second layer
'dim2' : 30, # number of neurons in 3rd layer
'dim3' : 30, # number of neurons in 4th layer. If 0, no layer
'dim4' : 30, # number of neurons in 5th layer. If 0, no layer
'dim5' : 0, # number of neurons in 6th layer. If 0, no layer
'ct' : 0.0, # contamination level (the same for all levels as it is now)
'n_epochs' : 100, # number of epochs in the unsupervised step
'batch_size' : 64, # batch size
'learning_rate' : .00001, # learning rate
'use_err' : False, #defines if bias errors are used in the chi^2
'selection' : 'original', #defines the data selection used for the analysis.
'version' : 'deep_reg_est', #version of learning code used. Can be '' (original), 'v4', 'ldae_semi', 'deep_reg', 'deep_reg_est', deep_regh', 'deep_regrh', 'deep_regrhw'
'z' : 0., #pcost for the supervised learning
'optimizer' : 'Adam', #optimizer used in machine learning proces, can be 'Adam' or 'GD'
'object' : 'KSB', #object from which the measurements of bias are done
'test' : False, #if True, it uses the data and paths from the test mode.
'lr_decay' : 15, #epochs after which to begin the learning rate decay
'make_plots' : True, #Specify if plots are created.
'extra_filter' : '', #Specify an extra selection cut. Can be '' (none), 'has_disk', 'has_no_disk', or 'sfrac#' (fraction of sing. Sersic), where # is in [0,100].
'show' : True, #Specify if plots are shown.
'save' : False, #Specify if output is saved.
'top_fs' : 0, #Specifies how many feature space properties are used for the training, taking the most important ones. 0 if all are taken.
}

par_types = {
'output_data' : str,
'n_train' : int,
'n_test' : int,
'dim1' : int,
'dim2' : int,
'dim3' : int,
'dim4' : int,
'dim5' : int,
'ct' : float,
'n_epochs' : int,
'batch_size' : int,
'learning_rate' : float,
'use_err' : bool, #defines if bias errors are used in the chi^2
'selection' : str,
'version' : str,
'z' : float,
'optimizer' : str,
'object' : 'KSB', #object from which the measurements of bias are done
'test' : bool,
'lr_decay' : int,
'make_plots' : bool,
'extra_filter' : str,
'small_m_diluted' : bool,
'show' : bool,
'save' : bool,
'top_fs' : int,
}

def update(params, argv):
	"""Updates parameters from default values and arguments."""
	if len(argv) > 1:
		for i in range(1,len(argv),2):
			if argv[i + 1] == 'max':
				params[argv[i]] = np.inf
			elif argv[i + 1] == 'min':
				params[argv[i]] = -np.inf
			elif argv[i + 1] == 'False':
				params[argv[i]] = False
			elif argv[i + 1] == 'True':
				params[argv[i]] = True
			else:
				params[argv[i]] = par_types[argv[i]](argv[i + 1])
	for i in params:
		if params[i] == 'max':
			params[i] = np.inf
		elif params[i] == 'min':
			params[i] = -np.inf