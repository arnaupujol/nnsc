"""
NAMES

This module defines the names of the files saves as output.

:Author: Arnau Pujol <arnaupv@gmail.com>

:Version: 1.0.1
"""

def fname(params):
	"""This method defines de fname of the model used in bias_ldae.py

    Parameters:
    -----------
    params: dict
        Dictionary specifying all the model parameters

    Returns:
    --------
    fname: str
        Name of the model

    """
	fname='MODEL' + params['version'] + '_' + params['selection'] + '_' + params['output_data'] + '_ntr_' + str(params['n_train'])+ '_nts_' + str(params['n_test']) + '_dim_' + str(params['dim1']) + '_'+ str(params['dim2'])
	if params['dim3'] > 0:
		fname += '_' + str(params['dim3'])
	if params['dim4'] > 0:
		fname += '_' + str(params['dim4'])
	if params['dim5'] > 0:
		fname += '_' + str(params['dim5'])
	fname+= '_ct_' + str(params['ct']) + '_nep_' + str(params['n_epochs']) + '_bs_' + str(params['batch_size']) + '_lr_' + str(params['learning_rate']) + '_opt' + params['optimizer']
	if 'deep_reg' in params['version']:
		fname+= '_lrdec_' + str(params['lr_decay'])
	if params['extra_filter'] != '':
		fname += '_' + params['extra_filter']
	if params['top_fs'] > 0:
		fname += '_topfs_' + str(params['top_fs'])
	return fname
