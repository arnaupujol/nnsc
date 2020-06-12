"""
IMPORT DATA

This module defines methods to import data.

:Author: Arnau Pujol <arnaupv@gmail.com>

:Version: 1.0.0
"""

import numpy as np

def get_extra_filter(extra_filter, path = ''):
    """
    This method loads the array of the extra filter applied to the data.

    Parameters:
    -----------
    extra_filter: str {'has_disk', 'has_no_disk'}, optional
        It specifies the filter

            'has_disk':
                The mask keeps objects with a disk and a bulge

            'has_no_disk':
                The mask keeps objects with a single Sersic profile

    path: str
        Path where the filter is.

    Results:
    --------
    filter_arr: boolean np.array
        The filter array
    """
    if extra_filter == 'has_disk':
        filter_arr = np.load(path + 'has_disk.npy')
    elif extra_filter == 'has_no_disk':
        filter_arr = np.load(path + 'has_no_disk.npy')
    else:
        return None
    return filter_arr

def get_input_data(extra_filter = None, selection = 'original', path = '', name = 'input_data.npy'):
    """
    This method returns the parameters measured for the training.

    Parameters:
    -----------
    extra_filter: boolean np.array or None
        A boolean array (if it is not None) defining a selection of galaxies
    selection: str {'original', 'qbeta', 'ellip', 'reduced', 'sel8'}
        It defines the property selection of data

            'original':
                Original selection of properties from Pujol et al. (2020)

            'qbeta':
                It selects ellipticity modulus and orientation angle parameters

            'ellip':
                It selects parameters related with galaxy shape

            'reduced':
                A reduced selection with 14 properties

            'sel8':
                A reduced selection with only 8 properties

    path: str
        Directory where the data is
    name: str
        Name of file

    Returns:
    --------
    X: np.ndarray
        Properties (rows) of the objects (columns)
    input_props: list of strings
        List with property names
    """
    if selection == 'original':
        list_elements = range(27)
    elif selection == 'qbeta':
        list_elements = [17, 18, 23, 24]
    elif selection == 'ellip':
        list_elements = [0, 1, 17, 18, 19, 20, 23, 24]
    elif selection == 'reduced':
        list_elements = [0, 1, 2, 10, 11, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    elif selection == 'sel8':
        list_elements = [0, 1, 2, 19, 20, 22, 23, 24]
    else:
        print("ERROR: invalid value of selection: " + selection)
    X = np.load(path + name)
    X = np.array([X[:,i] for i in list_elements])
    input_props = [['gFIT_final_e1', 'gFIT_final_e2', 'gFIT_out_flux', \
    'gFIT_out_gal_sigma_noise', 'gFIT_out_dre', 'gFIT_out_bre', 'gFIT_out_df', \
    'gFIT_out_nb_fev', 'gFIT_SE_GAL_FLUX', 'gFIT_SE_GAL_FLUX_RADIUS', 'gFIT_SE_GAL_SNR', \
    'gFIT_SE_GAL_MAG', 'gFIT_SE_PSF_FLUX', 'gFIT_SE_PSF_FLUX_RADIUS', 'gFIT_SE_PSF_SNR', \
    'gFIT_SE_PSF_MAG', 'gFIT_SE_PSF_FWHM_IMAGE', 'gFIT_out_beta', 'gFIT_out_q', 'KSB_final_e1', \
    'KSB_final_e2', 'KSB_out_scale', 'KSB_out_sn', 'KSB_out_beta', 'KSB_out_q','KSB_final_ep', \
    'KSB_final_ex'][i] for i in list_elements]
    if extra_filter is not None:
        X = np.array([i[extra_filter] for i in X]).T
    else:
        X = np.array([i for i in X]).T
    return X, input_props


def get_galsim_data(extra_filter = None, path = ''):
    """
    This method returns the parameters defined in galsim.

    Parameters:
    -----------
    extra_filter: boolean array or None
        A boolean array (if it is not None) defining a selection of galaxies
    path: str
        Directory where the data is

    Returns:
    --------
    X: np.ndarray
        Properties (rows) of the objects (columns)
    galsim_props: list of strings
        List with property names
    """
    X = np.load(path + 'galsim_data.npy')
    galsim_props = ['g1', 'g2', 'bulge_n', 'bulge_hlr', 'bulge_q', 'bulge_beta_radians', \
    'bulge_flux', 'disk_hlr', 'disk_q', 'disk_beta_radians', 'disk_flux', 'gal_sn', \
    'g1_intrinsic', 'g2_intrinsic', 'psf_theta', 'in_beta', 'in_q', 'gp', 'gx', 'gp_intrinsic', \
    'gx_intrinsic']
    if extra_filter is not None:
        X = np.array([i[extra_filter] for i in X])
    else:
        X = np.array([i for i in X])
    return X, galsim_props

def get_m(extra_filter = None, path = ''):
    """
    This method reads and returns the shear bias m and c measurements for
    individual images from KSB.

    Parameters:
    -----------
    extra_filter: boolean np.array or None
        A boolean array (if it is not None) defining a selection of galaxies
    path: str
        Directory where the data is

    Returns:
    --------
    A tuple with multiplicative and additive biases (m1, m2, c1, c2)
    """
    m1 = np.load(path + 'm1.npy')
    m2 = np.load(path + 'm2.npy')
    c1 = np.load(path + 'c1.npy')
    c2 = np.load(path + 'c2.npy')
    if extra_filter is not None:
        return np.array((m1[extra_filter], m2[extra_filter], c1[extra_filter], c2[extra_filter]))
    else:
        return np.array((m1, m2, c1, c2))
