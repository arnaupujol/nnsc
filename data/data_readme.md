This document contains information about the data from this directory.

The data refers to a random subsample of 309550 objects from the catalogue used
for Pujol et al. (2020). It contains:

`c1.npy`, `c2.npy`, `m1.npy`, `m2.npy`:

These are numpy arrays with the values of additive and multiplicative bias for
all the objects.

`galsim_data.npy`:

This 2d-array contains input simulation properties (from galsim) used to
generate the images. Its shape is (309550, 21), corresponding to the following
21 properties for the 309550 objects:

'g1': original shear $g_1$ from GREAT3 (not applied)

'g2': original shear $g_2$ from GREAT3 (not applied)

'bulge_n': Sersic index

'bulge_hlr': bulge half-light radius

'bulge_q': bulge axis ratio

'bulge_beta_radians': bulge orientation angle

'bulge_flux': bulge flux

'disk_hlr': disk half-light radius

'disk_q': disk axis ratio

'disk_beta_radians': disk orientation angle

'disk_flux': disk flux

'gal_sn': SNR

'g1_intrinsic': intrinsic ellipticity $e_1$

'g2_intrinsic': intrinsic ellipticity $e_2$

'psf_theta': PSF orientation angle

'in_beta': input orientation angle

'in_q': input axis ratio


'gp': shear $g_+$ with respect to PSF orientation angle (not applied)

'gx': shear $g_x$ with respect to PSF orientation angle (not applied)

'gp_intrinsic': intrinsic ellipticity $e_+$ with respect to PSF orientation angle

'gx_intrinsic': intrinsic ellipticity $e_x$ with respect to PSF orientation angle

`input_data.npy`:

This 2d-array contains measured properties used as input for the training.
Its shape is (309550, 27), corresponding to the following 27 properties
for the 309550 objects (also in Table 1 in Pujol et al. (2020)):

'gFIT_final_e1': galaxy ellipticity $e_1$ from gFIT

'gFIT_final_e2': galaxy ellipticity $e_2$ from gFIT

'gFIT_out_flux': galaxy flux from gFIT

'gFIT_out_gal_sigma_noise': noise level from gFIT

'gFIT_out_dre': disk size from gFIT

'gFIT_out_bre': bulge size from gFIT

'gFIT_out_df': disk to bulge ratio from gFIT

'gFIT_out_nb_fev': number of fitting iterations from gFIT

'gFIT_SE_GAL_FLUX': galaxy flux from SExtractor

'gFIT_SE_GAL_FLUX_RADIUS': galaxy radius from SExtractor

'gFIT_SE_GAL_SNR': galaxy SNR from SExtractor

'gFIT_SE_GAL_MAG': galaxy magnitude from SExtractor

'gFIT_SE_PSF_FLUX': PSF flux from SExtractor

'gFIT_SE_PSF_FLUX_RADIUS': PSF radius from SExtractor

'gFIT_SE_PSF_SNR': PSF SNR from SExtractor

'gFIT_SE_PSF_MAG': PSF magnitude from SExtractor

'gFIT_SE_PSF_FWHM_IMAGE': PSF FWHM from SExtractor

'gFIT_out_beta': galaxy orientation angle from gFIT

'gFIT_out_q': galaxy axis ratio from gFIT

'KSB_final_e1': galaxy ellipticity $e_1$ from KSB

'KSB_final_e2': galaxy ellipticity $e_2$ from KSB

'KSB_out_scale': scale of window function for KSB

'KSB_out_sn': galaxy SNR from KSB

'KSB_out_beta': galaxy orientation angle from KSB

'KSB_out_q': galaxy axis ratio from KSB

'KSB_final_ep': galaxy ellipticity $e_+$ from KSB

'KSB_final_ex': galaxy ellipticity $e_x$ from KSB

`has_disk.npy`, `has_no_disk.npy`:

Boolean numpy arrays specifying whether the simulated galaxies are single Sersic
profiles (they have no disk) or they are a combination of disk and bulge (they
have disk). The two arrays are opposite between them.
