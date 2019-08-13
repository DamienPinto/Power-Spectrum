import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from generate_2D_dist_from_p_spec import *
from mock_power_spectrum import make_power_spectrum
from power_spectrum_to_universe import make_fourier_space



def make_gauss_p_spec_database(param_path):

	param_file 		= open(param_path, "r")
	param_str  		= param_file.read()
	param_str_array = param_str.split("\n")
	param_file.close()

	'''
	The parameter file is expected to be in the format:
	[k1, k2, k3, ... kN] -> array specifying all wavenumber values for which you want to produce distributions, important to separate values with comma AND space.
	L N 				 -> two space-separated (as in literally separated by a single character that is a space) values specifying the length in whatever desired units of the space to be generated, and the number of sampling steps spanning that space in each orthogonal direction 
	sigma amp 			 -> two space-separated values indicating the standard deviatino and the amplitude of the gaussian profile used to model the power spectra of the distributions to be created
	num_imgs 			 -> number of distributions to be created per given wavenumber value

	An example txt file should have been given to you.
	'''

	p_spec_means   = np.fromstring(param_str_array[0].strip("[").strip("]"), sep=", ", dtype=np.float32)
	[L, N] 		   = np.fromstring(param_str_array[1], sep=" ", dtype=np.float32)
	[sigma, amp]   = np.fromstring(param_str_array[2], sep=" ", dtype=np.float32)
	num_imgs 	   = int(param_str_array[3])
	flat_img_array = []
	lbl_array 	   = []

	for k_loc in p_spec_means:

		fig = plt.figure()

		k, k_min, k_max, k_step = get_k_axis(L, N)
		p_spec 					= get_gauss_p_spec(k, k_loc, sigma, amp)
		print("k_loc: ", k_loc)

		for i in range(200):

			# ax = fig.add_subplot(1, 5, i+1)

			# dist 		 			 = get_2D_from_L_n_N(L, N, k_spike_loc=k_loc, sigma=sigma, amplitude=amp)
			# dist_fourier 			 = get_fourier_transf(dist, 2)
			# k, k_min, k_max, k_step  = get_k_axis(L, N)
			# dist_p_spec, dist_p_errs = make_power_spectrum(dist_fourier)

			# plot_2Ddist_n_p_spec(k, dist, dist_p_spec, dist_p_errs)

			# plt.imshow(np.real(dist))
			# print(i)

			dist = get_dist_from_p_spec(p_spec, 2)
			# print("dist shape: ", np.real(dist).shape)
			flat_img_array.append(np.real(dist).flatten().tolist())
			lbl_array.append(k_loc)

		# plt.show()

	if len(flat_img_array) == len(lbl_array):
		img_database_file = open("gauss_p_spec_database_1000imgs.txt", "w")
		lbl_database_file = open("gauss_p_spec_database_1000imgs_labels.txt", "w")
		img_database_file.write(str(flat_img_array))
		lbl_database_file.write(str(lbl_array))
		img_database_file.close()
		lbl_database_file.close()
	else:
		print("Shit fucked up son.")

	return

if __name__ == "__main__":
	make_gauss_p_spec_database("test_params.txt")		

