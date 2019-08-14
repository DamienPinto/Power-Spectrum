import math
import os
import sys
import numpy.random
import matplotlib.pyplot as plt
import numpy as np
from p_spec_tools import get_fourier_transf, get_k_axis, plot_2Ddist_n_p_spec, make_power_spectrum
from get_array_from_txt import get_array_from_txt_w_list, str_array_to_np_array


PARAM_DIR = "Parameter Files/"
DATA_DIR  = "Data Files/"
OUT_DIR   = "Output Files/"


def get_gaussian_space(parameter_array):

	# ds: array representing the step-size in each direction/dimension of the space to be generated.
	ds = np.fromstring(parameter_array[1], sep=" ", dtype=np.int16).tolist()
	# Ls: array representing the length the space is supposed to span in every direction.
	Ls = np.fromstring(parameter_array[2], sep=" ", dtype=np.int16).tolist()
	#mean, std_dev: mean and standard deviation used to parametrize the normal/Gaussian distribution from with the
	#space will be sampled.
	[mean, std_dev] = np.fromstring(parameter_array[3], sep=" ", dtype=np.float32).tolist()

	# Ns: array representing number of steps in each direction.
	# Ns = [int(round((1<<(int(Ls[i]/ds[i])).bit_length())/2)) for i in range(len(ds))]
	Ns = [int(round(Ls[i]/ds[i])) for i in range(len(ds))]
	print(ds)
	print(Ls)
	print(Ns)
	# Using the smallest value of ds because that will allow for the largest possibly detected k-value,
	# hence encompassing all the k-values measured in the other directions.
	min_idx = ds.index(min(ds))
	dim     = len(ds)
	k, k_min, k_max, k_step = get_k_axis(Ls[min_idx], Ns[min_idx], ds[min_idx], dim)

	# Create the distribution/ND-array sampled from a random Gaussian/normal distribution.
	distribution = np.random.normal(loc=mean, scale=std_dev, size=Ns)

	#Get the fourier transform of the distribution.
	fourier_dist = get_fourier_transf(distribution, dim)
	out_filename = parameter_array[0] + "_dist_m" + str(mean) + "_stddv" + str(std_dev)

	return k, distribution, fourier_dist, out_filename




def get_2Dperiodic_space(parameter_array):

	[k_val, L, N, direc] = np.fromstring(parameter_array[1], sep=" ", dtype=np.float32).tolist()

	dx		  = L/N
	x 		  = np.arange(0, L, dx)
	xy 		  = np.array([x]*x.shape[0])
	per_array = np.sin(k_val*xy)

	if direc == 1:
		distribution = per_array.T
	if direc == 2:
		distribution = np.array([np.roll(per_array[i], i) for i in range(len(per_array))])

	fourier_dist = get_fourier_transf(per_array, 2)
	out_filename = parameter_array[0] + "_dist_k" + str(k_val) + "_direc" + str(direc)
	k, k_min, k_max, k_step = get_k_axis(L, N, dx, 2)

	return k, distribution, fourier_dist, out_filename




def get_dist_from_file(parameter_array):

	filename = parameter_array[1]
	Ns 		 = str_array_to_np_array(parameter_array[2])
	ds 		 = str_array_to_np_array(parameter_array[3])
	Ls 		 = ds*Ns

	if os.path.ispath.file(filename):
		get_array_from_txt_w_list(DATA_DIR + filename)
	else:
		print("Couldn't find the file specified in the parameter file, please check the spelling/path and try again.")
		quit()

	min_idx = ds.index(min(ds))
	dim     = len(ds)
	k, k_min, k_max, k_step = get_k_axis(Ls[min_idx], Ns[min_idx], ds[min_idx], dim)

	distribution.reshape(Ns)
	fourier_dist = get_fourier_transf(distribution, dim)
	out_filename  = "dist_" + parameter_array[0] + parameter_array[1]

	return k, distribution, fourier_dist, out_filename




def main():

	# If the user input a command-line argument...
	if len(sys.argv) != 1:
		# ...try that argument as a parameter file.
		print(os.listdir(PARAM_DIR))
		if os.path.isfile(PARAM_DIR+sys.argv[1]):

			# Open and read parameter file.
			parameter_file  = open(PARAM_DIR + sys.argv[1], "r")
			parameter_str   = parameter_file.read()
			parameter_array = parameter_str.split("\n")
			print(parameter_array[0])

			# The first line in the parameter file is expected to be a string indicating the type of sidtribution to be
			# generated.
			if parameter_array[0] == "gaussian":
				# If this is where the first line of the file read, expect the rest of the file to be of the format:
				'''
				(2 or 3 floats separated by spaces)
				(2 or 3 floats separated by spaces)
				(2 floats separated by spaces)
				
				These will be used to create 2 or 3 dimensional array populated with values sampled from a 
				Gaussian/normal distribution.
				'''

				k, distribution, fourier_dist, out_filename = get_gaussian_space(parameter_array)

			elif parameter_array[0] == "2Dperiodic":

				'''
				(3 floats indicating k, dx, and direc)

				This will be used to created a 2D distribution representing either a vertical, horizontal, or diagonal 
				sin wave.
				k(float): wavenumber, follows k = 2*pi/(lambda) where lambda is the wavelength of the wave to be created.
				L(float): parameter indicating the length in appropriate units of the side-length of the space to be generated
				N(int)  : parameter indicating the number of steps/pixels per side of the space to be generated
				direc   : 0->horizontal, 1->vertical, 2-diagonal.
				'''

				k, distribution, fourier_dist, out_filename = get_2Dperiodic_space(parameter_array)

			elif parameter_array == "from_file":

				'''
				(string indicating filename to go get distribution from)
				(N floats separated by ", " indicating size of N dimensions distribution spans)
				(N floats separated by ", " indicating step-size between adjacent pixels in each dimension)
				'''

				k, distribution, fourier_dist, out_filename = get_dist_from_file(parameter_array)

		else:
			print("Couldn't find the file given as input. Please check spelling and or existence of the file and try again.")
			quit()

	power_spectrum, power_spectrum_errs = make_power_spectrum(fourier_dist)

	if len(np.shape(distribution)) == 2:
		distribution_slice = distribution
	elif len(np.shape(distribution)) == 3:
		distribution_slice = distribution[0,:,:]
	else:
		print("Your distribution is in some custom shape and you're going to have to write the code to output a slice of it here if you want to plot something of it.")
	
	print(len(k), len(power_spectrum), len(power_spectrum_errs))
	plot_2Ddist_n_p_spec(k, distribution_slice, power_spectrum, power_spectrum_errs)

	out_p_spec_file     = open(OUT_DIR + out_filename+".txt", "w")
	out_p_spec_err_file	= open(OUT_DIR + out_filename+"_errors.txt", "w")

	out_p_spec_file.write(str(power_spectrum))
	out_p_spec_err_file.write(str(power_spectrum_errs))

	out_p_spec_file.close()
	out_p_spec_err_file.close()		

if __name__ == '__main__':
	main()