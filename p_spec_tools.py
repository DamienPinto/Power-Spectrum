import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import sys
import torch




def make_power_spectrum(fourier_universe):

	#The biggest distance from the origin is the one at the furthest corner so the biggest k would be the norma at the 
	#max length in each dimension.
	k_max = np.sqrt(np.sum(np.square(fourier_universe.shape)))/2
	
	#Make the power spectrum have that many points. The power_spectrum_log keeps a log gives all the values of k an 
	#array and keeps track of all the values in the Fourier transformed universe with that value of k.
	power_spectrum_log = [[] for _ in range(int(k_max))]
	power_spectrum 	   = [0 for _ in range(int(k_max))]
	counting_errors	   = [0 for _ in range(int(k_max))]

	#Cycle through all the points in the Fourier transformed universe...
	#Since what is given to the make_power_spectrum function is the Fourier space *after* the shifts, then the central 
	#frequency should be the DC term, the first should be the most negative frequency, and the last should be the 
	#largest positive frequncy.
	# print("Fourier_shape:", fourier_universe.shape)
	for k1 in range(-int(fourier_universe.shape[0]/2-1), int(fourier_universe.shape[0]/2)):
		for k2 in range(-int(fourier_universe.shape[1]/2-1), int(fourier_universe.shape[1]/2)):
			if len(fourier_universe.shape) == 3:
				for k3 in range(-int(fourier_universe.shape[2]/2-1), int(fourier_universe.shape[2]/2)):
					# print (k1+len(fourier_universe)/2, k2+len(fourier_universe[k1])/2, k3/len(fourier_universe[k1][k2])/2)
					#...determine it's distance from the origin, its k...
					k = int(round(float(math.sqrt(np.abs(k1)**2 + np.abs(k2)**2 + np.abs(k3)**2))))
					# print k, len(power_spectrum), k1+len(fourier_universe)/2, len(fourier_universe), k2+len(fourier_universe)/2, len(fourier_universe[k1]), k2+len(fourier_universe)/2, len(fourier_universe[k1][k2])
					#...append the norm squared of that value to the correct array in the power_spectrum_log.
					power_spectrum_log[k].append(float(np.abs(fourier_universe[int(k1+len(fourier_universe)/2)][int(k2+len(fourier_universe[k1])/2)][int(k3+len(fourier_universe[k1][k2])/2)])**2))
					
			elif len(fourier_universe.shape) == 2:
				k = int(round(float(math.sqrt(np.abs(k1)**2 + np.abs(k2)**2))))
				power_spectrum_log[k].append(float(np.abs(fourier_universe[int(k1+len(fourier_universe)/2)][int(k2+len(fourier_universe[k1])/2)])))
					
	#For each value of k, average over the values of the points with that k and place the result in the power spectrum,
	for i in range(len(power_spectrum_log)):
		if len(power_spectrum_log[i]) > 0:
			power_spectrum[i]  = float(np.sum(power_spectrum_log[i])/len(power_spectrum_log[i]))
			counting_errors[i] = power_spectrum[i]*float(2/np.sqrt(len(power_spectrum_log[i])))
		else:
			power_spectrum[i]  = 0
			counting_errors[i] = 0

	#This is only true when producing the power spectrum of a gaussian noise distribution:
	#Power spectrum should be flat, and it's value should be the standard deviation of the Gaussian distribution used to
	#produce the initial universe/matter distribution but squared so posting the square root should return just 
	#(approximately) the standard deviation:
	# np.sqrt(float(np.sum(power_spectrum)/len(power_spectrum)))

	# print len(power_spectrum)

	return power_spectrum, counting_errors




def get_k_axis(L, N, dx, dim):

	'''
	Function made to get the k_axis specified by L and N (which specify dx, but that is included as a param simply to 
	avoid having to recalculate it. Q:is that more efficient or less? Or inconsequential?). In other words this function
	returns the range of possible wavenumbers k that could be detected given a space of length L and N regularly spaced
	sampling intervals.

	Input:
		-L(float) : Value indicating side-length (in some units) of space from which one would like to get a power
					spectrum.
		-N(int)   : Number of sampling intervals in said space.
		-dx(float): Distance between sampling intervals, dx = L/N

	Output:
		-k(np.array)  : Array specifiying the range of possibly sampled wavenumbers, partitionned with appropriate step 
						size given N, L, and dx.
		-k_min(float) : Smallest wavenumber possibly sampled given specifics of space (basically 0, because you can 
						always just have a space filled with the same value everywhere regardless of its size).
		-k_max(float) : Largest wavenumber possibly sampled based on specifics of space and Nyquist frequecy.
		-k_step(float): Step-size between possible 
	'''

	min_wvlgth = 2*dx #Smallest wavelength that can be detected given the parameters of the space.
	k_max 	   = 2*math.pi/min_wvlgth #Maximum wavelength based o Nyquist frequency.
	k_min 	   = 0
	k_step	   = (k_max - k_min)/(N/2) #N/2 because np.fft.fftn returns a space of the same size as the input space 
									   #and the origin of the fourier space is at (N/2, N/2), where every cell away from
									   #the origin represents a specific wavenumber vector who's magnitude is found in 
									   #[k_min, k_max] that was sampled.
	
	'''
	Given that the fourier space created by np.fft.fftn is the same size as the distribution it's given, but has its 
	origin in the middle of the space at the (N/2, N/2,... N/2) cell, the furthest distance that a point can be from the
	origin in fourier space is sqrt((N/2)^2 + (N/2)^2 + ...) = sqrt(num_dimensions)*N/2 cells away. So the furthest 
	distance from the origin.
	'''
	k = np.arange(k_min, np.sqrt(dim)*N/2*k_step, k_step)

	return k, k_min, k_max, k_step



def get_p_spec_idx(k_step, k_wanted):	
	idx = int(round(k_wanted/k_step))
	return idx



def get_1D_gauss(k, mu, sigma, mag):
	gauss_1D = stats.norm.pdf(k, mu, sigma)*mag
	return gauss_1D



def get_dist_from_p_spec(p_spec, dim):
	from power_spectrum_to_universe import make_fourier_space

	axes		  = tuple(list(range(dim)))
	fourier_space = make_fourier_space(p_spec, dim)
	print("Obtained Fourier Space with similar power spectrum.")
	#Normalize Fourier Space (assuming all dimensions are of same length and partitioned the same way).
	#This is based on numpy.fft documentation.
	fourier_space /= np.sqrt(len(fourier_space)**dim)
	dist_2D		  = np.fft.ifftshift(np.fft.ifftn(fourier_space, axes=axes))
	dist_2D 	  *= np.sqrt(len(fourier_space)**dim)

	return dist_2D



def get_2Ddist_w_k_spike_pspec(L, N, k_spike_loc=-1, amplitude=200, gauss=1, sigma=0.15):

	'''
	Function that makes a 2D distribution based on a power spectrum with a spike of a specified ampitude at a specified 
	value of k (at a specific wavenumber). 
	This can be a pure spike (0 everywhere except for specified location) or a gaussian with a specified standard 
	deviation (sigma).

	Input:
		-L(float)		   : Total length of space in the space's units.
		-N(int)			   : Number of steps/divisions spanning one direction of the space (assumes all directions have 
							 same number of steps)
		-k_spike_loc(float): Location in 1/(units of L) indicatig the value at which to create a spike in the power 
							 spectrum. The valid range for k-values given L and N will be computed and k_spike_loc must 
							 be within this range.
		-amplitude(float)  : Amplitude the spike in the power spectrum is to have. 
							 NOTE: Currently seems to produce distributions with power spectrum amplitudes 1/100th this??
		-gauss(int) 	   : On/Off on whether power spectrum spike is gaussian or just a pure delta.
		-sigma(float)	   : Value indicating the standard deviation of gaussian modeling the power spectrum spike.

	Output:
		-dist_2D (np.ndarray): 2D array that was created from the power spectrum specified by the input parameters and 
							   made to have a very similar power spectrum. Due to random elements in population/creation
							   of space, it's power will not be EXACTLY like the one used to creat the space, but it 
							   will have very similar features and characteristics.
	'''

	dx = L/N
	k, k_min, k_max, k_step = get_k_axis(L, N, dx, 2)

	if k_spike_loc == -1:
		input_query = "No value given for k-spike location. k range is [" + str(k_min) + ", " + str(k_max) + "]. Input a float that is inside that range."
		k_spike_loc = float(input(input_query))

	if not(k_min <= k_spike_loc <= k_max):
		print("The current location enterred for the k_pike_location is not within the possible k range of [" + str(k_min) + ", " + str(k_max) + "].")
		k_spike_loc = float(input("Please enter a k-value within that range: "))

	if k_min <= k_spike_loc <= k_max:

		k_spike_idx = get_p_spec_idx(k_step, k_spike_loc)
		k_spike_loc = k[k_spike_idx]
		if gauss == 1:
			p_spec 	= get_1D_gauss(k, k_spike_loc, sigma, amplitude)
		else:
			p_spec 	= np.zeros(len(k))
			p_spec[k_spike_idx] = magnitude
		# print("k_spike_loc: ", k_spike_loc)
		# plt.plot(k, p_spec)
		dist_2D 	= get_dist_from_p_spec(p_spec, 2)


	print("If you get a \"Referenced before declaration.\" error here w.r.t. dist_2D then you STILL didn\'t enter a good k_spike_location despite the warning. If you don\'t then you\'re all good for this section.")	

	return dist_2D



def get_fourier_transf(dist, dim):
	
	axes 		  = tuple(list(range(dim)))
	fourier_space = np.fft.ifftshift(np.fft.fftn(np.fft.fftshift(dist), axes=axes))
	#Normalize the fourier space (assuming all dimensions are of same length and partitioned the same way)
	fourier_space /= np.sqrt(len(fourier_space)**dim)

	return fourier_space



def plot_2Ddist_n_p_spec(k, dist, p_spec, p_spec_errs):

	'''
	Function made to makes side-by-side plts of a  2D distribution and its power spectrum (with errorbars).
	NOTE: Could add string input indicating units of values found in distribution and units of partitioning of space to 
		  add proper units to power spectrum axes.
	'''
	
	fig = plt.figure()

	ax = fig.add_subplot(1,2,1)
	ax.set_title("Distribution Slice")
	plt.imshow(np.real(np.fft.ifftshift(dist)))

	ax = fig.add_subplot(1,2,2)
	ax.set_title("Power Spectrum of Distribution")
	ax.set_ylabel("P(k)")
	ax.set_xlabel("k")
	# print(k.shape, np.array(p_spec).shape, np.array(p_spec_errs).shape)
	#kept that^ line because sometimes the rounding involved in getting the length of the power spectrum causes a 
	#difference in length by 1 between k and p_spec
	if len(k) != len(p_spec):
		print("The length of the k_axis and of the power spectrum do not match, not going to plot the \"excess\" on the larger one of the two.")
		short_idx = min(len(k), len(p_spec))
		ax.errorbar(k[:short_idx-1], p_spec[:short_idx-1], yerr=p_spec_errs[:short_idx-1], ecolor='r')
	else:
		ax.errorbar(k, p_spec, yerr=p_spec_errs, ecolor='r')

	plt.show()



if __name__ == "__main__":

	L  = 150
	N  = 100
	dx = L/N

	dist_2D 				= get_2Ddist_w_k_spike_pspec(L, N, k_spike_loc=1)
	k, k_min, k_max, k_step = get_k_axis(L, N, dx, 2)
	dist_2D_fourier_space   = get_fourier_transf(dist_2D, 2)

	dist_2D_p_spec, dist_2D_errs = make_power_spectrum(dist_2D_fourier_space)

	plot_2Ddist_n_p_spec(k, dist_2D, dist_2D_p_spec, dist_2D_errs)