import numpy as np
import tensorflow as tf
import mat73
# from pyfftw.interfaces import scipy_fftpack as fftw
import scipy.fft as fftw
import sys
import os

class InferenceDataGenerator:
    def __init__(self, config):
        self.config = config

        inference_in_file = os.path.join(self.config.data_dir,self.config.inference_input)
        inference_out_file = os.path.join(self.config.data_dir,self.config.inference_target_output)

        print('*** LOADING INFERENCE INPUT DATA ***')
        inference_in_dict = mat73.loadmat(inference_in_file)
        
        print('*** LOADING INFERENCE OUTPUT DATA ***')
        inference_out_dict = mat73.loadmat(inference_out_file)

        inference_in_key = list(inference_in_dict.keys())[0]
        inference_out_key = list(inference_out_dict.keys())[0]
        
        self.input = inference_in_dict[inference_in_key]
        self.output = inference_out_dict[inference_out_key]
        
        self.input = np.transpose(inference_in_dict[inference_in_key])
        self.output = np.transpose(inference_out_dict[inference_out_key])

        self.max_real = 534.5828247070312
        self.mean_real = 0.013785875402938082
        self.max_img = 291.32269287109375
        self.mean_img = 0.00027116766610798106

        self.mask = np.load('64_64_subsample_mask.npy') #64,64
        self.input_new = np.zeros((self.output.shape[0], 2*self.output.shape[1]))
        for idx in range(self.output.shape[0]):
            fft_im = fftw.fftshift( fftw.fft2(self.output[idx,:].reshape(self.config.im_h, self.config.im_w)) ) #64,64
            fft_im = np.multiply(self.mask, fft_im)
            row, col = np.repeat(np.arange(fft_im.shape[0]),fft_im.shape[0]).tolist(), np.tile(np.arange(fft_im.shape[1]),fft_im.shape[1]).tolist()
            samples = fft_im[row, col]
            samples_real = np.real(samples)
            samples_imag = np.imag(np.conj(samples))
            samples_concat = np.squeeze(np.concatenate( (samples_real, samples_imag) ))
            self.input_new[idx,:] = samples_concat

        self.input_new[:,:4096] = ( self.input_new[:,:4096] - self.mean_real ) / self.max_real #220719
        self.input_new[:,4096:] = ( self.input_new[:,4096:] - self.mean_img ) / self.max_img

        self.len = self.input.shape[0]

    def next_batch(self, ind_start, batch_size):
        idx = np.arange(ind_start,ind_start+batch_size)
        #yield self.input[idx], self.output[idx]
        yield self.input_new[idx], self.output[idx] #220714
