import numpy as np
import tensorflow as tf
import mat73
# from pyfftw.interfaces import scipy_fftpack as fftw
import scipy.fft as fftw
import sys
import os

class DataGenerator:
    def __init__(self, config):
        self.config = config

        train_in_file = os.path.join(self.config.data_dir,self.config.train_input)
        train_out_file = os.path.join(self.config.data_dir,self.config.train_output)

        print('*** LOADING TRAINING INPUT DATA ***')
        train_in_dict = mat73.loadmat(train_in_file)
        
        print('*** LOADING TRAINING OUTPUT DATA ***')
        train_out_dict = mat73.loadmat(train_out_file)

        train_in_key = list(train_in_dict.keys())[0]
        train_out_key = list(train_out_dict.keys())[0]
        
        self.input = train_in_dict[train_in_key]
        self.output = train_out_dict[train_out_key]
        
        self.input = np.transpose(train_in_dict[train_in_key])
        self.output = np.transpose(train_out_dict[train_out_key])
        
        self.mask = np.load('64_64_subsample_mask.npy') #64,64
        self.input_new = np.zeros((self.output.shape[0], 2*self.output.shape[1]))

        self.max_real = 534.5828247070312
        self.mean_real = 0.013785875402938082
        self.max_img = 291.32269287109375
        self.mean_img = 0.00027116766610798106

        
        # w = np.zeros((4096,4096)).astype(complex)
        # sub_matirx = scipy.linalg.dft(64)
        # for i in range(64):
        #     for j in range(64):    
        #         w[i*64:(i+1)*64, j*64:(j+1)*64] = sub_matirx[i][j] * sub_matirx
        
        # m = np.zeros((np.sum(self.mask),4096))
        # mask_shift = fftw.fftshift(self.mask)
        # s = 0
        # for i in range(64):
        #     for j in range(64):
        #         if mask_shift[i][j] == 1:
        #             m[s][64*i+j] = 1
        #             s += 1
        #         else: 
        #             continue

        # self.A = m @ w
        
        for idx in range(self.output.shape[0]):            
            
            #fft_im = fftw.fftshift( np.reshape(w @ self.output[idx,:], (self.config.im_h, self.config.im_w)) ) #fft1
            fft_im = fftw.fftshift( fftw.fft2(self.output[idx,:].reshape(self.config.im_h, self.config.im_w)) ) #fft2
            
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
        
    def next_batch(self, batch_size):
        idx = np.random.choice(self.len, batch_size)
        #yield self.input[idx], self.output[idx]
        yield self.input_new[idx], self.output[idx] ##220714



class ValDataGenerator:
    def __init__(self, config):
        self.config = config

        # test_in_file = os.path.join(self.config.data_dir, self.config.test_input)
        # test_out_file = os.path.join(self.config.data_dir, self.config.test_output)
        test_in_file = os.path.join("/home/sunyang/whyfail/data_64/", "test_input.mat")
        test_out_file = os.path.join("/home/sunyang/whyfail/data_64/", "test_x_real.mat")

        print('*** LOADING TESTING INPUT DATA ***')
        test_in_dict = mat73.loadmat(test_in_file)

        print('*** LOADING TESTING OUTPUT DATA ***')
        test_out_dict = mat73.loadmat(test_out_file)

        test_in_key = list(test_in_dict.keys())[0]
        test_out_key = list(test_out_dict.keys())[0]

        self.input = test_in_dict[test_in_key]
        self.output = test_out_dict[test_out_key]
        
        self.input = np.transpose(test_in_dict[test_in_key])
        self.output = np.transpose(test_out_dict[test_out_key])

        self.mask = np.load('64_64_subsample_mask.npy') #64,64
        self.input_new = np.zeros((self.output.shape[0], 2*self.output.shape[1]))

        self.max_real = 534.5828247070312
        self.mean_real = 0.013785875402938082
        self.max_img = 291.32269287109375
        self.mean_img = 0.00027116766610798106

        for idx in range(self.output.shape[0]):
            # fft_im = fftw.fftshift( fftw.fft2(self.output[idx,:].reshape(self.config.im_h, self.config.im_w)) ) #64,64
            fft_im = fftw.fftshift( fftw.fft2(self.output[idx,:].reshape(64, 64)) ) #64,64
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

    def next_batch(self, batch_size):
        idx = np.random.choice(self.len, batch_size)
        #yield self.input[idx], self.output[idx]
        yield self.input_new[idx], self.output[idx] #220714





if __name__ == '__main__':
    config = {

                "num_epochs": 10,
                "batch_size": 100, 

                "fc_input_dim": 8192,
                "fc_hidden_dim": 4096,
                "fc_output_dim": 4096,

                "im_h": 64,
                "im_w": 64,
                
                "data_dir": "/home/sunyang/whyfail/data_64/",
                "train_input": "train_input.mat",
                "train_output": "train_x_real.mat",
                "test_input": "test_input.mat",
                "test_output": "test_x_real.mat",

                "learning_rate": 0.0002
                
            }

    tr = ValDataGenerator(config)
    print(tr.len)
