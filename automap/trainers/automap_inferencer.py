import tqdm
import numpy as np
import tensorflow as tf
import scipy.io as sio
import sys
import os 
import pandas as pd
import scipy
# from pyfftw.interfaces import scipy_fftpack as fftw
import scipy.fft as fftw
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
from bm4d import bm4d
plt.rcParams["axes.grid"] = False

class AUTOMAP_Inferencer:

    def __init__(self, model, data, config, attack_iter=None, alpha=None, eps=None, 
        ascent=None, beta=None, vis=0, atk=0, atxs=0, smooth=0, sample=0, std=0, m_m='median', denoise=0):

        self.model =model
        self.config = config
        self.data = data
        self.epsilon = eps
        self.iter = attack_iter
        self.alpha = alpha
        self.asc = ascent
        self.beta = beta
        self.real_mean = np.mean(self.data.input_new[:,:4096])
        self.img_mean = np.mean(self.data.input_new[:,4096:])
        self.mask = np.load('64_64_subsample_mask.npy') #64,64
        self.A = self.matrix_A()
        self.visualize = vis

        self.max_real = 534.5828247070312
        self.mean_real = 0.013785875402938082
        self.max_img = 291.32269287109375
        self.mean_img = 0.00027116766610798106
        self.atk = atk

        self.smooth = smooth
        self.sample = sample
        self.std = std
        self.mean_median = m_m
        self.atk_x_smth = atxs
        self.ql_idx = -1
        self.qu_idx = -1
        self.dnis = denoise



    def estimate_ql_qu(self, eps, sample_count, sigma, conf_thres=.99999):
        theo_perc_u = stats.norm.cdf(eps/sigma)
        theo_perc_l = stats.norm.cdf(-eps / sigma)

        q_u_u = sample_count -1
        q_u_l = math.ceil(theo_perc_u*sample_count)
        q_l_u = math.floor(theo_perc_l*sample_count)
        q_l_l = 0
        
        q_u_final = q_u_u
        for q_u in range(q_u_l, q_u_u):
            conf = stats.binom.cdf(q_u-1, sample_count, theo_perc_u)
            if conf > conf_thres:
                q_u_final = q_u
                break

        q_l_final = q_l_l
        for q_l in range(q_l_u, q_l_l, -1):
            conf = 1-stats.binom.cdf(q_l-1, sample_count, theo_perc_l)
            if conf > conf_thres:
                q_l_final = q_l
                break
        
        return q_l_final, q_u_final

    
    def smooth_model(self, model, batch_size, ipt, noise_std, num_sample, mean_median, denoise=0, ql=None, qu=None):
        # final_prediction = np.zeros(ipt.shape)
        # for i in range(ipt.shape[-1]): #n
        #     mask = np.zeros(ipt.shape) #b,n
        #     mask[:,i] = 1
        #     pixel_i_batch = []
        #     for j in range(num_sample):
        #         print(f'pixel {i}/{ipt.shape[-1]}, sample {j}/{num_sample} started ...')
        #         noise_batch = tf.random.normal( shape=tf.shape(ipt), stddev=noise_std )
        #         slide_ipt = ipt + tf.math.multiply( noise_batch, mask )
        #         c_2, prediction = model(self.fft_A(slide_ipt), training=False)
        #         prediction = self.prediction_resize(prediction, batch_size)
        #         pixel_i_batch.append(prediction[:,i])
        #     sorted_pixel_i_batch = np.sort( np.transpose(np.array(pixel_i_batch)) )
        #     final_prediction[:,i] = sorted_pixel_i_batch[:,int(num_sample/2)]

        final_prediction = np.zeros(ipt.shape)
        samples = []
        for j in range(num_sample):
            print(f'sample {j}/{num_sample} started ...')
            noise_batch = tf.random.normal( shape=tf.shape(ipt), stddev=noise_std )
            slide_ipt = ipt + noise_batch #100,4096
            
            if denoise: 
                slide_ipt_np = slide_ipt.numpy()
                for i in range(slide_ipt_np.shape[0]):
                    print(f'sample: {j}/{num_sample}. denoise: {i}/{slide_ipt_np.shape[0]} completed ...')
                    denoised_i = bm4d(slide_ipt_np[i].reshape((64,64)), sigma_psd=noise_std) #(64,64,1)
                    slide_ipt_np[i] = np.squeeze(denoised_i).reshape(4096,)
                slide_ipt = tf.convert_to_tensor(slide_ipt_np)
            
            c_2, prediction = self.model(self.fft_A(slide_ipt), training=False)
            prediction = self.prediction_resize(prediction, batch_size)
            samples.append(prediction)
        if mean_median == 'median':
            sorted_batch = np.sort( np.transpose(np.array(samples), (1,2,0)) ) #(b,n,num_sample)
            median_prediction = np.squeeze( sorted_batch[:,:,int(num_sample/2)] ) #(b,n)
            final_prediction = median_prediction
            if (ql!=None) and (qu!=None):
                l_prediction = np.squeeze( sorted_batch[:,:,ql] )
                u_prediction = np.squeeze( sorted_batch[:,:,qu] )
                final_prediction = median_prediction, l_prediction, u_prediction
        elif mean_median == 'mean':
            final_prediction = np.squeeze(np.mean( np.transpose(np.array(samples), (1,2,0)), axis=-1, keepdims=True )) #b,n
            
        return final_prediction
        

    def matrix_A(self):
        # build A
        w = np.zeros((4096,4096)).astype(complex)
        sub_matirx = scipy.linalg.dft(64)
        for i in range(64):
            for j in range(64):    
                w[i*64:(i+1)*64, j*64:(j+1)*64] = sub_matirx[i][j] * sub_matirx
        m = np.zeros((int(np.sum(self.mask)),4096))
        mask_shift = fftw.fftshift(self.mask)
        s = 0
        for i in range(64):
            for j in range(64):
                if mask_shift[i][j] == 1:
                    m[s][64*i+j] = 1
                    s += 1
                else: 
                    continue
        A = m @ w # (m,n)
        A_tf = tf.convert_to_tensor(A, dtype=tf.complex64)
        return A_tf


    def fft_A(self, output):
        for idx in range(output.shape[0]):            
            fft_im = tf.signal.fftshift( tf.signal.fft2d( tf.cast( tf.reshape(output[idx,:], [self.config.im_h, self.config.im_w]), dtype=tf.complex64 ) ) ) #fft2
            fft_im = tf.math.multiply(self.mask, fft_im)
            row, col = np.repeat(np.arange(fft_im.shape[0]), fft_im.shape[0]).tolist(), np.tile(np.arange(fft_im.shape[1]),fft_im.shape[1]).tolist()
            samples = tf.gather_nd( fft_im, indices=[ [row[i], col[i]] for i in range(len(row)) ] )
            samples_real = tf.math.real(samples)
            samples_imag = tf.math.imag(tf.math.conj(samples))

            samples_real = ( samples_real - self.mean_real ) / self.max_real #220719
            samples_imag = ( samples_imag - self.mean_img ) / self.max_img

            samples_concat = tf.expand_dims( tf.squeeze(tf.concat( [samples_real, samples_imag], axis=0 )), 0 )
            if idx == 0:
                input_new = samples_concat
            else:
                input_new = tf.concat( [input_new, samples_concat], 0 )
        return input_new


    def inference_attack_step(self, ind_start, batch_size, attack_iter, alpha, eps, ascent, atk_Ax=0, atk_x=0, smooth=0):
        raw_data_input, targets = next(self.data.next_batch(ind_start, batch_size))

        #ordinary val
        c_2, predictions_ord = self.model(raw_data_input, training=False)

        #attack Ax
        if atk_Ax:
            x_input = tf.math.multiply(raw_data_input, tf.ones(shape=tf.shape(raw_data_input)))
            # input_mean = tf.squeeze(tf.concat( (tf.ones((batch_size,4096))*self.real_mean, tf.ones((batch_size,4096))*self.img_mean), axis=1 ))
            epsilon = eps# * input_mean 
            attack_scale = alpha# * input_mean 
            
            for _ in range(attack_iter):   
                with tf.GradientTape() as tape:
                    tape.watch(x_input)
                    c_2, predictions = self.model(x_input, training=False) #training=False --> No dropout
                    train_loss_for_gradient, train_loss_mse_reconstruction = self.custom_loss(targets, predictions, c_2, batch_size)
                gradients_ipt = tape.gradient(train_loss_for_gradient, x_input)

                if ascent == 'fgsm':
                    # Method 1 FGSM
                    x_input += attack_scale * np.sign(gradients_ipt) 
                elif ascent == 'pga':
                    # Method 2 PGA (without momentum)
                    x_input += alpha * gradients_ipt

                x_input = tf.clip_by_value(x_input, raw_data_input - epsilon, raw_data_input + epsilon) 

            c_2, predictions_atk = self.model(x_input, training=False)

        #attack x
        elif atk_x:
            x_input = tf.math.multiply(targets, tf.ones(shape=tf.shape(targets))) #b,n
            epsilon = eps# * input_mean 
            attack_scale = alpha# * input_mean 
            
            for it in range(attack_iter):   
                start = timer()

                with tf.GradientTape() as tape:
                    tape.watch(x_input)
                    c_2, predictions = self.model(self.fft_A(x_input), training=False) 
                    train_loss_for_gradient, train_loss_mse_reconstruction = self.custom_loss(targets, predictions, c_2, batch_size)
                gradients_ipt = tape.gradient(train_loss_for_gradient, x_input)

                if ascent == 'fgsm':
                    # Method 1 FGSM
                    x_input += attack_scale * np.sign(gradients_ipt) 
                elif ascent == 'pga':
                    # Method 2 PGA (without momentum)
                    x_input += alpha * gradients_ipt
                elif ascent == 'dag':
                    # Method 3 DAG
                    x_input += gradients_ipt/tf.norm(gradients_ipt, axis=1, keepdims=True) * .2 * epsilon

                # x_input = tf.clip_by_value(x_input, targets - epsilon, targets + epsilon) #A good practice to clip? #220804 Removed
                norm = tf.norm(x_input-targets, axis=1) #100
                div = tf.where(norm > epsilon, norm/epsilon, tf.ones_like(norm)) #100
                x_input = (x_input-targets) / tf.expand_dims(div, axis=-1) + targets #100,4096
                
                end = timer()
                perturbing_time = end - start
                print(f'Attack iter {it} done. Time cost {perturbing_time}')

            if atk_x and self.atk_x_smth: 
                ql_idx, qu_idx = self.estimate_ql_qu(epsilon, self.sample, self.std, conf_thres=.99999)
                predictions_atk = self.smooth_model(self.model, batch_size, x_input, self.std, self.sample, self.mean_median, denoise=self.dnis, ql=ql_idx, qu=qu_idx) #3 outputs 
                self.ql_idx, self.qu_idx = ql_idx, qu_idx
            else:
                c_2, predictions_atk = self.model(self.fft_A(x_input), training=False)   
        
        # #for smooth test
        # elif smooth:
        #     x_input = tf.math.multiply(targets, tf.ones(shape=tf.shape(targets))) #b,n
        #     predictions_atk = self.smooth_model(self.model, batch_size, x_input, self.std, self.sample, self.mean_median, denoise=self.dnis)

        #just purterb x
        else:
            x_input = tf.math.multiply(targets, tf.ones(shape=tf.shape(targets))) #b,n
            x_noise = tf.random.normal( shape=tf.shape(x_input), stddev=self.std )
            x_input += x_noise
            c_2, predictions_atk = self.model(self.fft_A(x_input), training=False) #b,n


        return predictions_ord, predictions_atk, targets


    def valcustom_loss(self, targets, predictions, batch_size):

        if predictions.shape != targets.shape:
            predictions = tf.reshape(predictions, [batch_size, self.config.im_h + 8, self.config.im_w + 8])
            predictions = tf.transpose(predictions, perm=[1,2,0])
            predictions = tf.image.resize_with_crop_or_pad(predictions, self.config.im_h, self.config.im_w)
            predictions = tf.transpose(predictions, perm=[2,0,1])
            predictions = tf.reshape(predictions, [batch_size, self.config.fc_output_dim])
        val_loss = tf.reduce_mean(tf.square(tf.norm(targets-predictions, axis=1)))
        return val_loss

    def valcustom_discrepency(self, targets, predictions_tuple, batch_size):
        
        for i, predictions in enumerate(predictions_tuple):
            if predictions.shape != targets.shape:
                predictions = tf.reshape(predictions, [batch_size, self.config.im_h + 8, self.config.im_w + 8])
                predictions = tf.transpose(predictions, perm=[1,2,0])
                predictions = tf.image.resize_with_crop_or_pad(predictions, self.config.im_h, self.config.im_w)
                predictions = tf.transpose(predictions, perm=[2,0,1])
                predictions = tf.reshape(predictions, [batch_size, self.config.fc_output_dim])
                predictions_tuple[i] = predictions
        val_loss = tf.reduce_mean(tf.square(tf.norm(predictions_tuple[0]-predictions_tuple[1], axis=1)))
        return val_loss



    def custom_loss(self, targets, predictions, c_2, batch_size, lbd=1e-4):

        predictions = tf.reshape(predictions, [batch_size, self.config.im_h + 8, self.config.im_w + 8])
        predictions = tf.transpose(predictions, perm=[1,2,0])
        predictions = tf.image.resize_with_crop_or_pad(predictions, self.config.im_h, self.config.im_w)
        predictions = tf.transpose(predictions, perm=[2,0,1])
        predictions = tf.reshape(predictions, [batch_size, self.config.fc_output_dim])
        
        act_loss = lbd*tf.reduce_sum(tf.abs(c_2)) #TODO: weight regilarization parameter
        loss_gradient = tf.reduce_mean(tf.square(tf.norm(targets-predictions, axis=1))) #+ act_loss #TODO: weight regilarization
        train_loss_mse_reconstruction = tf.reduce_mean(tf.square(tf.norm(targets-predictions, axis=1)))
        
        return loss_gradient, train_loss_mse_reconstruction


    def prediction_resize(self, predictions, batch_size):
        predictions = tf.reshape(predictions, [batch_size, self.config.im_h + 8, self.config.im_w + 8])
        predictions = tf.transpose(predictions, perm=[1,2,0])
        predictions = tf.image.resize_with_crop_or_pad(predictions, self.config.im_h, self.config.im_w)
        predictions = tf.transpose(predictions, perm=[2,0,1])
        predictions = tf.reshape(predictions, [batch_size, self.config.fc_output_dim])
        return predictions



    def inference_attack(self):

        train_info = self.config.checkpoint_dir.split('/')[-1].split('_')[1]# '(Spec)JA'/'(Spec)JAJ'/'adv'/'original/smt'
        if train_info == 'adv' or train_info == 'smt':
            train_info = '_'.join( self.config.checkpoint_dir.split('/')[-1].split('_')[1:] ) 

        output_array = np.zeros( int(np.ceil(self.data.len/self.config.batch_size)) ) 
        a_output_array = np.zeros( int(np.ceil(self.data.len/self.config.batch_size)) )

        a_dis_array = np.zeros( int(np.ceil(self.data.len/self.config.batch_size)) )
        a_lb_dis_array = np.zeros( int(np.ceil(self.data.len/self.config.batch_size)) )
        a_ub_dis_array = np.zeros( int(np.ceil(self.data.len/self.config.batch_size)) )

        vis_step = np.random.randint(int(np.ceil(self.data.len/self.config.batch_size)))
   
        for step in range(int(np.ceil(self.data.len/self.config.batch_size))): ###220714

            if step < np.ceil(self.data.len/self.config.batch_size)-1:
                batch_size = self.config.batch_size
            else:
                batch_size = self.data.len-self.config.batch_size*step
            
            ind_start = step*self.config.batch_size
            predictions, a_predictions, targets = self.inference_attack_step(ind_start, batch_size, self.iter, self.alpha, self.epsilon, self.asc, atk_x=self.atk, smooth=self.smooth)
            
            bs = targets.shape[0]
            
            #loss
            output_array[step] = self.valcustom_loss(targets, predictions, bs) 
            if len(a_predictions) == 3: #pred, l_pred, u_pred
                a_output_array[step] = self.valcustom_loss(targets, a_predictions[0], bs) 
            else:
                a_output_array[step] = self.valcustom_loss(targets, a_predictions, bs)

             
            #discrepency
            if len(a_predictions) == 3: #pred, l_pred, u_pred
                a_dis_array[step] = self.valcustom_discrepency(targets, [predictions, a_predictions[0]], bs) 
                a_lb_dis_array[step] = self.valcustom_discrepency(targets, [predictions, a_predictions[1]], bs) 
                a_ub_dis_array[step] = self.valcustom_discrepency(targets, [predictions, a_predictions[2]], bs) 
            else:
                a_dis_array[step] = self.valcustom_discrepency(targets, [predictions, a_predictions], bs) 
             
            #visualize
            if self.visualize:
                if step == vis_step:
                    predictions = self.prediction_resize(predictions, batch_size)
                    a_predictions = self.prediction_resize(a_predictions, batch_size)

                    fig, axs = plt.subplots(3, 4, figsize=(12, 9))
                    ids = [0,1,5,8]
                    for i in range(len(ids)):
                        opt = np.asarray(tf.reshape(predictions[ids[i]], [self.config.im_h, self.config.im_w]))
                        ptb = np.asarray(tf.reshape(a_predictions[ids[i]], [self.config.im_h, self.config.im_w]))
                        tgt = np.asarray(tf.reshape(targets[ids[i]], [self.config.im_h, self.config.im_w]))

                        org_los = tf.reduce_mean(tf.square(tf.norm(tgt-opt)))
                        atk_los = tf.reduce_mean(tf.square(tf.norm(tgt-ptb)))
                        atk_dis = tf.reduce_mean(tf.square(tf.norm(opt-ptb)))

                        axs[0,i].imshow(tgt, cmap='gray')
                        axs[0,i].axis('off')
                        img_name = f'plots/sample{i}_recons_ORG.png'
                        axs[0,i].title.set_text(f'Original')

                        axs[1,i].imshow(opt, cmap='gray')
                        axs[1,i].axis('off')
                        axs[1,i].title.set_text(f'Ordinary_loss: {str(org_los)[10:15]}')
                        
                        axs[2,i].imshow(ptb, cmap='gray')
                        axs[2,i].axis('off')
                        axs[2,i].title.set_text(f'Atk loss:{str(atk_los)[10:15]} / dis:{str(atk_dis)[10:15]}')

                    fig.savefig(f'plots/model_{train_info}_atk_{self.iter}_{self.alpha}_{self.epsilon}_{self.asc}_smt_{self.atk_x_smth}_{self.sample}_{self.std}.png')
                    
                    # ptb = np.asarray(tf.reshape(a_predictions[ids[0]], [self.config.im_h, self.config.im_w]))
                    # tgt = np.asarray(tf.reshape(targets[ids[0]], [self.config.im_h, self.config.im_w]))
                    # plt.imshow(ptb, cmap='gray')
                    # plt.imshow(tgt, cmap='gray')
                    # plt.imshow(pre, cmap='gray') #cmap='gray_r'

                    

        
        #table-lize
        mean_loss = np.mean( output_array ) 
        a_mean_loss = np.mean( a_output_array )
        a_mean_dis = np.mean( a_dis_array )
        a_lb_mean_dis = np.mean( a_lb_dis_array )
        a_ub_mean_dis = np.mean( a_ub_dis_array )
        
        if not os.path.exists(self.config.pickle_file_path):
            d = {'model_type':[train_info], 
                'atk_mode':[self.atk], 'atk_iter':[self.iter], 'atk_alp':[self.alpha], 'atk_elp':[self.epsilon], 'atk_asc':[self.asc], 
                'smooth':[self.atk_x_smth], 'atk_smp':[self.sample], 'atk_std':[self.std], 'mean_median':[self.mean_median], 'denoise':[self.dnis],
                'loss':[mean_loss], 'atk_loss':[a_mean_loss], 
                'ql':[self.ql_idx], 'qu':[self.qu_idx], 'atk_dis':[a_mean_dis], 'atk_lb_dis':[a_lb_mean_dis], 'atk_ub_dis':[a_ub_mean_dis]}
            df = pd.DataFrame(data=d)
            df.to_pickle(self.config.pickle_file_path)
        else:
            d = {'model_type':train_info, 
                'atk_mode':self.atk, 'atk_iter':self.iter, 'atk_alp':self.alpha, 'atk_elp':self.epsilon, 'atk_asc':self.asc, 
                'smooth':self.atk_x_smth, 'atk_smp':self.sample, 'atk_std':self.std, 'mean_median':self.mean_median, 'denoise':self.dnis,
                'loss':mean_loss, 'atk_loss':a_mean_loss, 
                'ql':self.ql_idx, 'qu':self.qu_idx, 'atk_dis':a_mean_dis, 'atk_lb_dis':a_lb_mean_dis, 'atk_ub_dis':a_ub_mean_dis}
            df = pd.read_pickle(self.config.pickle_file_path)
            df = df.append(d, ignore_index=True)
            df.to_pickle(self.config.pickle_file_path)

        print('Inference Done')




