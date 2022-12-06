import tqdm
import numpy as np
import tensorflow as tf
import pickle
import sys
import os
from timeit import default_timer as timer
import scipy
# from pyfftw.interfaces import scipy_fftpack as fftw
import scipy.fft as fftw


class AUTOMAP_Trainer:

    def __init__(self, model, data, valdata, config, attack_iter=None, alpha=None, eps=None, ascent=None, original=None, spectral=None, jacobian=None, jaj=None, bta=None, jcb_gamma=20, sample=0, std=0, std_step=0, wmstart=0, sample_smt=0):

        self.model = model
        self.config = config
        self.data = data
        self.valdata = valdata

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.attack_val_loss = tf.keras.metrics.Mean(name='attack_val_loss')

        self.attack_iter = attack_iter
        self.alpha = alpha
        self.eps = eps
        self.asc = ascent
        self.org = original

        self.jacobian = jacobian
        self.spectral = spectral
        self.mask = np.load('64_64_subsample_mask.npy') #64,64
        self.A = self.matrix_A()
        self.jaj = jaj
        self.bta = bta
        self.jcb_gamma = jcb_gamma

        self.max_real = 534.5828247070312
        self.mean_real = 0.013785875402938082
        self.max_img = 291.32269287109375
        self.mean_img = 0.00027116766610798106

        self.warm_start_epoch = wmstart
        self.sample = sample
        self.std = std
        self.stdstp = std_step
        self.sample_smt = sample_smt


    def custom_loss(self, targets, predictions, c_2):
        predictions = tf.reshape(predictions,[self.config.batch_size,self.config.im_h + 8,self.config.im_w + 8])
        predictions = tf.transpose(predictions,perm=[1,2,0])
        predictions = tf.image.resize_with_crop_or_pad(predictions, self.config.im_h, self.config.im_w)
        predictions = tf.transpose(predictions,perm=[2,0,1])
        predictions = tf.reshape(predictions,[self.config.batch_size,self.config.fc_output_dim])
        
        act_loss = 1e-4*tf.reduce_sum(tf.abs(c_2)) #TODO: weight regilarization parameter
        train_loss_with_regularization = tf.reduce_mean(tf.square(tf.norm(targets-predictions, axis=1))) #+ act_loss #TODO: weight regilarization
        train_loss_mse_reconstruction = tf.reduce_mean(tf.square(tf.norm(targets-predictions, axis=1)))

        return train_loss_with_regularization, train_loss_mse_reconstruction


    def valcustom_loss(self,targets,predictions):
        predictions = tf.reshape(predictions,[self.config.batch_size,self.config.im_h + 8,self.config.im_w + 8])
        predictions = tf.transpose(predictions,perm=[1,2,0])
        predictions = tf.image.resize_with_crop_or_pad(predictions, self.config.im_h, self.config.im_w)
        predictions = tf.transpose(predictions,perm=[2,0,1])
        predictions = tf.reshape(predictions,[self.config.batch_size,self.config.fc_output_dim])
        val_loss = tf.reduce_mean(tf.square(tf.norm(targets-predictions, axis=1)))
        return val_loss


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


    def batch_JA(self, J, A): 
        # (b,2n) -> (b,m) -> (b,m)@(m,n)=(b,n)
        bs, y_dim = J.shape[0], J.shape[1] # (b,2n)
        rea_l, img_l = [], []
        s = 0
        assert 4096 == int(y_dim/2)
        for i in range(4096):
            if self.mask.reshape(-1)[i] == 1:
                rea_l.append(J[:,i])
                img_l.append(J[:,4096+i])
                s += 1
            else:
                continue
        assert s == int(np.sum(self.mask))
        J_rea = tf.transpose( tf.stack(rea_l) )
        J_img = tf.transpose( tf.stack(img_l) )
        J_clx = tf.complex(J_rea, J_img)# (b,m)
        JA = tf.linalg.matmul(J_clx, A) # (b,n)
        return J_clx, JA


    def batch_random_constant(self, B, C): 
        v = tf.random.normal([B,C]) #(b,n)
        vnorm = tf.norm(v, 2, 1, True) #(b,1)
        nomrlised_v = tf.divide(v, vnorm) #(b,n)
        return nomrlised_v     
    

    def prediction_resize(self, predictions):
        predictions = tf.reshape(predictions, [predictions.shape[0],self.config.im_h + 8,self.config.im_w + 8])
        predictions = tf.transpose(predictions, perm=[1,2,0])
        predictions = tf.image.resize_with_crop_or_pad(predictions, self.config.im_h, self.config.im_w)
        predictions = tf.transpose(predictions, perm=[2,0,1])
        predictions = tf.reshape(predictions,[predictions.shape[0], self.config.fc_output_dim]) #(b,n)
        return predictions


    def train_step(self, epoch, optimizer, spectral=0, jacobian=0, jaj=0, gamma=20):

        raw_data, targets = next(self.data.next_batch(self.config.batch_size))
        cprob = 1  # multiplicative noise on (default during training)
        raw_data_input = tf.math.multiply(raw_data, tf.random.uniform(shape=tf.shape(raw_data), minval=0.99, maxval=1.01)) * cprob + raw_data * (1 - cprob)

        start = timer()

        with tf.GradientTape() as tape:
            c_2, predictions = self.model(raw_data_input, training=False) #training=False --> No dropout    
            loss_gradient, train_loss_mse_reconstruction = self.custom_loss(targets, predictions, c_2) 
        
        if jacobian: #return J_loss
            
            with tf.GradientTape() as tape_j:
                tape_j.watch(raw_data_input)
                _, outputs = self.model(raw_data_input, training=False) 
                outputs = self.prediction_resize(outputs) #(b,n)
            
            #JA
            vA = self.batch_random_constant(B=outputs.shape[0], C=outputs.shape[1]) #(b,n)
            grad_ipt = tape_j.gradient(outputs, raw_data_input, vA) # (b,n), (b,2n), (b,n) -> (b,2n) (in fact should be (b,m))
            J, JA = self.batch_JA(grad_ipt, self.A) # (b,m), (b,n)  
            JA_norm = outputs.shape[1] * tf.norm(JA)**2 / (1*self.config.batch_size) #TODO why still complex part exist? 
            beta_JA = tf.math.pow( 10, tf.math.floor(tf.math.log(loss_gradient/tf.math.real(JA_norm))) ) / gamma
            J_loss = beta_JA * tf.math.real(JA_norm)
            
            #J
            if jaj:
                J_norm = outputs.shape[1] * tf.norm(J)**2 / (1*self.config.batch_size) 
                beta_J = tf.math.pow( 10, tf.math.floor(tf.math.log(loss_gradient/tf.math.real(J_norm))) ) / gamma 
                J_loss += beta_J * tf.math.real(J_norm) 
                
        if spectral: #return J_loss  
            
            #initialize u, power method iter
            spec_iter = 3 # TODO
            uA = self.batch_random_constant(B=raw_data_input.shape[0], C=4096) #(b,n) 
            if jaj:
                u = self.batch_random_constant(B=raw_data_input.shape[0], C=4096) #(b,n) 

            #power method to compute largest singular value of |JA|
            for _ in range(spec_iter):
                
                #JA
                #vjp
                with tf.GradientTape() as tape_vjp:
                    tape_vjp.watch(raw_data_input)
                    _, outputs = self.model(raw_data_input, training=False) 
                    outputs = self.prediction_resize(outputs) #(b,n)
                grad_ipt = tape_vjp.gradient(outputs, raw_data_input, uA) # (b,n), (b,2n), (b,n) -> (b,2n) (in fact should be (b,m))
                J, JA = self.batch_JA(grad_ipt, self.A) # (b,m), (b,n), complex
                vA = tf.transpose(JA) # (n,b), complex
                
                #jvp
                with tf.GradientTape() as tape_jvp:
                    d = tf.Variable(self.batch_random_constant(B=raw_data_input.shape[0], C=4096)) #(b,n) 
                    tape_jvp.watch(d)
                    with tf.GradientTape() as sub_tape:
                        sub_tape.watch(raw_data_input)
                        _, outputs = self.model(raw_data_input, training=False) 
                        outputs = self.prediction_resize(outputs) #(b,n)
                    sub_grad_ipt = sub_tape.gradient(outputs, raw_data_input, d) # (b,n), (b,2n), (b,n) -> (b,2n) (in fact should be (b,m))
                    g, _ = self.batch_JA(sub_grad_ipt, self.A) # (b,m), _, complex
                uA = tape_jvp.gradient(g, d, tf.transpose(tf.linalg.matmul(self.A,vA))) # (b,m), (b,n), (b,m) -> (b,n) real

                #J
                if jaj:
                    #vjp
                    with tf.GradientTape() as tape_vjp:
                        tape_vjp.watch(raw_data_input)
                        _, outputs = self.model(raw_data_input, training=False) 
                        outputs = self.prediction_resize(outputs) #(b,n)
                    grad_ipt = tape_vjp.gradient(outputs, raw_data_input, u) # (b,n), (b,2n), (b,n) -> (b,2n) (in fact should be (b,m))
                    J, JA = self.batch_JA(grad_ipt, self.A) # (b,m), (b,n), complex
                    v = tf.transpose(J) # (m,b), complex
                    
                    #jvp
                    with tf.GradientTape() as tape_jvp:
                        d = tf.Variable(self.batch_random_constant(B=raw_data_input.shape[0], C=4096)) #(b,n) 
                        tape_jvp.watch(d)
                        with tf.GradientTape() as sub_tape:
                            sub_tape.watch(raw_data_input)
                            _, outputs = self.model(raw_data_input, training=False) 
                            outputs = self.prediction_resize(outputs) #(b,n)
                        sub_grad_ipt = sub_tape.gradient(outputs, raw_data_input, d) # (b,n), (b,2n), (b,n) -> (b,2n) (in fact should be (b,m))
                        g, _ = self.batch_JA(sub_grad_ipt, self.A) # (b,m), _, complex
                    u = tape_jvp.gradient(g, d, tf.transpose(v)) # (b,m), (b,n), (b,m) -> (b,n), complex? #real
        
            #final
            uv_A_norm = tf.norm(uA, axis=1) / tf.math.real(tf.norm(tf.transpose(vA), axis=1)) #(b,)
            JA_norm = tf.reduce_max(uv_A_norm) 
            beta_JA = tf.math.pow( 10, tf.math.floor(tf.math.log(loss_gradient/tf.math.real(JA_norm))) ) / gamma
            J_loss = beta_JA * tf.math.real(JA_norm)
            if jaj:
                uv_norm = tf.norm(u, axis=1) / tf.math.real(tf.norm(tf.transpose(v), axis=1)) #(b,)
                J_norm = tf.reduce_max(uv_norm) 
                beta_J = tf.math.pow( 10, tf.math.floor(tf.math.log(loss_gradient/tf.math.real(J_norm))) ) / gamma 
                J_loss += beta_J * tf.math.real(J_norm) 

        with tf.GradientTape() as tape:
            c_2, predictions = self.model(raw_data_input, training=False) #training=False --> No dropout    
            loss_gradient, train_loss_mse_reconstruction = self.custom_loss(targets, predictions, c_2) 
            if jacobian or spectral:
                loss_gradient_J = loss_gradient + J_loss
        if jacobian or spectral:
            gradients = tape.gradient(loss_gradient_J, self.model.trainable_variables)
        else:
            gradients = tape.gradient(loss_gradient, self.model.trainable_variables)
        
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        end = timer()
        optimising_time = end - start
        
        return train_loss_mse_reconstruction, optimising_time


    def attack_val_step(self, epoch, attack_iter, alpha, eps, ascent, atk_Ax=0, atk_x=1):
        raw_data_input, targets = next(self.valdata.next_batch(self.config.batch_size)) 
        
        #ordinary validate
        c_2, predictions = self.model(raw_data_input, training=False)
        valloss = self.valcustom_loss(targets, predictions)

        #attack validate
        if atk_Ax:
            x_input = tf.math.multiply(raw_data_input, tf.ones(shape=tf.shape(raw_data_input)))
            # input_mean = tf.squeeze(tf.concat( (tf.ones((self.config.batch_size,4096))*self.val_real_mean, tf.ones((self.config.batch_size,4096))*self.val_img_mean), axis=1 ))
            epsilon = eps# * input_mean 
            attack_scale = alpha# * input_mean 
            
            for i in range(attack_iter):   
                with tf.GradientTape() as tape:
                    tape.watch(x_input)
                    c_2, predictions = self.model(x_input, training=False) #training=False --> No dropout
                    train_loss_for_gradient, train_loss_mse_reconstruction = self.custom_loss(targets, predictions, c_2)
                gradients_ipt = tape.gradient(train_loss_for_gradient, x_input)

                if ascent == 'fgsm':
                    # Method 1 FGSM
                    x_input += attack_scale * np.sign(gradients_ipt) 
                elif ascent == 'pga':
                    # Method 2 PGA (without momentum)
                    x_input += alpha * gradients_ipt
                
                x_input = tf.clip_by_value(x_input, raw_data_input - epsilon, raw_data_input + epsilon) 

            c_2, predictions = self.model(x_input, training=False)
            
            #loss
            attack_valloss = self.valcustom_loss(targets, predictions)
            return valloss, attack_valloss

        elif atk_x:
            x_input = tf.math.multiply(targets, tf.ones(shape=tf.shape(targets))) #b,n
            epsilon = eps# * input_mean 
            attack_scale = alpha# * input_mean 
            
            for it in range(attack_iter):   
                with tf.GradientTape() as tape:
                    tape.watch(x_input)
                    c_2, predictions = self.model(self.fft_A(x_input), training=False) #training=False --> No dropout
                    train_loss_for_gradient, train_loss_mse_reconstruction = self.custom_loss(targets, predictions, c_2)
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

            c_2, predictions = self.model(self.fft_A(x_input), training=False)

            #loss
            attack_valloss = self.valcustom_loss(targets, predictions)
            return valloss, attack_valloss

        else:
            return valloss


    def attack_train_step(self, epoch, optimizer, attack_iter, alpha, eps, ascent, beta, atk_Ax=0, atk_x=1, smp=0, std=0): 

        #load raw data
        raw_data, targets = next(self.data.next_batch(self.config.batch_size))
        cprob = 1  # multiplicative noise on (default during training)
        raw_data_input = tf.math.multiply(raw_data, tf.random.uniform(shape=tf.shape(raw_data), minval=0.99, maxval=1.01)) * cprob + raw_data * (1 - cprob)
        
        # #ordinary training #temporary 220717
        # c_2, predictions = self.model(raw_data_input, training=False) #training=False --> No dropout    
        # loss_cln, train_loss_mse_reconstruction = self.custom_loss(targets, predictions, c_2) 
        
        #perturb 
        start = timer()
        
        if atk_Ax:
            x_input = raw_data_input
            epsilon = eps# * input_mean #tunable 220622
            attack_scale = alpha# * input_mean #tunable 220622
            
            for i in range(attack_iter):   
                with tf.GradientTape() as tape:
                    tape.watch(x_input)
                    c_2, predictions = self.model(x_input, training=False) #training=False --> No dropout
                    train_loss_for_gradient, train_loss_mse_reconstruction = self.custom_loss(targets, predictions, c_2)
                gradients_ipt = tape.gradient(train_loss_for_gradient, x_input)

                if ascent == 'fgsm':
                    # Method 1 FGSM
                    x_input += attack_scale * np.sign(gradients_ipt) 
                elif ascent == 'pga':
                    # Method 2 PGA (without momentum)
                    x_input += alpha * gradients_ipt
                
                x_input = tf.clip_by_value(x_input, raw_data_input - epsilon, raw_data_input + epsilon) 

        elif atk_x:
            
            if std != 0:
                x_input = tf.math.multiply(targets, tf.ones(shape=tf.shape(targets))) #b,n
                epsilon = eps# * input_mean 
                attack_scale = alpha# * input_mean 

                # noise_batch = []
                # for _ in range(smp):
                #     noise_batch.append(tf.random.normal( shape=tf.shape(targets), stddev=std )) #b,n
                
                for it in range(attack_iter):                    
                    with tf.GradientTape() as tape:
                        tape.watch(x_input)
                        for j in range(smp):
                            print(f'epoch {epoch}. attack {it}/{attack_iter}. sample {j}/{smp}...')  
                            noise_batch = tf.random.normal( shape=tf.shape(targets), stddev=std )
                            noisy_ipt = x_input + noise_batch
                            c_2, prediction = self.model(self.fft_A(noisy_ipt), training=False) #b,n       
                            if j == 0 :
                                predictions = prediction
                            else:
                                predictions += prediction 
                        predictions = predictions / smp
                        train_loss_for_gradient, train_loss_mse_reconstruction = self.custom_loss(targets, predictions, c_2)
                    gradients_ipt = tape.gradient(train_loss_for_gradient, x_input)        
                    if ascent == 'fgsm': # Method 1 FGSM
                        x_input += attack_scale * np.sign(gradients_ipt) 
                    elif ascent == 'pga': # Method 2 PGA (without momentum)
                        x_input += alpha * gradients_ipt
                    elif ascent == 'dag': # Method 3 DAG
                        x_input += gradients_ipt/tf.norm(gradients_ipt, axis=1, keepdims=True) * .2 * epsilon 
                    norm = tf.norm(x_input-targets, axis=1) #100
                    div = tf.where(norm > epsilon, norm/epsilon, tf.ones_like(norm)) #100
                    x_input = (x_input-targets) / tf.expand_dims(div, axis=-1) + targets #100,4096
                     
            else:
                x_input = tf.math.multiply(targets, tf.ones(shape=tf.shape(targets))) #b,n
                epsilon = eps# * input_mean 
                attack_scale = alpha# * input_mean 
                
                for it in range(attack_iter):   
                    with tf.GradientTape() as tape:
                        tape.watch(x_input)
                        c_2, predictions = self.model(self.fft_A(x_input), training=False) #training=False --> No dropout
                        train_loss_for_gradient, train_loss_mse_reconstruction = self.custom_loss(targets, predictions, c_2)
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

        #normal gradient 
        start = timer()
        
        with tf.GradientTape() as tape:
            if atk_Ax:
                c_2, predictions = self.model(x_input, training=False)
            elif atk_x:
                c_2, predictions = self.model(self.fft_A(x_input), training=False)
            loss_atk, train_loss_mse_reconstruction = self.custom_loss(targets, predictions, c_2)
            loss_gradient = loss_atk #loss_gradient = beta * loss_cln + (1-beta) * loss_atk #temporary 220717
        gradients = tape.gradient(loss_gradient, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables)) 
        
        end = timer()
        optimising_time = end - start
        
        return train_loss_mse_reconstruction, perturbing_time, optimising_time


    def smooth_train_step(self, epoch, optimizer, noise_std, noise_std_step, num_sample, attack_iter, alpha, epsilon, smp): 

        raw_data, targets = next(self.data.next_batch(self.config.batch_size))
        cprob = 1  # multiplicative noise on (default during training)
        raw_data_input = tf.math.multiply(raw_data, tf.random.uniform(shape=tf.shape(raw_data), minval=0.99, maxval=1.01)) * cprob + raw_data * (1 - cprob)

        start = timer()

        for i in range(1,noise_std_step+1):
            if attack_iter: #20220907          
                x_input = tf.math.multiply(targets, tf.ones(shape=tf.shape(targets))) #b,n
                if smp:
                    for it in range(attack_iter):                    
                        with tf.GradientTape() as tape:
                            tape.watch(x_input)
                            for j in range(smp):
                                print(f'Attack noise: epoch {epoch}. attack {it}/{attack_iter}. sample {j+1}/{smp}...')  
                                noise_batch = tf.random.normal( shape=tf.shape(targets), stddev=noise_std )
                                noisy_ipt = x_input + noise_batch
                                c_2, prediction = self.model(self.fft_A(noisy_ipt), training=False) #b,n       
                                if j == 0 :
                                    predictions = prediction
                                else:
                                    predictions += prediction 
                            predictions = predictions / smp
                            train_loss_for_gradient, train_loss_mse_reconstruction = self.custom_loss(targets, predictions, c_2)
                        gradients_ipt = tape.gradient(train_loss_for_gradient, x_input)        
                        x_input += alpha * gradients_ipt
                        norm = tf.norm(x_input-targets, axis=1) #100
                        div = tf.where(norm > epsilon, norm/epsilon, tf.ones_like(norm)) #100
                        x_input = (x_input-targets) / tf.expand_dims(div, axis=-1) + targets #100,4096
                else:
                    for it in range(attack_iter):   
                        with tf.GradientTape() as tape:
                            tape.watch(x_input)
                            c_2, predictions = self.model(self.fft_A(x_input), training=False) #training=False --> No dropout
                            train_loss_for_gradient, train_loss_mse_reconstruction = self.custom_loss(targets, predictions, c_2)
                        gradients_ipt = tape.gradient(train_loss_for_gradient, x_input)
                        x_input += alpha * gradients_ipt # PGA (without momentum)
                        norm = tf.norm(x_input-targets, axis=1) #100
                        div = tf.where(norm > epsilon, norm/epsilon, tf.ones_like(norm)) #100
                        x_input = (x_input-targets) / tf.expand_dims(div, axis=-1) + targets #100,4096

            gradients_list = []
            for j in range(num_sample):
                print(f'Smoothing noise: epoch {epoch}, step{i}/{noise_std_step}, sample {j+1}/{num_sample} ...')
                noise_batch = tf.random.normal( shape=tf.shape(targets), stddev=i*noise_std/noise_std_step ) #b,n
                noisy_ipt = x_input + noise_batch #b,n
                with tf.GradientTape() as tape: 
                    c_2, prediction = self.model(self.fft_A(noisy_ipt), training=False) #b,n
                    loss_atk, train_loss_mse_reconstruction = self.custom_loss(targets, prediction, c_2)
                gradients = tape.gradient(loss_atk, self.model.trainable_variables) #10,2n,n
                if j == 0:
                    gradients_list = gradients #10,2n,n
                else:
                    for k in range(len(gradients)):
                        gradients_list[k] += gradients[k] #10,2n,n
            for h in range(len(gradients_list)):
                gradients_list[h] = gradients_list[h] / (noise_std_step*num_sample)
            optimizer.apply_gradients(zip(gradients_list, self.model.trainable_variables)) 

        end = timer()
        optimising_time = end - start

        c_2, predictions = self.model(raw_data_input, training=False)
        _, train_loss_mse_reconstruction = self.custom_loss(targets, predictions, c_2)
        
        return train_loss_mse_reconstruction, optimising_time


    def jcb_train(self): 
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.config.learning_rate)

        loss_training = np.zeros((3,self.config.num_epochs))
        
        for epoch in range(self.config.num_epochs):

            pbar = tqdm.tqdm(total=self.data.len // self.config.batch_size, desc='Steps', position=0)
            train_status = tqdm.tqdm(total=0, bar_format='{desc}', position=1)

            epoch_train_loss, epoch_val_loss, epoch_attack_val_loss = 0, 0, 0
            for step in range(self.data.len // self.config.batch_size):  
                
                if self.jacobian or self.spectral:
                    loss, optimising_time = self.train_step(epoch, optimizer, self.spectral, self.jacobian, self.jaj, self.jcb_gamma)
                else:
                    loss, optimising_time = self.train_step(epoch, optimizer)

                # valloss, attack_valloss = self.attack_val_step(epoch, self.attack_iter, self.alpha, self.eps, self.asc)    
                # train_status.set_description_str(f'Epoch: {epoch} Step: {step} Train_Loss: {loss} Val_Loss: {valloss} Atk_Val_Loss: {attack_valloss} Optim_Time: {optimising_time}')
                train_status.set_description_str(f'Epoch: {epoch} Step: {step} Train_Loss: {loss} Optim_Time: {optimising_time}')
                pbar.update()

                epoch_train_loss += loss
                # epoch_val_loss += valloss
                # epoch_attack_val_loss += attack_valloss
            
            loss_training[0, epoch] = epoch_train_loss / (self.data.len // self.config.batch_size)
            # loss_training[1, epoch] = epoch_val_loss / (self.data.len // self.config.batch_size)
            # loss_training[2, epoch] = epoch_attack_val_loss / (self.data.len // self.config.batch_size)

        
        self.model.save(self.config.checkpoint_dir)

        with open(self.config.graph_file, 'wb') as f:
            np.save(f, loss_training)


    def attack_train(self):
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.config.learning_rate)

        loss_training = np.zeros((3,self.config.num_epochs))
        
        for epoch in range(self.config.num_epochs):

            pbar = tqdm.tqdm(total=self.data.len // self.config.batch_size, desc='Steps', position=0)
            train_status = tqdm.tqdm(total=0, bar_format='{desc}', position=1)

            epoch_train_loss, epoch_val_loss, epoch_attack_val_loss = 0, 0, 0
            
            for step in range(self.data.len // self.config.batch_size):
                if self.org == 1:
                    loss, optimising_time = self.train_step(epoch, optimizer)
                    perturbing_time = 'NA'
                else:
                    loss, perturbing_time, optimising_time = self.attack_train_step(epoch, optimizer, self.attack_iter, self.alpha, self.eps, self.asc, self.bta, smp=self.sample, std=self.std) 
                
                # valloss, attack_valloss = self.attack_val_step(epoch, self.attack_iter, self.alpha, self.eps, self.asc)
                # train_status.set_description_str(f'Epoch: {epoch} Step: {step} Train_Loss: {loss} Val_Loss: {valloss} Atk_Val_Loss: {attack_valloss} Optim_Time: {optimising_time}')
                train_status.set_description_str(f'Epoch: {epoch} Step: {step} Train_Loss: {loss} Optim_Time: {optimising_time}')
                pbar.update()

                epoch_train_loss += loss
                # epoch_val_loss += valloss
                # epoch_attack_val_loss += attack_valloss
            
            loss_training[0, epoch] = epoch_train_loss / (self.data.len // self.config.batch_size)
            # loss_training[1, epoch] = epoch_val_loss / (self.data.len // self.config.batch_size)
            # loss_training[2, epoch] = epoch_attack_val_loss / (self.data.len // self.config.batch_size)
        
        self.model.save(self.config.checkpoint_dir)
        # To save a different model/checkpoint at each epoch (will take up a lot more disk space!):
        # self.model.save(os.path.join(self.config.checkpoint_dir,str(epoch)+'.h5'))

        with open(self.config.graph_file, 'wb') as f:
            np.save(f, loss_training)


    def smooth_train(self): 
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.config.learning_rate)

        loss_training = np.zeros((3,self.config.num_epochs))
        pre_loss = 0
        
        for epoch in range(self.config.num_epochs):

            pbar = tqdm.tqdm(total=self.data.len // self.config.batch_size, desc='Steps', position=0)
            train_status = tqdm.tqdm(total=0, bar_format='{desc}', position=1)

            epoch_train_loss, epoch_val_loss, epoch_attack_val_loss = 0, 0, 0
                
            for step in range(self.data.len // self.config.batch_size):

                if epoch < self.warm_start_epoch:
                    loss, optimising_time = self.train_step(epoch, optimizer)
                else:
                    loss, optimising_time = self.smooth_train_step(epoch, optimizer, self.std, self.stdstp, self.sample, self.attack_iter, self.alpha, self.eps, self.sample_smt)
                
                # valloss = self.attack_val_step(epoch, self.attack_iter, self.alpha, self.eps, self.asc, atk_Ax=0, atk_x=0)
                # train_status.set_description_str(f'Epoch: {epoch} Step: {step} Train_Loss: {loss} Val_Loss: {valloss} Atk_Val_Loss: {attack_valloss} Optim_Time: {optimising_time}')
                train_status.set_description_str(f'Epoch: {epoch} Step: {step} Train_Loss: {loss} Optim_Time: {optimising_time}')
                pbar.update()

                epoch_train_loss += loss
                # epoch_val_loss += valloss

            
            loss_training[0, epoch] = epoch_train_loss / (self.data.len // self.config.batch_size)
            # loss_training[1, epoch] = epoch_val_loss / (self.data.len // self.config.batch_size)
        
        self.model.save(self.config.checkpoint_dir)

        with open(self.config.graph_file, 'wb') as f:
            np.save(f, loss_training)
