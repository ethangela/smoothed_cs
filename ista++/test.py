# -*- coding: utf-8 -*-
import os
from datetime import datetime

import cv2
import glob
import torch
import platform
import numpy as np
from tqdm import tqdm
from time import time
import torch.nn as nn
import scipy.io as sio
import scipy.stats as stats
import math
from argparse import ArgumentParser
from torch.utils.data import DataLoader
# from skimage.measure import compare_ssim as ssim
import pandas as pd
import math
import torch.nn.functional as F
from utils import RandomDataset, imread_CS_py, img2col_py, col2im_CS_py, psnr, add_test_noise, write_data,get_cond
import pickle as pkl
from torch.autograd import Variable



## -- Arguments -- ##
parser = ArgumentParser(description='ISTA-Net-plus')

parser.add_argument('--test_epoch', type=int, default=100, help='epoch number of start training')
parser.add_argument('--layer_num', type=int, default=20, help='phase number of ISTA-Net-plus')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--cs_ratio', type=int, default=10, help='from {1, 4, 10, 25, 40, 50}')

parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model_new', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--result_dir', type=str, default='result', help='result directory')
parser.add_argument('--test_name', type=str, default='Set11', help='name of test set')
parser.add_argument('--gpu', type=str, default='0', help='gpu index')
# parser.add_argument('--test_cycle', type=int, default=10, help='epoch number of each test cycle')

parser.add_argument('--atk', type=int, default=0, help='attack indicator')
parser.add_argument('--alp', type=float, default=0., help='attack step size')
parser.add_argument('--eps', type=float, default=0., help='attack norm')
parser.add_argument('--itr', type=int, default=0, help='attack iterations')

parser.add_argument('--jcb', type=int, default=0, help='jacobian indicator')
parser.add_argument('--spc', type=int, default=0, help='spectral norm iterations')
parser.add_argument('--gma', type=int, default=0, help='spectral norm iterations')

parser.add_argument('--smt', type=int, default=0, help='smoothing indicator')
parser.add_argument('--smp', type=int, default=0, help='smoothing samples')
parser.add_argument('--std', type=float, default=0., help='smoothing std')
parser.add_argument('--stp', type=int, default=0, help='smoothing training stpes')
parser.add_argument('--ex_smp', type=int, default=0, help='smoothing training Extreme samples')

parser.add_argument('--tatk', type=int, default=0, help='attack indicator')
parser.add_argument('--talp', type=float, default=0., help='attack step size')
parser.add_argument('--teps', type=float, default=0., help='attack norm')
parser.add_argument('--titr', type=int, default=0, help='attack iterations')
parser.add_argument('--tsmt', type=int, default=0, help='smoothing indicator')
parser.add_argument('--tsmp', type=int, default=0, help='smoothing samples')
parser.add_argument('--tstd', type=float, default=0., help='smoothing std')
parser.add_argument('--pkl', type=str, default='sep22', help='gpu index')

args = parser.parse_args()

test_epoch = args.test_epoch
# end_epoch = args.end_epoch
learning_rate = args.learning_rate
layer_num = args.layer_num
group_num = args.group_num
cs_ratio = args.cs_ratio
test_name = args.test_name
gpu = args.gpu
# test_cycle = args.test_cycle
test_dir = os.path.join(args.data_dir, test_name)
filepaths = glob.glob(test_dir + '/*.*')
result_dir = os.path.join(args.result_dir, test_name)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
ImgNum = len(filepaths)

attack = args.atk
epsilon = args.eps
alpha = args.alp
attack_iter = args.itr
smooth = args.smt
sample = args.smp
std = args.std

jacobian = args.jcb
spectral = args.spc
gamma = args.gma

smooth = args.smt
robust_sample = args.smp
robust_noise = args.std
robust_noise_step = args.stp
Eextreme_sample = args.ex_smp

test_attack = args.tatk
test_epsilon = args.teps
test_alpha = args.talp
test_attack_iter = args.titr
test_smooth = args.tsmt
test_sample = args.tsmp
test_std = args.tstd
pickle_file_path = '{}.pkl'.format(args.pkl)





## -- Params -- ##
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
device = torch.device('cuda', torch.cuda.current_device()) #cuda:0/1

ratio_dict =  {1: 10, 4: 43, 10: 109, 20: 218, 24:261, 25: 272, 27:294, 30: 327, 33:359, 36:392, 40: 436, 50: 545}
n_input = ratio_dict[cs_ratio]
n_output = 1089
batch_size = 64
total_phi_num = 50
rand_num = 1
test_cs_ratio_set = [cs_ratio] #TODO
test_sigma_set = [0.0] #TODO
if attack:
    if sample:
        model_name = f'smtADV_itr{attack_iter}_alp{alpha}_eps{epsilon}_smp{sample}_std{std}_ISTA_Net_pp'
    else:
        model_name = f'ADV_itr{attack_iter}_alp{alpha}_eps{epsilon}_ISTA_Net_pp'
elif jacobian:
    model_name = f'JCBgma{gamma}_ISTA_Net_pp'
elif smooth:
    if attack_iter:
        if Eextreme_sample:
            model_name = f'STHexatk_smp{robust_sample}_std{robust_noise}_stp{robust_noise_step}_itr{attack_iter}_alp{alpha}_eps{epsilon}_Ext_smp{Eextreme_sample}_std{std}_ISTA_Net_pp'
        else:
            model_name = f'STHatk_smp{robust_sample}_std{robust_noise}_stp{robust_noise_step}_itr{attack_iter}_alp{alpha}_eps{epsilon}_ISTA_Net_pp'
    else:
        model_name = f'STH_smp{robust_sample}_std{robust_noise}_stp{robust_noise_step}_ISTA_Net_pp'
else:
    model_name = 'ISTA_Net_pp'
Phi_all = {}
for cs_ratio in test_cs_ratio_set:
    size_after_compress = ratio_dict[cs_ratio]
    Phi_all[cs_ratio] = np.zeros((int(rand_num * 1), size_after_compress, 1089)) #(1,108,1089)
    Phi_name = './%s/phi_sampling_%d_%dx%d.npy' % (args.matrix_dir, total_phi_num, size_after_compress, 1089)
    Phi_data = np.load(Phi_name)
    for k in range(rand_num):
        Phi_all[cs_ratio][k, :, :] = Phi_data[k, :, :]
Qinit = None







## -- helper functions -- ##
def PhiTPhi_fun(x, PhiW, PhiTW):
    temp = F.conv2d(x, PhiW, padding=0,stride=33, bias=None)
    temp = F.conv2d(temp, PhiTW, padding=0, bias=None)
    return torch.nn.PixelShuffle(33)(temp)

class condition_network(nn.Module):
    def __init__(self):
        super(condition_network, self).__init__()

        self.fc1 = nn.Linear(1, 32, bias=True)
        self.fc2 = nn.Linear(32, 32, bias=True)
        self.fc3 = nn.Linear(32, 40, bias=True)

        self.act12 = nn.ReLU(inplace=True)
        self.act3 = nn.Softplus()

    def forward(self, x):
        x=x[:,0:1]
        x = self.act12(self.fc1(x))
        x = self.act12(self.fc2(x))
        x = self.act3(self.fc3(x))

        return x[0,0:20],x[0,20:40]


class ResidualBlock_basic(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_basic, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        cond = x[1]
        content = x[0]

        out = self.act(self.conv1(content))
        out = self.conv2(out)
        return content + out, cond


class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()
        self.head_conv = nn.Conv2d(2, 32, 3, 1, 1, bias=True)
        self.ResidualBlocks = nn.Sequential(
            ResidualBlock_basic(nf=32),
            ResidualBlock_basic(nf=32)
        )
        self.tail_conv = nn.Conv2d(32, 1, 3, 1, 1, bias=True)

    def forward(self, x, PhiWeight, PhiTWeight, PhiTb,lambda_step,x_step):
        x = x - lambda_step * PhiTPhi_fun(x, PhiWeight, PhiTWeight)
        x = x + lambda_step * PhiTb

        x_input = x
        sigma= x_step.repeat(x_input.shape[0], 1, x_input.shape[2], x_input.shape[3])
        x_input_cat = torch.cat((x_input,sigma),1)
        x_mid = self.head_conv(x_input_cat)
        cond = None

        x_mid, cond = self.ResidualBlocks([x_mid, cond])
        x_mid = self.tail_conv(x_mid)

        x_pred = x_input + x_mid
        return x_pred


class ISTA_Net_pp(torch.nn.Module): #ISTA-Net-pp
    def __init__(self, LayerNo):
        super(ISTA_Net_pp, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock())

        self.fcs = nn.ModuleList(onelayer)
        self.condition = condition_network()

    def forward(self, x, Phi, Qinit,n_input): 

        batch_x = x[0]  
        cond = x[1]  
        lambda_step,x_step = self.condition(cond)
        
        PhiWeight = Phi.contiguous().view(-1, 1, 33, 33)
        Phix = F.conv2d(batch_x, PhiWeight, padding=0, stride=33, bias=None)    # Get measurements

        # Initialization-subnet
        PhiTWeight = Phi.t().contiguous().view(n_output, n_input, 1, 1)
        PhiTb = F.conv2d(Phix, PhiTWeight, padding=0, bias=None)
        PhiTb = torch.nn.PixelShuffle(33)(PhiTb)
        x = PhiTb    # Conduct initialization

        for i in range(self.LayerNo):
            x = self.fcs[i](x, PhiWeight, PhiTWeight, PhiTb,lambda_step[i],x_step[i])

        x_final = x

        return x_final


def estimate_ql_qu(eps, sample_count, sigma, conf_thres=.99999):
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





## -- model restoring -- ##
model = ISTA_Net_pp(layer_num)
model = nn.DataParallel(model)
model = model.to(device)

print_flag = 0
if print_flag:
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model_dir = "./%s/%s_layer_%d_group_%d_ratio_all_lr_%.4f" % (args.model_dir, model_name, layer_num, group_num, learning_rate)
log_file_name = "./%s/%s_Log_testset_%s_layer_%d_group_%d_ratio_%d_lr_%.4f.txt" % (args.log_dir, model_name, args.test_name,layer_num, group_num, cs_ratio, learning_rate)
# if not os.path.exists(model_dir):
#     os.makedirs(model_dir)
model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir, test_epoch)))

Phi = {}
for cs_ratio in test_cs_ratio_set:
    Phi[cs_ratio] = torch.from_numpy(Phi_all[cs_ratio]).type(torch.FloatTensor) #(1,108,1089)
    Phi[cs_ratio] = Phi[cs_ratio].to(device)
cur_Phi = None 





def img_save(yuv, img):
    yuv[:,:,0] = img*255
    im_rec_rgb = cv2.cvtColor(yuv, cv2.COLOR_YCrCb2BGR)
    im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)
    return im_rec_rgb



## -- testing -- ##
def test_model(epoch_num, cs_ratio, sigma, model_name):
    PSNR_All = np.zeros([3, ImgNum], dtype=np.float32)
    LOSS_All = np.zeros([3, ImgNum], dtype=np.float32)
    DIS_All = np.zeros([2, ImgNum], dtype=np.float32)
    # SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)
    # COST_TIME_All = np.zeros([1, ImgNum], dtype=np.float32)

    rand_Phi_index = 0
    cur_Phi = Phi[cs_ratio][rand_Phi_index] #(108,1089)
    # print("(Test)CS reconstruction start, using Phi[%d][%d] to test" % (cs_ratio, rand_Phi_index))

    # for img_no in tqdm(range(ImgNum)):
    for img_no in range(ImgNum):
        imgName = filepaths[img_no]

        Img = cv2.imread(imgName, 1)

        Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
        Img_rec_yuv = Img_yuv.copy()

        Iorg_y = Img_yuv[:, :, 0] #(256,256)

        [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(Iorg_y) #padding
        Img_output = Ipad.reshape(1, 1, row_new, col_new)/255.0 #(1, 1, 264, 264) or (1, 1, 568, 568)
        # print('debug!!!!!!!!!!!!!!!! img, row, col', img_no, row, col)
        
        batch_x = torch.from_numpy(Img_output)
        batch_x = batch_x.type(torch.FloatTensor)
        batch_x = batch_x.to(device)
        n_input = ratio_dict[cs_ratio] #compress size
            
        ### normal test ###
        start_time = time()
        with torch.no_grad():
            ### ordinary ###
            x_input_x = [batch_x, get_cond(cs_ratio, sigma, 'org_ratio')]
            x_output = model(x_input_x, cur_Phi, Qinit, n_input)
            ord_output = x_output.cpu().data.numpy().squeeze()
            ord_rec = np.clip(ord_output[:row,:col], 0, 1).astype(np.float64)
            ord_PSNR, ord_loss = psnr(ord_rec*255, Iorg.astype(np.float64), row, col)
            del x_output

        ### attack ###
        ql, qu = 0, 0
        if test_attack:
            # batch_x.requires_grad = True
            # optimizer = torch.optim.SGD([batch_x], lr=test_alpha)
            # for i in range(test_attack_iter):
            #     x_input_x = [batch_x, get_cond(cs_ratio, sigma, 'org_ratio')]
            #     x_output = model(x_input_x, cur_Phi, Qinit, n_input)
            #     loss = -1. * torch.mean(torch.pow(x_output - batch_x, 2))
            #     optimizer.zero_grad()
            #     loss.backward(retain_graph=True)
            #     optimizer.step()
            #     del x_output
            #     norm = torch.norm(batch_x-torch.from_numpy(Img_output)) #s
            #     if norm
            #     div = torch.where(norm > epsilon, norm/epsilon, torch.ones_like(norm)) 
            #     x_input = (batch_x-torch.from_numpy(Img_output)) / tf.expand_dims(div, axis=-1) + targets #1,1,264,264
            # # batch_x = torch.clamp(batch_x, lb, ub) 
            adv_img = Variable(batch_x.clone().detach().type(torch.FloatTensor).to(device), requires_grad=True)
            optimizer = torch.optim.SGD([adv_img], lr=test_alpha)
            for i in range(test_attack_iter):
                x_output = model([adv_img, get_cond(cs_ratio, sigma, 'org_ratio')], cur_Phi, Qinit, n_input)
                loss = torch.mean(torch.pow(x_output - batch_x, 2))
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                adv_img.data = adv_img.data + test_alpha * adv_img.grad
                with torch.no_grad():
                    diff_ori = adv_img - batch_x
                    norm = torch.norm(diff_ori) #s
                    div = norm/test_epsilon if norm>test_epsilon else 1.
                    adv_img.data = diff_ori/div + batch_x
            batch_x = adv_img.clone().detach().type(torch.FloatTensor).to(device).requires_grad_(False)

            with torch.no_grad():
                if not test_smooth:
                    x_input_x = [batch_x, get_cond(cs_ratio, sigma, 'org_ratio')]
                    x_output = model(x_input_x, cur_Phi, Qinit, n_input)
                    atk_output = x_output.cpu().data.numpy().squeeze()
                    atk_rec = np.clip(atk_output[:row,:col], 0, 1).astype(np.float64)
                    atk_PSNR, atk_loss = psnr(atk_rec*255, Iorg.astype(np.float64), row, col)
                    atk_dis = np.mean((atk_output - ord_output) ** 2) / (row*col)
                    del x_output
                else:
                    ### smoothing ### 
                    ql, qu = estimate_ql_qu(test_epsilon, test_sample, test_std, conf_thres=.99999)
                    samples = []
                    for j in range(test_sample):
                        # print(f'sample {j}/{sample} started ...')
                        batch_n = torch.normal(0, test_std, size=batch_x.shape).type(torch.FloatTensor).to(device) #(1, 1, 264, 264)
                        batch_x_noisy = batch_x + batch_n 
                        x_input_x = [batch_x_noisy, get_cond(cs_ratio, sigma, 'org_ratio')]
                        x_output = model(x_input_x, cur_Phi, Qinit, n_input)
                        samples.append( x_output.cpu().data.numpy().squeeze().reshape(-1) ) #(s, row_new*col_new)
                        del x_output
                    sorted_batch = np.sort( np.transpose(samples) ) #(row_new*col_new, s)
                    smt_output = np.squeeze( sorted_batch[:,int(test_sample/2)] ).reshape(row_new, col_new) #(row_new, col_new)
                    smt_rec = np.clip(smt_output[:row,:col], 0, 1).astype(np.float64)
                    smt_dis = np.mean((smt_output - ord_output) ** 2) / (row*col)
                    smt_PSNR, smt_loss = psnr(smt_rec*255, Iorg.astype(np.float64), row, col)
                    if (ql!=None) and (qu!=None):                    
                        u_output = np.squeeze( sorted_batch[:,qu] ).reshape(row_new, col_new) #(row_new, col_new)
                        u_rec = np.clip(u_output[:row,:col], 0, 1).astype(np.float64)
                        u_dis = np.mean(np.power(u_output - ord_output, 2)) / (row*col)
                        u_PSNR, u_loss = psnr(u_rec*255, Iorg.astype(np.float64), row, col)

        end_time = time()

        PSNR_All[0, img_no] = ord_PSNR
        LOSS_All[0, img_no] = ord_loss
        if test_attack:
            if not test_smooth:
                PSNR_All[1, img_no] = atk_PSNR
                LOSS_All[1, img_no] = atk_loss
                DIS_All[0, img_no] = atk_dis
            else:
                PSNR_All[1, img_no] = smt_PSNR
                LOSS_All[1, img_no] = smt_loss
                DIS_All[0, img_no] = smt_dis
                PSNR_All[2, img_no] = u_PSNR
                LOSS_All[2, img_no] = u_loss
                DIS_All[1, img_no] = u_dis
        
        # COST_TIME_All[0, img_no] = end_time - start_time

        img_dir = imgName.replace(args.data_dir, args.result_dir) #result/set11/Monarch.tif
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        model_type = model_name.split('_')[0]
        cv2.imwrite("%s/%s_PSNR_%.2f.png" % (img_dir, model_type, ord_PSNR), img_save(Img_rec_yuv, ord_rec))
        if test_attack:
            model_dir = os.path.join(img_dir, model_type) #result/set11/Monarch.tif/JCBgmma10
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            if not test_smooth:
                cv2.imwrite("%s/ATK_itr_%d_alp_%.1f_eps_%.2f_SMT_smp_%d_std_%.2f_PSNR_%.2f.png" % (model_dir, test_attack_iter, test_alpha, test_epsilon, test_sample, test_std, atk_PSNR), img_save(Img_rec_yuv, atk_rec))
            else:
                cv2.imwrite("%s/ATKSMT_itr_%d_alp_%.1f_eps_%.2f_smp_%d_std_%.2f_PSNR_%.2f.png" % (model_dir, test_attack_iter, test_alpha, test_epsilon, test_sample, test_std, smt_PSNR), img_save(Img_rec_yuv, smt_rec))
    
        print(f'{img_no+1}/{ImgNum} imgs done.')
                
    # print_data = str(datetime.now()) + " atk_itr:%d, atk_alp:%.3f, atk_eps:%.3f, sth_smp:%d, sth_std:%.3f, avg PSNR:%.2f, avg time:%.4f \n" % (attack_iter, alpha, epsilon, sample, std, np.mean(PSNR_All), np.mean(COST_TIME_All))
    # print(print_data)
    # output_file_name = "%s/%s_Results_testset_%s_layer_%d_group_%d_ratio_%d_sigma_%d_lr_%.4f.txt" % (args.log_dir, model_name, args.test_name, layer_num, group_num, cs_ratio, sigma, learning_rate)
    # output_data = "train_epochs: %d, atk_itr:%d, atk_alp:%.3f, atk_eps:%.3f, sth_smp:%d, sth_std:%.3f, avg_PSNR:%.2f, avg_time:%.4f\n" % (epoch_num, attack_iter, alpha, epsilon, sample, std, np.mean(PSNR_All), np.mean(COST_TIME_All))
    # write_data(output_file_name, output_data)  

    if not os.path.exists(pickle_file_path):
        d = {'model_type':[model_type], 'test_set':[args.test_name], 'cs_ratio':[cs_ratio],
            'atk_mode':[test_attack], 'atk_iter':[test_attack_iter], 'atk_alp':[test_alpha], 'atk_elp':[test_epsilon], 
            'smooth':[test_smooth], 'smooth_sample':[test_sample], 'smooth_std':[test_std],
            'ord_psnr':[np.mean(PSNR_All[0,:])], 'atk_psnr':[np.mean(PSNR_All[1,:])], 
            'ord_loss':[np.mean(LOSS_All[0,:])], 'atk_loss':[np.mean(LOSS_All[1,:])], 'ub_loss':[np.mean(LOSS_All[2,:])],
            'atk_dis':[np.mean(DIS_All[0,:])], 'ub_dis':[np.mean(DIS_All[1,:])],
            'ql':[ql], 'qu':[qu]}
        df = pd.DataFrame(data=d)
        df.to_pickle(pickle_file_path)
    else:
        d = {'model_type':model_type, 'test_set':args.test_name, 'cs_ratio':cs_ratio,
            'atk_mode':test_attack, 'atk_iter':test_attack_iter, 'atk_alp':test_alpha, 'atk_elp':test_epsilon, 
            'smooth':test_smooth, 'smooth_sample':test_sample, 'smooth_std':test_std,
            'ord_psnr':np.mean(PSNR_All[0,:]), 'atk_psnr':np.mean(PSNR_All[1,:]), 
            'ord_loss':np.mean(LOSS_All[0,:]), 'atk_loss':np.mean(LOSS_All[1,:]), 'ub_loss':np.mean(LOSS_All[2,:]),
            'atk_dis':np.mean(DIS_All[0,:]), 'ub_dis':np.mean(DIS_All[1,:]),
            'ql':ql, 'qu':qu}
        df = pd.read_pickle(pickle_file_path)
        df = df.append(d, ignore_index=True)
        df.to_pickle(pickle_file_path)

    with open(pickle_file_path, "rb") as f:
        object = pkl.load(f)
    df = pd.DataFrame(object)
    df.to_csv(f'test_{args.pkl}.csv', index=False)

for cs_ratio in test_cs_ratio_set:
    for test_sigma in test_sigma_set:
        test_model(test_epoch, cs_ratio, test_sigma, model_name)
