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
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from utils import imread_CS_py, img2col_py, col2im_CS_py, psnr, add_test_noise, write_data,get_cond
import math
from torch.utils.data import Dataset, DataLoader
import csdata_fast
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable




## -- Arguments -- ##
parser = ArgumentParser(description='ISTA-Net-plus')
parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=100, help='epoch number of end training')
parser.add_argument('--layer_num', type=int, default=20, help='phase number of ISTA-Net-plus')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--cs_ratio', type=int, default=50, help='from {1, 4, 10, 25, 40, 50}')
parser.add_argument('--gpu_idx', type=str, default='0', help='gpu index')
parser.add_argument('--data_dir', type=str, default='cs_train400_png', help='training data directory')
parser.add_argument('--rgb_range', type=int, default=1, help='value range 1 or 255')
parser.add_argument('--n_channels', type=int, default=1, help='1 for gray, 3 for color')
parser.add_argument('--patch_size', type=int, default=33, help='from {1, 4, 10, 25, 40, 50}')

parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model_new', help='trained or pre-trained model directory') #TODO !!!!
parser.add_argument('--data_dir_org', type=str, default='data', help='training data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--ext', type=str, default='.png', help='training data directory')

parser.add_argument('--result_dir', type=str, default='result', help='result directory')
parser.add_argument('--test_name', type=str, default='Set11', help='name of test set')
parser.add_argument('--test_cycle', type=int, default=10, help='epoch number of each test cycle')

parser.add_argument('--atk', type=int, default=0, help='attack indicator')
parser.add_argument('--alp', type=float, default=0., help='attack step size')
parser.add_argument('--eps', type=float, default=0., help='attack norm')
parser.add_argument('--itr', type=int, default=0, help='attack iterations')
parser.add_argument('--smp', type=int, default=0, help='smoothing samples')
parser.add_argument('--std', type=float, default=0., help='smoothing std')

parser.add_argument('--jcb', type=int, default=0, help='jacobian indicator')
parser.add_argument('--spc', type=int, default=0, help='spectral norm iterations')
parser.add_argument('--gma', type=int, default=0, help='spectral norm iterations')

parser.add_argument('--smt', type=int, default=0, help='smoothing indicator')
parser.add_argument('--stp', type=int, default=0, help='smoothing training stpes')
parser.add_argument('--ex_smp', type=int, default=0, help='smoothing training Extreme samples')



args = parser.parse_args()

start_epoch = args.start_epoch
end_epoch = args.end_epoch
learning_rate = args.learning_rate
layer_num = args.layer_num
group_num = args.group_num
cs_ratio = args.cs_ratio
gpu_idx = args.gpu_idx
test_name = args.test_name
test_cycle = args.test_cycle
test_dir = os.path.join(args.data_dir_org, test_name)
filepaths = glob.glob(test_dir + '/*.tif')
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






## -- Params -- ##
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
device = torch.device('cuda', torch.cuda.current_device()) #cuda:0
ratio_dict = {1: 10, 4: 43, 10: 109, 20: 218, 25: 272, 30: 327, 40: 436, 50: 545}
n_input = ratio_dict[cs_ratio]
n_output = 1089
batch_size = 64
Phi_input = None
total_phi_num = 50
rand_num = 1
train_cs_ratio_set = [10, 20, 30, 40, 50] #TODO
if attack:
    if sample:
        model_name = f'smtADV_itr{attack_iter}_alp{alpha}_eps{epsilon}_smp{sample}_std{std}_ISTA_Net_pp'
    else:
        model_name = f'ADV_itr{attack_iter}_alp{alpha}_eps{epsilon}_ISTA_Net_pp'
elif jacobian:
    model_name = f'JCB_gma{gamma}_ISTA_Net_pp'
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

for cs_ratio in train_cs_ratio_set:
    size_after_compress = ratio_dict[cs_ratio]
    Phi_all[cs_ratio] = np.zeros((int(rand_num * 1), size_after_compress, 1089))
    Phi_name = './%s/phi_sampling_%d_%dx%d.npy' % (args.matrix_dir, total_phi_num, size_after_compress, 1089)
    Phi_data = np.load(Phi_name)

    for k in range(rand_num):
        Phi_all[cs_ratio][k, :, :] = Phi_data[k, :, :]

Qinit = None










## -- helper functions -- ##
def random_u(b,c): 
    v = torch.normal(0, 1, size=(b,c)) 
    vnorm = torch.norm(v, 2, 1, True) 
    nomrlised_v = torch.div(v, vnorm) #(w,h)
    return nomrlised_v
    
def PhiTPhi_fun(x, PhiW, PhiTW):
    temp = F.conv2d(x, PhiW, padding=0,stride=33, bias=None)
    temp = F.conv2d(temp, PhiTW, padding=0, bias=None)
    return torch.nn.PixelShuffle(33)(temp)

class condition_network(nn.Module):
    def __init__(self,LayerNo):
        super(condition_network, self).__init__()

        self.fc1 = nn.Linear(1, 32, bias=True, device='cuda:0')
        self.fc2 = nn.Linear(32, 32, bias=True, device='cuda:0')
        self.fc3 = nn.Linear(32, LayerNo+LayerNo, bias=True, device='cuda:0')

        self.act12 = nn.ReLU(inplace=True)
        self.act3 = nn.Softplus()

    def forward(self, x):
        x = x[:,0:1]

        x = self.act12(self.fc1(x))
        x = self.act12(self.fc2(x))
        x = self.act3(self.fc3(x))
        num=x.shape[1]
        num=int(num/2)
        return x[0,0:num],x[0,num:]

class ResidualBlock_basic(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_basic, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True, device='cuda:0')
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True, device='cuda:0')
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

        self.head_conv = nn.Conv2d(2, 32, 3, 1, 1, bias=True, device='cuda:0')
        self.ResidualBlocks = nn.Sequential(
            ResidualBlock_basic(nf=32),
            ResidualBlock_basic(nf=32)
        )
        self.tail_conv = nn.Conv2d(32, 1, 3, 1, 1, bias=True, device='cuda:0')


    def forward(self, x,  PhiWeight, PhiTWeight, PhiTb,lambda_step,x_step):
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

class ISTA_Net_pp(torch.nn.Module):
    def __init__(self, LayerNo):
        super(ISTA_Net_pp, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock())

        self.fcs = nn.ModuleList(onelayer)
        self.condition = condition_network(LayerNo)
    
    def forward(self, x, Phi, Qinit, n_input, grad=None): 

        batchx = x[0]  #(64,1,33,33) --> (64,1089)=(b,n)
        cond = x[1]  
        lambda_step,x_step = self.condition(cond)
        
        PhiWeight = Phi.contiguous().view(n_input, 1, 33, 33) #(m,1,33,33) --> (m,1089)=(m,n)
        Phix = F.conv2d(batchx, PhiWeight, padding=0, stride=33, bias=None) #(64,m,1,1) --> (64,m)=(b,m)      
        if grad:
            Phix.requires_grad = True

        # Initialization-subnet
        PhiTWeight = Phi.t().contiguous().view(n_output, n_input, 1, 1) #(n,m,1,1)
        PhiTb = F.conv2d(Phix, PhiTWeight, padding=0, bias=None) #(64,n,1,1)
        PhiTb = torch.nn.PixelShuffle(33)(PhiTb)
        x = PhiTb   

        for i in range(self.LayerNo):
            x = self.fcs[i](x, PhiWeight, PhiTWeight, PhiTb,lambda_step[i],x_step[i])

        x_final = x

        if grad:
            return x_final, Phix
        else:
            return x_final

class spc_ISTA_Net_pp(torch.nn.Module):
    def __init__(self, LayerNo):
        super(spc_ISTA_Net_pp, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock())

        self.fcs = nn.ModuleList(onelayer)
        self.condition = condition_network(LayerNo)

    def sett(self, x, Phi, Qinit, n_input):
        batchx = x[0]  #(64,1,33,33) --> (64,1089)=(b,n)
        cond = x[1]  
        self.lambda_step, self.x_step = self.condition(cond)

        self.Phi = Phi
        self.n_input = Phi
        
        PhiWeight = Phi.contiguous().view(n_input, 1, 33, 33) #(m,1,33,33) --> (m,1089)=(m,n)
        Phix = F.conv2d(batchx, PhiWeight, padding=0, stride=33, bias=None) #(64,m,1,1) --> (64,m)=(b,m)
        Phix.requires_grad = True
        
        return Phix

    def forward(Phix): 
        # Initialization-subnet
        PhiTWeight = self.Phi.t().contiguous().view(n_output, self.n_input, 1, 1) #(n,m,1,1)
        PhiTb = F.conv2d(Phix, PhiTWeight, padding=0, bias=None) #(64,n,1,1)
        PhiTb = torch.nn.PixelShuffle(33)(PhiTb)
        x = PhiTb   

        for i in range(self.LayerNo):
            x = self.fcs[i](x, PhiWeight, PhiTWeight, PhiTb, self.lambda_step[i], self.x_step[i])

        x_final = x

        return x_final

class RandomDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length

    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :]).float()

    def __len__(self):
        return self.len









## -- model restoring -- ##
model = ISTA_Net_pp(layer_num)
model = nn.DataParallel(model)
model = model.to(device)
# if spectral:
#     spc_model = spc_ISTA_Net_pp(layer_num)
#     spc_model = nn.DataParallel(spc_model)
#     spc_model = spc_model.to(device)

print_flag = 0

if print_flag:
    num_count = 0
    for para in model.parameters():
        num_count += 1
        print('Layer %d' % num_count)
        print(para.size())

# training_data = csdata.SlowDataset(args)
training_data = csdata_fast.SlowDataset(args)


if (platform.system() =="Windows"):
    rand_loader = DataLoader(dataset=training_data, batch_size=batch_size, num_workers=0, shuffle=True)
else:
    rand_loader = DataLoader(dataset=training_data, batch_size=batch_size, num_workers=8, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model_dir = "./%s/%s_layer_%d_group_%d_ratio_all_lr_%.4f" % (args.model_dir, model_name, layer_num, group_num, learning_rate)
log_file_name = "./%s/%s_Log_layer_%d_group_%d_ratio_all_lr_%.4f.txt" % (args.log_dir, model_name, layer_num, group_num, learning_rate)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if start_epoch > 0:
    model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir, start_epoch)))
Phi = {}
for cs_ratio in train_cs_ratio_set:
    Phi[cs_ratio] = torch.from_numpy(Phi_all[cs_ratio]).type(torch.FloatTensor)
    Phi[cs_ratio] = Phi[cs_ratio].to(device)
cur_Phi = None  










## -- training -- ##
for epoch_i in range(start_epoch + 1, end_epoch + 1):
    
    pbar = tqdm(total=len(rand_loader), desc='data_points', position=0)
    train_status = tqdm(total=0, bar_format='{desc}', position=1)
    
    for idx, data in enumerate(rand_loader):
        start_time = time()

        #x
        batch_x = data.view(-1, 1, 33, 33) #(b,1,33,33)
        batch_x = batch_x.to(device)

        #A
        rand_Phi_index = np.random.randint(rand_num * 1)
        rand_cs_ratio = np.random.choice(train_cs_ratio_set)
        cur_Phi = Phi[rand_cs_ratio][rand_Phi_index] #(m,n)
        n_input = ratio_dict[rand_cs_ratio]
        lb, ub = batch_x-epsilon, batch_x+epsilon

        #smooth train 
        if smooth: 
            times = robust_noise_step + 1
            in_times = robust_sample
            for j in range(times):
                optimizer.zero_grad()

                if attack_iter:
                    batch_x = data.view(-1, 1, 33, 33) #(b,1,33,33)
                    batch_x = batch_x.to(device)
                    batch_x.requires_grad = True
                    opt_x = torch.optim.SGD([batch_x], lr=alpha)
                    for i in range(attack_iter):
                        #smooth
                        if Eextreme_sample:
                            for j in range(Eextreme_sample):
                                batch_n = torch.normal(0, std, size=batch_x.shape).type(torch.FloatTensor).to(device) #(1, 1, 33, 33)
                                batch_x_noisy = batch_x + batch_n 
                                x_input_x = [batch_x_noisy, get_cond(cs_ratio, 0.0, 'org_ratio')]
                                if j==0:
                                    x_output = model(x_input_x, cur_Phi, Qinit, n_input)
                                else:
                                    x_output = x_output + model(x_input_x, cur_Phi, Qinit, n_input)
                            x_output = x_output / Eextreme_sample
                        else:    
                            x_input_x = [batch_x, get_cond(cs_ratio, 0.0, 'org_ratio')]
                            x_output = model(x_input_x, cur_Phi, Qinit, n_input)
                        loss = -1. * torch.mean(torch.pow(x_output - batch_x, 2))
                        opt_x.zero_grad()
                        loss.backward(retain_graph=True)
                        opt_x.step()
                    batch_x = torch.clamp(batch_x, lb, ub) #TODO to include spe12
                    batch_x.detach()

                for k in range(in_times):
                    batch_n = torch.normal(0, (robust_noise/robust_noise_step)*j, size=batch_x.shape).type(torch.FloatTensor).to(device) #(1, 1, 33, 33)
                    batch_x_noisy = batch_x + batch_n 
                    x_input_x = [batch_x_noisy, get_cond(rand_cs_ratio, 0.0, 'org_ratio')]
                    x_output = model(x_input_x, cur_Phi, Qinit, n_input)
                    loss_discrepancy = torch.mean(torch.pow(x_output - batch_x, 2))
                    loss_all = loss_discrepancy / (times * in_times)
                    loss_all.backward(retain_graph=True)
                optimizer.step()

        else:
            #attack train
            if attack:
                # batch_x.requires_grad = True
                # opt_x = torch.optim.SGD([batch_x], lr=alpha)
                # for i in range(attack_iter):
                #     #smooth
                #     if sample:
                #         for j in range(sample):
                #             batch_n = torch.normal(0, std, size=batch_x.shape).type(torch.FloatTensor).to(device) #(1, 1, 33, 33)
                #             batch_x_noisy = batch_x + batch_n 
                #             x_input_x = [batch_x_noisy, get_cond(cs_ratio, 0.0, 'org_ratio')]
                #             if j==0:
                #                 x_output = model(x_input_x, cur_Phi, Qinit, n_input)
                #             else:
                #                 x_output = x_output + model(x_input_x, cur_Phi, Qinit, n_input)
                #         x_output = x_output / sample
                #     else:    
                #         x_input_x = [batch_x, get_cond(cs_ratio, 0.0, 'org_ratio')]
                #         x_output = model(x_input_x, cur_Phi, Qinit, n_input)
                #     loss = -1. * torch.mean(torch.pow(x_output - batch_x, 2))
                #     opt_x.zero_grad()
                #     loss.backward(retain_graph=True)
                #     opt_x.step()
                #     batch_x = torch.clamp(batch_x, lb, ub) #TODO to include spe12
                # # batch_x = torch.clamp(batch_x, lb, ub) #TODO to include spe12
                # batch_x.detach()
                            # batch_x.requires_grad = True
                adv_img = Variable(batch_x.clone().detach().type(torch.FloatTensor).to(device), requires_grad=True)
                optimizer = torch.optim.SGD([adv_img], lr=alpha)
                for i in range(attack_iter):
                    x_output = model([adv_img, get_cond(cs_ratio, 0.0, 'org_ratio')], cur_Phi, Qinit, n_input)
                    loss = torch.mean(torch.pow(x_output - batch_x, 2))
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    adv_img.data = adv_img.data + alpha * adv_img.grad
                    with torch.no_grad():
                        diff_ori = adv_img - batch_x
                        norm = torch.norm(diff_ori) #s
                        div = norm/epsilon if norm>epsilon else 1.
                        adv_img.data = diff_ori/div + batch_x
                batch_x = adv_img.clone().detach().type(torch.FloatTensor).to(device).requires_grad_(False)
                
            #y and recover \hat(x)
            x_input_x = [batch_x, get_cond(rand_cs_ratio, 0.0, 'org_ratio')]
            x_output = model(x_input_x, cur_Phi, Qinit, n_input)
            loss_discrepancy = torch.mean(torch.pow(x_output - batch_x, 2))
            loss_all = loss_discrepancy
            
            #jacobian train
            if jacobian:
                zA = random_u(batch_size,33*33).view(-1, 1, 33, 33).to(device) #(b,1,33,33)
                z = random_u(batch_size,33*33).view(-1, 1, 33, 33).to(device) #(b,1,33,33) 
                x_hat, y = model([batch_x, get_cond(cs_ratio, 0.0, 'org_ratio')], cur_Phi, Qinit, n_input, jacobian)
                autograd.backward(x_hat, zA, retain_graph=True)
                J = y.grad #(b,m,1,1)

                J_norm = 33*33 * torch.norm(J)**2 / (1*batch_size)
                beta_J = torch.pow( 10, torch.floor(torch.log(loss_discrepancy/J_norm)) ) / gamma
                J_loss = beta_J * J_norm

                JA = torch.mm(torch.squeeze(J), cur_Phi) #(b,n)
                JA_norm = 33*33 * torch.norm(JA)**2 / (1*batch_size) 
                beta_JA = torch.pow( 10, torch.floor(torch.log(loss_discrepancy/JA_norm)) ) / gamma
                JA_loss = beta_JA * JA_norm

                loss_all = loss_all + J_loss + JA_loss
                # y.detach() 

            if spectral: #TODO
                uA = random_u(batch_size,33*33).view(-1, 1, 33, 33).to(device) #(b,1,33,33)
                u = random_u(batch_size,33*33).view(-1, 1, 33, 33).to(device) #(b,1,33,33)
                
                # for _ in range(spectral):                 
                #     #JA:vjp
                #     x_hat, y = model([batch_x, get_cond(cs_ratio, 0.0, 'org_ratio')], cur_Phi, Qinit, n_input, spectral)
                #     autograd.backward(x_hat, uA, retain_graph=True)
                #     grad_vjp = y.grad #(b,m,1,1)
                #     vA = torch.mm(torch.t(cur_Phi), torch.t(torch.squeeze(grad_vjp))) #(n,b)
                #     AvA = torch.unsqueeze(torch.unsqueeze(torch.t(torch.mm(cur_Phi, vA)), -1), -1) #(b,m,1,1)

                #     #JA:jvp
                #     x_hat, y = model([batch_x, get_cond(cs_ratio, 0.0, 'org_ratio')], cur_Phi, Qinit, n_input, spectral)
                #     d_JA = random_u(batch_size,33*33).view(-1, 1, 33, 33).to(device)
                #     d_JA.requires_grad = True
                #     autograd.backward(x_hat, d_JA, retain_graph=True)
                #     gA = y.grad #(b,m,1,1)
                #     AvA = torch.unsqueeze(torch.unsqueeze(torch.t(torch.mm(cur_Phi, vA)), -1), -1) #(b,m,1,1)
                #     gA.backward()
                #     autograd.backward(gA, AvA, retain_graph=True)
                #     uA = d_JA.grad #(b,1,33,33)

                #     # #JA:jvp second try
                #     # y = spc_model.sett([batch_x, get_cond(cs_ratio, 0.0, 'org_ratio')], cur_Phi, Qinit, n_input)
                #     # jvp(spc_model, y, AvA)
                
                # uv_A_norm = torch.norm(uA, dim=1) / torch.norm(torch.t(vA), dim=1) #(b,)
                # JA_norm = torch.mean(uv_A_norm) 
                # beta_JA = torch.pow( 10, torch.floor(torch.log(loss_discrepancy/JA_norm)) ) / gamma
                # JA_loss = beta_JA * JA_norm
                # uv_norm = torch.norm(u, dim=1) / torch.norm(torch.t(v), dim=1) #(b,)
                # J_norm = tf.reduce_max(uv_norm) 
                # beta_J = torch.pow( 10, torch.floor(torch.log(loss_discrepancy/J_norm)) ) / gamma
                # J_loss = beta_J * J_norm
                # loss_all = loss_all + JA_loss + J_loss
                # y.detach()
                
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

        end_time = time()

        train_status.set_description_str(f'Epoch: {epoch_i}/{end_epoch} Data: {idx}/{len(rand_loader)} Train_Loss: {loss_all.item()} Optim_Time: {end_time - start_time}')
        pbar.update()

    # output_data = str(datetime.now()) + " [%d/%d] Total loss: %.4f, discrepancy loss: %.4f\n" % (epoch_i, end_epoch, loss_all.item(), loss_discrepancy.item())
    # print(output_data)

    # write_data(log_file_name, output_data)

    if epoch_i % 10 == 0:
        torch.save(model.state_dict(), "./%s/net_params_%d.pkl" % (model_dir, epoch_i))  # save only the parameters



