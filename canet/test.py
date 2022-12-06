import os, glob, random, torch, cv2
import numpy as np
from argparse import ArgumentParser
from model_test import CASNet
from utils import *
import scipy.stats as stats
import math
from torch.autograd import Variable
import pandas as pd
import pickle as pkl

parser = ArgumentParser(description='CASNet')
parser.add_argument('--epoch', type=int, default=100) #TODO
parser.add_argument('--phase_num', type=int, default=13)
parser.add_argument('--block_size', type=int, default=32)
parser.add_argument('--model_dir', type=str, default='model_new') #TODO
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--testset_name', type=str, default='BSD68')
parser.add_argument('--result_dir', type=str, default='test_out')
parser.add_argument('--gpu', type=str, default='0')

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
epoch = args.epoch
N_p = args.phase_num
B = args.block_size

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

gpu = args.gpu

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
device = torch.device('cuda', torch.cuda.current_device()) #cuda:0/1

# # fixed seed for reproduction
# seed = 0 #TODO 20220919
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

img_nf = 1  # image channel number
N = B * B
# cs_ratio_list = [0.01, 0.04, 0.10, 0.25, 0.30, 0.40, 0.50]  # ratios in [0, 1] are all available
cs_ratio_list = [0.10] #TODO changeable

# create and initialize CASNet
if attack:
    if sample:
        model_name = f'smtADV_itr{attack_iter}_alp{alpha}_eps{epsilon}_smp{sample}_std{std}'
    else:
        model_name = f'ADV_itr{attack_iter}_alp{alpha}_eps{epsilon}'
elif jacobian:
    model_name = f'JCBgma{gamma}'
elif smooth:
    if attack_iter:
        if Eextreme_sample:
            model_name = f'STHexatk_smp{robust_sample}_std{robust_noise}_stp{robust_noise_step}_itr{attack_iter}_alp{alpha}_eps{epsilon}_Ext_smp{Eextreme_sample}_std{std}'
        else:
            model_name = f'STHatk_smp{robust_sample}_std{robust_noise}_stp{robust_noise_step}_itr{attack_iter}_alp{alpha}_eps{epsilon}'
    else:
        model_name = f'STH_smp{robust_sample}_std{robust_noise}_stp{robust_noise_step}'
else:
    model_name = 'ordinary'
model = CASNet(N_p, B, img_nf, torch.zeros(N, N))
model = torch.nn.DataParallel(model).to(device)
model_dir = '%s/%s_layer_%d_block_%d' % (args.model_dir, model_name, N_p, B)
model.load_state_dict(torch.load('%s/net_params_%d.pkl' % (model_dir, epoch), map_location=device))

# test set info
test_image_paths = glob.glob(os.path.join(args.data_dir, args.testset_name) + '/*')
test_image_num = len(test_image_paths)

output_dir = os.path.join(args.result_dir, args.testset_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)



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



def test(cs_ratio, epoch_num, rand_modes):
    
    PSNR_All = np.zeros([3, test_image_num], dtype=np.float32)
    LOSS_All = np.zeros([3, test_image_num], dtype=np.float32)
    DIS_All = np.zeros([2, test_image_num], dtype=np.float32)

    ql, qu = 0, 0

    for img_no in range(test_image_num):
        image_path = test_image_paths[img_no]
        test_image = cv2.imread(test_image_paths[img_no], 1)  # read test data from image file
        test_image_ycrcb = cv2.cvtColor(test_image, cv2.COLOR_BGR2YCrCb)
        
        img, old_h, old_w, img_pad, new_h, new_w = my_zero_pad(test_image_ycrcb[:,:,0])
        img_pad = img_pad.reshape(1, 1, new_h, new_w) / 255.0  # normalization
        # print('debug!!!!!!!!!!!! img_no old_h old_w', img_no, old_h, old_w)
        
        x_input = torch.from_numpy(img_pad)
        x_input = x_input.type(torch.FloatTensor).to(device)

        img_dir = image_path.replace(args.data_dir, args.result_dir)[:-4] #test_out/BSD68/test049
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        model_type = model_name.split('_')[0]

        ### ordinary ###
        with torch.no_grad():
            x_output = model(x_input, int(np.ceil(cs_ratio * N)), rand_modes)
            ord_output = x_output.cpu().data.numpy().squeeze()
            ord_rec = np.clip(ord_output[:old_h, :old_w], 0, 1).astype(np.float64) * 255.0
            ord_psnr = psnr(ord_rec, img)
            ord_loss = np.mean(np.power(ord_rec-img, 2))
            # test_image_ycrcb[:, :, 0] = ord_rec
            # test_image_rgb = cv2.cvtColor(test_image_ycrcb, cv2.COLOR_YCrCb2BGR).astype(np.uint8)
            # cv2.imwrite('%s/%s_ratio_%.2f_psnr_%.2f.png' % (img_dir, model_type, cs_ratio, ord_psnr), test_image_rgb)
        
        ### attack ###
        if test_attack:
            model_dir = os.path.join(img_dir, model_type) #test_out/BSD68/test049/JCBgmma10
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            adv_img = Variable(x_input.clone().detach().type(torch.FloatTensor).to(device), requires_grad=True)
            optimizer = torch.optim.SGD([adv_img], lr=test_alpha)
            for i in range(test_attack_iter):
                x_output = model(adv_img, int(np.ceil(cs_ratio * N)), rand_modes) #TODO int(np.ceil(cs_ratio * N)) 
                loss_x = (x_output - x_input).pow(2).mean()
                optimizer.zero_grad()
                loss_x.backward(retain_graph=True)
                adv_img.data = adv_img.data + test_alpha * adv_img.grad
                with torch.no_grad():
                    diff_ori = adv_img - x_input #(1,1,new_h, new_w)
                    norm = torch.norm(diff_ori) #s
                    div = norm/test_epsilon if norm>test_epsilon else 1.
                    adv_img.data = diff_ori/div + x_input
            x_input = adv_img.clone().detach().type(torch.FloatTensor).to(device).requires_grad_(False)

            with torch.no_grad():
                if not test_smooth:
                    x_output = model(x_input, int(np.ceil(cs_ratio * N)), rand_modes)
                    atk_output = x_output.cpu().data.numpy().squeeze()
                    atk_rec = np.clip(atk_output[:old_h, :old_w], 0, 1).astype(np.float64) * 255.0
                    atk_psnr = psnr(atk_rec, img)
                    atk_loss = np.mean(np.power(atk_rec-img, 2))
                    tord_loss = np.mean(np.power(ord_rec-img, 2))
                    atk_dis = np.mean((atk_output - ord_output) ** 2)
                    # test_image_ycrcb[:, :, 0] = atk_rec
                    # test_image_rgb = cv2.cvtColor(test_image_ycrcb, cv2.COLOR_YCrCb2BGR).astype(np.uint8)
                    # cv2.imwrite('%s/Atk_ratio_%.2f_itr_%d_alp_%.1f_eps_%.2f_psnr_%.2f.png' % (model_dir, cs_ratio, test_attack_iter, test_alpha, test_epsilon, atk_psnr), test_image_rgb)
            
                else:
                    ### smoothing ###
                    ql, qu = estimate_ql_qu(test_epsilon, test_sample, test_std, conf_thres=.99999)
                    samples = []
                    for j in range(test_sample):
                        # print(f'sample {j}/{test_sample} started ...')
                        x_noise = torch.normal(0, std, size=x_input.shape).type(torch.FloatTensor).to(device) #(1, 1, new_h, new_w)
                        x_input_noisy = x_input + x_noise 
                        x_output = model(x_input_noisy, int(np.ceil(cs_ratio * N)), rand_modes)
                        samples.append( x_output.cpu().data.numpy().squeeze().reshape(-1) ) #(s, new_h*new_w)
                        del x_output
                    sorted_batch = np.sort( np.transpose(samples) ) #(new_h*new_w, s)
                    smt_output = np.squeeze( sorted_batch[:,int(sample/2)] ).reshape(new_h, new_w) #(new_h, new_w)
                    smt_rec = np.clip(smt_output[:old_h,:old_w], 0, 1).astype(np.float64) * 255.0
                    smt_psnr = psnr(smt_rec, img)
                    smt_loss = np.mean(np.power(smt_rec-img, 2))
                    smt_dis = np.mean(np.power(smt_output - ord_output, 2))
                    # test_image_ycrcb[:, :, 0] = smt_rec
                    # test_image_rgb = cv2.cvtColor(test_image_ycrcb, cv2.COLOR_YCrCb2BGR).astype(np.uint8)
                    # cv2.imwrite('%s/SmtAtk_ratio_%.2f_smp%d_std%.2f_itr_%d_alp_%.1f_eps_%.2f_psnr_%.2f.png' % (model_dir, cs_ratio, test_sample, test_std, test_attack_iter, test_alpha, test_epsilon, smt_psnr), test_image_rgb)
                    
                    if (ql!=None) and (qu!=None):
                        u_output = np.squeeze( sorted_batch[:,qu] ).reshape(new_h, new_w) #(new_h, new_w)
                        u_rec = np.clip(u_output[:old_h,:old_w], 0, 1).astype(np.float64) * 255.0
                        u_psnr = psnr(u_rec, img)
                        u_loss = np.mean(np.power(u_rec-img, 2))
                        u_dis = np.mean(np.power(u_output - ord_output, 2))

        PSNR_All[0, img_no] = ord_psnr
        LOSS_All[0, img_no] = ord_loss
        if test_attack:
            if not test_smooth:
                PSNR_All[1, img_no] = atk_psnr
                LOSS_All[1, img_no] = atk_loss
                DIS_All[0, img_no] = atk_dis
            else:
                PSNR_All[1, img_no] = smt_psnr
                LOSS_All[1, img_no] = smt_loss
                DIS_All[0, img_no] = smt_dis
                PSNR_All[2, img_no] = u_psnr
                LOSS_All[2, img_no] = u_loss
                DIS_All[1, img_no] = u_dis

        print(f'{img_no+1}/{test_image_num} imgs done.')

    if not os.path.exists(pickle_file_path):
        d = {'model_type':[model_type], 'test_set':[args.testset_name], 'cs_ratio':[cs_ratio],
            'atk_mode':[test_attack], 'atk_iter':[test_attack_iter], 'atk_alp':[test_alpha], 'atk_elp':[test_epsilon], 
            'smooth':[test_smooth], 'smooth_sample':[test_sample], 'smooth_std':[test_std],
            'ord_psnr':[np.mean(PSNR_All[0,:])], 'atk_psnr':[np.mean(PSNR_All[1,:])], 
            'ord_loss':[np.mean(LOSS_All[0,:])], 'atk_loss':[np.mean(LOSS_All[1,:])], 'ub_loss':[np.mean(LOSS_All[2,:])],
            'atk_dis':[np.mean(DIS_All[0,:])], 'ub_dis':[np.mean(DIS_All[1,:])],
            'ql':[ql], 'qu':[qu]}
        df = pd.DataFrame(data=d)
        df.to_pickle(pickle_file_path)
    else:
        d = {'model_type':model_type, 'test_set':args.testset_name, 'cs_ratio':cs_ratio,
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
    df.to_csv(f'casnet_{args.pkl}.csv', index=False)
    
test_time = 1
for cs_ratio in cs_ratio_list:
    for i in range(test_time):
        rand_modes = [random.randint(0, 7) for _ in range(N_p)]  # randomly choose a transformation for each phase
        test(cs_ratio, epoch, rand_modes)
