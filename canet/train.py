import torch
import scipy.io as sio
import numpy as np
import os, glob, cv2, random
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser
from model_train import CASNet
from utils import *
# from skimage.metrics import structural_similarity as ssim
from time import time
from tqdm import tqdm

parser = ArgumentParser(description='CASNet')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--end_epoch', type=int, default=100) #TODO #320
parser.add_argument('--phase_num', type=int, default=13)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--block_size', type=int, default=32)
parser.add_argument('--model_dir', type=str, default='model_new') #TODO
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--log_dir', type=str, default='log')
parser.add_argument('--save_interval', type=int, default=20)
parser.add_argument('--testset_name', type=str, default='Set11')
parser.add_argument('--gpu_idx', type=str, default='0')

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

start_epoch, end_epoch = args.start_epoch, args.end_epoch
learning_rate = args.learning_rate
N_p = args.phase_num
B = args.block_size
gpu_idx = args.gpu_idx

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
device = torch.device('cuda', torch.cuda.current_device()) #cuda:0/1
torch.backends.cudnn.benchmark = True

# fixed seed for reproduction
# seed = 0 #TODO
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

img_nf = 1  # image channel number
patch_size = 128  # training patch size
patch_number = 25600   # number of training patches
batch_size = 16
N = B * B
d = patch_size // B
l = d * d
cs_ratio_list = [0.01, 0.04, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 1.00]    # ratios in [0, 1] are all available

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

training_set_name = 'Training_Data_size128_CS_T91andTrain400.mat'
training_set = sio.loadmat('%s/%s' % (args.data_dir, training_set_name))['labels']

# SVD-based initialization scheme, X = US(V^T)
x = torch.Tensor(training_set).view(patch_number, img_nf, patch_size, patch_size)
x = x.reshape(patch_number, img_nf, d, B, d, B).permute(0, 1, 3, 5, 2, 4)
x = x.reshape(patch_number, img_nf * N, l).permute(0, 2, 1)
x = x.reshape(patch_number * l, img_nf * N)
x_eig_values, x_eig_vectors = torch.linalg.eig(x.t().mm(x))
x_eig_values, x_eig_vectors = x_eig_values.real, x_eig_vectors.real
descending_k_indices = x_eig_values.sort(descending=True)[1]
Phi_init = x_eig_vectors.t()[descending_k_indices]

model = CASNet(N_p, B, img_nf, Phi_init)
model = torch.nn.DataParallel(model).to(device)

class MyDataset(Dataset):
    def __init__(self, data, length):
        self.data = torch.Tensor(data).float() / 255.0
        self.len = length

    def __getitem__(self, index):
        return self.data[index, :]

    def __len__(self):
        return self.len

my_loader = DataLoader(dataset=MyDataset(training_set, patch_number), batch_size=batch_size, num_workers=8, shuffle=True, pin_memory=True)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300], gamma=0.1, last_epoch=start_epoch)

if attack:
    if sample:
        model_name = f'smtADV_itr{attack_iter}_alp{alpha}_eps{epsilon}_smp{sample}_std{std}'
    else:
        model_name = f'ADV_itr{attack_iter}_alp{alpha}_eps{epsilon}'
elif jacobian:
    model_name = f'JCB_gma{gamma}'
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

model_dir = '%s/%s_layer_%d_block_%d' % (args.model_dir, model_name, N_p, B)
log_path = '%s/%s_layer_%d_block_%d.txt' % (args.log_dir, model_name, N_p, B)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
    
# test set info
test_image_paths = glob.glob(os.path.join(args.data_dir, args.testset_name) + '/*')
test_image_num = len(test_image_paths)

def test(cs_ratio, epoch_num, rand_modes):
    with torch.no_grad():
        PSNR_list = []
        for i in range(test_image_num):
            test_image = cv2.imread(test_image_paths[i], 1)  # read test data from image file
            test_image_ycrcb = cv2.cvtColor(test_image, cv2.COLOR_BGR2YCrCb)
            
            img, old_h, old_w, img_pad, new_h, new_w = my_zero_pad(test_image_ycrcb[:,:,0])
            img_pad = img_pad.reshape(1, 1, new_h, new_w) / 255.0  # normalization
            
            x_input = torch.from_numpy(img_pad)
            x_input = x_input.type(torch.FloatTensor).to(device)

            x_output = model(x_input, int(np.ceil(cs_ratio * N)), rand_modes)
            x_output = x_output.cpu().data.numpy().squeeze()
            x_output = np.clip(x_output[:old_h, :old_w], 0, 1).astype(np.float64) * 255.0

            PSNR = psnr(x_output, img)

            PSNR_list.append(PSNR)

    return float(np.mean(PSNR_list))

if start_epoch > 0:
    model.load_state_dict(torch.load('%s/net_params_%d.pkl' % (model_dir, start_epoch)))

print('start training...')
for epoch_i in range(start_epoch + 1, end_epoch + 1):

    pbar = tqdm(total=len(my_loader), desc='data_points', position=0)
    train_status = tqdm(total=0, bar_format='{desc}', position=1)

    loss_avg, iter_num = 0.0, 0
    for idx, data in enumerate(my_loader):
        start_time = time()

        x_input = data.view(-1, img_nf, patch_size, patch_size).to(device) #(b, 1, 128, 128)
        lb, ub = x_input-epsilon, x_input+epsilon
        
        if smooth: 
            #smooth train
            times = robust_noise_step + 1
            in_times = robust_sample
            for j in range(1,times):
                optimizer.zero_grad(set_to_none=True) #TODO
                if attack_iter:
                    x_input = data.view(-1, img_nf, patch_size, patch_size).to(device) #(b, 1, 128, 128)
                    x_input.requires_grad = True
                    opt_x = torch.optim.SGD([x_input], lr=alpha)
                    for i in range(attack_iter):   
                        q = random.randint(1, N)  # target average block measurement size, corresponding CS ratio is q/N
                        rand_modes = [random.randint(0, 7) for _ in range(N_p)]
                        x_output = model(x_input, q, rand_modes)
                        loss_x = -1. * torch.mean(torch.pow(x_output - x_input, 2))
                        opt_x.zero_grad()
                        loss_x.backward(retain_graph=True)
                        opt_x.step()
                    x_input = torch.clamp(x_input, lb, ub) #TODO to include spe12
                    x_input.detach()
                for k in range(in_times):
                    batch_n = torch.normal(0, (robust_noise/robust_noise_step)*j, size=x_input.shape).type(torch.FloatTensor).to(device) #(b, 1, 128, 128)
                    x_input_noisy = x_input + batch_n 
                    q = random.randint(1, N)  # target average block measurement size, corresponding CS ratio is q/N
                    rand_modes = [random.randint(0, 7) for _ in range(N_p)]
                    x_output = model(x_input_noisy, q, rand_modes)
                    loss = (x_output - x_input).abs().mean()  # L1 loss
                    #loss = (x_output - x_input).pow(2).mean()  # L2 loss
                    loss = loss / (times * in_times)
                    loss.backward(retain_graph=True)
                optimizer.step()

        else:
            #attack train
            if attack:
                x_input.requires_grad = True
                opt_x = torch.optim.SGD([x_input], lr=alpha)
                for i in range(attack_iter):
                    #smooth
                    if sample:
                        x_lis = []
                        loss_x_lis = []
                        for j in range(sample):
                            batch_n = torch.normal(0, std, size=x_input.shape).type(torch.FloatTensor).to(device) #(b, 1, 128, 128)
                            x_input_noisy = x_input + batch_n 
                            q = random.randint(1, N)  # target average block measurement size, corresponding CS ratio is q/N
                            rand_modes = [random.randint(0, 7) for _ in range(N_p)] #TODO
                            x_output = model(x_input_noisy, q, rand_modes)
                            loss_x = -1. * torch.mean(torch.pow(x_output - x_input, 2)) / sample
                            opt_x.zero_grad()
                            loss_x.backward(retain_graph=True)
                            opt_x.step()
                    else:    
                        q = random.randint(1, N)  # target average block measurement size, corresponding CS ratio is q/N
                        rand_modes = [random.randint(0, 7) for _ in range(N_p)]
                        x_output = model(x_input, q, rand_modes)
                        loss_x = -1. * torch.mean(torch.pow(x_output - x_input, 2))
                        opt_x.zero_grad()
                        loss_x.backward(retain_graph=True)
                        opt_x.step()
                x_input = torch.clamp(x_input, lb, ub) #TODO to include spe12
                x_input.detach()

            #ordinary train
            q = random.randint(1, N)  # target average block measurement size, corresponding CS ratio is q/N
            rand_modes = [random.randint(0, 7) for _ in range(N_p)]
            x_output = model(x_input, q, rand_modes)
            # loss = (x_output - x_input).abs().mean()  # L1 loss #TODO
            loss = (x_output - x_input).pow(2).mean()  # L2 loss

            # zero gradients, perform a backward pass, and update the weights
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        
        iter_num += 1
        loss_avg += loss.item()

        end_time = time()

        train_status.set_description_str(f'Epoch: {epoch_i}/{end_epoch} Data: {idx}/{len(my_loader)} Train_Loss: {loss.item()} Optim_Time: {end_time - start_time}')
        pbar.update()
    
    # scheduler.step()
    
    loss_avg /= iter_num
    log_data = '[%d/%d] Average loss: %.4f, time cost: %.2fs.' % (epoch_i, end_epoch, loss_avg, time() - start_time)
    with open(log_path, 'a') as log_file:
        log_file.write(log_data + '\n') 

    if epoch_i % args.save_interval == 0:
        torch.save(model.state_dict(), '%s/net_params_%d.pkl' % (model_dir, epoch_i))  # save only the parameters

    # for cs_ratio in cs_ratio_list:  # test at the end of each epoch
    #     rand_modes = [random.randint(0, 7) for _ in range(N_p)]
    #     cur_psnr = test(cs_ratio, epoch_i, rand_modes)
    #     log_data = 'CS ratio is %.2f, PSNR is %.2f.' % (cs_ratio, cur_psnr)
    #     print(log_data)
    #     with open(log_path, 'a') as log_file:
    #         log_file.write(log_data + '\n')
