import os, glob, random, cv2
import numpy as np
from argparse import ArgumentParser
from model_test import CASNet
from utils import *


os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

import torch
print('torch.cuda.current_device():', torch.cuda.current_device())
device = torch.device('cuda', torch.cuda.current_device())
print(device)

