import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
import utils

from natsort import natsorted

from glob import glob
from skimage import img_as_ubyte
from skimage import metrics

from basicsr.models import create_model
from basicsr.utils.options import dict2str, parse


parser = argparse.ArgumentParser(
    description='Evaluation')

parser.add_argument('--output_dir', default='./results/LOL-Blur',
                    type=str, help='Directory for output')
parser.add_argument(
    '--opt', type=str, default='./Options/LOL-Blur-T.yml', help='Path to option YAML file.')
parser.add_argument('--weights', default='./pretrained_models/VIVNet-LOLBlur-T.pth',
                    type=str, help='Path to weights')
parser.add_argument('--input_dir', default='./Datasets/', type=str, help='Directory of validation images')
parser.add_argument('--data', default='LOL-Blur', type=str, help='Dataset')

parser.add_argument('--gpus', type=str, default="0", help='GPU devices.')
parser.add_argument('--save_img', default=True, help='Use self-ensemble to obtain better results')

args = parser.parse_args()


weights = args.weights

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

opt = parse(args.opt, is_train=False)
opt['dist'] = False

x = yaml.load(open(args.opt, mode='r'), Loader=Loader)
s = x['network_g'].pop('type')
##########################


model_restoration = create_model(opt).net_g
checkpoint = torch.load(weights)

try:
    model_restoration.load_state_dict(checkpoint['params'])
except:
    new_checkpoint = {}
    for k in checkpoint['params']:
        new_checkpoint['module.' + k] = checkpoint['params'][k]
    model_restoration.load_state_dict(new_checkpoint)

print("===>Testing using weights: ", weights)
model_restoration.cuda()
model_restoration.eval()

factor = 32

output_dir = args.output_dir

if args.output_dir != '':
    os.makedirs(output_dir, exist_ok=True)

input_dir = os.path.join(args.input_dir, args.data, 'Test', 'low_blur_noise_all')
target_dir = os.path.join(args.input_dir, args.data, 'Test', 'high_sharp_scaled_all')

input_paths = natsorted(
    glob(os.path.join(input_dir, '*.png')) + glob(os.path.join(input_dir, '*.jpg')) + glob(os.path.join(input_dir, '*.tif')))

with torch.inference_mode():
    for inp_path in tqdm(input_paths, total=len(input_paths)):

        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        img = np.float32(utils.load_img(inp_path)) / 255.

        img = torch.from_numpy(img).permute(2, 0, 1)
        input_ = img.unsqueeze(0).cuda()

        # Padding in case images are not multiples of 32
        b, c, h, w = input_.shape
        H, W = ((h + factor) // factor) * \
            factor, ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

        restored = model_restoration(input_)
        restored = restored[:, :, :h, :w]

        restored = torch.clamp(restored, 0, 1).cpu(
        ).detach().permute(0, 2, 3, 1).squeeze(0).numpy()

        if args.save_img:
            utils.save_img((os.path.join(output_dir, os.path.splitext(
                os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(restored))
