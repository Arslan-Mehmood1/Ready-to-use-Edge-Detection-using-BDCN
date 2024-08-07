import os.path as osp
import re
import os
import sys
import cv2
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
from .bdcn import BDCN
import torch.optim as optim
from scipy.io import savemat
from torch.autograd import Variable
from torch.nn import functional as F
from matplotlib import pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

EDGE_MODEL_OUTPUT_DIR = 'bdcn_output'

def make_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def init_edge_detection_model(checkpoint_path:str='checkpoint/bdcn_pretrained_on_bsds500.pth',rate:int=4):        
    edge_model = BDCN(rate=rate)
    edge_model.load_state_dict(torch.load('%s' % (checkpoint_path), map_location = device))
    if torch.cuda.is_available():
        edge_model = edge_model.cuda()
    return edge_model

def convert_to_black_edges(edge_image_path:str)->str:
    ext = edge_image_path.split('/')[-1].split('.')[-1]
    black_edges_output_path = f'{EDGE_MODEL_OUTPUT_DIR}/' + edge_image_path.split('/')[-1].split('.')[0] + '_black_edges.' + ext

    #print("post_processed_edge_image_path", post_processed_edge_image_path)
    edge_image = cv2.imread(edge_image_path)
    # Convert the image to grayscale
    gray = cv2.cvtColor(edge_image, cv2.COLOR_BGR2GRAY)
    # Invert the colors
    inverted_image = cv2.bitwise_not(gray)
    cv2.imwrite(black_edges_output_path,inverted_image)

    return black_edges_output_path

def detect_edges(model, test_image_path, output_dir=f'{EDGE_MODEL_OUTPUT_DIR}/'):
    make_dir(output_dir)
    output_with_white_edges = os.path.join(output_dir,test_image_path.split('/')[-1])
    
    mean_bgr = np.array([104.00699, 116.66877, 122.67892])

    model.eval()

    start_time = time.time()
    all_t = 0
    data = cv2.imread(test_image_path)
    # data = cv2.bilateralFilter(data, d=9, sigmaColor=250, sigmaSpace=250) #bilateral filtering
    data = np.array(data, np.float32)
    data = data - mean_bgr
    data = data.transpose((2, 0, 1))
    data = torch.from_numpy(data).float().unsqueeze(0)
    if torch.cuda.is_available():
        data = data.cuda()
    data = Variable(data)
    
    # Perform Inference
    t1 = time.time()

    out = model(data)

    out = [F.sigmoid(out[-1]).cpu().data.numpy()[0, 0, :, :]]

    cv2.imwrite(output_with_white_edges, 255*out[-1])

    all_t += time.time() - t1

    print(all_t)
    print('Total Edge Detection Inference Time: ', time.time() - start_time)

    # Edge color conversion
    output_with_black_edges = convert_to_black_edges(output_edge_image_path)
    
    return output_with_white_edges, output_with_black_edges


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Edge Detection using BDCN model')
    parser.add_argument('--checkpoint', type=str, default='checkpoint/bdcn_pretrained_on_bsds500.pth', help='Path to the model checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--output', type=str, default='bdcn_output/', help='Directory to save the output images')

    args = parser.parse_args()

    model = init_edge_detection_model(checkpoint_path=args.checkpoint)
    output_image_path_with_white_edges, output_image_path_with_black_edges = detect_edges(model, args.image, args.output)

    print(f'Edge detection complete.\nWhite edges output saved to: {output_with_white_edges}\nBlack edges output saved to: {output_with_black_edges}')

