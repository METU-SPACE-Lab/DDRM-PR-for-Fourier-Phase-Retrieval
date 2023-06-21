import torch
import os
import scipy.io as scipy_io
import re
import matplotlib.pylab as plt
import PIL.Image as Image
import numpy as np
import cupy as cp
import mat73
import cv2
from numpy import linalg as LA
import torch
import numpy as np
import math
import argparse
import torchvision
import os
from argparse import Namespace
import skimage.metrics
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import math
import torchvision
from argparse import Namespace
import numpy as np
import cupy as cp
from numpy import linalg as LA
import mat73
import imageio
import matplotlib.pyplot as plt
import time
import skimage.metrics

def load_data(data_path, device, mode):
    data = mat73.loadmat(data_path)['data']

    noisy = data['post_hio']
    noisy = np.expand_dims(noisy, axis=0);
    orig = data['gt']
    orig = np.expand_dims(orig, axis=0);

    noisy = np.transpose(noisy, [3, 0, 1, 2])
    orig = np.transpose(orig, [3, 0, 1, 2])

    training_images_count = round(noisy.shape[0]*0.95)
    if mode == 'train':
        noisy = torch.tensor(noisy[0:training_images_count]).float().to(device)
        orig = torch.tensor(orig[0:training_images_count]).float().to(device)
    elif mode == 'eval':
        noisy = torch.tensor(noisy[training_images_count:]).float().to(device)
        orig = torch.tensor(orig[training_images_count:]).float().to(device)
    else:
        raise ValueError('mode should be train or test')
    return noisy, orig


def load_checkpoint(model, optimizer, checkpoint_dir, epoch_loss=False, CV_psnrs=False, CV_ssims=False):
    if not os.path.exists(checkpoint_dir):
        raise ValueError('checkpoint dir does not exist')

    checkpoint_list = [f for f in os.listdir(checkpoint_dir) if os.path.isfile(os.path.join(checkpoint_dir,f))]

    if len(checkpoint_list) > 0:
        checkpoint_list.sort(key=lambda x: int(re.findall(r"epoch-(\d+).pkl", x)[0]))

        last_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_list[-1])
        print('load checkpoint: %s' % last_checkpoint_path)

        model_ckpt = torch.load(last_checkpoint_path)
        model.load_state_dict(model_ckpt['state_dict'])

        if optimizer:
            optimizer.load_state_dict(model_ckpt['optimizer'])

        if epoch_loss:
          epoch_loss = model_ckpt['epoch_losses']

        if CV_psnrs:
          CV_psnrs = model_ckpt['cv_psnrs']

        if CV_ssims:
          CV_ssims = model_ckpt['cv_ssims']
          
        epoch = model_ckpt['epoch']
        return model, optimizer, epoch, epoch_loss, CV_psnrs, CV_ssims


def cmap_convert(image_tensor, isnumpy=False):
    if isnumpy:
      image = image_tensor
      image = image - image.min()
      image = image / image.max()
      return image*255
    else:
      image = image_tensor.detach().cpu().clone().numpy().squeeze()
      image = image - image.min()
      image = image / image.max()
      cmap_viridis = plt.get_cmap('viridis')
      image = cmap_viridis(image)
      image = Image.fromarray((image * 255).astype(np.uint8)).convert('L')
      return image


def rsnr(rec, oracle):
    "regressed SNR"
    sumP = sum(oracle.reshape(-1))
    sumI = sum(rec.reshape(-1))
    sumIP = sum(oracle.reshape(-1) * rec.reshape(-1) )
    sumI2 = sum(rec.reshape(-1)**2)
    A = np.matrix([[sumI2, sumI], [sumI, oracle.size]])
    b = np.matrix([[sumIP], [sumP]])
    c = np.linalg.inv(A)*b #(A)\b
    rec = c[0, 0]*rec+c[1, 0]
    err = sum((oracle.reshape(-1)-rec.reshape(-1))**2)
    SNR = 10.0*np.log10(sum(oracle.reshape(-1)**2)/err)

    if np.isnan(SNR):
        SNR = 0.0
    return SNR


def fienup_phase_retrieval(mag, mask=None, beta=0.9, steps=200, mode='hybrid', verbose=True, x_init=None, SamplingRateSqrt=1):
    assert beta > 0, 'step size must be a positive number'
    assert steps > 0, 'steps must be a positive number'
    assert mode == 'input-output' or mode == 'output-output'\
        or mode == 'hybrid',\
    'mode must be \'input-output\', \'output-output\' or \'hybrid\''
    
    if mask is None:
        mask = cp.ones(mag.shape)
        
    assert mag.shape == mask.shape, 'mask and mag must have same shape'
    
    mag = cp.array(mag)
    mask = cp.array(mask)
    if not(x_init is None): x_init = cp.array(x_init)
    
    # sample random phase and initialize image x 
    if x_init is None:
      y_hat = mag*cp.exp(1j*2*cp.pi*cp.random.rand(*mag.shape))
    else:
      y_hat = mag*cp.exp(1j*cp.angle(cp.fft.fft2(x_init)/(SamplingRateSqrt)))
    
    x = cp.zeros(mag.shape)
    
    # previous iterate
    x_p = None
    # main loop
    for i in range(1, steps+1):
        # show progress
        if i % 100 == 0 and verbose: 
            print("step", i, "of", steps)
        # inverse fourier transform
        y = cp.real(cp.fft.ifft2(y_hat)*(SamplingRateSqrt))
        # previous iterate
        if x_p is None:
          if x_init is None:
            x_p = y
          else:
            x_p = x_init
        else:
            x_p = x 
        
        # updates for elements that satisfy object domain constraints
        if mode == "output-output" or mode == "hybrid":
            x = y
            
        # find elements that violate object domain constraints 
        # or are not masked
        indices = cp.logical_or(cp.logical_and(y<0, mask), 
                                cp.logical_not(mask))
        # updates for elements that violate object domain constraints
        if mode == "hybrid" or mode == "input-output":
            x[indices] = x_p[indices]-beta*y[indices] 
        elif mode == "output-output":
            x[indices] = y[indices]-beta*y[indices] 
        # fourier transform
        x_hat = cp.fft.fft2(x)/(SamplingRateSqrt)
        # satisfy fourier domain constraints
        # (replace magnitude with input magnitude)
        y_hat = mag*cp.exp(1j*cp.angle(x_hat))

    return cp.asnumpy(x)


def cal_ssim(img1, img2):
    L = 255 # doğru olan 255 ama matlabla uyumlu 1
    C1 = (0.01 * L)**2
    C2 = (0.03 * L)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


import time


#random initializations
def random_best(magnitudes_oversampled___mask___mask_slice___normalization):
  
  magnitudes_oversampled, mask, mask_slice, normalization = magnitudes_oversampled___mask___mask_slice___normalization

  resid_best = float("inf")

  for i in range(50):
    cp.random.seed(2023)

    result_oversampled = cp.array(fienup_phase_retrieval(magnitudes_oversampled,
                                                          steps=50,
                                                          mask=mask,
                                                          verbose=False,
                                                          SamplingRateSqrt=normalization))
        
    resid = LA.norm(magnitudes_oversampled - cp.asnumpy(cp.fft.fft2(result_oversampled)/(normalization)), 2)
    if resid < resid_best:
      resid_best = resid
      x_init_best = result_oversampled

  return x_init_best

def psnr_correct_channelwise(image_tocorrect, image_groundtruth):
    for i in range(3):
        if(skimage.metrics.peak_signal_noise_ratio(np.asarray(image_tocorrect[i], dtype=np.uint8), np.flip(np.asarray(image_groundtruth[i], dtype=np.uint8))) > skimage.metrics.peak_signal_noise_ratio(np.asarray(image_tocorrect[i], dtype=np.uint8), np.asarray(image_groundtruth[i], dtype=np.uint8))):
            image_tocorrect[i] = np.flip(image_tocorrect[i]).copy()
        
    return image_tocorrect  

# first hio stage
def hio_stage(magnitudes_oversampled___mask___mask_slice___normalization___image):
  magnitudes_oversampled, mask, mask_slice, normalization, image_ = magnitudes_oversampled___mask___mask_slice___normalization___image
  
  for i in range(3):
    magnitudes_oversampled___mask___mask_slice___normalization_perchannel = (magnitudes_oversampled[i], mask, mask_slice, normalization)
    x_init_best = random_best(magnitudes_oversampled___mask___mask_slice___normalization_perchannel)

    result_oversampled = fienup_phase_retrieval(magnitudes_oversampled[i],
                                                steps=1000,
                                                mask=mask,
                                                verbose=False,
                                                x_init=x_init_best,
                                                SamplingRateSqrt=normalization)
    
    if(i==0):
        image_iter = result_oversampled[mask_slice]
        image_iter = np.repeat(np.expand_dims(image_iter, axis=(0,1)), 3, axis=1)
    else:
        image_iter[0,i,:,:] = result_oversampled[mask_slice]


  # if(skimage.metrics.peak_signal_noise_ratio(np.asarray(image_[i], dtype=np.uint8), np.flip(np.asarray(image_iter[0,i,:,:], dtype=np.uint8))) > skimage.metrics.peak_signal_noise_ratio(np.asarray(image_[i], dtype=np.uint8), np.asarray(image_iter[0,i,:,:], dtype=np.uint8))):
  #     image_iter[0,i,:,:] = np.flip(image_iter[0,i,:,:]).copy()
      
  image_iter[0,:,:,:] = psnr_correct_channelwise(image_iter[0,:,:,:], image_)
    
  return torch.tensor(image_iter, device=torch.device("cuda:0"), dtype=torch.float32)/255*2-1


def pr_encode(image_full):
    image_full = (image_full.squeeze().cpu().numpy()+1)/2*255
    y = []
    for i in range(3):
        image_ = image_full[i]
        SamplingRateSqrt = 2
        padded_zero = (np.array(image_.shape)*(SamplingRateSqrt-1)/2).astype(np.int)
        image_padded_ = np.pad(image_, padded_zero, 'constant') # oversampling
        mask = np.pad(np.ones(image_.shape), padded_zero, 'constant') # HIO mask
        mask_slice = (slice(padded_zero[0], padded_zero[0]+image_.shape[0]), slice(padded_zero[1], padded_zero[1]+image_.shape[1]))
        normalization = np.sqrt(image_.shape[0]*image_.shape[1]) * SamplingRateSqrt**2
        fourier_oversampled_ = (np.fft.fft2(image_padded_)/(normalization))
        # add noise
        alpha = 3
        intensity_noise = alpha*np.multiply(np.abs(fourier_oversampled_), np.random.randn(*fourier_oversampled_.shape))
        y2 = np.square(np.abs(fourier_oversampled_)) + intensity_noise
        y2 = np.multiply(y2, y2>0)
        y.append(np.sqrt(y2))
    
    return y, mask, mask_slice, normalization, image_full



def je(image_full):
    # print("55555",image_full.shape)
    image_full = (image_full.squeeze().cpu().numpy()+1)/2*255
    # print("55555",image_full.shape)
    y = []
    image_padded_ = []
    
    for i in range(3):
        image_ = image_full[i]
        # print("6666", image_.shape)
        SamplingRateSqrt = 2
        padded_zero = (np.array(image_.shape)*(SamplingRateSqrt-1)/2).astype(np.int)
        image_padded_.append(np.pad(image_, padded_zero, 'constant')) # oversampling
        mask = np.pad(np.ones(image_.shape), padded_zero, 'constant') # HIO mask
        mask_slice = (slice(padded_zero[0], padded_zero[0]+image_.shape[0]), slice(padded_zero[1], padded_zero[1]+image_.shape[1]))
        normalization = np.sqrt(image_.shape[0]*image_.shape[1]) * SamplingRateSqrt**2
        fourier_oversampled_ = (np.fft.fft2(image_padded_[-1])/(normalization))
        # add noise
        alpha = 0 #  3 yapinca daha iyi aldım ama .02 puan
        intensity_noise = alpha*np.multiply(np.abs(fourier_oversampled_), np.random.randn(*fourier_oversampled_.shape))
        y2 = np.square(np.abs(fourier_oversampled_)) + intensity_noise
        y2 = np.multiply(y2, y2>0)
        y.append(np.sqrt(y2))
    #     print("77", y2.shape)
    # print("XXXXX",len(y))
    return y, mask, mask_slice, normalization, image_padded_, image_full

def jd(magnitudes_oversampled___mask___mask_slice___normalization___image_):
    magnitudes_oversampled, mask, mask_slice, normalization, image_, image_full = magnitudes_oversampled___mask___mask_slice___normalization___image_


    for i in range(3):
        x_init_best = image_[i]
        # print("*******55******")
        # print(mask.shape)
        # print(x_init_best.shape)
        # print(magnitudes_oversampled[i].shape)
        # print("*************")

        # result_oversampled = x_init_best
        result_oversampled = fienup_phase_retrieval(magnitudes_oversampled[i],
                                                    steps=1,
                                                    mask=mask,
                                                    verbose=False,
                                                    x_init=x_init_best,
                                                    SamplingRateSqrt=normalization)

        if(i==0):
            image_iter = result_oversampled[mask_slice]
            image_iter = np.repeat(np.expand_dims(image_iter, axis=(0,1)), 3, axis=1)
        else:
            image_iter[0,i,:,:] = result_oversampled[mask_slice]

        # print("xxxx",image_full[i].shape)
        # print("xxxx",image_iter[0,i,:,:].shape)

        # if(skimage.metrics.peak_signal_noise_ratio(np.asarray(image_full[i], dtype=np.uint8), np.flip(np.asarray(image_iter[0,i,:,:], dtype=np.uint8))) > skimage.metrics.peak_signal_noise_ratio(np.asarray(image_full[i], dtype=np.uint8), np.asarray(image_iter[0,i,:,:], dtype=np.uint8))):
        #     image_iter[0,i,:,:] = np.flip(image_iter[0,i,:,:]).copy()
            
    image_iter[0,:,:,:] = psnr_correct_channelwise(image_iter[0,:,:,:], image_full)
        
        # print("onurkaya",image_iter.shape)
    return torch.tensor(image_iter, device=torch.device("cuda:0"), dtype=torch.float32)/255*2-1