import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_gaussian_kernel(size, sigma):
    if isinstance(size, int):
        kx = cv2.getGaussianKernel(size, sigma)
        ky = cv2.getGaussianKernel(size, sigma)
        return np.multiply(kx,np.transpose(ky))
    elif isinstance(size, list):
        if len(size) == 2:
            kx = cv2.getGaussianKernel(size[0], sigma)
            ky = cv2.getGaussianKernel(size[1], sigma)
            return np.multiply(kx,np.transpose(ky))
        
    return None

def imfilter(input, size, sigma):
    return cv2.GaussianBlur(input, (size, size), sigma)

def imfilter_mask(input, mask, size, sigma):
    z = cv2.GaussianBlur(mask, (size, size), sigma)
    output = cv2.GaussianBlur(input * mask, (size, size), sigma)
    return output / (z + np.spacing(1))

def laplacian_pyramid(input, level, mask):
    pyramids = [input]
    
    for i in range(2, level + 1):
        sigma = 2 ** i

        size = sigma * 5 - 1
        pyramids.append(imfilter_mask(input, mask, size, sigma))
        
    for i in range(level - 1):
        pyramids[i] = (pyramids[i] - pyramids[i + 1]) * mask
    
    pyramids[level - 1] *= mask

    return pyramids

def sum_pyramid(pyr_out):
    output = np.zeros(pyr_out[0].shape)

    for p in pyr_out:
        output += p
    
    return output
