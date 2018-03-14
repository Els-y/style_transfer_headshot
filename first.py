import cv2
import numpy as np
import matplotlib.pylab as plt
import scipy.io as sio
import config
from utils import *
from morph import morph

def style_transfer(style_in, im_in_name, style_ex, im_ex_name):
    im_in = im2double(bgr2rgb(cv2.imread(get_img_path(style_in, im_in_name))))
    im_ex = im2double(bgr2rgb(cv2.imread(get_img_path(style_ex, im_ex_name, True))))

    mask_in = im2double(bgr2rgb(cv2.imread(get_mask_path(style_in, im_in_name))))
    mask_ex = im2double(bgr2rgb(cv2.imread(get_mask_path(style_ex, im_ex_name))))

    bgs_ex = im2double(bgr2rgb(cv2.imread(get_bgs_path(style_ex, im_ex_name))))

    if config.debug:    
        plt.subplot(231), plt.imshow(im_in)
        plt.subplot(232), plt.imshow(im_ex)
        plt.subplot(233), plt.imshow(mask_in)
        plt.subplot(234), plt.imshow(mask_ex)
        plt.subplot(235), plt.imshow(bgs_ex)
        plt.show()

    if config.recomp:
        vxm, vym = morph(style_ex, im_ex_name, style_in, im_in_name)
        im_ex_w, mask = warp_image(im_ex, vxm, vym)
        np.savez('morph.npz', vxm=vxm, vym=vym, im_ex_w=im_ex_w)
    else:
        morph_data = np.load('morph.npz')
        vxm = morph_data['vxm']
        vym = morph_data['vym']
        im_ex_w = morph_data['im_ex_w']
    
    sio.savemat(config.first_mat_file, {
        'im_ex_w': im_ex_w,
        'im_in': im_in,
        'mask_in': mask_in,
        'mask_ex': mask_ex,
        'vxm': vxm,
        'vym': vym})

if __name__ == '__main__':
    style_transfer(config.style_in,
                   config.im_in_name,
                   config.style_ex,
                   config.im_ex_name)
