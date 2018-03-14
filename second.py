import cv2
import numpy as np
import matplotlib.pylab as plt
import scipy.io as sio
from scipy import misc

import config
import pyramid
import iris
from utils import *
from morph import morph

def style_transfer(style_in, im_in_name, style_ex, im_ex_name):
    im_in = im2double(bgr2rgb(cv2.imread(get_img_path(style_in, im_in_name))))
    im_ex = im2double(bgr2rgb(cv2.imread(get_img_path(style_ex, im_ex_name, True))))

    mask_in = im2double(bgr2rgb(cv2.imread(get_mask_path(style_in, im_in_name))))

    bgs_ex = im2double(bgr2rgb(cv2.imread(get_bgs_path(style_ex, im_ex_name))))

    data = sio.loadmat(config.second_mat_file)
    vx, vy = data['vx'], data['vy']
    vxm, vym = data['vxm'], data['vym']
    bin_alpha_in, bin_alpha_ex = data['bin_alpha_in'], data['bin_alpha_ex']

    vxf, vyf = thresh_v(vx + vxm, vy + vym)

    if config.debug:
        im_ex_wf, mask = warp_image(im_ex, vxf, vyf)
        plt.figure()
        plt.imshow(0.5 * (im_in + im_ex_wf))
        plt.show()

    im_in = mask_in * im_in + (1 - mask_in) * bgs_ex

    if config.debug:
        plt.figure()
        plt.imshow(im_in)
        plt.show()
    
    im_in = rgb2lab(im_in)
    im_ex = rgb2lab(im_ex)

    level = 6
    height, width, channels = im_in.shape
    im_out = np.zeros(im_in.shape)

    if config.recomp:
        for c in range(channels):
            pyr_in = pyramid.laplacian_pyramid(im_in[:, :, c], level, bin_alpha_in)
            pyr_ex = pyramid.laplacian_pyramid(im_ex[:, :, c], level, bin_alpha_ex)

            pyr_out = []
            for i in range(level - 1):
                r = 2**(i + 2)

                l_in = pyr_in[i]
                l_ex = pyr_ex[i]
                l_ex, _ = warp_image(l_ex, vxf, vyf)

                e_in = pyramid.imfilter(l_in ** 2, 6 * r - 1, r)
                e_ex = pyramid.imfilter(l_ex ** 2, 6 * r - 1, r)
                gain = (e_ex / (e_in + config.e_0)) ** 0.5

                for x in range(height):
                    for y in range(width):
                        gain[x, y] = max(min(gain[x, y], config.gain_max), config.gain_min)

                l_new = l_in * gain
                pyr_out.append(l_new)

            last, _ = warp_image(pyr_ex[level - 1], vxf, vyf)
            pyr_out.append(last)
            im_out[:, :, c] = pyramid.sum_pyramid(pyr_out)

        im_out = lab2rgb(im_out)
        im_in = lab2rgb(im_in)

        im_out = mask_in * im_out + (1 - mask_in) * bgs_ex
        sio.savemat(config.img_out_mat_file, {'im_out': im_out})
    else:
        im_out = sio.loadmat(config.img_out_mat_file)['im_out']

    if config.transfer_eye:
        alpha_l = im2double(bgr2rgb(cv2.imread(get_alpha_path(style_ex, True))))
        alpha_r = im2double(bgr2rgb(cv2.imread(get_alpha_path(style_ex, False))))

        fg_l = im2double(bgr2rgb(cv2.imread(get_fl_path(style_ex, True))))
        fg_r = im2double(bgr2rgb(cv2.imread(get_fl_path(style_ex, False))))

        model = load_model(style_in, im_in_name, True)
        leye_center = np.round(np.mean(model[36:42], 0))
        reye_center = np.round(np.mean(model[42:48], 0))

        half_width = 75
        half_height = 50

        leye_raw = im_in[int(leye_center[1]) - half_height - 1: int(leye_center[1]) + half_height,
                         int(leye_center[0]) - half_width - 1: int(leye_center[0]) + half_width]
        reye_raw = im_in[int(reye_center[1]) - half_height - 1: int(reye_center[1]) + half_height,
                         int(reye_center[0]) - half_width: int(reye_center[0]) + half_width]
        leye = im_out[int(leye_center[1]) - half_height - 1: int(leye_center[1]) + half_height,
                      int(leye_center[0]) - half_width - 1: int(leye_center[0]) + half_width]
        reye = im_out[int(reye_center[1]) - half_height - 1: int(reye_center[1]) + half_height,
                      int(reye_center[0]) - half_width: int(reye_center[0]) + half_width]
        
        leye_new = iris.eye_transfer(leye, leye_raw, alpha_l, fg_l)
        reye_new = iris.eye_transfer(reye, reye_raw, alpha_r, fg_r)

        im_out[int(leye_center[1]) - half_height - 1: int(leye_center[1]) + half_height,
            int(leye_center[0]) - half_width - 1: int(leye_center[0]) + half_width] = leye_new
        im_out[int(reye_center[1]) - half_height - 1: int(reye_center[1]) + half_height,
            int(reye_center[0]) - half_width: int(reye_center[0]) + half_width] = reye_new

    plt.figure()
    plt.imshow(im_out)
    plt.show()

    if config.save_output_img:
        misc.imsave(config.img_out_path, matrix2image(im_out))

if __name__ == '__main__':
    style_transfer(config.style_in,
                   config.im_in_name,
                   config.style_ex,
                   config.im_ex_name)