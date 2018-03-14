import math
import numpy as np
import scipy as sp
import scipy.signal
import scipy.ndimage
from scipy.misc import imresize
from scipy.interpolate import griddata
import skimage.color

import pyramid
from utils import *

def eye_transfer(im2c, im2c_raw, alpha_map, fg):
    im2c[im2c > 1] = 1
    im2c[im2c < 0] = 0

    rmin = 20
    rmax = 200
    delta_r = 5

    ci2, cp2, o2 = thresh(skimage.color.rgb2grey(im2c_raw), rmin, rmax)
    ci2 = np.array(ci2, np.int)

    r = np.int(ci2[2] - delta_r)
    cm = circleMask(r)
    irisM = np.dstack([cm, cm, cm])

    spatch2 = im2c[ci2[0] - r - 1:ci2[0] + r, ci2[1] - r - 1:ci2[1] + r, :]
    spatch2_raw = im2c_raw[ci2[0] - r - 1:ci2[0] + r, ci2[1] - r - 1:ci2[1] + r, :]
    skinM2 = skin(255 * spatch2_raw)

    spatch2_lab = rgb2lab(spatch2)
    spatch2_l = spatch2_lab[:, :, 0]
    hm = (1 - skinM2) * (spatch2_l > 60)
    highlightM = np.dstack([hm, hm, hm]) * irisM
    spatch2_lab[highlightM != 0] = np.nan
    spatch2_lab[:, :, 0] = interp_nan(spatch2_lab[:, :, 0])
    spatch2_lab[:, :, 1] = interp_nan(spatch2_lab[:, :, 1])
    spatch2_lab[:, :, 2] = interp_nan(spatch2_lab[:, :, 2])
    spatch2 = lab2rgb(spatch2_lab)
    im2c[ci2[0] - r - 1:ci2[0] + r, ci2[1] - r - 1:ci2[1] + r, :] = spatch2

    h, w = np.size(spatch2, 0), np.size(spatch2, 1)
    alpha_map = im2double(imresize(alpha_map, [h, w]))
    alpha_map = alpha_map * (1 - np.dstack([skinM2, skinM2, skinM2])) * irisM
    fg = im2double(imresize(fg, [h, w]))

    eye2 = im2c[ci2[0] - r -1: ci2[0] + r, ci2[1] - r -1: ci2[1] + r, :]
    im2c[ci2[0] - r - 1: ci2[0] + r, ci2[1] - r - 1: ci2[1] + r, :] = (1 - alpha_map) * eye2 +  alpha_map * fg

    return im2c


def skin(I):
    I = im2double(I)
    hsv = skimage.color.rgb2hsv(I)
    hue = hsv[:, :, 0]

    cb = 0.148 * I[:, :, 0] - 0.291 * I[:, :, 1] + 0.439 * I[:, :, 2] + 128
    cr = 0.439 * I[:, :, 0] - 0.368 * I[:, :, 1] -0.071 * I[:, :, 2] + 128
    b = I[:, :, 2]
    w, h = b.shape

    segment = np.zeros([w, h])
    for i in range(w):
        for j in range(h):
            if 140 <= cr[i, j] and cr[i, j] <= 165 and \
               140 <= cb[i, j] and cb[i, j] <= 195 and \
               0.01 <= hue[i, j] and hue[i, j] <= 0.1 and b[i, j] < 0.7*255:
                segment[i, j] = 1
            else:
                segment[i, j] = 0

    return segment

def thresh(I, rmin, rmax):
    I = im2double(I)
    pimage = np.copy(I)

    I = 1 - imfill(1 - I)
    rows = np.size(I, 0)
    cols = np.size(I, 1)

    X, Y = np.where(I < 0.5)
    delete_i = []

    for i in range(np.size(X)):
        if X[i] > rmin and Y[i] > rmin and X[i] <= (rows - rmin) and Y[i] < (cols - rmin):
            A = I[X[i] - 2:X[i] + 1, Y[i] - 2:Y[i] + 1]
            M = np.min(A)

            if I[X[i] - 1, Y[i] - 1] != M:
                delete_i.append(i)

    X = np.delete(X, delete_i)
    Y = np.delete(Y, delete_i)

    delete_i = np.where(X <= rmin)
    X = np.delete(X, delete_i)
    Y = np.delete(Y, delete_i)

    delete_i = np.where(Y <= rmin)
    X = np.delete(X, delete_i)
    Y = np.delete(Y, delete_i)

    delete_i = np.where(X > (rows - rmin))
    X = np.delete(X, delete_i)
    Y = np.delete(Y, delete_i)

    delete_i = np.where(Y > (cols - rmin))
    X = np.delete(X, delete_i)
    Y = np.delete(Y, delete_i)

    N = np.size(X)
    maxb = np.zeros([rows, cols])
    maxrad = np.zeros([rows, cols])

    for j in range(N):
        b, r, blur = partiald(I, [X[j], Y[j]], rmin, rmax, 'inf', 600, 'iris')
        maxb[X[j], Y[j]] = b
        maxrad[X[j], Y[j]] = r

    x, y = np.where(maxb == np.max(maxb))
    ci = search(I, rmin, rmax, np.int(x[0]), np.int(y[0]), 'iris')
    cp = search(I, np.int(np.round(0.1 * r)), np.int(np.round(0.8 * r)), ci[0], ci[1], 'pupil')

    out = drawcircle(pimage, [ci[0], ci[1]], ci[2], 600)
    out = drawcircle(out, [cp[0], cp[1]], cp[2], 600)

    return ci, cp, out

def imfill(test_array, h_max=255):
    input_array = np.copy(test_array)
    el = sp.ndimage.generate_binary_structure(2,2).astype(np.int)
    inside_mask = sp.ndimage.binary_erosion(~np.isnan(input_array), structure=el)
    output_array = np.copy(input_array)
    output_array[inside_mask]=h_max
    output_old_array = np.copy(input_array)
    output_old_array.fill(0)
    el = sp.ndimage.generate_binary_structure(2,1).astype(np.int)
    while not np.array_equal(output_old_array, output_array):
        output_old_array = np.copy(output_array)
        output_array = np.maximum(input_array,sp.ndimage.grey_erosion(output_array, size=(3,3), footprint=el))
    return output_array

def search(im, rmin, rmax, x, y, option):
    rows = np.size(im, 0)
    cols = np.size(im, 1)

    maxrad = np.zeros([rows, cols])
    maxb = np.zeros([rows, cols])

    for i in range(np.int(x) - 5, np.int(x) + 5):
        for j in range(np.int(y) - 5, np.int(y) + 5):
            b, r, blur = partiald(im, [i, j], rmin, rmax, 0.5, 600, option)
            maxrad[i, j] = r
            maxb[i, j] = b

    B = np.max(maxb)
    X, Y = np.where(maxb == B)
    radius = maxrad[X[0], Y[0]]
    cp = [X[0], Y[0], radius]

    return np.array(cp)

def partiald(I, C, rmin, rmax, sigma, n, part):
    R = np.arange(rmin, rmax)
    L = []

    for k in range(np.size(R)):
        l = lineint(I, C, R[k], n, part)
        if l == 0:
            break
        else:
            L.append(l)

    D = np.diff(L)
    D = np.insert(D, 0, 0)

    if sigma == 'inf':
        f = np.ones(7) / 7
    else:
        f = pyramid.get_gaussian_kernel([1, 5], sigma)[0]

    blur = np.abs(scipy.signal.convolve(D, f, mode='same'))
    max_index = np.int(np.where(blur == np.max(blur))[0][0])

    r = R[max_index]
    b = blur[max_index]

    return b, r, blur

def lineint(I, C, r, n, part):
    theta = (2 * math.pi) / n

    rows, cols = I.shape

    angle = np.arange(theta, 2 * math.pi + theta / 2, theta)
    x = C[0] - r * np.sin(angle)
    y = C[1] + r * np.cos(angle)

    if np.any(x >= rows) or np.any(y >= cols) or np.any(x <= 1) or np.any(y <= 1):
        return 0

    s = 0
    if part == 'pupil':
        for i in range(n):
            s += I[np.int(round(x[i])) - 1, np.int(round(y[i])) - 1]
        L = s / n
    elif part == 'iris':
        for i in range(round(n / 8)):
            s += I[np.int(round(x[i])) - 1, np.int(round(y[i])) - 1]

        for i in range(round(3 * n / 8), round(5 * n / 8)):
            s += I[np.int(round(x[i])) - 1, np.int(round(y[i])) - 1]

        for i in range(round(7 * n / 8), n):
            s += I[np.int(round(x[i])) - 1, np.int(round(y[i])) - 1]

        L = 2 * s / n

    return L

def drawcircle(I, C, r, n=600):
    theta = 2 * math.pi / n
    O = np.copy(I)

    rows = np.size(I, 0)
    cols = np.size(I, 1)
    angle = np.arange(theta, 2 * math.pi + theta / 10, theta)

    x = C[0] - r * np.sin(angle)
    y = C[1] + r * np.cos(angle)

    if np.any(x >= rows) or np.any(y >= cols) or np.any(x <= 1) or np.any(y <= 1):
        return O

    for i in range(n):
        O[np.int(round(x[i])), np.int(round(y[i]))] = 1

    return O

def circleMask(r):
    mask = np.zeros([2 * r + 1, 2 * r + 1])

    x = np.arange(-r, r + 1)
    xx, yy = np.meshgrid(x, x)

    mask[xx * xx + yy * yy <= (r ** 2)] = 1

    return mask

def interp_nan(input):
    h, w = np.size(input, 0), np.size(input, 1)
    XI, YI = meshgrid(w, h)

    out = np.copy(input)
    grid = griddata(np.dstack([XI[~np.isnan(input)], YI[~np.isnan(input)]])[0],
                   input[~np.isnan(input)],
                   (XI[np.isnan(input)], YI[np.isnan(input)]),
                   method='cubic')
    out[np.isnan(input)] = grid

    return out