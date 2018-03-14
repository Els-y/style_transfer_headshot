import os
import math
import cv2
import csv
import numpy as np

import config

def meshgrid(w, h):
    x = np.linspace(1, w, w)
    y = np.linspace(1, h, h)
    
    return np.meshgrid(x, y)

def im2double(input):
    if np.max(input) > 1:
        return np.float64(input) / 255
    else:
        return np.float64(input)

def bgr2rgb(image):
    (b, g, r) = cv2.split(image)
    return cv2.merge([r, g, b])

def rgb2gray(rgb):
    # r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
 
    # return gray

    return np.dot(rgb, [0.2989, 0.5870, 0.1140])

def get_img_path(folder, filename, model=False):
    if model:
        return os.path.join(config.root, folder, 'imgs', filename + '.png')
    else:
        return os.path.join(config.root, folder, 'fgs', filename + '.png')

def get_mask_path(folder, filename):
    return os.path.join(config.root, folder, 'masks', filename + '.png')

def get_bgs_path(folder, filename):
    return os.path.join(config.root, folder, 'bgs', filename + '.jpg')

def get_landmarks_path(folder, filename):
    return os.path.join(config.root, folder, 'landmarks', filename + '.lm')

def get_alpha_path(folder, left=True):
    if left:
        return os.path.join(config.root, 'eyes', folder, '001_alpha_l.png')
    else:
        return os.path.join(config.root, 'eyes', folder, '001_alpha_r.png')

def get_fl_path(folder, left=True):
    if left:
        return os.path.join(config.root, 'eyes', folder, '001_fg_l.png')
    else:
        return os.path.join(config.root, 'eyes', folder, '001_fg_r.png')

def load_model(folder, name, matrix=False):
    with open(get_landmarks_path(folder, name)) as csv_file:
        model = list(csv.reader(csv_file, delimiter=','))
    
    if matrix:
        rows = len(model)
        cols = len(model[0])

        out = np.zeros([rows, cols])
        for r in range(rows):
            for c in range(cols):
                out[r, c] = model[r][c]

        return out
    else:
        return model

def load_face_model(folder, name):
    with open('../code/face.con') as csv_file:
        con = list(csv.reader(csv_file, delimiter=','))

    model = load_model(folder, name)

    segs = []
    for i in range(len(con)):
        p = model[int(con[i][0])]
        q = model[int(con[i][1])]
        tmp = {
            'p': {
                'x': float(p[0]),
                'y': float(p[1])
            },
            'q': {
                'x': float(q[0]),
                'y': float(q[1])
            }
        }
        segs.append(tmp)

    return segs

def rgb2lab(rgb):
    T = 0.008856
    [M, N, C] = rgb.shape
    s = M * N

    R = np.zeros((M, N))
    G = np.zeros((M, N))
    B = np.zeros((M, N))

    for x in range(M):
        for y in range(N):
            R[x][y] = rgb[x][y][0]
            G[x][y] = rgb[x][y][1]
            B[x][y] = rgb[x][y][2]

    if np.max(R) > 2 or np.max(G) > 2 or np.max(B) > 2:
        R /= 255
        G /= 255
        B /= 255

    RGB = np.vstack((np.reshape(R, [1, s], 'F'),
                     np.reshape(G, [1, s], 'F'),
                     np.reshape(B, [1, s], 'F')))

    MAT = np.array([[0.412453, 0.357580, 0.180423],
                    [0.212671, 0.715160, 0.072169],
                    [0.019334, 0.119193, 0.950227]])
    XYZ = np.matmul(MAT, RGB)

    X = XYZ[0] / 0.950456
    Y = XYZ[1]
    Z = XYZ[2] / 1.088754
    
    XT = X > T
    YT = Y > T
    ZT = Z > T

    Y3 = Y ** (1/3); 

    fX = XT * (X**(1 / 3)) + (~XT) * (7.787 * X + 16 / 116)
    fY = YT * Y3 + (~YT) * (7.787 * Y + 16 / 116)
    fZ = ZT * (Z**(1 / 3)) + (~ZT) * (7.787 * Z + 16 / 116)

    L = np.reshape(YT * (116 * Y3 - 16.0) + (~YT) * (903.3 * Y), [M, N], 'F')
    a = np.reshape(500 * (fX - fY), [M, N], 'F')
    b = np.reshape(200 * (fY - fZ), [M, N], 'F')

    return np.dstack([L, a, b])

def lab2rgb(lab):
    T1 = 0.008856
    T2 = 0.206893

    M, N, C = lab.shape
    s = M * N

    L = np.reshape(lab[:,:,0], [1, s], 'F')[0]
    a = np.reshape(lab[:,:,1], [1, s], 'F')[0]
    b = np.reshape(lab[:,:,2], [1, s], 'F')[0]

    fY = ((L + 16) / 116) ** 3
    YT = fY > T1
    fY = (~YT) * (L / 903.3) + YT * fY
    Y = fY

    fY = YT * (fY ** (1 / 3)) + (~YT) * (7.787 * fY + 16 / 116)

    fX = a / 500 + fY
    XT = fX > T2
    X = (XT * (fX ** 3) + (~XT) * ((fX - 16 / 116) / 7.787))

    fZ = fY - b / 200
    ZT = fZ > T2
    Z = (ZT * (fZ ** 3) + (~ZT) * ((fZ - 16 / 116) / 7.787))

    X = X * 0.950456
    Z = Z * 1.088754

    MAT = np.array([[3.240479, -1.537150, -0.498535],
                    [-0.969256, 1.875992, 0.041556],
                    [0.055648, -0.204043, 1.057311]])

    XYZ = np.vstack([X, Y, Z])
    RGB = np.matmul(MAT, XYZ)

    for x in range(RGB.shape[0]):
        for y in range(RGB.shape[1]):
            RGB[x][y] = max(min(RGB[x][y], 1), 0)

    R = np.reshape(RGB[0], [M, N], 'F')
    G = np.reshape(RGB[1], [M, N], 'F')
    B = np.reshape(RGB[2], [M, N], 'F') 

    return np.dstack([R, G, B])

def matrix2image(data):
    data = data * 255
    img = data.astype(np.uint8)
    return img

def warp_image(im_ex, vx, vy, offset=True):
    height2, width2 = np.size(im_ex, 0), np.size(im_ex, 1)
    height1, width1 = vx.shape

    xx, yy = meshgrid(width2, height2)
    XX, YY = meshgrid(width1, height1)

    if offset:
        XX = XX + vx
        YY = YY + vy
    else:
        XX = vx
        YY = vy
    
    a = XX < 1
    b = XX > width2
    c = YY < 1
    d = YY > height2
    mask = a | b | c | d
    
    for x in range(height2):
        for y in range(width2):
            XX[x][y] = min(max(XX[x][y], 1), width2)
            YY[x][y] = min(max(YY[x][y], 1), height2)

    foo = np.zeros([height2, width2])
    index_x, index_y = np.where(mask == 1)

    warped_image = bilinear_interp(im_ex, XX, YY)
    for j in range(index_x.shape[0]):
        warped_image[index_x[j],index_y[j],:] = 0.6
    
    mask = 1 - mask

    return warped_image, mask

def bilinear_interp(img, xx, yy):
    if img.ndim == 3:
        height, width, channels = img.shape
    else:
        height, width = img.shape
        channels = 1

    output = np.zeros(img.shape)

    for i in range(height):
        for j in range(width):
            x = min(max(math.floor(xx[i,j] - 1), 0), width - 2)
            y = min(max(math.floor(yy[i,j] - 1), 0), height - 2)
            
            p = xx[i,j] - 1 - x
            q = yy[i,j] - 1 - y

            if img.ndim == 3:
                for c in range(channels):
                    output[i,j,c] = (img[y,x,c] * (1 - p) * (1 - q) + 
                                    img[y+1,x,c] * (1 - p) * q + 
                                    img[y,x+1,c] * p * (1 - q) + 
                                    img[y+1,x+1,c] * p * q)
            else:
                output[i,j] = (img[y,x] * (1 - p) * (1 - q) + 
                               img[y+1,x] * (1 - p) * q + 
                               img[y,x+1] * p * (1 - q) + 
                               img[y+1,x+1] * p * q)

    return output

def thresh_v(vx, vy):
    height, width = vx.shape
    warp, mask = warp_image(np.zeros([height, width, 3]), vx, vy)

    mask = 1 - mask
    for x in range(height):
        for y in range(width):
            if mask[x][y] == 1:
                vx[x][y] = 0
                vy[x][y] = 0

    return vx, vy