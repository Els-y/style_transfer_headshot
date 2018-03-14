import cv2
import numpy as np
from utils import *

def morph(style_src, im_src_name, style_dst, im_dst_name):
    im_src = im2double(bgr2rgb(cv2.imread(get_img_path(style_src, im_src_name, True))))

    height, weight, channel = im_src.shape
    xx, yy = meshgrid(weight, height)

    segs_src = add_boundary(load_face_model(style_src, im_src_name), height, weight)
    segs_dst = add_boundary(load_face_model(style_dst, im_dst_name), height, weight)

    xsum = xx * 0
    ysum = yy * 0
    wsum = xx * 0

    for i in range(len(segs_src)):
        u, v = get_uv(xx, yy, segs_src[i])
        x, y = get_xy(u, v, segs_dst[i])
        weights = get_weight(xx, yy, segs_src[i])
        wsum += weights
        xsum += weights * x
        ysum += weights * y

    x_m = xsum / wsum
    y_m = ysum / wsum
    vx = xx - x_m
    vy = yy - y_m

    index_x, index_y = np.where(x_m < 1)
    for i in range(index_x.shape[0]):
        vx[index_x[i]][index_y[i]] = 0

    index_x, index_y = np.where(x_m > weight)
    for i in range(index_x.shape[0]):
        vx[index_x[i]][index_y[i]] = 0

    index_x, index_y = np.where(y_m < 1)
    for i in range(index_x.shape[0]):
        vy[index_x[i]][index_y[i]] = 0

    index_x, index_y = np.where(y_m > height)
    for i in range(index_x.shape[0]):
        vy[index_x[i]][index_y[i]] = 0

    return vx, vy

def add_boundary(face_model, height, weight):
    pb = np.array([[1, 1],[1, height], [weight, height], [weight, 1]])
    qb = np.array([[1, height], [weight, height], [weight, 1], [1, 1]])

    segs_b = construct_segs(pb, qb)
    face_model.extend(segs_b)

    return face_model

def draw_segs(img, segs):
    for seg in segs:
        img = cv2.print(img, (seg['p']['x'], seg['p']['y']), (seg['q']['x'], seg['q']['y']))

    return img

def construct_segs(pp, qq):
    segs = []

    for i in range(pp.shape[0]):
        tmp = {
            'p': {
                'x': pp[i][0],
                'y': pp[i][1]
            },
            'q': {
                'x': qq[i][0],
                'y': qq[i][1]
            }
        }

        segs.append(tmp)

    return segs

def get_weight(x, y, line):
    a, b, p, d = 10, 1, 1, 1
    u, v = get_uv(x, y, line)
    d1 = ((x - line['q']['x'])**2 + (y - line['q']['y'])**2)**0.5
    d2 = ((x - line['p']['x'])**2 + (y - line['p']['y'])**2)**0.5
    d = np.abs(v)

    index_x, index_y = np.where(u > 1)
    for i in range(index_x.shape[0]):
        d[index_x[i]][index_y[i]] = d1[index_x[i]][index_y[i]]

    index_x, index_y = np.where(u < 0)
    for i in range(index_x.shape[0]):
        d[index_x[i]][index_y[i]] = d2[index_x[i]][index_y[i]]

    pq_x = line['q']['x'] - line['p']['x']
    pq_y = line['q']['y'] - line['p']['y']
    len_1 = (pq_x**2 + pq_y**2)**0.5

    return (len_1**p / (a + d))**b

def get_uv(x, y, line):
    p = line['p']
    q = line['q']

    pq_x = q['x'] - p['x']
    pq_y = q['y'] - p['y']

    len_2 = pq_x**2 + pq_y**2
    len_1 = len_2**0.5

    ret_u = ((x - p['x']) * pq_x + (y - p['y']) * pq_y) / len_2

    perp_pq_x = -pq_y
    perp_pq_y = pq_x

    ret_v = ((x - p['x']) * perp_pq_x + (y - p['y']) * perp_pq_y) / len_1

    return (ret_u, ret_v)

def get_xy(u, v, line):
    p = line['p']
    q = line['q']

    pq_x = q['x'] - p['x']
    pq_y = q['y'] - p['y']

    perp_pq_x = -pq_y
    perp_pq_y = pq_x

    len_1 = (pq_x**2 + pq_y**2)**0.5

    x = p['x'] + u * pq_x + (v * perp_pq_x) / len_1
    y = p['y'] + u * pq_y + (v * perp_pq_y) / len_1

    return x, y
