import os

root = '../data'

style_in = 'flickr2'
im_in_name = '2239739044_c8e129a57e_o'
style_ex = 'martin'
im_ex_name = '15'

debug = False
recomp = True
save_output_img = True
transfer_eye = True

e_0 = 1e-4
gain_max = 2.8
gain_min = 0.9

first_mat_file = 'first_data.mat'
second_mat_file = 'sift_flow.mat'
img_out_mat_file = 'img_out.mat'

output_folder = 'output'
if transfer_eye:
    output_file = im_in_name + '_' + style_ex.split('/')[-1] + '_eye.png'
else:
    output_file = im_in_name + '_' + style_ex.split('/')[-1] + '.png'

img_out_path = os.path.join(output_folder, output_file)
