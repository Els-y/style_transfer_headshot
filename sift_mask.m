% Ex: style_transfer flickr2 2187287549_74951db8c2_o martin 0;

tmp_data = 'first_data.mat';
load(tmp_data);

[vx vy] = sift_flow(im_in, im_ex_w);

bin_alpha_in = bin_alpha(mask_in(:,:,1));
bin_alpha_ex = bin_alpha(mask_in(:,:,1)) | bin_alpha(mask_ex(:,:,1));

save('sift_flow.mat', 'vx', 'vy', 'vxm', 'vym', 'bin_alpha_in', 'bin_alpha_ex');

function output=bin_alpha(input) 
    output=input;
    % In case the mask is not perfec / too small
    output(output<0.5)=0;
    output(output>=0.5)=1;
    se = strel('disk', 71);  
    output = imdilate(output,se);

    % Add a small number to avoid crazy
    eps=1e-2;
    output = output + eps;
end