function regular_testing(video)


% load('net');
% net.layers{end-1} = struct('type', 'conv', ...
%     'filters', randn(1,1,64,1, 'single'), ...
%     'biases', zeros(1, 1, 'single'), ...
%     'stride', 1, ...
%     'pad', 0) ;

method = 'manual';
frames = 200;

load(['net/' method '/' num2str(frames) '/' video '/net-epoch-20']);



net.layers{end} = struct('name', 'data_hat_sigmoid', ...
    'type', 'sigmoid'         );

net = vl_simplenn_move(net,'gpu');
load meanPixel;

%imagePath= video;
%S = regexp(imagePath, '/', 'split');
%category = S{1};    video_i = S{2};
% load(['../meanPixel/', category, '_', video_i, '_meanPixel']);

half_size = 15;


imgDir = ['../'  video '/Test/input'];
resDir = ['../Result/' method '/' num2str(frames) '/' video];

mkdir(resDir);

images = dir([imgDir '/*.jpg']);

for kk = 1 : numel(images)
    %fprintf('%d:',kk);
    imagename = images(kk).name;
    im = single(imread([imgDir '/' imagename]));
    
    if size(im,1) > 400 || size(im,2) >400
        im = imresize(im, 0.5, 'nearest');
    end
    
    
    im_large = padarray(im,[half_size,half_size],'symmetric');
    im_large = bsxfun(@minus, im_large, meanPixel);
    im_large = gpuArray(im_large);
    
    %tic;
    A = vl_simplenn(net, im_large);
    B = gather(A(end).x);
    %toc;
    
    map_im = uint8(B * 255);    
    if size(im,1) > 400 || size(im,2) >400
        map_im = imresize(im, [size(im,1),size(im,2)], 'nearest');
    end
    imagename = strrep(imagename, '.jpg', '.png');
    imwrite(map_im,[resDir '/' imagename]);
end
