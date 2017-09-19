scale = [2, 3, 4];
dataset = 'DIV2K';
apath = '../../../../dataset';
quality = 100; 
hrDir = fullfile(apath, dataset, 'DIV2K_train_HR');
lrDir = fullfile(apath, dataset, ['DIV2K_train_LR_bicubic', num2str(quality)]);

if ~exist(lrDir, 'dir')
    mkdir(lrDir)
end

for sc = 1:length(scale)
    lrSubDir = fullfile(lrDir, sprintf('X%d', scale(sc)));
    if ~exist(lrSubDir, 'dir')
        mkdir(lrSubDir);
    end
end

hrImgs = dir(fullfile(hrDir, '*.png'));
for idx = 1:length(hrImgs)
    imgName = hrImgs(idx).name;
    try
        hrImg = imread(fullfile(hrDir, imgName));
    catch
        disp(imgName);
        continue;
    end
    [h, w, ~] = size(hrImg);
    for sc = 1:length(scale)
        ch = floor(h / scale(sc)) * scale(sc);
        cw = floor(w / scale(sc)) * scale(sc);
        cropped = hrImg(1:ch, 1:cw, :);
        lrImg = imresize(cropped, 1 / scale(sc), 'bicubic');
        [~, woExt, ext] = fileparts(imgName);
        lrName = sprintf('%sx%d%s', woExt, scale(sc), '.jpeg');
        imwrite( ...
            lrImg, ...
            fullfile(lrDir, sprintf('X%d', scale(sc)), lrName), ...
            'quality', quality);
    end
    if mod(idx, 100) == 0
        fprintf('Processed %d / %d images\n', idx, length(hrImgs));
    end
end
