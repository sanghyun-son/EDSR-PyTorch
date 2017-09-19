scale = [2, 4, 8];
dataset = 'imagenet_val';
apath = '../../../../dataset';
hrDir = fullfile(apath, dataset, 'original');
lrDir = fullfile(apath, dataset, 'sr');

if ~exist(lrDir, 'dir')
    mkdir(lrDir)
end

for sc = 1:length(scale)
    lrSubDir = fullfile(lrDir, sprintf('X%d', scale(sc)));
    if ~exist(lrSubDir, 'dir')
        mkdir(lrSubDir);
    end
end

hrImgs = dir(fullfile(hrDir, '*.JPEG'));
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
        lrName = sprintf('%sx%d%s', woExt, scale(sc), ext);
        imwrite(lrImg, fullfile(lrDir, sprintf('X%d', scale(sc)), lrName));
    end
    if mod(idx, 100) == 0
        fprintf('Processed %d / %d images\n', idx, length(hrImgs));
    end
end
