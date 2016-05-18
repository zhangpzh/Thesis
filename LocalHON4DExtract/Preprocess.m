function [gx,gy,gz,frameInfo] = Preprocess(depthImageDirectPth, boundingBoxDirectPth)
% ��ȡDepthImage Ŀ¼������ depth ͼƬ������
depthImagesName = dir(fullfile(depthImageDirectPth,'*.png'));
% ��ʼ�����ͼ������
depthImageSize=[374 512];
depth = zeros(depthImageSize(1),depthImageSize(2),length(depthImagesName));

% Read images
for i=1:length(depthImagesName)
    filePth = fullfile(depthImageDirectPth,depthImagesName(i).name);
    image = imread(filePth);
    image = image(:,:);
    % ԭʼdepth��������ȥ��bodyIndex
    image = bitshift(image,-3);
    depth(:,:,i) = image;
end

% Compute derivatives of depth
num_frames = size(depth,3);
gx =zeros(size(depth,1),size(depth,2),size(depth,3)-1);
gy =zeros(size(depth,1),size(depth,2),size(depth,3)-1);
gz =zeros(size(depth,1),size(depth,2),size(depth,3)-1);
for indFrame=1:num_frames-1
    im1 = medfilt2(depth(:,:,indFrame), [5 5]);
    im2 = medfilt2(depth(:,:,indFrame+1), [5 5]);
    [dx,dy,dz] = gradient(cat(3,im1,im2),1,1,1);
    
    gx(:,:,indFrame) = dx(:,:,1);
    gy(:,:,indFrame) = dy(:,:,1);
    gz(:,:,indFrame) = dz(:,:,1);
end

% Compute bounding boxes' information of current video
frameInfo = [];

% Read bounding boxes
fid=fopen(boundingBoxDirectPth,'r');
% frame number, personId, topLeftPointX, topLeftPointY, width, height
contents = textscan(fid,'%s%u%u%u%u%u');
frameNumber_vector = contents{1};
personId_vector = contents{2};
topLeftPointX_vector = contents{3};
topLeftPointY_vector = contents{4};
width_vector = contents{5};
height_vector = contents{6};
fclose(fid);

% Initialize frameInfo
for i = 1:num_frames
    frameInfo(i).boxNumber =0;
    frameInfo(i).boxes = [];
end
    
%num2str(i,'%02d');

% Iterate all records(each takes a single line) in BBoxes.txt
for i = 1:size(frameNumber_vector,1)
    % ��ȡ��ǰ��¼��֡��
    frameNumberOfCurRecord = str2num(char(frameNumber_vector(i)));
    % ֡��Ϊ frameNumberOfCurRecord ����һ�� bounding box
    frameInfo(frameNumberOfCurRecord).boxNumber = frameInfo(frameNumberOfCurRecord).boxNumber + 1;
    % ��¼�е� index �Ǵ�0�����, ��matlab �Ǵ�1�����
    beginColumn = topLeftPointX_vector(i)+1;
    beginRow = topLeftPointY_vector(i)+1;
    endColumn = topLeftPointX_vector(i) + width_vector(i);
    endRow = topLeftPointY_vector(i) + height_vector(i);
    
    tmpBox = [personId_vector(i) beginColumn endColumn beginRow endRow];
    
    frameInfo(frameNumberOfCurRecord).boxes = cat(1,frameInfo(frameNumberOfCurRecord).boxes, tmpBox);
end

