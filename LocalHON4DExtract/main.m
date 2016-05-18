datasetRootDirectPth='I:\kinect-dataset(multi-view for activity)\view-yongyi';
depthSubDirect='DepthImage';
boundingBoxesLocalPth='BBoxes.txt';

% HON4D parameters
DIM = 120; % number of 4D projectors
P = loadProjector('000.txt',DIM);

% 一个 patch 大小为 126 x 42, 分为 3x1 个 cell, 每个 cell为 42 x 42
patch = [];
patch.cellNum = 3;
patch.height = 126;
patch.width = 42;

% 一个 cell 大小为 42 x 42,分为 3x3 个 grid, 每个 grid 为 14 x 14
cell = [];
cell.numR = 3;
cell.numC = 3;
cell.numD = 1;
cell.width = 42;
cell.height = 42;
cell.depth = 4;
cell.DIM = DIM;

% 用于提取 HON4D 的连续视频段的帧数
tStep = cell.depth/cell.numD;

% Involved video number: 1~6, 15~95
videoNumberInvolved = zeros(1,87);
for i=1:6
    videoNumberInvolved(i)=i;
end
for i=7:size(videoNumberInvolved,2)
    videoNumberInvolved(i)=i+8;
end

% video01~06, 15~95 的 label
% 1->talking
% 2->fighting
% 3->following
% 4->waiting in line
% 5->entering
% 6->gathering
% 7->dismissing

% video01~06, 15~18 -> talking
labels = [1,1,1,1,1,1,1,1,1,1];
% video19 -> fighting
labels = [labels,2];
% video20~35 -> following
tmp = zeros(1,16);
tmp = tmp+3;
labels = [labels,tmp];
% video36~39 -> waiting in line
tmp = zeros(1,4);
tmp = tmp+4;
labels = [labels,tmp];
% video40~47 -> following
tmp = zeros(1,8);
tmp = tmp+3;
labels = [labels,tmp];
% video48~51 -> fighting
tmp = zeros(1,4);
tmp = tmp+2;
labels = [labels,tmp];
% video52~57 -> waiting in line
tmp = zeros(1,6);
tmp = tmp+4;
labels = [labels,tmp];
% video58~62 -> fighting only
tmp = zeros(1,5);
tmp = tmp+2;
labels = [labels,tmp];
% video63~75 -> entering
tmp = zeros(1,13);
tmp = tmp + 5;
labels = [labels,tmp];
% video76~95 -> gathering and dismissing in turn
tmp = [6 7 6 7 6 7 6 7 6 7 6 7 6 7 6 7 6 7 6 7];
labels = [labels,tmp];

for i=1:size(videoNumberInvolved,2)
    % 当前video所提取出来的 连续视频段 HON4D 特征的二维数组, 每一行是一个 连续视频段 HON4D 特征
    HON4D_OfCurVideo = [];
    % 'videoxx'
    videoName=strcat('video',num2str(videoNumberInvolved(i),'%02d'));
    % Get depth image directory path
    depthImageDirectPth = fullfile(datasetRootDirectPth,videoName,depthSubDirect);
    % Get bounding boxes path
    boundingBoxDirectPth = fullfile(datasetRootDirectPth,videoName,boundingBoxesLocalPth);
    
    % Preprocessing
    [gx,gy,gz,frameInfo] = Preprocess(depthImageDirectPth, boundingBoxDirectPth);
    % 用于记录当前 video 中连续视频段的 HON4D 特征的数目
    cnt_segmentHON4D_OfCurrentVideo = 0;
    depthImagesName = dir(fullfile(depthImageDirectPth,'*.png'));
    num_frames = length(depthImagesName);
    for f=1:num_frames
        Sum_HON4D_OfBoundingBoxes = zeros(1,120*3*3*3);
        
        % 用于判断当前帧是否有合法的 HON4D 特征被提取出来
        tag = 0;
        
        for ct=0:cell.numD-1 %这里仅循环一次
            % Compute tmin and tmax
            nPointt = f - ((cell.numD-1)/2)*tStep - .5*tStep;
            tmin = nPointt + tStep*ct;
            tmax = tmin + tStep-1;
            % 如果 tmin 和 tmax 在[1,num_frames-1]之间 且 f 帧有 bounding box 则进一步计算
            if tmin >= 1 && tmax <= num_frames-1 && frameInfo(f).boxNumber > 0
                tag = frameInfo(f).boxNumber;
                %对于 frameInfo(f) 中的 每一个 bounding box
                for index=1:frameInfo(f).boxNumber
                    % 获得 一个 bounding box
                    box = frameInfo(f).boxes;
                    box = box(index,:);
                    % 剪出梯度立方体 gx_box, gy_box, gz_box. 厚度均为 tmax-tmin+1
                    beginColumn = box(2);
                    endColumn = box(3);
                    beginRow = box(4);
                    endRow = box(5);
                    gx_box = gx(beginRow:endRow,beginColumn:endColumn,tmin:tmax);
                    gy_box = gy(beginRow:endRow,beginColumn:endColumn,tmin:tmax);
                    gz_box = gz(beginRow:endRow,beginColumn:endColumn,tmin:tmax);
                    % 将梯度立方体 gx_box, gy_box, gz_box 的每一层矩阵 resize 成 126 x 42
                    % 的二维数组, resize时 默认使用了'bicubic' 双三次插值方法生成新梯度
                    gx_box = imresize(gx_box,floor([patch.height patch.width]));
                    gy_box = imresize(gy_box,floor([patch.height patch.width]));
                    gz_box = imresize(gz_box,floor([patch.height patch.width]));
                    % 将 patch 分成 3x1 个 cell, 将每个 cell 分成 3x3 个 grid, 计算这总共
                    % 27 个 grid 的 1x120 HON4D 特征并联结成一个 1x(120x3x3x3) 的行向量
                    HON4D_OfBoundingBox = [];
                    HON4D_OfBoundingBox = [HON4D_OfBoundingBox, GetHON4D_OfCell(gx_box(1:42,1:42,:),...
                            gy_box(1:42,1:42,:),gz_box(1:42,1:42,:), cell, P)]; % 第一个 cell
                    HON4D_OfBoundingBox = [HON4D_OfBoundingBox, GetHON4D_OfCell(gx_box(43:84,1:42,:),...
                            gy_box(43:84,1:42,:),gz_box(43:84,1:42,:), cell, P)]; % 第二个 cell
                    HON4D_OfBoundingBox = [HON4D_OfBoundingBox, GetHON4D_OfCell(gx_box(85:126,1:42,:),...
                        gy_box(85:126,1:42,:),gz_box(85:126,1:42,:), cell, P)]; % 第三个 cell
                    % bounding box 的 HON4D 特征向量累加
                    Sum_HON4D_OfBoundingBoxes = Sum_HON4D_OfBoundingBoxes + HON4D_OfBoundingBox;
                end
            end
        end
        % 若本帧有合法的 HON4D 特征被提取出来
        if tag ~= 0
            % 求平均
            Sum_HON4D_OfBoundingBoxes = Sum_HON4D_OfBoundingBoxes./tag;
            % 将这个以帧 f 为中心的连续视频段的 HON4D特征 (1x(120x3x3x3)) 存储到二维数组中
            HON4D_OfCurVideo = [HON4D_OfCurVideo;Sum_HON4D_OfBoundingBoxes];
        end
    end
    
    current_video_data= [];
    current_video_data.feature = HON4D_OfCurVideo;
    current_video_data.label = labels(i);
    
    % 将 'current_video_data' 存储为 videoxx_HON4D_feature.mat 文件到对应的 video 的文件夹中
    fileName = strcat(videoName,'_HON4D_feature.mat');
    savePth = fullfile(datasetRootDirectPth,videoName,fileName);
    save(savePth,'current_video_data');
end