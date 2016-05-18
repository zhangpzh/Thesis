datasetRootDirectPth='I:\kinect-dataset(multi-view for activity)\view-yongyi';
depthSubDirect='DepthImage';
boundingBoxesLocalPth='BBoxes.txt';

% HON4D parameters
DIM = 120; % number of 4D projectors
P = loadProjector('000.txt',DIM);

% һ�� patch ��СΪ 126 x 42, ��Ϊ 3x1 �� cell, ÿ�� cellΪ 42 x 42
patch = [];
patch.cellNum = 3;
patch.height = 126;
patch.width = 42;

% һ�� cell ��СΪ 42 x 42,��Ϊ 3x3 �� grid, ÿ�� grid Ϊ 14 x 14
cell = [];
cell.numR = 3;
cell.numC = 3;
cell.numD = 1;
cell.width = 42;
cell.height = 42;
cell.depth = 4;
cell.DIM = DIM;

% ������ȡ HON4D ��������Ƶ�ε�֡��
tStep = cell.depth/cell.numD;

% Involved video number: 1~6, 15~95
videoNumberInvolved = zeros(1,87);
for i=1:6
    videoNumberInvolved(i)=i;
end
for i=7:size(videoNumberInvolved,2)
    videoNumberInvolved(i)=i+8;
end

% video01~06, 15~95 �� label
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
    % ��ǰvideo����ȡ������ ������Ƶ�� HON4D �����Ķ�ά����, ÿһ����һ�� ������Ƶ�� HON4D ����
    HON4D_OfCurVideo = [];
    % 'videoxx'
    videoName=strcat('video',num2str(videoNumberInvolved(i),'%02d'));
    % Get depth image directory path
    depthImageDirectPth = fullfile(datasetRootDirectPth,videoName,depthSubDirect);
    % Get bounding boxes path
    boundingBoxDirectPth = fullfile(datasetRootDirectPth,videoName,boundingBoxesLocalPth);
    
    % Preprocessing
    [gx,gy,gz,frameInfo] = Preprocess(depthImageDirectPth, boundingBoxDirectPth);
    % ���ڼ�¼��ǰ video ��������Ƶ�ε� HON4D ��������Ŀ
    cnt_segmentHON4D_OfCurrentVideo = 0;
    depthImagesName = dir(fullfile(depthImageDirectPth,'*.png'));
    num_frames = length(depthImagesName);
    for f=1:num_frames
        Sum_HON4D_OfBoundingBoxes = zeros(1,120*3*3*3);
        
        % �����жϵ�ǰ֡�Ƿ��кϷ��� HON4D ��������ȡ����
        tag = 0;
        
        for ct=0:cell.numD-1 %�����ѭ��һ��
            % Compute tmin and tmax
            nPointt = f - ((cell.numD-1)/2)*tStep - .5*tStep;
            tmin = nPointt + tStep*ct;
            tmax = tmin + tStep-1;
            % ��� tmin �� tmax ��[1,num_frames-1]֮�� �� f ֡�� bounding box ���һ������
            if tmin >= 1 && tmax <= num_frames-1 && frameInfo(f).boxNumber > 0
                tag = frameInfo(f).boxNumber;
                %���� frameInfo(f) �е� ÿһ�� bounding box
                for index=1:frameInfo(f).boxNumber
                    % ��� һ�� bounding box
                    box = frameInfo(f).boxes;
                    box = box(index,:);
                    % �����ݶ������� gx_box, gy_box, gz_box. ��Ⱦ�Ϊ tmax-tmin+1
                    beginColumn = box(2);
                    endColumn = box(3);
                    beginRow = box(4);
                    endRow = box(5);
                    gx_box = gx(beginRow:endRow,beginColumn:endColumn,tmin:tmax);
                    gy_box = gy(beginRow:endRow,beginColumn:endColumn,tmin:tmax);
                    gz_box = gz(beginRow:endRow,beginColumn:endColumn,tmin:tmax);
                    % ���ݶ������� gx_box, gy_box, gz_box ��ÿһ����� resize �� 126 x 42
                    % �Ķ�ά����, resizeʱ Ĭ��ʹ����'bicubic' ˫���β�ֵ�����������ݶ�
                    gx_box = imresize(gx_box,floor([patch.height patch.width]));
                    gy_box = imresize(gy_box,floor([patch.height patch.width]));
                    gz_box = imresize(gz_box,floor([patch.height patch.width]));
                    % �� patch �ֳ� 3x1 �� cell, ��ÿ�� cell �ֳ� 3x3 �� grid, �������ܹ�
                    % 27 �� grid �� 1x120 HON4D �����������һ�� 1x(120x3x3x3) ��������
                    HON4D_OfBoundingBox = [];
                    HON4D_OfBoundingBox = [HON4D_OfBoundingBox, GetHON4D_OfCell(gx_box(1:42,1:42,:),...
                            gy_box(1:42,1:42,:),gz_box(1:42,1:42,:), cell, P)]; % ��һ�� cell
                    HON4D_OfBoundingBox = [HON4D_OfBoundingBox, GetHON4D_OfCell(gx_box(43:84,1:42,:),...
                            gy_box(43:84,1:42,:),gz_box(43:84,1:42,:), cell, P)]; % �ڶ��� cell
                    HON4D_OfBoundingBox = [HON4D_OfBoundingBox, GetHON4D_OfCell(gx_box(85:126,1:42,:),...
                        gy_box(85:126,1:42,:),gz_box(85:126,1:42,:), cell, P)]; % ������ cell
                    % bounding box �� HON4D ���������ۼ�
                    Sum_HON4D_OfBoundingBoxes = Sum_HON4D_OfBoundingBoxes + HON4D_OfBoundingBox;
                end
            end
        end
        % ����֡�кϷ��� HON4D ��������ȡ����
        if tag ~= 0
            % ��ƽ��
            Sum_HON4D_OfBoundingBoxes = Sum_HON4D_OfBoundingBoxes./tag;
            % �������֡ f Ϊ���ĵ�������Ƶ�ε� HON4D���� (1x(120x3x3x3)) �洢����ά������
            HON4D_OfCurVideo = [HON4D_OfCurVideo;Sum_HON4D_OfBoundingBoxes];
        end
    end
    
    current_video_data= [];
    current_video_data.feature = HON4D_OfCurVideo;
    current_video_data.label = labels(i);
    
    % �� 'current_video_data' �洢Ϊ videoxx_HON4D_feature.mat �ļ�����Ӧ�� video ���ļ�����
    fileName = strcat(videoName,'_HON4D_feature.mat');
    savePth = fullfile(datasetRootDirectPth,videoName,fileName);
    save(savePth,'current_video_data');
end