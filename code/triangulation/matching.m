function [validity_KLT_points, all_KLT_points, all_candidates]=...
    matching(img_previous,img_query,keypoints_database,maxBidirectionalError,threshold,score_limit,block_size,choice,dataset)
%MATCHING tracks the given keypoints from the database image to the query
%image using KLT and generates new candidates based on a HARRIS search in
%the query image, which then go through a closeness selection based on a
%given threshold
%   INPUT:
%   img_database, img_query - MxN images
%   keypoints_database      - 2xk1 matrix
%   threshold               - scalar
%   OUTPUT:
%   validity_KLT_points     - bolean vector k1x1
%   all_KLT_points          - vector 2xk1
%   all candidates          - vector 2xk2 where k2 is the number of candidates

if size(keypoints_database, 2) < 50
    img_previous = histeq(img_previous,64); 
    img_query = histeq(img_query,64); 
    maxBidirectionalError = 2;
    block_size = 33;
    threshold = 9;
else
    img_previous = histeq(img_previous,128); 
    img_query = histeq(img_query,128); 
end

tracker = vision.PointTracker('MaxBidirectionalError',maxBidirectionalError,'BlockSize',[block_size,block_size]);
%Initialize the tracker with the database keypoints and the database image
initialize(tracker,keypoints_database',img_previous);
%Find the corresponding features in the query_image and return the query keypoints, their validity and their score
[all_KLT_points,validity_KLT_points,score] = tracker(img_query);
validity_KLT_points = (validity_KLT_points & (score>score_limit))';
all_KLT_points = all_KLT_points';

% if sum(validity_KLT_points) < 50
%     disp(['Number of validity KLT points: ',num2str(sum(validity_KLT_points))]);
%     score'
% end

%% Finding keypoint candidates

%Generate possible new keypoints
keypoint_candidates = feature_detection_NMS...
    (img_query,dataset.configurations.detection{:},ceil(dataset.configurations.matching{4}/2),choice);

%Calculate distances to existing valid keypoints
distances = pdist2(all_KLT_points(:,validity_KLT_points)',keypoint_candidates');

%Find indices of candidates close to existing keypoints
%This is done by thresholding the distance matrix and then checking
%whether the candidates were close to any other existing keypoint
indices_noncandidates = any(distances<threshold,1);

all_candidates = keypoint_candidates(:,~indices_noncandidates);

end

