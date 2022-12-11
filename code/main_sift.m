%% Setup
clc;
clear;

addpath(genpath('.'));

use_p3p = true; %else use DLT
enable_BA = false;
enable_3D = false;

ds = 0; % 0: KITTI, 1: Malaga, 2: parking, 3: indoor_ros, 4: indoor_img_notag
        % 5: indoor_img_tag 6: indoor_stair
feature_method = 0; % 0:Harris Features  1:BRISK Features  2:FAST Features
                    % 3:ORB Features  4:KAZE Features  5: MinEigen Features
                    
% Set paths of datasets
kitti_path = 'datasets/kitti';
malaga_path = 'datasets/malaga';
parking_path = 'datasets/parking';
indoor_path = 'datasets/mydataset';

all_configurations = jsondecode(fileread('configurations.json'));

if ds == 0
    assert(exist('kitti_path', 'var') ~= 0);
    dataset.name = 'kitti';
    dataset.path = kitti_path;
    dataset.first_frame = 0;
    dataset.last_frame = 4540;
    dataset.K = [7.188560000000e+02 0 6.071928000000e+02
                 0 7.188560000000e+02 1.852157000000e+02
                 0 0 1];
    dataset.has_ground_truth = true;
    ground_truth = load([kitti_path '/poses/00.txt']);
    dataset.ground_truth = permute(reshape(ground_truth',4,3,size(ground_truth,1)),[2 1 3]);
    dataset.configurations = all_configurations.kitti;
    
elseif ds == 1
    assert(exist('malaga_path', 'var') ~= 0);
    dataset.name = 'malaga';
    dataset.path = malaga_path;
    dataset.first_frame = 1;
    images = dir([malaga_path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images']);
    dataset.left_images = images(3:2:end);
    dataset.last_frame = length(dataset.left_images);
    dataset.K = [621.18428 0 404.0076
                 0 621.18428 309.05989
                 0 0 1];
    dataset.has_ground_truth = false;
    dataset.configurations = all_configurations.malaga;
    
elseif ds == 2
    assert(exist('parking_path', 'var') ~= 0);
    dataset.name = 'parking';
    dataset.path = parking_path;
    dataset.first_frame = 0;
    dataset.last_frame = 598;
    dataset.K = load([parking_path '/K.txt']);
    dataset.has_ground_truth = true; 
    ground_truth = load([parking_path '/poses.txt']);
    dataset.ground_truth = permute(reshape(ground_truth',4,3,size(ground_truth,1)),[2 1 3]);
    dataset.configurations = all_configurations.parking;

elseif ds == 3

    dataset.name = 'indoor_ros_img';
    dataset.path = indoor_path;
    dataset.first_frame = 1;
    all_images = dir([indoor_path '/indoor_ros_img']);
    dataset.images = all_images(3:end);
    dataset.last_frame = length(dataset.images);
    dataset.K = [310.76491497 0 330.72740083
                 0 308.63860267 221.16349911
                 0 0 1];
    dataset.has_ground_truth = false; 
    dataset.configurations = all_configurations.mydataset_ros;    
    
elseif ds == 4

    dataset.name = 'indoor_img_notag';
    dataset.path = indoor_path;
    dataset.first_frame = 0;
    images = dir([indoor_path ...
        '/indoor_img_notag_st10']);
    dataset.images = images(3:end);
    dataset.last_frame = length(dataset.images)-1;
    dataset.K = load([indoor_path '/K.txt']);
    dataset.has_ground_truth = false;
    dataset.configurations = all_configurations.mydataset_indoor;
    mydataset = true;
    
elseif ds == 5
%     assert(exist('malaga_path', 'var') ~= 0);
    dataset.name = 'indoor_img_tag';
    dataset.path = indoor_path;
    dataset.first_frame = 0;
    images = dir([indoor_path ...
        '/indoor_img_tag_st10']);
    dataset.images = images(3:end);
    dataset.last_frame = length(dataset.images)-1;
    dataset.K = load([indoor_path '/K.txt']);
    dataset.has_ground_truth = false;
    dataset.configurations = all_configurations.mydataset_indoor;
    mydataset = true;
    
elseif ds == 6
    dataset.name = 'indoor_stair';
    dataset.path = indoor_path;
    dataset.first_frame = 1;
    images = dir([indoor_path ...
         '/indoor_img_stairs_st10']);
    dataset.images = images(4:end);
    dataset.last_frame = length(dataset.images);
    dataset.K = load([indoor_path '/K.txt']);
    dataset.has_ground_truth = false;
    dataset.configurations = all_configurations.mydataset_indoor;
    mydataset = true;
else
    assert(false);
end

dataset.configurations.detection = struct2cell(dataset.configurations.detection);
dataset.configurations.matching = struct2cell(dataset.configurations.matching);
dataset.configurations.localization = struct2cell(dataset.configurations.localization);

%% Bootstrap

first_frame_index = dataset.first_frame;

%1. Get frames
first_frame = get_frame(dataset, first_frame_index);

% Find keypoints in first image
keypoints_first_frame = feature_detection_NMS(...
    first_frame,dataset.configurations.detection{:},ceil(dataset.configurations.matching{4}/2),feature_method);

tracked_keypoints_from_first_frame = keypoints_first_frame;
prev_frame = first_frame;

candidate_index = first_frame_index + 1;
found_next_keyframe = false;

while(~found_next_keyframe)

    candidate_frame = get_frame(dataset, candidate_index);

    %Track keypoints across frames
    [matched_logic_keyframe, all_KLT_keypoints, all_candidates] = matching(...
        prev_frame,candidate_frame,tracked_keypoints_from_first_frame,dataset.configurations.matching{:},feature_method,dataset);

    % extract the matched keypoints with help of the logic vectors
    keypoints_first_frame = keypoints_first_frame(:, matched_logic_keyframe);
    matched_keypoints_first_frame = keypoints_first_frame;
    tracked_keypoints_from_first_frame = all_KLT_keypoints(:, matched_logic_keyframe);
    
    %Find relative pose
    [F, inliers] = estimateFundamentalMatrix(matched_keypoints_first_frame',tracked_keypoints_from_first_frame','NumTrials',1000);
    inliers = inliers'; 

    [R_CW, t_CW] = decompose_fundamental_matrix(F,dataset.K,matched_keypoints_first_frame(:,inliers),tracked_keypoints_from_first_frame(:,inliers));


    % Transform to homogenous transformation
    T_CW = [R_CW t_CW];

    % Projection matrices for images 1 and 2
    M_1 = dataset.K * [eye(3) zeros(3,1)];   % set first camera frame as world frame
    M_2 = dataset.K * T_CW;

    % Triangulate landmarks in inertial frame, also outliers
    W_landmarks = linear_triangulation(matched_keypoints_first_frame,tracked_keypoints_from_first_frame, M_1, M_2);

    % Remove landmark that are not reasonable
    % e.g. behind camera or to far away or outliers
    is_reasonable_landmark = sanity_check_landmarks(T_CW, W_landmarks, dataset.configurations.triangulation.sanity_check_factor);
    W_landmarks = W_landmarks(:, is_reasonable_landmark & inliers);


    % Decide if we should take candiate as an keyframe
    average_depth = mean(W_landmarks(3,:));       % Compute average depth of triangulated landmarks
    keyframe_distance = norm(t_CW);               % Compute distance between keyframe and candidate keyframe
    ratio = keyframe_distance / average_depth;  

    if( ratio > 0.01)
        found_next_keyframe = true;
        disp('***************** BOOTSTRAPPING **********************');
        display(['Found new keyframe with index ', num2str(candidate_index) , newline, ...
            'Average depth = ', num2str(average_depth), '  ratio = ', num2str(ratio)]);
        disp('******************************************************');
    else
        display(['Still no keyframe with index ', num2str(candidate_index) , newline, ...
            'Average depth = ', num2str(average_depth), '  ratio = ', num2str(ratio)]);
        found_next_keyframe = false;
        candidate_index = candidate_index + 1; 
    end
    
end

% plotMatches(first_frame,candidate_frame,matched_keypoints_first_frame,tracked_keypoints_from_first_frame)

% **************************** CONVENTION *********************************
% state_prev.P - Previously matched keypoints
% state_prev.X - Landmarks corresponding to matched keypoints
% state_prev.C - Candidate keypoints
% state_prev.F - Image coordinates of first occurence of candidate keypoint
% state_prev.T - Pose at first occurence of candidate keypoint
% T in R^{3x4xN_T}
% extend T -> T(:,:,end+1:end+n_t) = repmat(Pose,1,1,n_t);
% state_prev.frame - number of treated image frame
% state_prev.Pose - Pose as homogeneous transformation in R^{3x4} = [R|t] omitting the last line.
% state_prev_first_observation_frame - frame in which the landmark was observed for the first time
% *************************************************************************
state_prev.P = tracked_keypoints_from_first_frame(: , inliers & is_reasonable_landmark);
state_prev.X = W_landmarks;
state_prev.C = [all_candidates, tracked_keypoints_from_first_frame(:, ~inliers) ];
state_prev.F = state_prev.C;
state_prev.Pose = invert_homo_trans(T_CW);
state_prev.T = repmat(state_prev.Pose, 1, 1, size(state_prev.C,2));  % 3x4xsize(state_prev.C,2)
state_prev.frame = candidate_index;
state_prev.first_observation_frame = state_prev.frame*ones(1,size(state_prev.C,2));
first_state = state_prev;
%% Continuous operation

% Initialize iteration, make robust for repetition without bootstrapping

% if starting from the beginning:
state_prev = first_state;
state_hist = first_state;

range = (candidate_index+1):dataset.last_frame;

BA_counter = 2;
no_keypoint_counter = 0;
BA_window_size = dataset.configurations.bundle_adjustment.window_size;
BA_periode = dataset.configurations.bundle_adjustment.periode;

for i = range
    
    frame = i;
    fprintf('\n\nProcessing frame %d\n=====================\n', frame);

    % Get new frame
    state_curr.frame = frame;
 
    curr_frame = get_frame(dataset, state_curr.frame);
    prev_frame = get_frame(dataset, state_prev.frame);

    % Match features from current frame against keypoints AND candidates from prev state
    [matched_logic_points, all_KLT_points, new_candidates] = matching(...
        prev_frame, curr_frame, [state_prev.P state_prev.C] , dataset.configurations.matching{:},feature_method,dataset);
    % Extract only keypoints
    all_KLT_keypoints = all_KLT_points(:, 1:size(state_prev.P,2));
    matched_logic_keypoints = matched_logic_points(:, 1:size(state_prev.P,2));
    % Extract only candidates
    all_KLT_candidates = all_KLT_points(:, size(state_prev.P,2)+1:end);
    matched_logic_candidates = matched_logic_points(:, size(state_prev.P,2)+1:end);

    % Fill keypoints and landmarks of current state
    state_curr.P = all_KLT_keypoints(:, matched_logic_keypoints);
    state_curr.X = state_prev.X(:, matched_logic_keypoints);

    % Sanity check: Do we still have valid keypoints?
%     if isempty(state_curr.P)
    if size(state_curr.P,2) < 8 || size(state_curr.X,2) < 8

        warning(['you have lost all zour keypoints at frame ',num2str(frame)])
        disp('Use SIFT')
        
        prev_frame = histeq(prev_frame,128); 
        curr_frame = histeq(curr_frame,128); 
        % SIFT
        [fc, dc] = vl_sift(single(prev_frame),'PeakThresh',1);
        [fd, dd] = vl_sift(single(curr_frame),'PeakThresh',1);
        
        [matched_logic_keyframe, scores] = vl_ubcmatch(dc, dd, 1.25);
        
        matched_keypoints_prev_frame = fc(1:2,matched_logic_keyframe(1,:));
        tracked_keypoints_curr_frame = fd(1:2,matched_logic_keyframe(2,:));
        
        new_candidates = fd(1:2,setdiff(1:size(fd,2),matched_logic_keyframe(2,:)));
        
        ransac_threshold = 2;
        [inliers, F] = ransac8pF_adaptive_iter([matched_keypoints_prev_frame;ones(1,size(matched_keypoints_prev_frame,2))],[tracked_keypoints_curr_frame;ones(1,size(tracked_keypoints_curr_frame,2))], ransac_threshold);
        matched_keypoints_prev_frame = matched_keypoints_prev_frame(1:2, inliers);
        tracked_keypoints_curr_frame = tracked_keypoints_curr_frame(1:2, inliers);
        
        [R_CW, t_CW] = decompose_fundamental_matrix(F,dataset.K,matched_keypoints_first_frame,tracked_keypoints_from_first_frame);
        T_CW = [R_CW t_CW];
        
        % Projection matrices for images 1 and 2
%         M_1 = dataset.K * state_prev.Pose;  % T_WC to T_CW
        M_1 = dataset.K * invert_homo_trans(state_prev.Pose);
        M_2 = dataset.K * invert_homo_trans(state_prev.Pose) * [T_CW;0 0 0 1];

        % Triangulate landmarks in inertial frame, also outliers
        W_landmarks = linear_triangulation(matched_keypoints_prev_frame,tracked_keypoints_curr_frame, M_1, M_2); 
%         T_WC_curr = invert_homo_trans(T_CW); % If not take last pose as current pose
        
        state_curr.P = tracked_keypoints_curr_frame;
        state_curr.X = W_landmarks;
%         [T_WC_curr, inlier_mask_localization, num_iterations] = ransacLocalization(state_curr.P, state_curr.X, dataset.K, dataset.configurations.localization{:}, use_p3p);
        state_curr.Pose = invert_homo_trans(T_CW); % Set current pose
%         inlier_mask_localization = inliers;
%         num_best_iterations = 0;
        
    else
        % Localization with P3P & RANSAC to detect inlier, followed by DLT on inliers and pose refinement minimizing the reprojection error
        [T_WC_curr, inlier_mask_localization, num_iterations] = ransacLocalization(state_curr.P, state_curr.X, dataset.K, dataset.configurations.localization{:}, use_p3p);
        state_curr.Pose = T_WC_curr; % Set current pose
        
        % Removal of outlier landmarks
        state_curr.X = state_curr.X(:, inlier_mask_localization);
        state_curr.P = state_curr.P(:, inlier_mask_localization);
    end
    
    if isempty(state_curr.P)
        no_keypoint_counter = no_keypoint_counter + 1;
        if no_keypoint_counter > 5
           error('stop it, tune your pipeline once more!!!') 
        end
    end
        


    % Check if remaining landmarks are in front of the camera and not too far away
    valid_landmarks = sanity_check_landmarks(invert_homo_trans(state_curr.Pose), state_curr.X, dataset.configurations.triangulation.sanity_check_factor);
    disp(['The number of valid landmarks is: ' num2str(sum(valid_landmarks))]);
    disp(['The number of invalid landmarks is: ' num2str(size(state_curr.X,2)-sum(valid_landmarks))]);
    
    
    % Remove points that do not pass sanity check
    state_curr.X = state_curr.X(:, valid_landmarks);
    state_curr.P = state_curr.P(:, valid_landmarks);
    
    % Remove too close keypoints if there are a lot of keypoints
    if size(state_curr.P, 2)>500
        is_keeper = sparsify_keypoints(state_curr.P);
        state_curr.X = state_curr.X(:, is_keeper);
        state_curr.P = state_curr.P(:, is_keeper);
    end

    % Fill in the candidates features, based on which features you have matched with previous candidates features
    state_curr.C = all_KLT_candidates(:, matched_logic_candidates); 
    state_curr.T = state_prev.T(:, :, matched_logic_candidates);
    state_curr.F = state_prev.F(:, matched_logic_candidates);
    state_curr.first_observation_frame = state_prev.first_observation_frame(matched_logic_candidates);

    % Check if current candidates are okay to use as landmarks
    [is_valid_landmark, is_wrong_triangulated, triangulated_candidates,reprojection_error,angles] = check_candidates(state_curr, dataset.K, dataset.configurations.triangulation.angle_treshold, dataset.configurations.triangulation.reprojection_error_treshold);
    is_reasonable_landmark = sanity_check_landmarks(invert_homo_trans(state_curr.Pose), triangulated_candidates, dataset.configurations.triangulation.sanity_check_factor);
    new_found_landmarks = triangulated_candidates(: ,is_valid_landmark & is_reasonable_landmark);
    
    if sum(is_valid_landmark) < 20
        disp(['reprojection_error: ',num2str(sum(reprojection_error<50))]);
        disp(['angles: ',num2str(angles)]);
    end
    
    if sum(is_reasonable_landmark) < 20
        disp('Problem in sanity check');
    end
    
    new_found_keypoints = state_curr.C(: ,is_valid_landmark & is_reasonable_landmark);

    disp(['Number of added landmarks: ',num2str(sum(is_valid_landmark & is_reasonable_landmark))]);

    % Remove new triangulated landmarks from candidates and add them to landmarks and keypoints
    state_curr.P = [state_curr.P new_found_keypoints];
    state_curr.X = [state_curr.X new_found_landmarks];
    
    % Remove also wrongly triangulated points
    state_curr.C = state_curr.C(:, ~(is_valid_landmark & is_reasonable_landmark) & ~is_wrong_triangulated);
    state_curr.T = state_curr.T(:, :, ~(is_valid_landmark & is_reasonable_landmark) & ~is_wrong_triangulated);
    state_curr.F = state_curr.F(:, ~(is_valid_landmark & is_reasonable_landmark) & ~is_wrong_triangulated);
    state_curr.first_observation_frame = state_curr.first_observation_frame(~(is_valid_landmark&is_reasonable_landmark) & ~is_wrong_triangulated);

    % Add new candidates to state
    state_curr.C = [state_curr.C, new_candidates];
    state_curr.F = [state_curr.F, new_candidates];
    state_curr.first_observation_frame = [state_curr.first_observation_frame, state_curr.frame*ones(1,size(new_candidates,2))];
    state_curr.T(:, :, end+1:end+size(new_candidates,2)) = repmat(state_curr.Pose, 1, 1, size(new_candidates,2));

    % Makes sure that plots refresh.
    pause(0.01);

    state_prev = state_curr;
    state_hist = [state_hist, state_curr];
    
    % Bundle Adjustment
    BA_flag = false;
    if enable_BA == true && mod(BA_counter,BA_periode)==0 && BA_counter >= BA_window_size
        BA_flag = true;
        state_hist(1,end-BA_window_size+1:end) = bundle_adjustment(state_hist(1,end-BA_window_size+1:end), dataset.K, dataset.configurations.bundle_adjustment.max_iteration);
        state_prev = state_hist(1,end);
    end
    BA_counter = BA_counter + 1;

    plot_reconstruction_live(dataset, state_hist, enable_3D, BA_flag);
        
end
