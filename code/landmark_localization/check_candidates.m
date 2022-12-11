function [is_valid_landmark, is_wrong_triangulated, landmark_keypoints, reprojection_error,angles] = check_candidates(state_curr, K, ang_thr, reprojection_thr)
%
% IS_VALID_LANDMARK determines which matched keypoints are usable as landmarks by calculating
% the change of angle between the first and current observation of the keypoint
%
% INPUT:
%     state_curr: current state - state struct
%     K: camera intrinsics
%     ang_thr: threshold of angle for valid landmark - scalar, degree
%     age_thr: introduce long tracked keypoints that moved through frame as landmarks
%     px_thr: introduce long tracked keypoints that moved through frame as landmarks
% OUTPUT:
%     is_valid_landmark: boolean array of valid landmarks - 1xn matrix

T = state_curr.T; % pose of camera when landmark was first observed
F = state_curr.F; % coordinates of keypoints when first obsevred, ordered
first_obs = state_curr.first_observation_frame;

T_CW_curr = invert_homo_trans(state_curr.Pose);
C = state_curr.C; % current coordinates of keypoints, ordered

angles = zeros([1 size(T,3)]);

landmark_keypoints = zeros([3 size(T,3)]);

init_ray = zeros([3 size(T,3)]);
curr_ray = zeros([3 size(T,3)]);

for i = 1:size(angles,2)
    % triangulate keypoints (2xN, 3x4)
    landmark_keypoints(:,i) = linear_triangulation(C(:,i), F(:,i), K*T_CW_curr, K*invert_homo_trans(T(:,:,i)));
    
    % create connecting lines
    init_ray(:,i) = landmark_keypoints(:,i)-T(:,4,i);
    curr_ray(:,i) = landmark_keypoints(:,i)-state_curr.Pose(:,4);
    
    % calculate cosine
    % anglescos(i) = abs(dot(init_ray,curr_ray)/(norm(init_ray) * norm(curr_ray) ) );
    
    % calulate angle
    angles(i) = abs(atan2d(norm(cross(init_ray(:,i),curr_ray(:,i))),dot(init_ray(:,i),curr_ray(:,i))));
end

% Check if the reprojection error is too big
landmarks_keypoints_camera = T_CW_curr*[landmark_keypoints;ones(1,size(landmark_keypoints,2))];
landmarks_keypoints_camera = landmarks_keypoints_camera(1:3,:);
projected_points = projectPoints(landmarks_keypoints_camera, K);

reprojection_error = vecnorm(projected_points - C,2,1);

px_dist = vecnorm(F-C,2,1);

% apply threshold
is_wrong_triangulated = reprojection_error >= reprojection_thr;
% is_wrong_triangulated = reprojection_error >= 100;

% sum((px_dist > px_thr & state_curr.frame-first_obs > age_thr & reprojection_error < 10));
% is_valid_landmark = (angles > ang_thr & reprojection_error < 30) | (px_dist > px_thr & state_curr.frame-first_obs > age_thr & reprojection_error < 30);
is_valid_landmark = (angles > ang_thr & reprojection_error < reprojection_thr);
% is_valid_landmark = (angles >= ang_thr & reprojection_error < 100);


end
