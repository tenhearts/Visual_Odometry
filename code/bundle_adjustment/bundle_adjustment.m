function adjusted_state_history = bundle_adjustment(state_history, K, max_iter )
% This function takes a history of states and put them into the same
% formulation of exercise 9(bundle ajustment). then runs lsqnonlin to
% reduce the total reprojection error. the same landmarks in different
% states should have the same coordinates, otherwise it does not work. The
% sliding window has to be therefore dense or smartly select only
% keyframes.
%
%   INPUT:
%           -state_history - 1xn - state structs
%           
%   OUTPUT:
%           -state_history - 1xn - state structs- adjusted pose and
%                                                 landmarks
%
    adjusted_state_history = state_history;
    
    % n - number of states
    n = size(state_history,2);
    ind_state2queue = cell(1,n); % stores for each state where it is in the queue
    num_keypoints = cell(1,n); % stores number of keypoints for each state 
    keypoints = cell(1,n); % stores keypoints for each state 
    landmarks = cell(1,n); % stores all landmarks for each state 
    poses = cell(1,n); % stores all poses for each state
    poses_adjusted = cell(1,n);
    start_index = 0;
    
    
    % extract all twists and concatenate all landmarks/keypoints and build
    % index
    for i = 1:n
        state = state_history(i);
        num_keypoints{i} = size(state.P,2);
        keypoints{i} = double(state.P);
        landmarks{i} = state.X;
        poses{i} = state.Pose;
        ind_state2queue{i} = (start_index+1):(start_index+num_keypoints{i});
        % 这12个keyframe中的所有keypoints
        start_index = start_index + num_keypoints{i};
    end
    
    % build unique landmark queue and its indices
    queue_landmark = cell2mat(landmarks);
    [queue_landmark_unique,index_to_unique, index_from_unique] = unique(queue_landmark','rows','stable');
    % 去掉这12个keyframe中所有重复的3D landmarks
    % queue_landmark_unique = queue_landmark(index_to_unique)
    % index_to_unique(:) = queue_landmark_unique(index_from_unique)
    
    queue_landmark_unique = queue_landmark_unique';
    index_to_unique = index_to_unique';
    index_from_unique = index_from_unique';
    m = size(queue_landmark_unique,2);
    queue_indices_unique = 1:m;
    queue_indices = queue_indices_unique(index_from_unique); % find which landmark index fits to which keypoints
    % 其实queue_indices也就等于index_from_unique
    
    observations = [n; m];
    hidden_state = zeros(6*n+3*m, 1);
    % 存储12个keyframe下的pose(用6*1的twist代替3*4的矩阵)和所有landmarks的3D坐标
    hidden_state(n*6+1:end, 1) = queue_landmark_unique(:);
    
    % fill in the information in observations and state
    for i = 1:n
       % fill in twist
       hidden_state((i-1)*6+1:i*6, 1) = HomogMatrix2twist(poses{i});
       
       O_i = [num_keypoints{i}; keypoints{i}(:); queue_indices(ind_state2queue{i})']; % ?x1 matrix
       observations = [observations;O_i];       
    end
        
    % runBA
    hidden_state_adjusted = runBA(hidden_state, poses{1}, poses{2}, observations, K, max_iter);
    
    queue_landmarks_adjusted_unique = reshape(hidden_state_adjusted(n*6+1:end), 3, m);
    queue_landmarks_adjusted = queue_landmarks_adjusted_unique(:, index_from_unique);
    
    % put adjusted hidden_state back into state history
    for i = 1:n
        state_adjusted = state_history(i);
        
        % 要在state中更新相机Pose，landmarks和T
        HomogMatrix = twist2HomogMatrix(hidden_state_adjusted((i-1)*6+1:i*6, 1));
        state_adjusted.Pose = HomogMatrix(1:3,1:4);
        state_adjusted.X = queue_landmarks_adjusted(:, ind_state2queue{i});
        poses_adjusted{i} = HomogMatrix(1:3,1:4);
        adjusted_state_history(i) = state_adjusted;
    end    
    
    % Replace all adjusted poses in the newest state
    T = state_history(1,end).T;
    % 只更新最后一个keyframe中每个candidate points第一次出现时的相机pose
    % 将其全部更新成最后一个keyframe用BA调整后的相机pose
    T_adjusted = T;
    n_T = size(T,3);
    for j = 1:n_T
       for i = 1:n
        if isequal(T(:,:,j),poses{i}) 
            T_adjusted(:,:,j) = poses_adjusted{i};
        end
       end
    end
    state_history(1,end).T = T_adjusted;
    
end