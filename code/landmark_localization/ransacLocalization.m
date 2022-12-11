function [T_WC, best_inlier_mask, num_iterations] ...
    = ransacLocalization(p_query, W_landmarks, K, max_num_iterations, max_dist, use_p3p)

if use_p3p
    k = 3;
else
    k = 6;
end


% Initialize RANSAC.
best_inlier_mask = [];
max_num_inlier = 0;
num_iterations = 0;

if  size(W_landmarks,2) < 60 && size(W_landmarks,2) >= 40
    max_num_iterations = 10000;
    disp('Increasing RANSAC now: level1')
elseif size(W_landmarks,2) < 40 && size(W_landmarks,2) >= 20
    max_num_iterations = 20000;
    disp('Increasing RANSAC now: level2')
elseif size(W_landmarks,2) < 20
    max_num_iterations = 30000;
    disp('Increasing RANSAC now: level3')
end


for i = 1:max_num_iterations
    % Model from k samples (DLT or P3P)
    [query_samples, idx] = datasample(p_query, k, 2, 'Replace', false);
    landmark_samples = W_landmarks(:, idx);
    
    if use_p3p
        % convert p_query to bearing vectors
        p_query_homo = K \ [query_samples ; ones(1,size(query_samples,2))];
        % normalize bearing vectors
        p_query_homo = p_query_homo ./ vecnorm(p_query_homo,2,1);
        
        M_p3p = real(p3p(landmark_samples, p_query_homo)); 
        M_p3p = reshape(M_p3p,3,4,4);
        
        % check the four solutions of P3P
        M_test = eye(3,4);
        for j = 1:4
            M_test(:,1:3) = M_p3p(:,2:4,j)';
            M_test(:,4) = - M_p3p(:,2:4,j)'*M_p3p(:,1,j); % T_WC to T_CW

            % rotate landmarkes into camera frame
            C_landmarks = M_test * [W_landmarks; ones(1,size(W_landmarks,2))];

            % project matched 3D landmarks to 2D camera plane
            p_landmarks_proj = projectPoints(C_landmarks, K);

            % get inliers
            inlier_mask = (sum((p_landmarks_proj - p_query).^2,1) < (max_dist^2));
            num_inliers = sum(inlier_mask);

            % check if we could get more inliers
            if num_inliers > max_num_inlier
                max_num_inlier = num_inliers;
                best_inlier_mask = inlier_mask;
                num_iterations = i;
            end
        end

    else
        T_CW = dlt(query_samples, landmark_samples, K);
        C_landmarks = T_CW * [W_landmarks; ones(1,size(W_landmarks,2))];
        p_landmarks_proj = projectPoints(C_landmarks, K);

        % get inliers
        inlier_mask = (sum((p_landmarks_proj - p_query).^2,1) < (max_dist^2));
        num_inliers = sum(inlier_mask);

        % check if we could get more inliers
        if num_inliers > max_num_inlier
            max_num_inlier = num_inliers;
            best_inlier_mask = inlier_mask;
            num_iterations = i;
        end
    end

end
    T_CW = dlt(p_query(:,best_inlier_mask),W_landmarks(:,best_inlier_mask), K);
    T_WC = invert_homo_trans(T_CW);
    initial_guess = HomogMatrix2twist(T_WC);
    error_terms = @(x) pose_refinement_error(x, W_landmarks(:,best_inlier_mask) , double(p_query(:,best_inlier_mask)), K);
    options = optimoptions(@lsqnonlin, 'Display','Off','MaxIter', 50);
    refined_twist = lsqnonlin(error_terms, initial_guess, [], [], options); 

    T_WC = twist2HomogMatrix(refined_twist);
    T_WC = T_WC(1:3,:);

end

