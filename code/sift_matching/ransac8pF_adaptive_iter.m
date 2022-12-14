% Compute the fundamental matrix using the eight point algorithm and RANSAC
% Input
%   x1, x2 	  Point correspondences 3xN
%   threshold     RANSAC threshold
%
% Output
%   best_inliers  Boolean inlier mask for the best model
%   best_F        Best fundamental matrix
%
function [best_inliers, best_F] = ransac8pF_adaptive_iter(x1, x2, threshold)

%iter = 1000;
iter_count=0;
in_ratio=0;

num_pts = size(x1, 2); % Total number of correspondences
best_num_inliers = 0; best_inliers = [];
best_F = 0;

% while((1-(1-in_ratio^8)^iter_count)<0.95) % 0.99
while(iter_count<10000)
    
    % Randomly select 8 points and estimate the fundamental matrix using these.
    random_sample=randperm(num_pts);
    sample_8=random_sample(1:8);
    
    [Fh, F] = fundamentalMatrix(x1(:,sample_8), x2(:,sample_8));
    FF=Fh;
    
    % Compute the Sampson error.
    d_s=distPointsLines(x1,FF'*x2)+distPointsLines(x2,FF*x1);
    
    % Compute the inliers with errors smaller than the threshold.
    inlier_idx=find(d_s<threshold);
    inlier_count=size(inlier_idx,2);    
    
    % Update the number of inliers and fitting model if the current model
    % is better.
    if(inlier_count>best_num_inliers)
        best_num_inliers=inlier_count;
        best_inliers=inlier_idx;
        best_F=FF;
        in_ratio=best_num_inliers/num_pts;
        
        d_s_inlier=d_s(inlier_idx);
        mean_d_s_inlier=mean(d_s_inlier);
    end
    
    iter_count=iter_count+1;
   
end

end


