function keypoint_collection = feature_detection_NMS(img,rows,cols,radius,npoints,...
    bins,minQuality,FilterSize,MinContrast,MetricThreshold,Threshold,NumOctaves,NumScaleLevels,...
    ScaleFactor,NumLevels,border_treshold,choice)

%Find keypoints in image by dividing it into multiple rectangular regions of interest
%   INPUT
%   rows    -   number of rectangular regions in vertical direction
%   cols    -   number of rectangular regions in horizontal direction
%   radius  -   percentage of included keypoints per rectangle
%   npoints -   number of points per region
%   OUTPUT
%   keypoints - 2xk matrix

% Division into multiple regions of interest
[M,N] = size(img);
% take floor, lost pixels are neglected
column_width = floor(N/cols);
row_width = floor(M/rows);

% Define regions of interest
col_indices = 1:column_width:N;
row_indices = 1:row_width:M;

img = histeq(img,bins); 

% imshow(img); hold on;

keypoint_collection = [];
for ic = 1:cols
    for ir = 1:rows
        if choice == 0
            point_struct = detectHarrisFeatures(img,'ROI',...
                [col_indices(ic) row_indices(ir) column_width row_width],...
                'MinQuality',minQuality,'FilterSize',FilterSize);
        elseif choice == 1
            point_struct = detectBRISKFeatures(img,'ROI',...
                [col_indices(ic) row_indices(ir) column_width row_width],...
                'MinContrast', MinContrast, 'MinQuality',minQuality);
        elseif choice == 2
            point_struct = detectFASTFeatures(img,'ROI',...
                [col_indices(ic) row_indices(ir) column_width row_width],...
                'MinContrast', MinContrast, 'MinQuality',minQuality);
        elseif choice == 3
            point_struct = detectORBFeatures(img,...%'ROI',...[col_indices(ic) row_indices(ir) column_width row_width],...       
                'ScaleFactor',ScaleFactor, 'NumLevels', NumLevels);
        elseif choice == 4
            point_struct = detectKAZEFeatures(img,...%'ROI',...[col_indices(ic) row_indices(ir) column_width row_width],...
                'Threshold', Threshold, 'NumOctaves', NumOctaves, 'NumScaleLevels', NumScaleLevels);
        else
            point_struct = detectMinEigenFeatures(img,'ROI',...
                [col_indices(ic) row_indices(ir) column_width row_width],...
                'MinQuality',minQuality,'FilterSize',FilterSize);
        end
        
        % maxima supression
        keypoints = [];
        flag = false;
        for ip = 1:npoints
            %find maximum only in unselected and undeleted points
            [~,ind_max] = max(point_struct.Metric);
            %include the newly found maximum if not to close to image border
            candidate = point_struct.Location(ind_max,:)';
            if isempty(candidate(1,:))
                break;
            end
            if candidate(1) > border_treshold && ...
               candidate(1) < N-border_treshold && ...
               candidate(2) > border_treshold && ...
               candidate(2) < M-border_treshold %&& ...
%                 candidate(2) < 285
           
               keypoints = [keypoints candidate];
            end
            %find close points
            distances = pdist2(point_struct.Location(ind_max,:)...
                ,point_struct.Location);
            ind_suppressed = any(distances<radius,1).';
            %check whether deleting more points depletes the reservoir
            if (numel(point_struct.Metric) - sum(ind_suppressed)) < npoints && ~flag
                flag = true;
            end
            %exclude the newly suppressed points
            point_struct(ind_suppressed) = [];
            
        end
        
        keypoint_collection = [keypoint_collection keypoints];

    end
%          scatter(keypoint_collection(1,:),keypoint_collection(2,:)); %plotting
%          legend('KAZE Feature')
%          hold on;

end
end