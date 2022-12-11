function is_reasonable_landmark = sanity_check_landmarks(T_CW, W_landmarks,sanity_check_factor)
% This function checks if the landmarks are reasonable:
% In front of the camera and not to far away
%
% INPUT
%       T_C_W - 3x4 - homogenous transformation from world frame to camera
%       W_landmarks - 3xN - position of N landmark in world frame

C_landmarks = T_CW * [ W_landmarks ; ones( 1 , size(W_landmarks,2) ) ];


is_positive_landmark = (C_landmarks(3,:) > 0);
C_new_landmarks = C_landmarks(:,is_positive_landmark);

    
average_depth = median(C_new_landmarks(3,:));

is_reasonable_landmark = (C_landmarks(3,:) > 0) & (C_landmarks(3,:) < average_depth*sanity_check_factor);
% if (C_landmarks(3,:) < 0 & abs(C_landmarks(3,:) < 1e-04))
%     C_landmarks(3,:) = 0;
% end
% is_reasonable_landmark = (C_landmarks(3,:) >= 0);


disp(['C_landmarks is: ' num2str(sum(is_reasonable_landmark))]);
disp(['average depth * sanity_check_factor is:' num2str(average_depth*sanity_check_factor)]);


end