function img = get_frame(obj, frame_index)
    if( frame_index <= obj.last_frame )
        if isequal(obj.name, 'kitti')
            img = imread([obj.path '/00/image_0/' sprintf('%06d.png',frame_index)]);
        elseif isequal(obj.name, 'malaga')
            img = rgb2gray(imread([obj.path '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' obj.left_images(frame_index).name]));
        elseif isequal(obj.name, 'parking')
            img = rgb2gray(imread([obj.path sprintf('/images/img_%05d.png',frame_index)]));
        elseif isequal(obj.name, 'indoor_ros_img')
            img = rgb2gray(imread([obj.path sprintf('/indoor_ros_img/testimgs_ns%04d.png', frame_index+9500)]));
%             img = rgb2gray(imread([obj.path '/Images/' obj.Images(frame_index).name]));
        elseif isequal(obj.name, 'indoor_img_notag')
            img = rgb2gray(imread([obj.path sprintf('/indoor_img_notag_st10/mytest%d.jpg', frame_index)]));
        elseif isequal(obj.name, 'indoor_img_tag')
            img = rgb2gray(imread([obj.path sprintf('/indoor_img_tag_st10/mytest%d.jpg', frame_index)]));
            % 0 - Last time fail at 87
%             img = rgb2gray(imread([obj.path sprintf('/Images_4/mytest%d.jpg', frame_index+112)]));
            % 112 - Last time fail at 71
%             img = rgb2gray(imread([obj.path sprintf('/Images_4/mytest%d.jpg', frame_index+112+68)]));
            % 112+68 - Last time fail at 39
%             img = rgb2gray(imread([obj.path sprintf('/Images_4/mytest%d.jpg', frame_index+112+68+33  +100)]));
            % 112+68+33+100 - Last time fail at 508
%             img = rgb2gray(imread([obj.path sprintf('/Images_4/mytest%d.jpg', frame_index+112+68+33+100 +490)]));
            % 112+68+33+100+490 - Last time fail at 152
%             img = rgb2gray(imread([obj.path sprintf('/Images_4/mytest%d.jpg', frame_index+112+68+33+100+490 +142)]));
        elseif isequal(obj.name, 'indoor_stair')
%             img = rgb2gray(imread([obj.path '/outdoor_img_st30/' [obj.images(frame_index+278+116+50+80+50+29+125+100).name(1:end-4),'.jpg']]));
              img = rgb2gray(imread([obj.path '/indoor_img_stairs_st10/' [obj.images(frame_index).name(1:end-4),'.jpg']]));


        else
            error('Something is gone wrong')
        end
    else
        error('You want to get an frame after the last frame');
    end
end