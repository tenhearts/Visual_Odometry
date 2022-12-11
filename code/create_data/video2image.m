function [] = video2image(input_path,output_path)
 vid=VideoReader(input_path);
 numFrames = round(vid.Duration*vid.FrameRate);
 n=numFrames;
 j = 0;
 
 for i = 1:10:n
 frames = read(vid,i);
 newname = sprintf('%04d',j);
 imwrite(frames,[output_path newname '.jpg']);
 j = j+1;
 end
end