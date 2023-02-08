%% Code to stitch all tiles together

clc;
clear all
close all
warning('off')
blsh = '\';

% Asking for the folder where tiles are stored
atlas_filename = uigetdir("","ATLAS file pointer"); % filename for all folders
path_atlas = strcat(atlas_filename,"\");

% Creating empty big empty atlas array
ATLAS = uint16(zeros(450000,900000));
disp("Empty array created!");

% Running the loop
i = 0;
j = 0;
while i < 359
    while j < 179
        name = strcat(string(i),"_",string((j-90)),".tif");
        img = imread(strcat(path_atlas,string(name)));
        img = imresize(img,0.1);
        ATLAS((45000-(j+2)*250+1:45000-j*250) , (i*250+1:(i+2)*250)) = img;
        j = j+2;
    end
    j = 0;
    disp(i);
    i = i+2;
end

% Writing final atlas
imwrite(ATLAS,"Final_Atlas.tif");
disp("ATLAS made and saved!");

% Clearing RAM
clear all;
close all;