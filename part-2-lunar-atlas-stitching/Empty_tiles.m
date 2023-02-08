%% Code to create blank tiles

clc;
clear all;
close all;
warning('off')

% Size of the blank tiles
blank_tile = (zeros(5000, 5000));

% Asking for the tilename where to store the empty tiles, i.e. tilename for all folders
atlas_filename = uigetdir;

% Looping to create 2 degree by 2 degree files, at a resolution half that of the original TMC files
i = 0;
j = -90;
while i < 359
    while j < 89
        thisimage = strcat(string(i), "_", string(j), ".tif");

        % Naming file relative to that directory
        fulldestination = fullfile(atlas_filename, thisimage);
        imwrite(uint16(blank_tile), fulldestination);
        j = j+2;
    end
    disp(i);
    i = i+2;
    j= -90;
end

clear blank_tile;
disp("BLANK OBJECT CREATION COMPLETED");