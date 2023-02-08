%% Code to populate the empty tiles with TMC Images

clc;
clear all;
close all;
warning('off');
blsh = '\';

% Reading the details about the TMC images from excel sheet
exf = readtable("TMC_oth_data.xlsx");

% Empty array to store images that can potentially cause error
large_file = [];        % Too Large Error
absent_file = [];       % File detail not present in the spreadsheet

% Extracting the details present in the spreadsheet
names = exf.PRODUCT_ID;
UL_long = exf.UL_LON;       % Upper left longitude of the image
UR_long = exf.UR_LON;       % Upper right longitude of the image
LL_long = exf.BL_LON;       % Bottom left longitude of the image
LR_long = exf.BR_LON;       % Bottom right longitude of the image
UL_lat = exf.UL_LAT;        % Upper left latitude of the image
UR_lat = exf.UR_LAT;        % Upper right latitude of the image
LL_lat = exf.BL_LAT;        % Bottom left latitude of the image
LR_lat = exf.BR_LAT;        % Bottom right latitude of the image

data_filename = uigetdir("", "TMC folder");              % Filename for all folders
topLevelFolder = data_filename;                         % Get a list of all files and folders in this folder.
files = dir(topLevelFolder);                            % Get a logical vector that tells which is a directory.
dirFlags = [files.isdir];                               % Extract only those that are directories.
subFolders = files(dirFlags);                           % A structure with extra info. Get only the folder names into a cell array.
subFolderNames = {subFolders(3:end).name};              % Start at 3 to skip .
lenn = length(subFolderNames);                          % Numbers of sub-folders present
atlas_filename = uigetdir("", "Blank file pointer");     % Filename for all folders
path_atlas = strcat(atlas_filename, "\");

disp("BLANK ATLAS POINTING IMAGES IDENTIFIED");
disp("..........................................................");

% Processing the images and converting them into tiles
for index = 1:lenn
    path_in = strcat(data_filename, blsh, subFolderNames(index), blsh, "data", blsh, "derived");
    bottomLevelFolder = string(path_in);
    files_in = dir(bottomLevelFolder); 
    dirFlags_in = [files_in.isdir];
    subFolders_in = files_in(dirFlags_in);  
    subFolderNames_in = {subFolders_in(3:end).name};
    path = strcat(path_in, blsh, subFolderNames_in(1), blsh);
    path = string(path);
    tif_files = dir(fullfile(path, '*.tif'));

    try
        img = imread(fullfile(path,tif_files(1).name));
    catch me
        large_file = [large_file; string(subFolderNames(index))];
        disp("encountered large file skipped iteration");
        disp("........................................");
        continue;
    end

    disp(".tif image found and read successfully");
    try
        img = img * 128;
    catch me
        large_file = [large_file; string(subFolderNames(index))];
        disp("encountered large file skipped iteration");
        disp("........................................");
        continue;
    end

    disp("image colour calibrated and resized");
    count = find(names == string(subFolderNames(index)));
    
    if(isempty(count))
        absent_file = [absent_file; string(subFolderNames(index))];
        disp("file skipped because name not present in the list");
        disp(".................................................");
        continue;
    end
    
    count = count(1);                       % Removing duplicated
    imagelong = floor([LL_long(count), LR_long(count), UL_long(count), UR_long(count)]);
    imagelat = floor([LL_lat(count), LR_lat(count), UL_lat(count), UR_lat(count)]);
    ilon = imagelong < 0;
    imagelong = imagelong+ilon*360;
    imagelong1 = [LL_long(count), LR_long(count), UL_long(count), UR_long(count)];
    ilon = imagelong1<0;
    imagelong1 = ilon*360 + [LL_long(count), LR_long(count), UL_long(count), UR_long(count)];
    imagelat1 = [LL_lat(count), LR_lat(count), UL_lat(count), UR_lat(count)];
        clear ilon;
    min_long = min(imagelong);
    max_long = max(imagelong);
    min_lat = min(imagelat);
    max_lat = max(imagelat);
    
    if mod(min_long,2) == 1
        min_long = min_long -1;
    end
    if mod(min_lat,2) == 1
        min_lat = min_lat -1;
    end
    
    atlas_img_names = [];
    latc = 0;
    longc = 0;
    min_long1 = min_long;
    min_lat1 = min_lat;
    while min_long <= max_long
        while min_lat <= max_lat
            atlas_img_names = [atlas_img_names; strcat(string(min_long), "_", string(min_lat))];
            min_lat = min_lat+2;
            latc = latc+1;
        end
        min_long =  min_long+2;
        min_lat = min_lat1;
        longc = longc +1;
    end
    
    latc = latc/longc;
    try
        atlas_temp = uint16(zeros(5000*latc, 5000*longc));
    catch me
        disp("ARRAY CANNOT BE DECLARED");
        large_file = [large_file; string(subFolderNames(index))];
        disp("........................");
        continue;
    end
    
    for ii = 1:numel(atlas_img_names)
        c = floor((ii-1)/latc);
        jj = ii-c*latc;
        atimg = imread(strcat(path_atlas, atlas_img_names(ii), ".tif"));  %% check value added
        atlas_temp(5000*latc-jj*5000+1:5000*latc-(jj-1)*5000, c*5000+1:(c+1)*5000) = atimg; 
            clear atimg;
    end
    
    disp("blank atlas image created and corresponded");
    imagelong1 = floor(imagelong1*2500) - 2500*min_long1;
    imagelat1 = floor(imagelat1*2500) - 2500*min_lat1;
        clear min_long;
        clear max_long;
        clear min_lat;
        clear max_lat;
    
    targetHeight = imagelat1(3)-imagelat1(1);
    targetWidth = imagelong1(4)-imagelong1(3);
    targetSize = [targetHeight targetWidth];
    img = imresize(img, targetSize); 
        clear targetSize;
        clear targetHeight;
        clear heightI;
        clear targetWidth;
    
    try
        binmir = uint16(atlas_temp((5000*latc-imagelat1(3)+1:5000*latc-imagelat1(1)), (imagelong1(3)+1:imagelong1(4)))>0) + uint16(img>0);
    catch me
        disp("binary image not decleared");
        large_file = [large_file; string(subFolderNames(index))];
        continue;
    end
    
    atlas_temp((5000*latc-imagelat1(3)+1:5000*latc-imagelat1(1)), (imagelong1(3)+1:imagelong1(4))) = atlas_temp((5000*latc-imagelat1(3)+1:5000*latc-imagelat1(1)),(imagelong1(3)+1:imagelong1(4)))./binmir+uint16(img)./binmir;
        clear img;
    for ii = 1:numel(atlas_img_names)
        c = floor((ii-1)/latc);
        jj = ii-c*latc;
        atimg = atlas_temp(5000*latc-jj*5000+1:5000*latc-(jj-1)*5000, c*5000+1:(c+1)*5000);
        delete(strcat(path_atlas,atlas_img_names(ii), ".tif"));
        imwrite(uint16(atimg),strcat(path_atlas,atlas_img_names(ii), ".tif"));
            clear atimg;
    end

        clear atlas_temp;
    disp("old images deleted and new images added");
    fprintf("%s data concatenation completed (%d/%d)\n",string(subFolderNames(index)), index, lenn);
    disp("..........................................................");
end

% Noting the error files for debugging
disp("large files are as follows :");
disp(large_file);
disp("...............................");
disp("absent name files are as follows :");
disp(absent_file);

% Clearing RAM
clear all;
close all;