%% run_bwskel.m
% Generate bwskel (morphological) skeletons from LKL images
% 
% Usage:
%   1. Set dataset to 'pmd' or 'stp'
%   2. Run this script in MATLAB from WholeBrainProject directory
%
% Outputs skeleton images to outputs/{dataset}/bwskel/

%% Configuration
dataset = 'pmd';  % Change to 'stp' for STP dataset

% Paths (relative to WholeBrainProject root)
BASE_DIR = pwd;
lkl_dir = fullfile(BASE_DIR, 'data', dataset, 'lkl');
output_dir = fullfile(BASE_DIR, 'outputs', dataset, 'bwskel');

%% Create output directory
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% Get all TIFF files
files = dir(fullfile(lkl_dir, '*.tif'));
fprintf('Processing %d images from %s\n', length(files), lkl_dir);
fprintf('Output directory: %s\n\n', output_dir);

%% Process each image
for i = 1:length(files)
    filename = files(i).name;
    fprintf('[%d/%d] %s\n', i, length(files), filename);
    
    % Read LKL image
    lkl = imread(fullfile(lkl_dir, filename));
    
    % Threshold to binary
    if max(lkl(:)) > 1
        lkl_bin = lkl > 20;  % Same threshold as other methods
    else
        lkl_bin = lkl > 0;
    end
    
    % Apply morphological skeleton
    skel = bwskel(lkl_bin);
    
    % Save skeleton
    output_path = fullfile(output_dir, filename);
    imwrite(uint8(skel) * 255, output_path);
    
    fprintf('  Skeleton pixels: %d\n', sum(skel(:)));
end

fprintf('\nDone! Skeletons saved to: %s\n', output_dir);
