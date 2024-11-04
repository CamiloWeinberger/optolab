clear all
close all
clc

addpath functions_system_phase_pyr;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
date_f = 'Nov_2022';  % Nov_2022 (all range), Apr_2023 (low range)
Include_BLT = 0; % for phase over the system
random_BLT_phase = 1;   % different phases in each frame
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if Include_BLT == 1;
    name_var = '_piston';
else
    name_var = '';
end



path_load = ['/home/bizon/Desktop/CamiloWein/Dataset_Phasemap/Datasets_phasemap_D1.5_' date_f '/'];
path_save = ['/home/bizon/Desktop/CamiloWein/Dataset_pyramid/Datasets_phasemap_D1.5_' date_f name_var '_splitted/'];
path_train = [path_save 'train/'];
path_val = [path_save 'val/'];
if ~exist(path_save, 'dir')
    mkdir(path_save)
end
if ~exist(path_train, 'dir')
    mkdir(path_train)
end
if ~exist(path_val, 'dir')
    mkdir(path_val)
end

directory = (dir(path_load));
www = waitbar(0,'splitting data');
count = 1;
split_val = 0.7;
display('loading')

for N_file = 3:size(directory,1);
    clc
    display(['Splitting files([' repelem(['/','-'],[floor(N_file/2), floor((size(directory,1)-N_file)/2)]) '])     ' num2str(floor((N_file-1)/size(directory,1))) '%'])
    file_name = directory(N_file).name;
    load([path_load file_name]);

    if Include_BLT == 1
        [X_phase,Dpiston] = include_BLT(X_phase,random_BLT_phase);
        Y_z = cat(2,Dpiston,Y_z);
    end
    Y_z = Y_z*550/2/pi;

    total_file = size(X_phase,3);
    for idx = 1:total_file
        waitbar(count/(total_file*190),www)
        if idx > split_val*total_file
            path = path_val;
            Xs = X_phase(:,:,idx);
            Yz = Y_z(idx,:); 
            save([path num2str(count) '.mat'], 'Xs','Yz');
        else
            path = path_train;
            Xs = X_phase(:,:,idx);
            Yz = Y_z(idx,:); 
            save([path num2str(count) '.mat'], 'Xs','Yz');
        end
        count = count + 1;
    end

end