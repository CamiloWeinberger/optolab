%% MANAGE workspaces
close all; % close all figures
clear all; % clear the workspace
clc
addpath(genpath('./oomao-master'))
addpath(genpath('./OOMAO'))
addpath functions_system_phase_pyr;
%load('sys268_struct.mat')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Params! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Mod = 0;  % modulation
Include_BLT = 0; % for phase over the system
random_BLT_phase = 1;   % different phases in each frame
date_f = 'Nov_2022';  % Nov_2022 (all range), Apr_2023 (low range)
type = 'train' % train or test
resol = 268;%560; % resolution
Photon_noise = 1;
D_tel = 1.5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


sys = Generate_system_mod(Mod,resol);
%sys.pyr.camera.photonNoise = Photon_noise;   %%%% IMPORTANT!!!!
%sys.pyr.modulation = Mod;


if strcmp(date_f,'Nov_2022')
    r0_init = .01:.01:.2;
    %r0_init = 0.08:0.02:0.2
elseif strcmp(date_f,'Apr_2023')
    r0_init = .01:.01:.06;
end
V=5;

if strcmp(type,'test')
    parts = 1;
else
    if length(r0_init) == 6
        parts = 33;
    else
        parts = 10;
    end
end

if Include_BLT == 1
    name_var = '_piston_splitted';
else
    name_var = '_splitted';
end

if length(r0_init) == 6
    name_var = ['_lowr' name_var];
end

if strcmp(type,'train')
    path_save = ['/home/bizon/Desktop/CamiloWein/Dataset_pyramid/Datasets_phase2pyr_D' num2str(D_tel) '_M' num2str(Mod) '_RandMag' name_var '/'];
    path_train = [path_save 'train/'];
    path_val = [path_save 'val/'];
    path_val_phase = [path_save 'val_phase/'];
    if ~exist(path_save, 'dir')
        mkdir(path_save)
    end
    if ~exist(path_train, 'dir')
        mkdir(path_train)
    end
    if ~exist(path_val, 'dir')
    mkdir(path_val)
    end
    if ~exist(path_val_phase, 'dir')
        mkdir(path_val_phase)
    end
end
count = 1;
split_val = 0.8;


for D = D_tel
    %  carpeta_1  = ['Dataset_Phasemap/Datasets_phasemap_D' num2str(D) '_Nov_2022_test/']; %phasemap 
    %  carpeta = ['Dataset_pyramid/Datasets_phase2pyr_D' num2str(D) '_Nov_2022_M' num2str(Mod) name_var]; % to save
    
    carpeta_1  = ['Dataset_Phasemap/Datasets_phasemap_D' num2str(D) '_' date_f '/']; %phasemap
    
    if strcmp(type,'test')
        carpeta = ['Dataset_pyramid/Datasets_phase2pyr_D' num2str(D) '_' date_f '_M' num2str(Mod) name_var '']; % to save                
        if ~exist(['./' carpeta], 'dir')
            mkdir(['./' carpeta])
        end
    end

    for part = 1:parts
        display(part)
        for r0 = r0_init
            r0
            name_in = ['./' carpeta_1 '/Phase_' num2str(resol) 'px_r0_' num2str(r0) '_part_' num2str(part) '.mat'];
            load(name_in)

            


            %%%%%%%%%%%%%%%%% Piston for BLT
            if Include_BLT == 1
                [X_phase,Dpiston] = include_BLT(X_phase,random_BLT_phase);
                Y_z = cat(2,Dpiston,Y_z);
            end
            
            Y_z = Y_z*550/2/pi;

            max(Y_z(:))

            %Y_z = Y_z*2*pi/550; % set in rads
            
            [X_s,~] = Pyr2raw_randMag(X_phase,size(X_phase,3),sys);
            %[X_s,~] = Pyr2raw_randMag(X_phase(:,:,1),1,sys);


            if strcmp(type,'test')
                name_out = ['./' carpeta '/Raw_' num2str(resol) 'px_r0_' num2str(r0) '_part_' num2str(part) '.mat'];
                save(name_out,'X_s','Y_z','r0_var','wvl_factor','X_phase');
            
            elseif strcmp(type,'train')
                total_file = size(X_s,3);
                for idx = 1:total_file
                    if idx > split_val*total_file
                        path1 = path_val;
                        path2 = path_val_phase;
                        Xs = single(X_s(:,:,idx));
                        Yz = Y_z(idx,:); 
                        X_ph = X_phase(:,:,idx);
                        save([path1 num2str(count) '.mat'], 'Xs','Yz');
                        %save([path2 num2str(count) '.mat'], 'X_ph');
                    else
                        path1 = path_train;
                        Xs = single(X_s(:,:,idx));
                        Yz = Y_z(idx,:); 
                        save([path1 num2str(count) '.mat'], 'Xs','Yz');
                    end
                count = count + 1;

                end
            end
        end
    end
end

