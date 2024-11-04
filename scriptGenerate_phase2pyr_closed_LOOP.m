%% MANAGE workspaces
close all; % close all figures
clear all; % clear the workspace
clc
addpath(genpath('./oomao-master'))
addpath(genpath('./OOMAO'))
addpath functions_system_phase_pyr;
%load('sys268_struct.mat')
Mod = 0;
resol = 268;
sys = Generate_system_mod(Mod);
%sys.pyr.modulation = Mod;

r0_init = [.01:.01:.2];
V=5;
parts = 1;


% load iMat_268.mat
zernRec = load('./Dataset_pyramid/zernRec.mat').zernRec;
if Mod == 0
    pyr2zern = load('./Dataset_pyramid/iMat_268.mat').pyr2zern;
end

for D = [1.5]
    carpeta_1  = ['Dataset_Phasemap/Datasets_phasemap_D' num2str(D) '_Nov_2022_test/']; %phasemap
    carpeta = ['Dataset_pyramid/Datasets_phase2pyr_D' num2str(D) '_Nov_2022_M' num2str(Mod) '_test_closedLOOP']; % to save
                
    if ~exist(['./' carpeta], 'dir')
        mkdir(['./' carpeta])
    end

    for part = 1:parts
        part
        for r0 = r0_init
            r0
            name_in = ['./' carpeta_1 '/Phase_' num2str(resol) 'px_r0_' num2str(r0) '_part_' num2str(part) '.mat'];
            name_out = ['./' carpeta '/Raw_' num2str(resol) 'px_r0_' num2str(r0) '_part_' num2str(part) '.mat'];
            load(name_in)
            %Y_z = Y_z*2*pi/550; % set in rads
            [X_s,~] = Pyr2raw(X_phase,size(X_phase,3),sys);
            [Y_p] = predict(X_s,pyr2zern);
            X_r = reconstruction(Y_p,zernRec);
            Y_p = Y_p';
            Y_z(:,1:209) = Y_z(:,1:209)-Y_p(:,1:209)*550/2/pi;
            X_phase = X_phase-X_r;
            [X_s,~] = Pyr2raw(X_phase,size(X_r,3),sys);
            save(name_out,'X_s','Y_z','r0_var','wvl_factor','X_phase');
        end
    end
end