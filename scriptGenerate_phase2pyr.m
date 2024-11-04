%% MANAGE workspaces
close all; % close all figures
clear all; % clear the workspace
clc
addpath(genpath('./oomao-master'))
addpath(genpath('./OOMAO'))
addpath functions_system_phase_pyr;
%load('sys268_struct.mat')


Mod = 0;
Include_BLT = 0;



resol = 268;
sys = Generate_system_mod(Mod,resol);


%sys.pyr.modulation = Mod;

r0_init = .01:.01:.2;
%r0_init = .01:.01:.06;
%r0_init = .07:.01:.2;
V=5;
parts = 1;
photonNoise = 1;
sys.pyr.camera.photonNoise = photonNoise;
for D = [1.5]
    for Mag = [0 2 4 5 6]
        sys.ngs.magnitude = Mag;
        %sys = Generate_system_Mag(Mag);
        %sys.pyr.camera.photonNoise = photonNoise;
        carpeta_1  = ['Dataset_Phasemap/Datasets_phasemap_D' num2str(D) '_Nov_2022_test/']; %phasemap
        %carpeta = ['Dataset_pyramid/Datasets_phase2pyr_D' num2str(D) '_Nov_2022_M' num2str(Mod) '_ron' num2str(ron) '_photonN' num2str(photonNoise) '_test']; % to save
        carpeta = ['Dataset_pyramid/Datasets_phase2pyr_D' num2str(D) '_Nov_2022_M' num2str(Mod) '_Mag' num2str(Mag) '_noises_test']; % to save
        
        if Include_BLT == 1
            carpeta = [carpeta '_piston'];
        end
        
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
                
                %Piston for BLT
                if Include_BLT == 1
                    [X_phase,Dpiston] = include_BLT(X_phase,1);Y_z = cat(2,Dpiston,Y_z);
                end
                
                %X_phase = X_phase;
                %Y_z = Y_z*2*pi/550; % set in rads
                [X_s,~] = Pyr2raw(X_phase,size(X_phase,3),photonNoise,sys);
                save(name_out,'X_s','Y_z','r0_var','wvl_factor','X_phase');
            end
        end
    end
end 
