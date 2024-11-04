% MANAGE workspaces random r_0
close all; % close all figures
% clear all; % clear the workspace
clc;


addpath functions_system_phase_pyr
addpath(genpath('./oomao-master'))
addpath(genpath('./OOMAO'))

wvl_factor = 550/2/pi;
% READ parfile
parFileName = 'parFileLOOPS4';
eval(parFileName);
check = false;

clc

% GROUND TRUTH
nAll        = .1*1e3;          % Number of simulations 1250000
%nAll        = 100;
genType     = 'FOURIER';        %can be Zernike, or Fourier
nZern       = 500;               % number of Zernike to be reconstructed for the Fourier generaton
jIndex      = 2:nZern+1;          % Nolls indexes of the Zernike modes to be simulated for the Zernike Generator - 1 is piston
noiseVar    = 0.1;              % WFS noise level in rad^2
% wSpeed      = 15;               % Wind speed in m/s
resol = 560;
V=5; 
r0_init = .01:.01:.2;
%r0_init = 0.08:0.02:0.2
%r0_init = .01:.01:.02;
r0_init = [.3]
length(r0_init)

if nAll > 1000
N = nAll/5; %number of examples
part = 1;
else 
N = nAll;
end

% INIT LOOP MEASURE
for D = [1.5];        % Telescope size
    
    carpeta  = ['Dataset_Phasemap/Datasets_phasemap_D' num2str(D) '_Nov_2022_exp'];
    %carpeta  = ['Dataset_Phasemap/Datasets_phasemap_D' num2str(D) '_Apr_2023'];
    %carpeta  = ['Dataset_Phasemap/Datasets_phasemap_D' num2str(D) '_Sep_2023'];
    if ~exist(['./' carpeta], 'dir')
       mkdir(['./' carpeta])
    end
    [X_phase,Y_z] = Generate_phasemap_var(r0_var,jIndex,r0,resol,D);
    X_phase = [];r0_var = [];

    for r0 = r0_init
        for part = 1%6:20
            count = 1;
            r0
            Y_z = zeros(N,jIndex(end)-1);Y_kl=Y_z;
            r0_var= round(rand(N,1)*.005+r0,4);  
            [X_phase,Y_z] = Generate_phasemap_var(r0_var,jIndex,r0,resol,D);
            %[X_phase,Dpiston] = include_BLT(X_phase,1);
            %Y_z = cat(2,Dpiston,Y_z);
            name = ['./' carpeta '/Phase_' num2str(resol) 'px'...
                '_r0_' num2str(r0) '_part_' num2str(part) '.mat'];
            save(name,'X_phase','Y_z','r0_var','wvl_factor');
        end
    end
end










%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Params! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nAll        = 1*1e3;          % Number of simulations 1250000
Mod = 0;  % modulation
Include_BLT = 0; % for phase over the system
random_BLT_phase = 1;   % different phases in each frame
date_f = 'Nov_2022';  % Nov_2022 (all range), Apr_2023 (low range)
type = 'train' % train or test
resol = 512; % resolution
Photon_noise = 1;
D_tel = 1.5;
genType     = 'FOURIER';        %can be Zernike, or Fourier
nZern       = 500;               % number of Zernike to be reconstructed for the Fourier generaton
jIndex      = 2:nZern+1;          % Nolls indexes of the Zernike modes to be simulated for the Zernike Generator - 1 is piston
noiseVar    = 0.1;              % WFS noise level in rad^2
resol = 560;
V=5; 
r0_init = .01:.01:.2;
r0_init = .01:.01:.06;
r0_init = [.3]
length(r0_init)
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
    path_save = ['./Dataset_pyramid/Datasets_phase2pyr_D' num2str(D_tel) '_M' num2str(Mod) '_RandMag' name_var '/'];
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
            
            r0_var= round(rand(N,1)*.005+r0,4);  
            [X_phase,Y_z] = Generate_phasemap_var(r0_var,jIndex,r0,resol,D);

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


                count = count + 1
                end
            end
        end
    end
end

