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
nAll        = 80;          % Number of simulations 1250000
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
    
    carpeta  = ['Dataset_Phasemap/Datasets_phasemap_D' num2str(D) '_Nov_2022_exp_test'];
    %carpeta  = ['Dataset_Phasemap/Datasets_phasemap_D' num2str(D) '_Apr_2023'];
    %carpeta  = ['Dataset_Phasemap/Datasets_phasemap_D' num2str(D) '_Sep_2023'];
    if ~exist(['./' carpeta], 'dir')
       mkdir(['./' carpeta])
    end

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
