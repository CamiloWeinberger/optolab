addpath(genpath('./OOMAO'))
addpath ./functions_system_phase_pyr

Include_BLT = 0;
random_BLT_phase = 0;   % different phases in each frame
N = 300;
jIndex = 2:210;
resol = 560;%268;
part = 1;
sampling = 1/1000;% 500Hz

if Include_BLT == 1
    name_var = '_piston';
else
    name_var = '';
end


%D:
% Paranal: 8.2m
% Las campanas: 1 y 2.5 m
% GMT: 25.5
% La silla:  3.6 m
% Tololo: 2.2 a 4 m
% SOAR: 4.1 m
% Gemini Sur: 8.1 m
% E-ELT: 42 m
% APEX: 12 m
% some tels: https://www.conicyt.cl/documentos/Fichasobservatorios.pdf


%[0.01 0.02 0.04 0.06 0.08 0.1]
for D = 1.5% [2.5 4 12 25.5 42]
    carpeta  = ['Dataset_Phasemap/Datasets_phasemap_D' num2str(D) '_sequential_Jun_2023' name_var '_experimental'];

    if ~exist(carpeta)
        mkdir(carpeta)
    end
    
    for r0 = [.03 .04 .05] %[.15 .16 .18]
        [X_phase,Y_z] = Generate_phasemap_sequential(N,jIndex,r0,resol,D, sampling);

        
        %%%%%%%%%%%%%%%%% Piston for BLT
        if Include_BLT == 1
            [X_phase,Dpiston] = include_BLT(X_phase,random_BLT_phase);
            Y_z = cat(2,Dpiston,Y_z);
        end


        name = ['./' carpeta '/Phase_' num2str(resol) 'px_r0_' num2str(r0) '_part_' num2str(part) '.mat'];
        save(name,'X_phase','Y_z');
        disp(['saved ' num2str(r0)])
    end
end
exit