%% MANAGE workspaces
close all; % close all figures
clear all; % clear the workspace
clc
addpath(genpath('./oomao-master'))
addpath(genpath('./OOMAO'))
addpath functions_system_phase_pyr;

pup = load(['Dataset_pyramid/I0_raw_M20.mat']).I_0;
pup = pup/max(pup(:))>.1;

Mod = 5;
resol = 268;
Zn = 201;
sys = Generate_system_mod(Mod);


zern = zernike(2:Zn+1,'resolution',resol).modes;

phase = 0.1*reshape(zern,[268,268,Zn]);

[I_0,~] = Pyr2raw(phase(:,:,1)~=0,1,sys);
nPhotons = sum(I_0(:));
I_0 = I_0/nPhotons;
save(['Dataset_pyramid/I0_raw_M' num2str(Mod) '.mat'],'I_0')

[sp ,~] = Pyr2raw(phase,Zn,sys);
[sm ,~] = Pyr2raw(-phase,Zn,sys);
iMat = 0.5*(sp-sm)/0.1;
iMat = iMat.*pup;
iMat = reshape(iMat,[size(sp,1).^2,Zn]);
pyr2zern = pinv(iMat);
save(['Dataset_pyramid/iMat_' num2str(resol) '_M' num2str(Mod) '.mat'],'iMat','pyr2zern')
