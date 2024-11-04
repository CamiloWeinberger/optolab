clear all
close all
clc
addpath(genpath('OOMAO'));
main = 'Dataset_Exp/';
impath = '004_Exp_Data_DP/';
phpath = '002_GroundTruth (GT)';
addpath(genpath(main))
files = dir([main phpath]);
pyr2zern = pinv(load('imat.mat').iMat);
io = load('imat.mat').I_0;

zern = zernike(2:501,'resolution',560).modes;
ph2zern = pinv(zern);
   
fold_out = ['Dataset_pyramid/Datasets_phase2pyr_D1.5_Nov_2022_M0_exp'];
if ~exist(fold_out)
    mkdir(fold_out)
end
%%
%figure(1);imagesc(I_0)
I_0 = crop(io);
save('Dataset_pyramid/I0_raw_M0_exp.mat','I_0');
%figure(2);imagesc(I_0);axis image
drawnow
%%
for idx = 1:length(files)-2
    clear X_phase X_s Y_z
    name = files(idx+2).name;
    pos_r = strfind(name,'r0_')+3;
    ro = str2num(name(pos_r:pos_r+5));
    X_phase = load(name).X_phase;
    Y_z2 = load(name).Y_z;
    xs = load([main impath name(1:end-4) 'Test_Axicon_Zern_mod1.mat']).X_s;
    Y_p = -py2z(xs,io,pyr2zern);
    Y_z = p2z(X_phase,ph2zern);
    X_s = crop(xs);
    file_name = [fold_out '/Raw_268px_r0_' num2str(ro) '_part_1.mat'];
    save(file_name,'Y_z','X_s','X_phase','Y_p');
    display(['r_0' num2str(ro) ', done'])
end
%%
% dim = 2
% ang = 180
% idx = 50
% x3 = flipdim(imrotate(imresize(X_s,[134 134]),ang),dim);
% io2 = flipdim(imrotate(imresize(I_0,[134 134]),ang),dim);
% Y_p2 = py2z(x3,io2,pyr2zern2);
% plot(1:50,Y_z(idx,1:50),1:50,Y_p(idx,1:50),1:50,Y_p2(idx,1:50),'linewidth',2);
% legend('gt','exp pyr','simul pyr')
%% check_ rotations
load('/home/bizon/Desktop/CamiloWein/Dataset_Exp/imat.mat')
figure(2)

%%
idx = 5
imag = reshape(iMat(:,idx),size(I_0));
%imag = imrotate(imag,-180);
%imag = flipdim(imag,1);
subplot(121);imagesc(reshape(zern(:,idx),[560 560]));axis image;subplot(122);imagesc(imag);axis image;


%%

function [out] = p2z(X,Y)
for idx = 1:size(X,3)
    x = X(:,:,idx);
    out(idx,:) = Y*x(:);
end
end

function [out] = py2z(X,I_0,Y)
I_0 = I_0/sum(I_0(:));
for idx = 1:size(X,3)
    x = X(:,:,idx);
    x = x/sum(x(:))-I_0;
    out(idx,:) = Y*x(:);
end
end


    
    
function [out] = crop(input)
D_sz = 301-91;
%xx = linspace(-1,1,D_sz+1);
%[X,Y] = meshgrid(xx,xx);
%mask = (sqrt(X.^2+Y.^2))<=1;
for idx = 1:size(input,3)
    x11 = [7,95];
    x12 = [1,473];
    x21 = [382,97];
    x22 = [379,473];
    i1 = input(x11(1):x11(1)+D_sz,x11(2):x11(2)+D_sz,idx);
    i2 = input(x12(1):x12(1)+D_sz,x12(2):x12(2)+D_sz,idx);
    i3 = input(x21(1):x21(1)+D_sz,x21(2):x21(2)+D_sz,idx);
    i4 = input(x22(1):x22(1)+D_sz,x22(2):x22(2)+D_sz,idx);
    im(:,:) = [i1 i2; i3 i4];
    im = imrotate(im,180);
    im = flipdim(im,2);
    out(:,:,idx) = im/sum(im(:));
end
end