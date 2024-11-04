function [Y_pz,Y_pk,I_0] = predict_Pyr(input,X_s)
pyr = input.pyr;
ngs = input.ngs;
tel = input.tel;
QE = input.QE;
wvl_factor = input.wvl_factor;
pyr2zern = input.pyr2zern;
pyr2kl = input.pyr2kl;
I_0 = input.I_0;
if size(X_s,3) == 1
    nAll = size(X_s,4);
    mode = 1;
else
    nAll = size(X_s,3);
    mode = 2;
end
idx1 = ((pyr.c-1)/2) * pyr.nLenslet + 1 : ((pyr.c-1)/2 + 1) * pyr.nLenslet;
idx2 = ((pyr.c-1)/2 + pyr.c) * pyr.nLenslet + 1 : ((pyr.c-1)/2 + pyr.c + 1) * pyr.nLenslet;

[Mm Mn] = size(pyr2zern);
% pp2=reshape(pyr2zern,[Mm,64,64]);
% pp21 = pp2(:,idx1,idx1);pp22 = pp2(:,idx2,idx1);pp23 = pp2(:,idx1,idx2);pp24 = pp2(:,idx2,idx2);
% pp3 = cat(2,pp21,pp22);pp4=cat(2,pp23,pp24);pp5 =cat(3,pp3,pp4);
% pyr2zern2=pp5(:,:);
% I_0 = [I_0(idx1,idx1),I_0(idx2,idx1);I_0(idx1,idx2),I_0(idx2,idx2)];
for kIter = 1:nAll
    
    % I_0 = imresize(I_0,[size(X_s,1) size(X_s,1)]);
    if mode == 1
        pyr_frame   = double(X_s(:,:,1,kIter));pyr_frame=pyr_frame/sum(pyr_frame(:))-I_0;
    else
        pyr_frame   = double(X_s(:,:,kIter));pyr_frame=pyr_frame/sum(pyr_frame(:))-I_0;
        
    end
    % pyr2zern = imresize(pyr2zern,[size(pyr2zern,1) size(X_s,1).^2]);
    % pyr2kl = imresize(pyr2kl,[size(pyr2kl,1) size(X_s,1).^2]);
    
    % Ground-truth - FOURIER GENERATOR CASE
    zCoefs_pyr =  wvl_factor*pyr2zern*pyr_frame(:);
    zCoefs_kl =  wvl_factor*pyr2kl*pyr_frame(:);
    % crop image
    Y_pz(kIter,:) = single(zCoefs_pyr)';
    Y_pk(kIter,:) = single(zCoefs_kl)';
end

return