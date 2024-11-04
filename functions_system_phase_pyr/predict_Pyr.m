function [Y_p,I_0] = predict_Pyr(input,X_s)
pyr = input.pyr;
ngs = input.ngs;
tel = input.tel;
QE = input.QE;
wvl_factor = input.wvl_factor;
pyr2zern = input.pyr2zern;
I_0 = input.I_0;
nAll = size(X_s,4);

idx1 = ((pyr.c-1)/2) * pyr.nLenslet + 1 : ((pyr.c-1)/2 + 1) * pyr.nLenslet;
idx2 = ((pyr.c-1)/2 + pyr.c) * pyr.nLenslet + 1 : ((pyr.c-1)/2 + pyr.c + 1) * pyr.nLenslet;

[Mm Mn] = size(pyr2zern);
pp2=reshape(pyr2zern,[Mm,64,64]);
pp21 = pp2(:,idx1,idx1);pp22 = pp2(:,idx2,idx1);pp23 = pp2(:,idx1,idx2);pp24 = pp2(:,idx2,idx2);
pp3 = cat(2,pp21,pp22);pp4=cat(2,pp23,pp24);pp5 =cat(3,pp3,pp4);
pyr2zern2=pp5(:,:);
I_0 = [I_0(idx1,idx1),I_0(idx2,idx1);I_0(idx1,idx2),I_0(idx2,idx2)];
for kIter = 1:nAll 
        
        
pyr_frame   = X_s(:,:,kIter);pyr_frame=pyr_frame/sum(pyr_frame(:))-I_0;

% Ground-truth - FOURIER GENERATOR CASE
zCoefs_pyr =  wvl_factor*pyr2zern2*pyr_frame(:);
% crop image
Y_p(kIter,:) = single(zCoefs_pyr)';
  
end

return