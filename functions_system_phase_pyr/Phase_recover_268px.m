function [out,out2] = Phase_recover_268px(Y,zernRec)
%
%Recover the phase from the zernikes modes in nm
%
%INPUTS:
% - Y = sernike mode
% - input = generated system
%
% OUTPUT:
% - out = fasemap
% - out2 = alphaData pupil

    zernRec = zernRec;
    if length(Y)>=size(zernRec,2)
        Y=Y(1:size(zernRec,2));
    end
    out       = reshape(zernRec(:,1:length(Y))*Y(:),[268 268]);


end