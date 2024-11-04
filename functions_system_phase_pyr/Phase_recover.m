function [out,out2] = Phase_recover(Y,input)
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

tel = input.tel;
zernRec = input.zernRec;
if length(Y)>=size(zernRec.modes,2)
Y=Y(1:size(zernRec.modes,2));
end
out       = reshape(zernRec.modes*Y(:),tel.resolution,[]);
out2 = tel.pupilLogical;
end