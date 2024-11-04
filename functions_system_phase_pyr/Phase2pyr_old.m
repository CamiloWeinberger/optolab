function [X_s,Y_p] = Phase2pyr(X_phase,Y_s,ron1,r0,wSpeed,magNGS1)
parFileName = 'parFileLOOPS4';
[nAll m] = size(Y_s);
eval(parFileName);
check = false;
ron = ron1;
magNGS = magNGS1;
% source
ngs = source('wavelength',photoNGS,'magnitude',magNGS);

% telescope
tel = telescope(D,'obstructionRatio',cobs,'resolution',nPxPup);
pupZeroPad = padarray(tel.pupil,[(Samp+1)*nPxPup/2,(Samp+1)*nPxPup/2]);

atm = atmosphere(photoNGS,r0,L0,'altitude',0,'fractionnalR0',1,...
    'windSpeed',wSpeed,'windDirection',wDirection*pi/180);

pyr = pyramid(nLenslet,nPxPup,'modulation',pyrMod,'binning',pyrBinning,'c',Samp);
ngs = ngs.*tel*pyr;


pyr.INIT
I_0 = pyr.camera.frame./sum(pyr.camera.frame(:));

wvl = ngs.wavelength;


%% ZERNIKE RECONSTRUCTION MATRIX : FULL-FRAME TO ZERNIKE MODE

% define Zernike Modes
zernRec  = zernike(jIndex,tel.D,'resolution',tel.resolution);
iMat     = interaction_matrix(ngs,tel,pyr,zernRec.modes);
pyr2zern = pinv(iMat);
wvl_factor  = wvl*1e9/2/pi; % from rad 2 nm


%% LOOP - MULTI CPU
% indexes for cropping the pyramid camera frame

% Give workers access to OOMAO functions
addAttachedFiles(gcp,{'telescope.m','telescopeAbstract.m','pyramid.m','source.m'})


% Manage noise
if ron
    pyr.camera.readOutNoise = ron;
    pyr.camera.photonNoise = true;
    pyr.camera.quantumEfficiency = QE;
end



idx1 = ((pyr.c-1)/2) * pyr.nLenslet + 1 : ((pyr.c-1)/2 + 1) * pyr.nLenslet;
idx2 = ((pyr.c-1)/2 + pyr.c) * pyr.nLenslet + 1 : ((pyr.c-1)/2 + pyr.c + 1) * pyr.nLenslet;


for kIter = 1:nAll 
        
        
% propagate through the pyramid
n2          = times(ngs,tel);
n2.phase    = X_phase(:,:,kIter);
n2          = mtimes(n2,pyr);
pyr_frame   = pyr.camera.frame./sum(pyr.camera.frame(:))-I_0;    

% Ground-truth - FOURIER GENERATOR CASE
zCoefs_pyr =  wvl_factor*pyr2zern*pyr_frame(:);
% crop image
pyr_frame = ([pyr.camera.frame(idx1,idx1),pyr.camera.frame(idx2,idx1);pyr.camera.frame(idx1,idx2),pyr.camera.frame(idx2,idx2)]);

X_s(:,:,kIter) = single(pyr_frame);
Y_p(kIter,:) = single(zCoefs_pyr)';
  
end

return