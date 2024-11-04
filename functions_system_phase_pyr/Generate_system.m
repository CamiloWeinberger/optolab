function [out] = Generate_system


parFileName = 'parFileLOOPS4';

eval(parFileName);
check = false;
% source
%ngs = source('wavelength',photoNGS,'magnitude',magNGS);
ngs = source('wavelength',photoNGS);

% telescope
tel = telescope(D,'obstructionRatio',cobs,'resolution',nPxPup);
pupZeroPad = padarray(tel.pupil,[(Samp+1)*nPxPup/2,(Samp+1)*nPxPup/2]);

% atm = atmosphere(photoNGS,r0,L0,'altitude',0,'fractionnalR0',1,...
%     'windSpeed',wSpeed,'windDirection',wDirection*pi/180);
atm = atmosphere(photoNGS,1,L0,'altitude',0,'fractionnalR0',1);
pyr = pyramid(nLenslet,nPxPup,'modulation',pyrMod,'binning',pyrBinning,'c',Samp);

ngs = ngs.*tel*pyr;
pyr.INIT;
ngs = ngs.*tel*pyr;

I_0 = pyr.camera.frame./sum(pyr.camera.frame(:));

wvl = ngs.wavelength;


%% ZERNIKE RECONSTRUCTION MATRIX : FULL-FRAME TO ZERNIKE MODE

% define Zernike Modes
zernRec  = zernike(jIndex,tel.D,'resolution',tel.resolution);
iMat    = interaction_matrix(ngs,tel,pyr,zernRec.modes); %pyr 2 zern

pyr2zern = pinv(iMat);

wvl_factor  = wvl*1e9/2/pi; % from rad 2 nm

% define Zernike Modes
ph2zern     = pinv(zernRec.modes);

%% LOOP - MULTI CPU
% indexes for cropping the pyramid camera frame

% Give workers access to OOMAO functions
% addAttachedFiles(gcp,{'telescope.m','telescopeAbstract.m','pyramid.m','source.m'});

out.pyr=pyr;
out.ngs=ngs;
out.tel=tel;
out.QE=QE;
out.wvl_factor=wvl_factor;
out.pyr2zern=pyr2zern;
out.ph2zern=ph2zern;
out.I_0=I_0;
out.zernRec=zernRec;
out.tag='OpticalSys';
return
