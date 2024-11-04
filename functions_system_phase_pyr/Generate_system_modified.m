function [out] = Generate_system_modified


parFileName = 'parFileLOOPS4';
jIndex =2:200
eval(parFileName);
check = false;
nPxPup = 64*4;


jIndex =2:200;
nZern = length(jIndex);
% source
%ngs = source('wavelength',photoNGS,'magnitude',magNGS);
ngs = source('wavelength',photoNGS);

% telescope
tel = telescope(D,'obstructionRatio',cobs,'resolution',nPxPup);
pupZeroPad = padarray(tel.pupil,[(Samp+1)*nPxPup/2,(Samp+1)*nPxPup/2]);

% atm = atmosphere(photoNGS,r0,L0,'altitude',0,'fractionnalR0',1,...
%     'windSpeed',wSpeed,'windDirection',wDirection*pi/180);
atm = atmosphere(photoNGS,1,L0,'altitude',0,'fractionnalR0',1);
pyr = pyramid(nPxPup/4,nPxPup,'modulation',pyrMod,'binning',pyrBinning,'c',Samp);

ngs = ngs.*tel*pyr;
pyr.INIT;
ngs = ngs.*tel*pyr;

I_0 = pyr.camera.frame./sum(pyr.camera.frame(:));

wvl = ngs.wavelength;


%% ZERNIKE RECONSTRUCTION MATRIX : FULL-FRAME TO ZERNIKE MODE

% define Zernike Modes
zernRec  = zernike(jIndex,tel.D,'resolution',tel.resolution);
dm = deformableMirror(17*17,'modes',zernRec,'resolution',nPxPup)

% iMat    = interaction_matrix(ngs,tel,pyr,zernRec.modes); %pyr 2 zern
iMat    = interaction_totalNorm(ngs,tel,pyr,zernRec.modes,nPxPup/4); %pyr 2 zern

% [kl_basis, M2C, eig, coef] = KLBasisDecomposition(tel,atm,dm,Samp);

kl_basis =KL_basis(tel,atm,nZern).KL_pure;
% iMat2    = interaction_matrix(ngs,tel,pyr,kl_basis); %pyr 2 kl
iMat2    = interaction_totalNorm(ngs,tel,pyr,zernRec.modes,nPxPup/4); %pyr 2 zern

pyr2zern = pinv(iMat);
pyr2kl = pinv(iMat2);

wvl_factor  = wvl*1e9/2/pi; % from rad 2 nm

% define Zernike Modes
ph2zern     = pinv(zernRec.modes);
ph2kl     = pinv(kl_basis);
%% LOOP - MULTI CPU
% indexes for cropping the pyramid camera frame

% Give workers access to OOMAO functions
% addAttachedFiles(gcp,{'telescope.m','telescopeAbstract.m','pyramid.m','source.m'});

out.pyr=pyr;
out.ngs=ngs;
out.tel=tel;
out.atm = atm;
out.QE=QE;
out.wvl_factor=wvl_factor;
out.pyr2zern=pyr2zern(1:end,:);
out.ph2zern=ph2zern(1:end,:);
out.I_0=I_0;
out.pyr2kl = pyr2kl;
out.ph2kl = ph2kl;
out.zernRec = zernRec;
out.kl_basis = kl_basis;
out.iMat = iMat;
out.iMat2 = iMat2;
out.tag='OpticalSys';
return
