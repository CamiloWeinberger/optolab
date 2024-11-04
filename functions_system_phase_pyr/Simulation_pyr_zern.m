function [X_phase,X_s,Y_s] = Simulation_pyr_zern(nAll,jIndex,noiseVar,r0,wSpeed,Y_s)
parFileName = 'parFileLOOPS4';
eval(parFileName);
check = false;

% source
ngs = source('wavelength',photoNGS,'magnitude',magNGS);

% telescope
tel = telescope(D,'obstructionRatio',cobs,'resolution',nPxPup);
pupZeroPad = padarray(tel.pupil,[(Samp+1)*nPxPup/2,(Samp+1)*nPxPup/2]);

atm = atmosphere(photoNGS,r0,L0,'altitude',0,'fractionnalR0',1,...
    'windSpeed',wSpeed,'windDirection',wDirection*pi/180);

% wavefront-sensor
pyr = pyramid(nLenslet,nPxPup,'modulation',pyrMod,'binning',pyrBinning,'c',Samp);
% Flat for FULL FRAME
ngs = ngs.*tel*pyr;
pyr.INIT
ngs = ngs.*tel*pyr;
I_0 = pyr.camera.frame./sum(pyr.camera.frame(:));

wvl = ngs.wavelength;

%% Phase screen generator


    % ZERNIKE CLASS%
     zernGen = zernike(jIndex,tel.D,'resolution',tel.resolution);
     nZern   = numel(jIndex); % ZERNIKE POLYNOMES  - ARRAY OF SIZE nPix^2 x nModes
     zModes  = zernGen.modes;
%     % DISTRIBUTION OF AMPLITUDE
     if numel(zStdMax) == 1
         zStdMax = zStdMax * ones(nZern,1);
     else
         zStdMax = reshape(zStdMax,nZern,1);
    end
    if numel(zMean) == 1
        zMean = zMean * ones(nZern,1);
    else
        zMean = reshape(zMean,nZern,1);
    end
%     
%     if strcmpi(zDistrib,'NORMAL')
%         % Normal distribution of each of the nZ_Gen modes
%         zAmplitude =  bsxfun(@plus,zMean,bsxfun(@times,zStdMax,randn(nZern,nAll)));
%     elseif strcmpi(zDistrib,'UNIFORM')
%         zAmplitude = zMean - 5*zStdMax + 2*(5*zStdMax - zMean)*rand(nZern,nAll);
%     end
zAmplitude = Y_s;
% zMean = zAmplitude + 5*zStdMax + 2*(5*zStdMax - zMean)*rand(nZern,nAll);

%% ZERNIKE RECONSTRUCTION MATRIX : FULL-FRAME TO ZERNIKE MODE

% define Zernike Modes
zernRec  = zernike(jIndex,tel.D,'resolution',tel.resolution);
iMat     = interaction_matrix(ngs,tel,pyr,zernRec.modes);
pyr2zern = pinv(iMat);
wvl_factor  = wvl*1e9/2/pi; % from rad 2 nm
ph2zern     = pinv(zernRec.modes);

% if you change the phase amplitude, you'll see that the pyramid-based
% Zernike reconstruction degrades -> optical gain issue
if check
    close all;
    amp_z       = 1000/wvl_factor; %50 nm in amplitude
    z_true      = randn(nZern,1)*amp_z;
    ngs         = ngs.*tel;
    ngs.phase   = reshape(zernRec.modes*z_true,tel.resolution,[]);
    ngs         = ngs*pyr;
    
    % phase map to zernike
    c1          = wvl_factor*ph2zern*ngs.phase(:);
    zMap1       = reshape(sum(zernRec.modes*c1,2),tel.resolution,[]);
    zMap1       = zMap1 - mean(zMap1(tel.pupilLogical(:)));
    
    % pyramid full-frame to zernike
    pyr_frame   = pyr.camera.frame./sum(pyr.camera.frame(:))-I_0;
    c2          = wvl_factor*pyr2zern*pyr_frame(:);
    zMap2       = reshape(sum(zernRec.modes*c2,2),tel.resolution,[]);
    zMap2       = zMap2 - mean(zMap2(tel.pupilLogical(:)));
    
    % check numbers
    sqrt(ngs.var)*wvl_factor
    sqrt(sum(c1.^2))
    sqrt(sum(c2.^2))
end
clear pyr_frame
%% LOOP - MULTI CPU
% indexes for cropping the pyramid camera frame

% Give workers access to OOMAO functions
addAttachedFiles(gcp,{'telescope.m','telescopeAbstract.m','pyramid.m','source.m'});



 clear fx fr fxExt fy fyExt index psdKolmo photoNGS toolboxesList zernRec
[X_phase,X_s,Y_s] = Iter_simul_zern(nAll,atm,pyr,tel,ngs,zAmplitude,nPxPup,noiseVar,wvl_factor,ph2zern,pyr2zern,I_0,wvl,zModes);


return
