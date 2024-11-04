function [X_phase,X_s,Y_s,Y_p] = Simulation_pyr(nAll,genType,jIndex,ron1,r0,wSpeed)
parFileName = 'parFileLOOPS4';
eval(parFileName);
check = false;
ron = ron1;
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
pyr.INIT;
ngs = ngs.*tel*pyr;
I_0 = pyr.camera.frame./sum(pyr.camera.frame(:));

wvl = ngs.wavelength;

%% Phase screen generator

if strcmpi(genType,'FOURIER')
    fao = spatialFrequencyAdaptiveOptics(tel,atm,nLenslet+1,noiseVar,...
        loopGain, sampTime, latency, resAO,'pyramid',0,'modulation',pyrMod,...
        'nTimes',nTimes);
    %close all;
    close Figure 1
    [fxExt,fyExt] = freqspace(size(fao.fx,1)*fao.nTimes,'meshgrid');
    fxExt = fxExt*fao.fc*fao.nTimes;
    fyExt = fyExt*fao.fc*fao.nTimes;
    index = abs(fxExt)<fao.fc & abs(fyExt)<fao.fc;
    psdKolmo = fao.pistonFilter(hypot(fxExt,fyExt)).*phaseStats.spectrum(hypot(fxExt,fyExt),fao.atm);
    
    if loopGain == 0
        psd = psdKolmo;
        
    else
        psd        = zeros(size(fxExt));
        psd(index) = fao.noisePSD(fao.fx,fao.fy) + fao.anisoServoLagPSD(fao.fx,fao.fy);
        psd        = psd + fao.fittingPSD(fao.fx,fao.fy);
        
    end
    % Zernike modes to be reconstructed
    jIndex = 2:nZern+1;
   
    % few parameters to create the phase screen from the PSD
    N       = 2*Samp*nPxPup;
    L       = (N-1)*tel.D/(nPxPup-1);
    [fx,fy] = freqspace(N,'meshgrid');
    [~,fr]  = cart2pol(fx,fy);
    fr      = fftshift(fr.*(N-1)/L./2);
    [idx]           = find(fr==0);
    fourierSampling = 1./L;
    
    % Check the presence of the Control System tool box
    toolboxesList = ver;
    flagControlToolBox = any(strcmp(cellstr(char(toolboxesList.Name)), 'Control System Toolbox'));
    
    % Initialization of system parameters
    nPts        = size(fao.fx,1);
    RTF         = fao.atf;
    pupil       = tel.pupil;
    rngStream   = atm.rngStream;
    
else
    % ZERNIKE CLASS
    zernGen = zernike(jIndex,tel.D,'resolution',tel.resolution);
    nZern   = numel(jIndex);
    % ZERNIKE POLYNOMES  - ARRAY OF SIZE nPix^2 x nModes
    zModes  = zernGen.modes;
    % DISTRIBUTION OF AMPLITUDE
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
    
    if strcmpi(zDistrib,'NORMAL')
        % Normal distribution of each of the nZ_Gen modes
        zAmplitude =  bsxfun(@plus,zMean,bsxfun(@times,zStdMax,randn(nZern,nAll)));
    elseif strcmpi(zDistrib,'UNIFORM')
        zAmplitude = zMean - 5*zStdMax + 2*(5*zStdMax - zMean)*rand(nZern,nAll);
    end
end

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
    sqrt(ngs.var)*wvl_factor;
    sqrt(sum(c1.^2));
    sqrt(sum(c2.^2));
end
clear pyr_frame
%% LOOP - MULTI CPU
% indexes for cropping the pyramid camera frame

% Give workers access to OOMAO functions
addAttachedFiles(gcp,{'telescope.m','telescopeAbstract.m','pyramid.m','source.m'});


% Manage noise
if ron
    pyr.camera.readOutNoise = ron;
    pyr.camera.photonNoise = true;
    pyr.camera.quantumEfficiency = QE;
end
 clear fx fr fxExt fy fyExt index psdKolmo photoNGS toolboxesList zernRec
[X_phase,X_s,Y_s,Y_p] = Iter_simul(nAll,psd,atm,pyr,tel,ngs,rngStream,nPxPup,noiseVar,wvl_factor,ph2zern,pyr2zern,idx,N,pupil,fourierSampling,I_0);


return
