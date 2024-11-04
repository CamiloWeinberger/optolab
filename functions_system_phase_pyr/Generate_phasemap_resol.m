function [X_phase,Y_s] = Generate_phasemap_resol(nAll,jIndex,r0,resol,tel_size)
    parFileName = 'parFileLOOPS4';
    %parFileName = 'parFileLOOPS_Nov2021';
    eval(parFileName);
    check = false;
    nZern = length(jIndex);
    wSpeed = 5;
    % source
    rng('shuffle');
    D = tel_size;
    nPxPup = resol;
    fovInPixel  = nPxPup*2*Samp;    % number of pixel to describe the PSD
    nTimes      = fovInPixel/resAO;

    ngs = source('wavelength',photoNGS,'magnitude',magNGS);
    
    % telescope
    tel = telescope(D,'obstructionRatio',cobs,'resolution',nPxPup);
    pupZeroPad = padarray(tel.pupil,[(Samp+1)*nPxPup/2,(Samp+1)*nPxPup/2]);
    
    atm = atmosphere(photoNGS,r0,L0,'altitude',0,'fractionnalR0',1,...
        'windSpeed',wSpeed,'windDirection',wDirection*pi/180);
    
    % wavefront-sensor
    % Flat for FULL FRAME
    ngs = ngs.*tel;
    
    wvl = ngs.wavelength;
    
    %% Phase screen generator
    
        fao = spatialFrequencyAdaptiveOptics(tel,atm,nLenslet+1,0.1,...
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
        
    
    %% ZERNIKE RECONSTRUCTION MATRIX : FULL-FRAME TO ZERNIKE MODE
    
    % define Zernike Modes
    zernRec  = zernike(jIndex,tel.D,'resolution',tel.resolution);
    wvl_factor  = wvl*1e9/2/pi; % from rad 2 nm
    ph2zern     = pinv(zernRec.modes);
    
    %% LOOP - MULTI CPU
    % indexes for cropping the pyramid camera frame
    
    X_phase = zeros(nPxPup,nPxPup,nAll,'single');
    Y_s  = zeros(nAll,nZern,'single');
    
    
    for kIter = 1:nAll 
    % Give workers access to OOMAO functions
    %addAttachedFiles(gcp,{'telescope.m','telescopeAbstract.m','source.m'})
    map = real(ifft2(idx.*sqrt(fftshift(psd)).*fft2(randn(N))./N).*fourierSampling).*N.^2;        
    phaseMap = pupil.*map(1:nPxPup,1:nPxPup);  
    
    
    
    n2          = times(ngs,tel);
    n2.phase    = phaseMap;
    % Ground-truth - FOURIER GENERATOR CASE
    zCoefs = wvl_factor*ph2zern*n2.phase(:);
    
    X_phase(:,:,kIter)  = phaseMap;
    Y_s(kIter,:) = zCoefs;
    
    
    end
    
    return