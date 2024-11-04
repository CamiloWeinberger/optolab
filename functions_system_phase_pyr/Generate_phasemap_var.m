function [X_phase,Y_s] = Generate_phasemap_var(r0_vars,jIndex,r0,resol,tel_size)
    parFileName = 'parFileLOOPS4';
    %parFileName = 'parFileLOOPS_Nov2021';
    eval(parFileName);
    check = false;
    nZern = length(jIndex);
    wSpeed = 5;
    % source
    D = tel_size;
    nPxPup = resol;
    fovInPixel  = nPxPup*2*Samp;    % number of pixel to describe the PSD
    nTimes      = fovInPixel/resAO;

    ngs = source('wavelength',photoNGS);
    
    % telescope
    tel = telescope(D,'obstructionRatio',cobs,'resolution',nPxPup);
    pupZeroPad = padarray(tel.pupil,[(Samp+1)*nPxPup/2,(Samp+1)*nPxPup/2]);
    
    stream = RandStream('mt19937ar','Seed','shuffle');
    atm = atmosphere(photoNGS,r0,L0,'altitude',5000,'fractionnalR0',1,...
        'windSpeed',wSpeed,'windDirection',wDirection*pi/180,'randStream',stream);
    % Flat for FULL FRAME
    ngs = ngs.*tel;
    
    wvl = ngs.wavelength;
    tel = tel + atm;

    % define Zernike Modes
    zernRec  = zernike(jIndex,tel.D,'resolution',tel.resolution);
    wvl_factor  = wvl*1e9/2/pi; % from rad 2 nm
    ph2zern     = pinv(zernRec.modes);

    %% Phase screen generator
    nAll = length(r0_vars);
    X_phase = zeros(nPxPup,nPxPup,nAll,'single');
    Y_s  = zeros(nAll,nZern,'single');

    for kIter = 1:nAll
        ngs=ngs.*+tel;
        turbPhase = ngs.meanRmPhase;
        zCoefs = ph2zern*turbPhase(:);
        X_phase(:,:,kIter)  = turbPhase;
        Y_s(kIter,:) = zCoefs;
    end
    
return