function [X_phase,Y_s] = Generate_phasemap_sequential(nAll,jIndex,r0,resol,tel_size,tel_tsamp)
    %parFileName = 'parFileLOOPS_Nov2021';
    nZern = length(jIndex);
    % source
    D = tel_size;
    nPxPup = resol;

    
    % telescopetel_D        = 8;           % telescope primary mirror diameter
    %tel_tsamp    = 1/1000;       % WFS sampling time (seconds)
    tel_obst_rat = 0;           % 0.14;   % central obscuration ratio
    tel = telescope(D, ...
        'resolution',resol,...
        'obstructionRatio',tel_obst_rat, ...
        'samplingTime',tel_tsamp);

    atm_r0   = r0;
    atm_L0   = 25;
    atm_h    = [1 6]*1e3;
    atm_fr0  = [0.7 0.3];
    atm_wspd = [5 10];
    atm_wdir = [3*pi/4 pi/4];
    atm = atmosphere(photometry.V, atm_r0, atm_L0,...
        'fractionnalR0',atm_fr0, ...
        'altitude',atm_h,  ...
        'windSpeed',atm_wspd, ...
        'windDirection',atm_wdir);


    % Create a star to capture the wavefront
    star = source;

    % define Zernike Modes
    zernRec  = zernike(jIndex,tel.D,'resolution',tel.resolution);
    %wvl_factor  = 550*1e9/2/pi; % from rad 2 nm
    ph2zern     = pinv(zernRec.modes);

    tel = tel+atm;
    % +tel; %% draw another wavefront


    % Propagate the light from the star down to the telescope pupil
    % and get the wavefront
    star = star.*tel;
    wavefront = star.meanRmPhase;

    %% Phase screen generator
    % nAll = length(r0_vars);
    X_phase = zeros(nPxPup,nPxPup,nAll,'single');
    Y_s  = zeros(nAll,nZern,'single');

    for kIter = 1:nAll
        +tel;

        % propagation
        star = star.*tel;
        turbPhase = star.meanRmPhase;
        zCoefs = ph2zern*turbPhase(:);
        X_phase(:,:,kIter)  = turbPhase;
        Y_s(kIter,:) = zCoefs;
    end
    
return