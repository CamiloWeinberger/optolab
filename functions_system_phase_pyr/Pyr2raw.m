function [X_s,Y_p] = Pyr2raw(X_phase,nAll,condition_PhN,input)
    pyr = input.pyr;
    ngs = input.ngs;
    tel = input.tel;
    QE = input.QE;
    wvl_factor = input.wvl_factor;
    %pyr2zern = input.pyr2zern;
    I_0 = input.I_0;
    % ngs.magnitude=0;
    
    pyr.camera.frame = pyr.camera.frame;
    %%% CHECK THIS %%%%%
    %im = im_0 x QE x 10^(-0.4xmagNGS) x Texp x Flux_0
    %im_noise = poissrnd(im) + randn(size(im))*ron
    
    % Manage noise
    % pyr.camera.readOutNoise = 0;
    
    
    
    %  if ron
    pyr.camera.readOutNoise = 0;
    
    %condition_PhN = pyr.camera.photonNoise;
    pyr.camera.photonNoise = 0;
    pyr.camera.quantumEfficiency = 1;%0.48;
    %  end
    
    
    mag = ngs.magnitude;
    relac  = 10^(0.4*(0-mag));
    
    idx1 = floor(((pyr.c-1)/2) * pyr.nLenslet + 1 : ((pyr.c-1)/2 + 1) * pyr.nLenslet)+1;
    idx2 = floor(((pyr.c-1)/2 + pyr.c) * pyr.nLenslet + 1 : ((pyr.c-1)/2 + pyr.c + 1) * pyr.nLenslet)+1;
    X_s = [];
    Y_p = [];
    X_s = [];
    for kIter = 1:nAll
    
        %for nnn = 1:100
        %parfor kI = 1:10
        %    kIter = kI+(nnn-1)+10;
        % propagate through the pyramid
        n2          = ngs.*tel;
        n2.phase    = X_phase(:,:,kIter);
        n2          = n2*pyr;%imagesc(pyr.camera.frame);axis image;colorbar
        % pyr_frame   = pyr.camera.frame./sum(pyr.camera.frame(:))-I_0;
    
    
        % Ground-truth - FOURIER GENERATOR CASE
        %zCoefs_pyr =  wvl_factor*pyr2zern*pyr_frame(:);
        % crop image
        RAW = ([pyr.camera.frame(idx1,idx1),pyr.camera.frame(idx1,idx2);...
            pyr.camera.frame(idx2,idx1),pyr.camera.frame(idx2,idx2)]);
        if condition_PhN == 1
            %RAW = poissrnd(RAW*relac+1e-3*0)-1e-3*0 + randn(size(RAW)).*0.01;
            RAW = AddPhotonNoise(RAW*relac,0) + randn(size(RAW)).*0.01;
        end
        X_s(:,:,kIter) = single(RAW);%/sum(RAW(:)));
        %Y_p1(kIter,:) = single(zCoefs_pyr)';
        %end
        %X_s = cat(3,X_s,X_s1);
    end
    
    return
    
    
        function out = AddPhotonNoise(x,bknoise)
            buffer = x + bknoise;
            x = x + randn(size(x)).*(x + bknoise);
            index     = x<0;
            x(index) = buffer(index);
            out = x;
        return
    