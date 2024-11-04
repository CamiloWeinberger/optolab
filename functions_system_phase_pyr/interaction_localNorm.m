function [ iMat ] = interaction_localNorm(ngs,tel,wfs,modalBasis)
% Compute intereaction matrix with push pull method around flat wavefront
% - signal: full frame
siz = 32;
    amplitude = 0.1;%Need to be small (for noise-free esystem)
    modalBasis = reshape(modalBasis,tel.resolution,tel.resolution,size(modalBasis,2));%Reshape modes
    % ------- PUSH ---------
    ngs = ngs.*tel;
    ngs.phase = amplitude*modalBasis;
    ngs = ngs*wfs;
    camera = reshape(wfs.camera.frame,2*wfs.c*wfs.nLenslet,2*wfs.c*wfs.nLenslet,[]);
    % Normalisation
    sp= camera;%./sum(sum(camera(:,:,1)));
    [r,pup,~] = find_subpupils2(camera(:,:,1),.1,1); %simul
    pupil = imresize(pup,[siz siz]);pupil = [pupil pupil; pupil pupil];
    [sp,~ ] = crop_norm(sp,r,0,siz); sp = double(sp)/4;
    % ------- PULL ---------
    ngs = ngs.*tel;
    ngs.phase = -amplitude*modalBasis;
    ngs = ngs*wfs;     
    camera = reshape(wfs.camera.frame,2*wfs.c*wfs.nLenslet,2*wfs.c*wfs.nLenslet,[]);
    % Normalisation
    sm= camera;%./sum(sum(camera(:,:,1)));
    [r,~,~] = find_subpupils2(camera(:,:,1),.1,1); %simul
    [sm,~ ] = crop_norm(sm,r,0,siz); sm = double(sm)/4;
    
    % PUSH-PULL COMPUTATION
    iMat=0.5*(sp-sm)/amplitude;
    
    % Reshape in 2-D
    iMat = reshape(iMat,(2*siz)^2,[]);
  
end