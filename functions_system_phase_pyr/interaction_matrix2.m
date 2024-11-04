function [ iMat ] = interaction_matrix2(ngs,tel,wfs,modalBasis,amplitude)
% Compute intereaction matrix with push pull method around flat wavefront
% - signal: full frame
    modalBasis = reshape(modalBasis,tel.resolution,tel.resolution,size(modalBasis,2));%Reshape modes
    % ------- PUSH ---------
    ngs = ngs.*tel;
    ngs.phase = amplitude*modalBasis;
    ngs = ngs*wfs;
    camera = reshape(wfs.camera.frame,2*wfs.c*wfs.nLenslet,2*wfs.c*wfs.nLenslet,[]);
    % Normalisation
    sp= camera./sum(sum(camera(:,:,1)));
    
    % ------- PULL ---------
    ngs = ngs.*tel;
    ngs.phase = -amplitude*modalBasis;
    ngs = ngs*wfs;     
    camera = reshape(wfs.camera.frame,2*wfs.c*wfs.nLenslet,2*wfs.c*wfs.nLenslet,[]);
    % Normalisation
    sm= camera./sum(sum(camera(:,:,1)));
    
    % PUSH-PULL COMPUTATION
    iMat=0.5*(sp-sm)/amplitude;
    
    % Reshape in 2-D
    iMat = reshape(iMat,(2*wfs.c*wfs.nLenslet)^2,[]);
end