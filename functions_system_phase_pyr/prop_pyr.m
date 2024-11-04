function [cam,cam2] = prop_pyr(phase,ngs,tel,wfs)
    ngs = ngs.*tel*wfs;
    ngs.phase = phase;%*550/2/pi;
    ngs = ngs*wfs;
    camera = wfs.camera.frame;
%     camera = reshape(wfs.camera.frame,2*wfs.c*wfs.nLenslet,2*wfs.c*wfs.nLenslet,[]);
%     [r,pup,~] = find_subpupils2(camera(:,:,1),.1,1); %simul
    % Normalisation
    cam     = camera./sum(sum(camera(:,:,1)));
    cam = cam/sum(cam(:));
    diam = 134/2;
    frac = floor(268/8);
    dx1= frac+1:frac+diam;
    dx2 = frac+134+1:frac+134+diam;
    cam2     = 0;%[cam(dx1,dx1) cam(dx1,dx2); cam(dx2,dx1) cam(dx2,dx2)];
return