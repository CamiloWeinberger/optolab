function [ iMat , I_0,Sp] = interaction_totalNorm(ngs,tel,wfs,modalBasis,resol,amplitude)
% Compute intereaction matrix with push pull method around flat wavefront
% - signal: full frame
    siz = resol;total = size(modalBasis,2);
    %amplitude = 0.1;%Need to be small (for noise-free esystem)
    modalBasis = reshape(modalBasis,tel.resolution,tel.resolution,size(modalBasis,2));%Reshape modes
    waitt = waitbar(0,'working');
    ngs = ngs.*tel;
    ngs.phase = (modalBasis(:,:,1)~=0);
    ngs = ngs*wfs;
    camera = reshape(wfs.camera.frame,2*wfs.c*wfs.nLenslet,2*wfs.c*wfs.nLenslet,[]);
%     [r,pup,~] = find_subpupils2(camera(:,:,1),.1,1); %simul
    % Normalisation
    I_0= camera./sum(sum(camera(:,:,1)));
    pup = I_0/max(I_0(:))>=0.1;
%     I_0 = [I_0(r(1,4:end-1),r(2,4:end-1)),I_0(r(3,4:end-1),r(4,4:end-1));I_0(r(5,4:end-1),r(6,4:end-1)),I_0(r(7,4:end-1),r(8,4:end-1))];
    % ------- PUSH ---------
    for idx=1:total
        if idx == 1
            imm = camera(:,:,1);
%             imm = imm(pup);
            fact = sum(imm(:));
        end
    ngs = ngs.*tel;
    ngs.phase = amplitude*modalBasis(:,:,idx);
    ngs = ngs*wfs;
    camera = wfs.camera.frame;
%     camera = [camera(r(1,4:end-1),r(2,4:end-1)),camera(r(3,4:end-1),r(4,4:end-1));camera(r(5,4:end-1),r(6,4:end-1)),camera(r(7,4:end-1),r(8,4:end-1))];
    camera = reshape(camera,2*wfs.c*wfs.nLenslet,2*wfs.c*wfs.nLenslet,[]);
%     camera = reshape(camera,wfs.c*wfs.nLenslet,wfs.c*wfs.nLenslet,[]);

    % Normalisation
    sp= camera(:)./fact;
%     if ~exist('r')
%     [r,pup,~] = find_subpupils2(camera(:,:,1),.1,1); %simul
%     end
%     pupil = imresize(pup,[siz siz]);pupil = [pupil pupil; pupil pupil];
%     [sp,~ ] = crop_resize(sp,r,0,siz); sp = single(sp).*pupil;sp = sp./sum(sp,[1 2]);
    Sp(:,idx) = sp;
    % ------- PULL ---------
    ngs = ngs.*tel;
    ngs.phase = -amplitude*modalBasis(idx);
    ngs = ngs*wfs;
    camera = wfs.camera.frame;
%     camera = [camera(r(1,4:end-1),r(2,4:end-1)),camera(r(3,4:end-1),r(4,4:end-1));camera(r(5,4:end-1),r(6,4:end-1)),camera(r(7,4:end-1),r(8,4:end-1))];
    camera = reshape(camera,2*wfs.c*wfs.nLenslet,2*wfs.c*wfs.nLenslet,[]);
%     camera = reshape(camera,wfs.c*wfs.nLenslet,wfs.c*wfs.nLenslet,[]);
    % Normalisation
    sm= camera(:)./fact;
    
%     [sm,~ ] = crop_resize(sm,r,0,siz);sm = single(sm).*pupil;sm = sm./sum(sm,[1 2]);
    Sm(:,idx) = sm;
%     subplot(121);imagesc(modalBasis(:,:,idx));axis image;axis off;
%     subplot(122);imagesc(sp);axis image;axis off;title(num2str(idx));drawnow

    waitbar(idx/total,waitt,'working');
    end
    % PUSH-PULL COMPUTATION
    iMat=0.5*(Sp-Sm)/amplitude;
    
    % Reshape in 2-D
%     iMat = reshape(iMat,size(iMat,1)^2,[]);
  
end