function [X_phase,X_s,Y_s] = Iter_simul_zern(nAll,atm,pyr,tel,ngs,zAmplitude,nPxPup,noiseVar,wvl_factor,ph2zern,pyr2zern,I_0,wvl,zModes)
idx1 = ((pyr.c-1)/2) * pyr.nLenslet + 1 : ((pyr.c-1)/2 + 1) * pyr.nLenslet;
idx2 = ((pyr.c-1)/2 + pyr.c) * pyr.nLenslet + 1 : ((pyr.c-1)/2 + pyr.c + 1) * pyr.nLenslet;


for kIter = 1:nAll 
        

% ------------------- FOURIER GENERATOR
% Get the phase map
%map = real(ifft2(idx.*sqrt(fftshift(psd)).*fft2(randn(rngStream,N))./N).*fourierSampling).*N.^2;        
%phaseMap = pupil.*map(1:nPxPup,1:nPxPup);     
zCoefs = zAmplitude(kIter,:)';
phaseMap = 2*pi*1e-9/wvl*reshape(zModes*zCoefs,nPxPup,nPxPup);

% propagate through the pyramid
n2          = times(ngs,tel);
n2.phase    = phaseMap;
n2          = mtimes(n2,pyr);
pyr_frame   = pyr.camera.frame/sum(pyr.camera.frame(:))-I_0;    

% Ground-truth - FOURIER GENERATOR CASE
zCoefs = wvl_factor*ph2zern*n2.phase(:);

% Pyramid-based Zernike reconstruction in nm
%        zCoefs_pyr =  wvl_factor*pyr2zern*pyr_frame(:);

% wavefront-error
%        wfe = sqrt(n2.var)*wvl_factor;

% crop image
pyr_frame  = pyr.camera.frame;
pyr_frame = ([pyr_frame(idx1,idx1),pyr_frame(idx1,idx2);pyr_frame(idx2,idx1),pyr_frame(idx2,idx2)]);

%elapsed = toc(t_i);
%fprintf(['Time remaining :',num2str(elapsed*nAll/kIter/3600),' h\n']);
    
X_phase(:,:,kIter) = single(phaseMap); %
%X_phase =1
X_s(:,:,kIter) = single(pyr_frame);
Y_s(kIter,:) = single(zCoefs)';

%Y_p=1;
  
end

return