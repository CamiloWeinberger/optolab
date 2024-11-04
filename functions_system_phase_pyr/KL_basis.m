
function modeOut=KL_basis(tel,atm,nKL)

% DESCRIPTION:
% Build a KL basis diagonalizing the covariance matrix of a Zernike basis.
% The first two modes are pure Zernikes: Tip and Tilt
%
% ARGUMENTS:
%   _ 'tel' is the telescope; it will give the properties (resolution,
%     diameter,etc) for the generated modes
%   _ 'atm' is the atmosphere, it will be used to compute the covariance
%     matrix of the Zernikes
%   _ 'nZern' is the number Zernikes used for the KL generation

%   _ 'nKL' is the number of KL modes generated.


nPup=sum(sum(tel.pupilLogical));
nRes=tel.resolution;
%% Pure KL Modes
% Take a high number of Zernikes
% jj=round(3+(sqrt(10+8*nKL)));
% nZern=(1+jj*(jj-3)/2);
nZern=2*nKL;
% Define the zernike basis (TT and piston are removed)
zern=zernike(tel,4:nZern-3);


%%
Z=zern.modes;

tiptilt=zernike(tel,2:3);
TT=tiptilt.modes;
% load the covariance matrix of the zernikes
C=phaseStats.zernikeCovariance(zern,atm);
% Apply a svd to the covariance matrix:
% C=blkdiag(C);
[U,S]=svd(C);

% Preallocate memory
modeKL=zeros(size(zern.modes,1),nKL);
% Generation of the KL modes:
for i=1:2
        modeKL(:,i)=TT(:,i);
end
for i=1:nKL-2
    for j=1:zern.nMode
        modeKL(:,i+2)=modeKL(:,i+2)+U(j,i)*Z(:,j);
    end
end
%     Normalization
for i=1:nKL
    %remove piston
    modeKL(:,i)=modeKL(:,i)-mean(modeKL(tel.pupilLogical,i));    
%     normalize
    modeKL(:,i)=modeKL(:,i)./rms(modeKL(tel.pupilLogical,i));
end
% save pure KL modes
modeOut.KL_pure=modeKL;
% 
% %% Fit the KL to the DM influence
% % Influence functions of the DM
% N=full(dm.modes.modes(:,:));
% % Projection of the KL in the DM space
% Mf=pinv(N)*modeKL(:,:);
% % Back in the Phase space
% Kf=N*Mf;
% % Autocovariance Matrix 
% Df=Kf'*Kf/nPup;
% %Cholesky transformation
% [L,nKLmax]=chol(Df,'lower');
% nKLmax
% if nKLmax ==0
%     nKLmax=nKL
% else 
%     nKLmax=nKLmax-1;
% end
% 
% % Orthogonal Modes in the DM space
% Kfo=Kf(:,1:nKLmax)*(pinv(L)');
% 
% % Compute KL modes orthogonalized in DM space
% KLortho=zeros(nRes*nRes,nKLmax);
% KLortho(tel.pupilLogical,:)=Kfo(tel.pupilLogical,:);
% KLortho=Kfo;
% 
% modeOrtho=zeros(nRes*nRes,nKLmax);
% 
% % normalize  the re-othogonalised KL modes
% for i=1:nKLmax
%     modeOrtho(:,i)=KLortho(:,i)-mean(KLortho(tel.pupilLogical,i));
% 
%     modeOrtho(:,i)=modeOrtho(:,i)./rms(modeOrtho(tel.pupilLogical,i));
% end
% modeOut.KL_ortho=KLortho;
% modeOut.KL_orthoNorm=modeOrtho;


end
