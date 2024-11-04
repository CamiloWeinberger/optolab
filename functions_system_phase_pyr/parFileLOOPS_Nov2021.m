%% MAIN PARAMETERS
L0          = 25;               % outer scale in meters
wDirection  = [0 pi/4 3*pi/6];                % Wind direction in degrees
% pyrMod      = 3;                % Modulation rate on top of the pyramid in lambda/D units
% loopGain    = 0.5;              % AO loop main gain, if 0, open-loop

%% GUIDE STAR
photoNGS    = photometry.V; %550nm
display(photoNGS.wavelength);
%% TELESCOPE
D           = 1.5;              % telescope diameter in meter
cobs        = 0;                % telescope central obstruction in percent
nPxPup      = 268;               % number of pixels to describe the pupil



% % %% GROUND TRUTH
%  nAll        = 1e2;          % Number of simulations 1250000
%  genType     = 'FOURIER';        %can be Zernike, or Fourier
  jIndex      = 2:55;          % Noll's indexes of the Zernike modes to be simulated for the Zernike Generator - 1 is piston
   nZern       = 54;               % number of Zernike to be reconstructed for the Fourier generator
%  zStdMax     = [500];            % Vector of size (1,numel(jIndex)) containing Std of the Zernike amplitude in nm
%  zMean       = [0];              % Vector of size (1,numel(jIndex)) containing mean of the Zernike amplitude in nm
%  zDistrib    = 'Normal';         % Normal -> N(zMean, zStdMax) or uniform -> [zMean-5*zStdMax,zMean+5*zStdMax]