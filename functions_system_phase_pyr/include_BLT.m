function [BLT,Delta_piston] = include_BLT(phase,random_cases)
%random_cases = 1;
[px, py, N] = size(phase);
pup = phase(:,:,1)~=0;
BLT = ones(px,py/2,N);
if random_cases == 0
    P1 = 1.5*pi*rand;
    P2 = 1.5*pi*rand;
    BLT = cat(2,BLT*P1, BLT*P2).*pup;
    Delta_piston = (P1-P2)*ones(N,1);
else
    P1 = 1.5*pi*rand(1,1,N);
    P2 = 1.5*pi*rand(1,1,N);
    BLT = cat(2,P1.*BLT, P2.*BLT).*pup;
    Delta_piston = (P2(:)-P1(:));
end
BLT = BLT+phase;
end