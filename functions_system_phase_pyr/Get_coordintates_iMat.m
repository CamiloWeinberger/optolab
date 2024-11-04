function [r_i] = Get_coordintates_iMat(input)
resol =sqrt(size(input,2));
Ii = reshape(input(1,:),[resol resol]); Ii = abs(Ii)/max(abs(Ii(:)));
tol         = .1; 
rad_fact    =1;
[r_i,pupil,angle] = find_subpupils2(Ii,tol,rad_fact);
return