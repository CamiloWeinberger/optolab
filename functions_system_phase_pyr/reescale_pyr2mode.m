function [out] = reescale_pyr2mode(r_i,input,resol)

for idx = 1:size(input,1)
    X_s = reshape(input(idx,:),[sqrt(size(input,2)) sqrt(size(input,2))]);
    [Xs,I]  = crop_resize(X_s,r_i,0,resol);
    out(idx,:) = Xs(:);
end

return