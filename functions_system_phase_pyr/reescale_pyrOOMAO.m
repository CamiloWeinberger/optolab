function [I_0,pyr2zern,pyr2kl] = reescale_pyrOOMAO(input)
pyr = input.pyr;
idx1 = floor(((pyr.c-1)/2) * pyr.nLenslet + 1 : ((pyr.c-1)/2 + 1) * pyr.nLenslet);
idx2 = floor(((pyr.c-1)/2 + pyr.c) * pyr.nLenslet + 1 : ((pyr.c-1)/2 + pyr.c + 1) * pyr.nLenslet);
pyr2zern2 = input.pyr2zern;
pyr2kl2 = input.pyr2kl;
for idx = 1:size(pyr2zern2,1)
    im = reshape(pyr2zern2(idx,:),size(input.I_0));
    im = ([im(idx1,idx1),im(idx1,idx2);im(idx2,idx1),im(idx2,idx2)]);
    pyr2zern(idx,:) = im(:);
end

for idx = 1:size(pyr2kl2,1)
    im = reshape(pyr2kl2(idx,:),size(input.I_0));
    im = ([im(idx1,idx1),im(idx1,idx2);im(idx2,idx1),im(idx2,idx2)]);
    pyr2kl(idx,:) = im(:);
end

I_0 = ([input.I_0(idx1,idx1),input.I_0(idx1,idx2);input.I_0(idx2,idx1),input.I_0(idx2,idx2)]);
pyr2zern = pyr2zern;
pyr2kl = pyr2kl;
return