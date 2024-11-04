function [out,I] = crop_resize(input,r_i,angle,siz)
% (input, r_i, angle, new size)
input = single(input);
len = size(r_i,2);
cant = size(r_i,1);
input = single(input);
pup = fspecial('disk',len/2); pup = pup/max(pup(:)); pup = pup>0.8;
I=[];
if size(input,3)>1
for jj = 1:size(input,3)
for i = 1:cant/2
    kk      = i*2-1;
    ff      = imrotate(input(r_i(kk,:),r_i(kk+1,:),jj),angle,'bicubic');
    ff(ff==0) = 1e-8;
    len2    = size(ff,2);
    mm      = floor((len2-len+1)/2+.5);
    MM      = floor((len2+len)/2+.5);
    I{i}    = imresize(ff(mm:MM,mm:MM),[siz siz]);

end

out(:,:,jj) = [I{1},I{2};I{3},I{4}];
end

else
for jj = 1:size(input,4)
for i = 1:cant/2
    kk      = i*2-1;
    ff      = imrotate(input(r_i(kk,:),r_i(kk+1,:),1,jj),angle,'bicubic');
    ff(ff==0) = 1e-8;
    len2    = size(ff,2);
    mm      = floor((len2-len+1)/2+.5);
    MM      = floor((len2+len)/2+.5);
    I{i}    = imresize(ff(mm:MM,mm:MM),[siz siz]);

end

out(:,:,1,jj) = [I{1},I{2};I{3},I{4}];
end
end
end