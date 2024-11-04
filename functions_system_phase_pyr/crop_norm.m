function [out,I,pupil] = crop_norm(input,r_i,angle,siz_input)
% (input, r_i, angle, new size)

siz=siz_input;
len = size(r_i,2);
cant = size(r_i,1);
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
    I{i}    = single(imresize(ff(mm:MM,mm:MM),[siz siz]));
    I{i}    = I{i}/sum(I{i},'all');

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
    I{i}    = single(imresize(ff(mm:MM,mm:MM),[siz siz]));
    I{i}    = I{i}/sum(I{i},'all');

end
pup = fspecial('disk',(1+size(I{1},1))/2); pup = imresize(pup,[size(I{1},1) size(I{1},1)]);
pup = pup/max(pup(:)); pup = pup>0.8;
pupil = [pup pup;pup pup];
out(:,:,1,jj) = [I{1},I{2};I{3},I{4}];

end
end
end