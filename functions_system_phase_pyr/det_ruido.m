function [img_out,img_snr] = det_ruido(img_in,nread,npoiss,mag)

img_out= 10^(-0.4*mag) * img_in;
img_or=img_out;
if npoiss
    img_out=poissrnd(img_out);
end
[sa,sb] = size(img_or);
img_out=abs(img_out+nread*randn(sa,sb));
if img_out-img_or
    img_snr = std(img_or)/std(img_out-img_or);
else
    img_snr = inf;
end