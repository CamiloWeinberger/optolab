function Phase_comparison(sys,Xphase,Y,Z,h1)
% X = Ground truth Phase
% Y = Zernikes predicted 1
% Z = Zernikes predicted 2 (optional)
% sys = system generated


X= Xphase;
Y_n  = Y;

if exist('Z');Y_p  = Z;end
h1.Position = [70 55 1140 710];
imGT = (X-mean(X(:)))*sys.wvl_factor;
[imN, Adat]= Phase_recover(Y,sys);imP= Phase_recover(Z,sys);
cmap = [min([imGT(:);imN(:);imP(:)]) max([imGT(:);imN(:);imP(:)])];
cmapres = [min([imGT(:)-imN(:);imGT(:)-imP(:)]) max([imGT(:)-imN(:);imGT(:)-imP(:)])];

subplot(231);imagesc(imGT,'AlphaData',Adat);axis image;colorbar;title('Ground Truth');axis off
subplot(232);imagesc(imN,'AlphaData',Adat,cmap);axis image;colorbar;title('Net recosntruction');axis off
subplot(235);imagesc((imN-imGT),'AlphaData',Adat,cmapres);axis image;colorbar;title('residual Net-GT');axis off

if exist('Z')
subplot(233);imagesc(imP,'AlphaData',Adat,cmap);axis image;colorbar;title('Pyr reconstruction');axis off
subplot(236);imagesc((imP-imGT),'AlphaData',Adat,cmapres);axis image;colorbar;title('residual Pyr-GT'); axis off
end
