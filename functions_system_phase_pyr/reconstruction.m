function [out] = reconstruction(X,zernRec)
    Y = zernRec(:,2:size(X,1)+1)*X;
    for i = 1:size(Y,2)
        out(:,:,i) = reshape(Y(:,i),sqrt([size(Y,1),size(Y,1)]));
    end

return
