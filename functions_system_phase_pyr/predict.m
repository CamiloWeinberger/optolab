function [out] = predict(X,pyr2zern)
    X = reshape(X,[size(X,1)*size(X,1),size(X,3)]);
    out = pyr2zern*X;
return