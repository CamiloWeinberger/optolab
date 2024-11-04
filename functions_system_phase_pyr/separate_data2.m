function [XTrain XVal YTrain YVal] = separate_data2(X,Y,porcentaje)

tam = size(X,4);
rng(7)
idx = randperm(size(X,4),size(X,4));

split = round(length(idx)*porcentaje);

XTrain = X(:,:,:,idx(1:split));
XVal = X(:,:,:,idx(split+1:end));
YTrain = Y(:,:,:,idx(1:split));
YVal = Y(:,:,:,idx(split+1:end));


end
