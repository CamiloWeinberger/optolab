M = 134;
N = 134;
Zs = 199;
lgraph = layerGraph();tempLayers = imageInputLayer([M N 1],"Name","imageinput");
lgraph = addLayers(lgraph,tempLayers);tempLayers = convolution2dLayer([1 12],128,"Name","conv1a_2","Padding","same");
lgraph = addLayers(lgraph,tempLayers);tempLayers = [
    reluLayer("Name","relu_1a_2")
    batchNormalizationLayer("Name","batchnorm_4")
    maxPooling2dLayer([2 2],"Name","maxpool_1a_2","Padding","same","Stride",[2 2])
    convolution2dLayer([1 9],64,"Name","conv2a_3","Padding","same")
    reluLayer("Name","relu_2a_2")
    batchNormalizationLayer("Name","batchnorm_5")
    maxPooling2dLayer([2 2],"Name","maxpool_2a_2","Padding","same","Stride",[2 2])
    convolution2dLayer([1 3],32,"Name","conv2a_4","Padding","same")
    reluLayer("Name","relu_3a_2")
    batchNormalizationLayer("Name","batchnorm_6")
    maxPooling2dLayer([2 2],"Name","maxpool_3a_2","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);tempLayers = convolution2dLayer([12 1],128,"Name","conv1a_3","Padding","same");
lgraph = addLayers(lgraph,tempLayers);tempLayers = [
    depthConcatenationLayer(2,"Name","depthcat_2")
    convolution2dLayer([5 5],256,"Name","conv","Padding","same","Stride",[4 4])
    reluLayer("Name","relu_2a_3_2")
    batchNormalizationLayer("Name","batchnorm_8_2")
    maxPooling2dLayer([2 2],"Name","maxpool_2a_3_2","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);tempLayers = [
    reluLayer("Name","relu_1a_3")
    batchNormalizationLayer("Name","batchnorm_7")
    maxPooling2dLayer([2 2],"Name","maxpool_1a_3","Padding","same","Stride",[2 2])
    convolution2dLayer([9 1],64,"Name","conv2a_5","Padding","same")
    reluLayer("Name","relu_2a_3_1")
    batchNormalizationLayer("Name","batchnorm_8_1")
    maxPooling2dLayer([2 2],"Name","maxpool_2a_3_1","Padding","same","Stride",[2 2])
    convolution2dLayer([3 1],32,"Name","conv2a_6","Padding","same")
    reluLayer("Name","relu_3a_3")
    batchNormalizationLayer("Name","batchnorm_9")
    maxPooling2dLayer([2 2],"Name","maxpool_3a_3","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);tempLayers = [
    depthConcatenationLayer(3,"Name","depthcat_1")
    crossChannelNormalizationLayer(4,"Name","crossnorm")
    transposedConv2dLayer([3 3],96,"Name","transposed-conv_1","Cropping","same","Stride",[2 2])
    transposedConv2dLayer([9 9],64,"Name","transposed-conv_2","Cropping","same","Stride",[2 2])
    transposedConv2dLayer([12 12],32,"Name","transposed-conv_3","Cropping","same","Stride",[2 2])
    reluLayer("Name","relu_5")
    fullyConnectedLayer(256,"Name","fc3")
    fullyConnectedLayer(64,"Name","fc2")
    fullyConnectedLayer(Zs,"Name","fc1")
    regressionLayer("Name","regressionoutput")];
lgraph = addLayers(lgraph,tempLayers);% clean up helper variable
clear tempLayers;lgraph = connectLayers(lgraph,"imageinput","conv1a_2");
lgraph = connectLayers(lgraph,"imageinput","conv1a_3");
lgraph = connectLayers(lgraph,"conv1a_2","relu_1a_2");
lgraph = connectLayers(lgraph,"conv1a_2","depthcat_2/in2");
lgraph = connectLayers(lgraph,"conv1a_3","depthcat_2/in1");
lgraph = connectLayers(lgraph,"conv1a_3","relu_1a_3");
lgraph = connectLayers(lgraph,"maxpool_2a_3_2","depthcat_1/in3");
lgraph = connectLayers(lgraph,"maxpool_3a_2","depthcat_1/in2");
lgraph = connectLayers(lgraph,"maxpool_3a_3","depthcat_1/in1");

