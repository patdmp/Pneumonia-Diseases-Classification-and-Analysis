function net3D = TL3D_adjusted(net2D,numClasses,inputSize)
% 功能：TL3D，Extend any 2d convolutional neural network to 3D convolution，把任意2d卷积神经网络扩展为3D卷积
% input:
%       net2D,          (positial)2D CNN network
%       numClasses,     (optional)numbers of classes
%       inputSize,       (optional)network input size,  H*W*L*C*Bs order,Bs samples
% output:
%      net3D,          layerGraph,return 3D CNN network
%
% cuixingxing150@gmail.com
% 2021.2.18
% 
arguments
    net2D (1,1)
    numClasses (1,1) double =1000
    inputSize (1,4) double = [224,224,16,3] % H*W*L*C, one sample
end
%% input
%lgraph = layerGraph(net2D);
lgraph = net2D;
inputLayer = image3dInputLayer(inputSize,...
    'Name','VideoInputLayer',...
     'Normalization', 'rescale-zero-one');
lgraph = replaceLayer(lgraph, lgraph.InputNames{1}, inputLayer);
%% conv,pool,concat,...
convIdxes = findLayers(lgraph,'nnet.cnn.layer.Convolution2DLayer');
for ii = 1:numel(convIdxes)
    lgraph = replaceConvLayer(lgraph,lgraph.Layers(convIdxes(ii)));
end
bnIdxes = findLayers(lgraph,'nnet.cnn.layer.BatchNormalizationLayer');
for ii = 1:numel(bnIdxes)
    lgraph = replaceBNLayer(lgraph,lgraph.Layers(bnIdxes(ii)));
end
normIdxes = findLayers(lgraph,'nnet.cnn.layer.CrossChannelNormalizationLayer');
removeNames = cell(numel(normIdxes),1);
for ii = 1:numel(normIdxes)
    removeNames{ii} = lgraph.Layers(normIdxes(ii)).Name;
end
for ii = 1:numel(removeNames)
    lgraph = removeCrossNormLayer(lgraph,removeNames{ii});
end
poolIdxes = findLayers(lgraph, 'nnet.cnn.layer.MaxPooling2DLayer');
for ii = 1:numel(poolIdxes)
    lgraph = replacePoolLayer(lgraph,lgraph.Layers(poolIdxes(ii)));
end
avgpoolIdxes = findLayers(lgraph,'nnet.cnn.layer.AveragePooling2DLayer');
for ii = 1:numel(avgpoolIdxes)
    lgraph = replaceAvgPoolLayer(lgraph,lgraph.Layers(avgpoolIdxes(ii)));
end
concatIdxes = findLayers(lgraph, 'nnet.cnn.layer.DepthConcatenationLayer');
for ii = 1:numel(concatIdxes)
    lgraph = replaceConcatLayer(lgraph,lgraph.Layers(concatIdxes(ii)));
end
gPoolIdx = find(arrayfun(@(x)isa(x,'nnet.cnn.layer.GlobalAveragePooling2DLayer'), lgraph.Layers));
gPool = lgraph.Layers(gPoolIdx);
gPoolRep = globalAveragePooling3dLayer('Name',gPool.Name + "_inflated");
lgraph = replaceLayer(lgraph,gPool.Name, gPoolRep);
fcIdx = findLayers(lgraph, 'nnet.cnn.layer.FullyConnectedLayer');
fcLayer = lgraph.Layers(fcIdx);
newFc = fullyConnectedLayer(numClasses, ...
        'Name','newFullyConnectedLayer', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
lgraph = replaceLayer(lgraph,fcLayer.Name, newFc);
net3D = lgraph;
end
%% support functions
function lgraph = replaceConvLayer(lgraph, convLayer)
filterSize = [convLayer.FilterSize, convLayer.FilterSize(2)];
outputChannels = convLayer.NumFilters;
name = convLayer.Name + "_Inflated";
stride = [convLayer.Stride, convLayer.Stride(2)];
w = convLayer.Weights;
temporal = filterSize(end);
nw = repmat(w,1,1,1,1,temporal);% H*W*C*B*L
nw = nw ./ temporal;
nw = permute(nw, [1,2,5,3,4]); % H*W*C*B*L--->H*W*L*C*B
b = convLayer.Bias;
b = reshape(b,[1,size(b)]);
convLayerRep = convolution3dLayer(...
        filterSize,...
        outputChannels,... % NumFilters
        'Name', name, ...
        'Stride', stride,...
        'Weights', nw,...
        'Bias', b,...
        'Padding', 'same');
lgraph = replaceLayer(lgraph, convLayer.Name, convLayerRep);
end
function lgraph = replaceBNLayer(lgraph,bnlayer)
TrainedMean = reshape(bnlayer.TrainedMean,[1,size(bnlayer.TrainedMean)]);
TrainedVar = reshape(bnlayer.TrainedVariance,[1,size(bnlayer.TrainedVariance)]);
offset = reshape(bnlayer.Offset,[1,size(bnlayer.Offset)]);
scale = reshape(bnlayer.Scale,[1,size(bnlayer.Scale)]);
name = bnlayer.Name + "_Inflated";
bnRep = batchNormalizationLayer(...
    'name',name,...
    'TrainedMean',TrainedMean,...
    'TrainedVariance',TrainedVar,...
    'offset',offset,...
    'Scale',scale);
lgraph = replaceLayer(lgraph, bnlayer.Name, bnRep);
end
function lgraph = replacePoolLayer(lgraph, poolLayer)
poolSize = [poolLayer.PoolSize, poolLayer.PoolSize(2)];
stride = [poolLayer.Stride, poolLayer.Stride(2)];
maxPool = maxPooling3dLayer(poolSize, ...
        'Name', poolLayer.Name + "_Inflated", ...
        'Stride', stride,...
        'Padding', 'same');
lgraph = replaceLayer(lgraph, poolLayer.Name, maxPool);
end
function lgraph = replaceAvgPoolLayer(lgraph, poolLayer)
poolSize = [poolLayer.PoolSize, poolLayer.PoolSize(2)];
stride = [poolLayer.Stride, poolLayer.Stride(2)];
avgPool = averagePooling3dLayer(poolSize, ...
        'Name', poolLayer.Name + "_Inflated", ...
        'Stride', stride,...
        'Padding', 'same');
lgraph = replaceLayer(lgraph, poolLayer.Name, avgPool);
end
function lgraph = removeCrossNormLayer(lgraph, normLayerName)
connections = lgraph.Connections;
sourceNames = connections.Source(find(strcmp(connections.Destination,normLayerName)));
destNames = connections.Destination(find(strcmp(connections.Source,normLayerName)));
lgraph = removeLayers(lgraph, normLayerName);
lgraph = connectLayers(lgraph,sourceNames{1},destNames{1});
end
function lgraph = replaceConcatLayer(lgraph,depthcatLayer)
concat = concatenationLayer(4,numel(depthcatLayer.InputNames),'Name', depthcatLayer.Name + "_Inflated");%concatenation channels
lgraph = replaceLayer(lgraph, depthcatLayer.Name, concat);
end
function idxes = findLayers(lgraph, name)
idxes = find(arrayfun(@(x)isa(x,name), lgraph.Layers));
end