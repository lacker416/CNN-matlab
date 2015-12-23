clc
clear
x=[];
y=[];
%you should download cifar10 dataset first
load ./cifar-10-batches-mat/data_batch_1.mat;
x = [x;data];
y= [y;labels];
load ./cifar-10-batches-mat/data_batch_2.mat;
x = [x;data];
y= [y;labels];
load ./cifar-10-batches-mat/data_batch_3.mat;
x = [x;data];
y= [y;labels];
load ./cifar-10-batches-mat/data_batch_4.mat;
x = [x;data];
y= [y;labels];
load ./cifar-10-batches-mat/data_batch_5.mat;
x = [x;data];
y= [y;labels];
x=double(x);
y=double(y);

load ./cifar-10-batches-mat/test_batch.mat;
test_x = double(data);
test_y_t = double(labels);

z = (y == 0);
y = y + z.*10;

train_x = double(reshape(x',32,32,3,50000))/255;
train_x = permute(train_x,[2 1 3 4]);

test_x = double(reshape(test_x',32,32,3,10000))/255;
test_x = permute(test_x,[2 1 3 4]);

z = (test_y_t == 0);
test_y_t = test_y_t + z.*10;

train_y = full(sparse(y, 1:50000, 1));
test_y = full(sparse(test_y_t, 1:10000, 1));

cnn.layers = {
    struct('type', 'd','channel',3) %input layer
    struct('type', 'c', 'outputmaps', 32, 'kernelsize', 5,'activetype','relu','pad',0) %convolution layer
    struct('type', 's', 'scale', 2,'pooltype','mean') %sub sampling layer
    struct('type', 'c', 'outputmaps', 32, 'kernelsize', 5,'activetype','relu','pad',0) %convolution layer
    struct('type', 's', 'scale', 2,'pooltype','mean') %subsampling layer    
    struct('type', 'i', 'mapsize', 128,'activetype','relu')%interconnect layer
    struct('type', 'o','loss','LLH') %LLH has softmax.and MSD has sigmoid.
};

cnn.lamda = 3e-4;
cnn.momentun = 0.9;
cnn.bl = 0.1;

opts.batchsize = 100;
opts.numepochs = 10;

for i=1:1
    cnn = skynet_setup(cnn, train_x, train_y);
    cnn = skynet_train(cnn, train_x, train_y, opts);
    [er, bad] = skynet_test(cnn, test_x, test_y);
    str = ['error rate£º' num2str(er*100),'%'];
    disp(str);
end


