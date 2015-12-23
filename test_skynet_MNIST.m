clc
clear
load mnist_uint8;

train_x = double(reshape(train_x',28,28,60000))/255;
test_x = double(reshape(test_x',28,28,10000))/255;
train_y = double(train_y');
test_y = double(test_y');

cnn.layers = {
    struct('type', 'd','channel',1) %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5,'activetype','relu','pad',0) %convolution layer
    struct('type', 's', 'scale', 2,'pooltype','mean') %sub sampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5,'activetype','relu','pad',0) %convolution layer
    struct('type', 's', 'scale', 2,'pooltype','mean') %subsampling layer
    struct('type', 'i', 'mapsize', 256,'activetype','relu')%interconnect layer
    struct('type', 'o','loss','LLH') %LLH has softmax.and MSD has sigmoid.
};

cnn.lamda = 1e-4;
cnn.momentun = 0.9;
cnn.bl = 0.1;

opts.batchsize = 50;
opts.numepochs = 1;

for i=1:1
    cnn = skynet_setup(cnn, train_x, train_y);
    cnn = skynet_train(cnn, train_x, train_y, opts);
    [er, bad] = skynet_test(cnn, test_x, test_y);
    str = ['error rate£º' num2str(er*100),'%'];
    disp(str);
end


