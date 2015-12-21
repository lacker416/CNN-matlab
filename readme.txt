this project is rewrited on the matlab deeplearntool box.
I keep a single M_file (expand.m).you can find URL for original project in that file.


1.this project is used for learning cnn. It's slower than caffe.If you want to train something.please use caffe.
2.first layer must be data layer,and last layer must be output layer.
3.after interconnect layer,only interconnect layer or output layer is valid.conv layer or pooling layer after a interconnect layer is not supported.
4.pooling layer's inputmap size should be div by pooling layer's scale.It won't add padding auto.
5.if activetype set as relu.learning rate should be smaller(0.1).If set as sigmoid,learning rate should be bigger(1).
6.output layer for MSD has a interconnect layer with sigmoid in it.and LLH has a interconnect layer with softmax in it.

Author: Lacker
Date:12/21/2015