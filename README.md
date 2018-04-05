# 4cnn-cifar10

Note

this code is to finfish classify based on the dataset of cifar-10 the model is created by myself 



1.模型很小时，即使是在train时，也很容易过拟合：几个epoch之后train_acc开始下降

2.模型很大时，为解决过拟合问题，在dense层加入了dropout，在loss中加入了regularization

3.没有batchnorm时，val_acc只能勉强到0.7，加入batchnorm后，直接到了0.8。这里要注意conv后接spatial batchnorm；而dense后接vanilla batchnor

Environment

tensorflow with python 2.7
