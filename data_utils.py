import numpy as np 
import platform
import  pickle
import os 
def load_pickle(f):
	version = platform.python_version_tuple()
	if version[0] == '2':
		return pickle.load(f)
	elif version[0] == '3':
		return pickle.load(f,encoding='latin1')
	raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
	with open(filename,'rb') as f :
		datadict = load_pickle(f)
		X = datadict['data']
		Y = datadict['labels']
		X = X.reshape(10000,3,32,32).transpose(0,2,3,1).astype("float")
		Y = np.array(Y)
		return X,Y

def load_CIFAR10(ROOT):
	xs=[]
	ys=[] 
	for i in range(1,6):
		f = os.path.join(ROOT,'data_batch_%d'%(i,))
		X,Y = load_CIFAR_batch(f)
		xs.append(X)
		ys.append(Y)
	Xtr = np.concatenate(xs)
	Ytr = np.concatenate(ys)
	del X,Y
	Xte,Yte = load_CIFAR_batch(os.path.join(ROOT,'test_batch'))
	return Xtr,Ytr,Xte,Yte

def get_CIFAR10_data(num_training=49000,num_validiation=1000,num_test= 1000,
					subtract_mean=True):
	cifar10_path = 'datasets/cifar-10-batches-py'
	x_train,y_train,x_test,y_test = load_CIFAR10(cifar10_path)
	mask = list(range(num_training,num_training+num_validiation))
	x_val = x_train[mask]
	y_val = y_train[mask]
	mask1 = list(range(num_training))
	x_train = x_train[mask1]
	y_train = y_train[mask1]
	mask2 = list(range(num_test))
	x_test = x_test[mask2]
	y_test = y_test[mask2]

	if subtract_mean:
		mean = np.mean(x_train,axis=0)
		x_val -= mean
		x_test -= mean
		x_train -= mean 
	# Transpose so that channels come firs
	# x_train = x_train.transpose(0,3,1,2).copy()
	# x_val = x_val.transpose(0,3,1,2).copy()
	# x_test = x_test.transpose(0,3,1,2).copy()
	return x_train, y_train,x_val, y_val,x_test, y_test
