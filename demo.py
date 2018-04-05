import numpy as np 
from data_utils import * 

x_train,y_train,x_val,y_val,x_test,y_test = get_CIFAR10_data()
print('Train data shape: ', x_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', x_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', x_test.shape)
print('Test labels shape: ', y_test.shape)

def my_model(X,y,is_training):  
      
    # [conv-relu-conv-relu-pool]  out=14x14  
    conv1 = tf.layers.conv2d(X,128,kernel_size=[3,3],strides=(1,1),activation=tf.nn.relu)  
    ba1   = tf.layers.batch_normalization(conv1,training=is_training)  
    conv2 = tf.layers.conv2d(ba1,256,[3,3],activation=tf.nn.relu)  
    ba2   = tf.layers.batch_normalization(conv2,training=is_training)  
    pool1 = tf.layers.max_pooling2d(ba2,pool_size=[2,2],strides=2)  
    #[conv-relu-conv-relu-pool]  out=5x5  
    conv3 = tf.layers.conv2d(pool1,512,[3,3],activation=tf.nn.relu)  
    ba3   = tf.layers.batch_normalization(conv3,training=is_training)  
    conv4 = tf.layers.conv2d(ba3,256,[3,3],activation=tf.nn.relu)  
    ba4   = tf.layers.batch_normalization(conv4,training=is_training)  
    pool2 = tf.layers.max_pooling2d(ba4,pool_size=[2,2],strides=2)  
    #[dense-relu]x2 layer  
    pool2_flat = tf.reshape(pool2,[-1,5*5*256])  
    dense1 =tf.layers.dense(pool2_flat,units=512,activation=tf.nn.relu)  
    ba5 = tf.layers.batch_normalization(dense1,center=False,scale=False,training=is_training)  
    dropout1 = tf.layers.dropout(ba5,training=is_training)  
    dense2 = tf.layers.dense(dropout1,units=128,activation=tf.nn.relu)  
    ba6 = tf.layers.batch_normalization(dense2,center=False,scale=False,training=is_training)  
    dropout2 = tf.layers.dropout(ba6,training=is_training)  
    #logit out  
    logits = tf.layers.dense(dropout2,units=10)  
    return logits

tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)
y_out = my_model(X,y,is_training)
total_loss= tf.losses.softmax_cross_entropy(tf.one_hot(y,10),y_out)+tf.losses.get_regularization_loss()
mean_loss = tf.reduce_mean(total_loss)
optimizer = tf.train.RMSPropOptimizer(1e-3,decay=0.90,momentum=0.1)
# batch normalization in tensorflow requires this extra dependency
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
	train_step = optimizer.minimize(mean_loss)

def early_stopping(accuracy,EARLY_STOPPING):
        stop_cnt=0
        stop_max = np.argmax(accuracy)
        stop_len = len(accuracy)
        for i in range(stop_len-1,max(stop_max,stop_len-1-EARLY_STOPPING),-1):
            if accuracy[i]<accuracy[stop_max]:
                stop_cnt=stop_cnt+1
        if stop_cnt>=EARLY_STOPPING:
            return 1
        else:
            return 0


def run_model(session, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False):
    # have tensorflow compute accuracy
    correct_prediction = tf.equal(tf.argmax(predict,1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)

    training_now = training is not None
    
    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [mean_loss,correct_prediction,accuracy]
    if training_now:
        variables[-1] = training
    #counter
    iter_cnt=0
    
    #add by ljj
    EARLY_STOPPING = 5
    eStopAcc = []
    for e in range(epochs):
        # keep track of losses and accuracy
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
            # generate indicies for the batch
            start_idx = (i*batch_size)%X_train.shape[0]
            idx = train_indicies[start_idx:start_idx+batch_size]
            
            # create a feed dictionary for this batch
            feed_dict = {X: Xd[idx,:],
                         y: yd[idx],
                         is_training: training_now }
            # get batch size
            actual_batch_size = yd[i:i+batch_size].shape[0]
            
            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, corr, _ = session.run(variables,feed_dict=feed_dict)
            
            # aggregate performance stats
            losses.append(loss*actual_batch_size)
            correct += np.sum(corr)
            
            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
                      .format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
            iter_cnt += 1
        total_correct = correct/Xd.shape[0]
        total_loss = np.sum(losses)/Xd.shape[0]
        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
              .format(total_loss,total_correct,e+1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e+1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()
        #for early stopping add bi ljj 
        eStopAcc.append(total_correct)
        if early_stopping(eStopAcc,EARLY_STOPPING)==1:
            break
    return total_loss,total_correct


sess = tf.Session()
sess.run(tf.global_variables_initializer())
print('Training')
run_model(sess,y_out,mean_loss,X_train,y_train,20,64,100,train_step,True)
print('Validation')
run_model(sess,y_out,mean_loss,X_val,y_val,1,64)