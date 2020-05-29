'''
@Author: CodeOfSH
@Github: https://github.com/CodeOfSH
@Date: 2020-05-29 16:17:50
@LastEditors: CodeOfSH
@LastEditTime: 2020-05-29 19:08:06
@Description: 
'''

import os
import scipy.io as sio
import numpy as np
import numpy.random as rng
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter

import time

import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model

from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform

from keras.engine.topology import Layer
from keras.regularizers import l2
from keras import backend as K

from sklearn.utils import shuffle

param_path = 'D:/Data/param_050220/'
model_path = 'D:/Codes/ml_localization/model'
motion_files = ["jwj_hori1.mat","sh_verti2.mat","shjwj_vh1.mat","jwj_hori2.mat","sh_walk1.mat","shjwj_vs1.mat","jwj_rand1.mat","sh_walk2.mat","shjwj_vv1.mat","jwj_rand2.mat","sh_walk3.mat","shjwj_vw1.mat","jwj_static1.mat","sh_walk4.mat","shjwj_vw2.mat","jwj_static2.mat","shjwj_hh1.mat","shjwj_wh1.mat","jwj_verti1.mat","shjwj_hs1.mat","shjwj_wh2.mat","jwj_verti2.mat","shjwj_hv1.mat","shjwj_ws1.mat","jwj_walk1.mat","shjwj_hw1.mat","shjwj_ws2.mat","jwj_walk2.mat","shjwj_hw2.mat","shjwj_ws3.mat","jwj_walk3.mat","shjwj_rand1.mat","shjwj_ws4.mat","jwj_walk4.mat","shjwj_rand2.mat","shjwj_wv1.mat","shjwj_rand3.mat","shjwj_wv2.mat","shjwj_sh1.mat","shjwj_ww1.mat","sh_hori1.mat","shjwj_ss1.mat","shjwj_ww2.mat","sh_hori2.mat","shjwj_ss2.mat","shjwj_ww3.mat","sh_rand1.mat","shjwj_sv1.mat","shjwj_ww4.mat","sh_rand2.mat","shjwj_sw1.mat","shjwj_ww5.mat","sh_static1.mat","shjwj_sw2.mat","shjwj_ww6.mat","sh_static2.mat","shjwj_sw3.mat","sh_verti1.mat","shjwj_sw4.mat"]
xrange = [-3.5,3.5]
yrange = [-2.5,2.5]

def read_data():
    X_train, Y_train, X_test, Y_test = [],[],[],[]
    partTrain, partPercent = True, 0.7
    allTrain, allPercent = False, 0.7
    # for fi in range(len(motion_files)):
    for fi in range(4,5):
        fname = param_path+'feature_'+motion_files[fi]
        feature = sio.loadmat(fname) #It is a dict with 2 keys: pos 936*2, pos_feature 936*600*9

        fname = param_path+'pos_'+motion_files[fi]
        gt = sio.loadmat(fname) #It is a dict with 1 key: pos_gt 6000*6

        labelMat = getLabel(xrange,yrange,gt['pos_gt']) # 0/1 matrix of size 936*600
        featureMat = feature['pos_feature']

        if partTrain:
            X_train.append(np.transpose(featureMat[:,:int(featureMat.shape[1]*partPercent),:],(1,0,2)).reshape((-1,9)))
            Y_train.append(np.transpose(labelMat[:,:int(labelMat.shape[1]*partPercent)],(1,0)).flatten())
            X_test.append(np.transpose(featureMat[:,int(featureMat.shape[1]*partPercent):,:],(1,0,2)).reshape(-1,9))
            Y_test.append(np.transpose(labelMat[:,int(labelMat.shape[1] * partPercent):],(1,0)).flatten())
        elif allTrain:
            if fi<=len(motion_files)*allPercent:
                X_train.append(np.transpose(featureMat,(1,0,2)).reshape((-1,9)))
                Y_train.append(np.transpose(labelMat,(1,0)).flatten())
            else:
                X_test.append(np.transpose(featureMat,(1,0,2)).reshape((-1,9)))
                Y_test.append(np.transpose(labelMat,(1,0)).flatten())

    X_train = np.concatenate(X_train)
    Y_train = np.concatenate(Y_train)
    X_test = np.concatenate(X_test)
    Y_test = np.concatenate(Y_test)

    return X_train,Y_train,X_test,Y_test
    


def getLabel(xrange,yrange, pos_gt):
    nSample = pos_gt.shape[0]//10
    nY = int((yrange[1]-yrange[0])/0.2)+1
    nX = int((xrange[1]-xrange[0])/0.2)+1
    ret = np.zeros((nX*nY, nSample))
    for i in range(pos_gt.shape[0]):
        x1,y1,x2,y2 = pos_gt[i,0],pos_gt[i,1],pos_gt[i,3],pos_gt[i,4]
        if x1 !=0 or y1 !=0:
            idx = int((x1-xrange[0])/0.2)*nY+int((y1-yrange[0])/0.2)
            ret[idx,i//10]=1
        if x2!=0 or y2!=0:
            idx = int((x2 - xrange[0]) / 0.2) * nY + int((y2 - yrange[0]) / 0.2)
            ret[idx, i // 10] = 1
    return ret

def initialize_weights(shape, name=None):
    return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)

def initialize_bias(shape, name=None):
    return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)

def get_siamese_model(input_shape):
    """
        Model architecture based on the one provided in: http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
    """
    
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    
    # # Convolutional Neural Network
    # model = Sequential()
    # model.add(Conv2D(64, (10,10), activation='relu', input_shape=input_shape,
    #                kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
    # model.add(MaxPooling2D())
    # model.add(Conv2D(128, (7,7), activation='relu',
    #                  kernel_initializer=initialize_weights,
    #                  bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    # model.add(MaxPooling2D())
    # model.add(Conv2D(128, (4,4), activation='relu', kernel_initializer=initialize_weights,
    #                  bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    # model.add(MaxPooling2D())
    # model.add(Conv2D(256, (4,4), activation='relu', kernel_initializer=initialize_weights,
    #                  bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    # model.add(Flatten())
    # model.add(Dense(4096, activation='sigmoid',
    #                kernel_regularizer=l2(1e-3),
    #                kernel_initializer=initialize_weights,bias_initializer=initialize_bias))
    
    # Convolutional Neural Network
    model = Sequential()
    model.add(Conv2D(4, (2,2), activation='relu', input_shape=input_shape,
                   kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(16, activation='sigmoid',
                   kernel_regularizer=l2(1e-3),
                   kernel_initializer=initialize_weights,bias_initializer=initialize_bias))
    

    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    
    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid',bias_initializer=initialize_bias)(L1_distance)
    
    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    
    # return the model
    return siamese_net

def get_batch(X,Y,batch_size,way='train'):
    '''
    @description: create batch of n pairs, half same class, half different class
    @param {type} 
    @return: pairs, labels 
    '''
    
    n_examples,n_features= X.shape

    # init empty array for pair and targets
    pairs = [np.zeros((batch_size,3,3,1)) for i in range(2)]
    targets = np.zeros((batch_size,))

    # make half of the batch to be 1 to balance the data
    targets[batch_size//2:] = 1

    # cut the X_train data
    X_categories_0=X[Y==0]
    len_cate_0=X_categories_0.shape[0]
    X_categories_1=X[Y==1]
    len_cate_1=X_categories_0.shape[1]

    for i in range(batch_size):
        index_1 = rng.randint(0,len_cate_1)
        pairs[0][i,:,:,:]=X_categories_1[index_1].reshape(3,3,1)

        if i >= batch_size//2:
            index_2 = rng.randint(0,len_cate_1)
            pairs[1][i,:,:,:]=X_categories_1[index_2].reshape(3,3,1)
        else:
            index_2 = rng.randint(0,len_cate_0)
            pairs[1][i,:,:,:]=X_categories_0[index_2].reshape(3,3,1)
    
    pairs[0],pairs[1], targets = shuffle(pairs[0],pairs[1],targets)
    
    return pairs, targets

def generate(batch_size, s="train"):
    """a generator for batches, so model.fit_generator can be used. """
    while True:
        pairs, targets = get_batch(batch_size,s)
        yield (pairs, targets)

def generate_test_batch(X_train,Y_train,X_test,Y_test):
    '''
    @description: generate test data for test in each iteration
    @param {type} 
    @return: pairs, labels
    '''
    len_test=X_test.shape[0]

    # init empty array for pair and targets
    pairs = [np.zeros((len_test,3,3,1)) for i in range(2)]
    targets = np.zeros((len_test,))

    # make half of the len_test to be 1 to balance the data
    targets[len_test//2:] = 1
    
    # cut the X_train data
    X_categories_0=X_train[Y_train==0]
    len_cate_0=X_categories_0.shape[0]
    X_categories_1=X_train[Y_train==1]
    len_cate_1=X_categories_0.shape[1]
    
    for i in range(len_test):
        pairs[0][i,:,:,:]=X_test[i].reshape(3,3,1)
        if i >= len_test//2:
            index_2 = rng.randint(0,len_cate_1)
            pairs[1][i,:,:,:]=X_categories_1[index_2].reshape(3,3,1)
        else:
            index_2 = rng.randint(0,len_cate_0)
            pairs[1][i,:,:,:]=X_categories_0[index_2].reshape(3,3,1)

    pairs[0],pairs[1], targets = shuffle(pairs[0],pairs[1],targets)
    
    return pairs, targets

def test_oneshot(model, test_pairs,test_labels,verbose=True):
    # """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
    n_correct = 0
    print("Evaluating model on test tasks ... \n")
    probs = model.predict(test_pairs)
    probs=np.squeeze(probs)
    print(probs)
    # print(probs.shape)
    # print(test_labels.shape)
    n_correct=np.sum((probs>0.5)==test_labels)
    percent_correct=n_correct/(test_labels.shape[0])*100.0
    print("Got an average of {}% learning accuracy with correnct num of {}\n".format(percent_correct, n_correct))
    return percent_correct

if __name__=='__main__':
    
    X_train,Y_train,X_test,Y_test = read_data()

    test_pairs,test_labels=generate_test_batch(X_train,Y_train,X_test,Y_test)

    model = get_siamese_model((3,3,1))
    model.summary()

    optimizer= Adam(lr=0.00005)
    model.compile(loss="binary_crossentropy", optimizer=optimizer)
    
    # Hyper parameters
    evaluate_every = 200 # interval for evaluating on one-shot tasks
    batch_size = 128
    n_iter = 20000 # No. of training iterations
    N_way = 20 # how many classes for testing one-shot tasks
    n_val = 250 # how many one-shot tasks to validate on
    best = -1

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    print("Starting training process!")
    print("-------------------------------------")
    t_start = time.time()
    for i in range(1, n_iter+1):
        (inputs,targets) = get_batch(X_train,Y_train,batch_size)
        loss = model.train_on_batch(inputs, targets)
        if i % evaluate_every == 0:
            print("\n ------------- \n")
            print("Time for {0} iterations: {1} mins".format(i, (time.time()-t_start)/60.0))
            print("Train Loss: {0}".format(loss)) 
            val_acc = test_oneshot(model, test_pairs, test_labels, verbose=True)
            model.save_weights(os.path.join(model_path, 'weights.{}.h5'.format(i)))
            if val_acc >= best:
                print("Current best: {0}, previous best: {1}".format(val_acc, best))
                best = val_acc