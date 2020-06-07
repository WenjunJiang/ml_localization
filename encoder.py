'''
@Author: CodeOfSH
@Github: https://github.com/CodeOfSH
@Date: 2020-05-30 15:36:40
@LastEditors: CodeOfSH
@LastEditTime: 2020-06-03 23:08:10
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
from keras.optimizers import *
from keras.layers import *
from keras.models import Model, load_model

from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, MaxPooling1D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform

from keras.engine.topology import Layer
from keras.regularizers import l2
from keras import backend as K

from keras.callbacks import TensorBoard

from sklearn.utils import shuffle

param_path = 'D:/Data/param_050220/'
model_path = './model/'
save_path = './output/'
motion_files = ["jwj_hori1.mat","sh_verti2.mat","shjwj_vh1.mat","jwj_hori2.mat","sh_walk1.mat","shjwj_vs1.mat","jwj_rand1.mat","sh_walk2.mat","shjwj_vv1.mat","jwj_rand2.mat","sh_walk3.mat","shjwj_vw1.mat","jwj_static1.mat","sh_walk4.mat","shjwj_vw2.mat","jwj_static2.mat","shjwj_hh1.mat","shjwj_wh1.mat","jwj_verti1.mat","shjwj_hs1.mat","shjwj_wh2.mat","jwj_verti2.mat","shjwj_hv1.mat","shjwj_ws1.mat","jwj_walk1.mat","shjwj_hw1.mat","shjwj_ws2.mat","jwj_walk2.mat","shjwj_hw2.mat","shjwj_ws3.mat","jwj_walk3.mat","shjwj_rand1.mat","shjwj_ws4.mat","jwj_walk4.mat","shjwj_rand2.mat","shjwj_wv1.mat","shjwj_rand3.mat","shjwj_wv2.mat","shjwj_sh1.mat","shjwj_ww1.mat","sh_hori1.mat","shjwj_ss1.mat","shjwj_ww2.mat","sh_hori2.mat","shjwj_ss2.mat","shjwj_ww3.mat","sh_rand1.mat","shjwj_sv1.mat","shjwj_ww4.mat","sh_rand2.mat","shjwj_sw1.mat","shjwj_ww5.mat","sh_static1.mat","shjwj_sw2.mat","shjwj_ww6.mat","sh_static2.mat","shjwj_sw3.mat","sh_verti1.mat","shjwj_sw4.mat"]
xrange = [-3.5,3.5]
yrange = [-2.5,2.5]

def read_data():
    X_train, Y_train, X_test, Y_test = [],[],[],[]
    partTrain, partPercent = True, 0.7
    allTrain, allPercent = False, 0.7
    # for fi in range(len(motion_files)):
    for fi in range(len(motion_files)-5):
        fname = param_path+'feature_'+motion_files[fi]
        feature = sio.loadmat(fname) #It is a dict with 2 keys: pos 936*2, pos_feature 936*600*9

        fname = param_path+'pos_'+motion_files[fi]
        gt = sio.loadmat(fname) #It is a dict with 1 key: pos_gt 6000*6

        labelMat = getLabel(xrange,yrange,gt['pos_gt']) # 0/1 matrix of size 936*600
        featureMat = feature['pos_feature']

        if partTrain:
            X_train.append(np.transpose(featureMat[:,:int(featureMat.shape[1]*partPercent),:],(1,0,2)).reshape((-1,36,26,9)))
            Y_train.append(np.transpose(labelMat[:,:int(labelMat.shape[1]*partPercent)],(1,0)).reshape(-1,36,26,1))
            X_test.append(np.transpose(featureMat[:,int(featureMat.shape[1]*partPercent):,:],(1,0,2)).reshape(-1,36,26,9))
            Y_test.append(np.transpose(labelMat[:,int(labelMat.shape[1] * partPercent):],(1,0)).reshape(-1,36,26,1))
        elif allTrain:
            if fi<=len(motion_files)*allPercent:
                X_train.append(np.transpose(featureMat,(1,0,2)).reshape((-1,36,26,9)))
                Y_train.append(np.transpose(labelMat,(1,0)).reshape(-1,36,26,1))
            else:
                X_test.append(np.transpose(featureMat,(1,0,2)).reshape((-1,36,26,9)))
                Y_test.append(np.transpose(labelMat,(1,0)).reshape(-1,36,26,1))

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
    
def get_model(input_shape,output_shape):
    # input_img = Input(shape=input_shape)
    # encoded = Flatten()(input_img)
    # encoded = Dense(512, activation='relu')(encoded)
    # encoded = Dense(256, activation='relu')(encoded)
    # encoded = Dense(128, activation='relu')(encoded)
    # encoded = Dense(64, activation='relu')(encoded)
    # encoded = Dense(32, activation='relu')(encoded)

    # decoded = Dense(64, activation='relu')(encoded)
    # decoded = Dense(128, activation='relu')(decoded)
    # decoded = Dense(256, activation='relu')(decoded)
    # decoded = Dense(512, activation='relu')(decoded)
    # decoded = Dense(936, activation='sigmoid')(decoded)
    # decoded = Reshape((36,26))(decoded)

    # autoencoder = Model(input=input_img, output=decoded)
    # return autoencoder

    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(9,9),activation='relu',padding='same'))
    model.add(Conv2D(64, kernel_size=(3,3),activation='relu',padding='same'))
    model.add(Conv2D(128, kernel_size=(3,3),activation='relu',padding='same'))
    model.add(Dense(64,activation='relu'))
    model.add(Conv2DTranspose(128, kernel_size=(3,3),activation='relu',padding='same'))
    model.add(Conv2DTranspose(64, kernel_size=(3,3),activation='relu',padding='same'))
    model.add(Conv2DTranspose(32, kernel_size=(9,9),activation='relu',padding='same'))
    model.add(Conv2DTranspose(1, kernel_size=(3,3),activation='sigmoid',padding='same'))

    return model

    # input_data = Input(input_shape)
    # x = Convolution2D(32, (9, 9), activation='relu', padding='same')(input_data)  
    # x = MaxPooling2D((2, 2), padding='same')(x)  
    # x = Convolution2D(64, (3, 3), activation='relu', padding='same')(x)  
    # # x = MaxPooling2D((2, 2), padding='same')(x)  
    # x = Convolution2D(128, (3, 3), activation='relu', padding='same')(x)
    # # x = MaxPooling2D((2, 2), padding='same')(x)  
    # x = Dense(256,activation='relu')(x)
    # x = Convolution2D(128, (3, 3), activation='relu', padding='same')(x)  
    # # x = UpSampling2D((2, 2))(x)  
    # x = Convolution2D(64, (3, 3), activation='relu', padding='same')(x)  
    # x = UpSampling2D((2, 2))(x)  
    # x = Convolution2D(32, (9, 9), activation='relu',padding='same')(x)  
    # # x = UpSampling2D((2, 2))(x)  
    # decoded = Convolution2D(1, (3, 3), activation='sigmoid', padding='same')(x)  
    
    # autoencoder = Model(inputs=input_data, outputs=decoded)
    # return autoencoder


def test_task(model, X_test, Y_test, mode=0):
    print("---Evaluating model on test tasks---")
    if mode == 0:
        probs = model.predict(X_test)
        probs=np.squeeze(probs)
        print(probs)
        n_correct=np.sum((probs>0.5)==Y_test)
        percent_correct=n_correct/(Y_test.shape[0])*100.0
        print("Got an average of {}% learning accuracy with correnct num of {}".format(percent_correct, n_correct))
        return percent_correct
    else:
        FF,FT,TF,TT = 0,0,0,0
        Y_pred = model.predict(X_test)
        print(Y_pred)
        for i in range(Y_pred.shape[0]):
            if Y_test[i]==0:
                if Y_pred[i] < 0.5:
                    FF+=1
                else:
                    FT+=1
            else:
                if Y_pred[i] < 0.5:
                    TF += 1
                else:
                    TT += 1
        print('FF:{},FT:{},TF:{},TT:{}'.format(FF,FT,TF,TT))

def plot_map(file_name, predict, groundtruth):
    local_save_path=save_path+file_name+"/"
    if not os.path.exists(local_save_path):
        os.makedirs(local_save_path)
    length = predict.shape[0]
    fig,axes = plt.subplots(1,2)
    save_file_name = local_save_path+file_name
    print("Now processing: "+str(save_file_name))
    for i in range(length):

        print("Step: %04d"%i)
        matrix_predict=predict[i].reshape(36,26)
        matrix_groundtruth=groundtruth[i].reshape(36,26)
        
        axes[0].imshow(matrix_predict)
        # plt.sca(axes[0])
        # plt.xticks(np.arange(1,26,1),np.arange(-2.5, 2.5, step=0.2))
        # plt.yticks(np.arange(1,36,1),np.arange(-3.5, 3.5, step=0.2))
        
        axes[1].matshow(matrix_groundtruth)
        # plt.sca(axes[1])
        # plt.xticks(np.arange(1,26,1),np.arange(-2.5, 2.5, step=0.2))
        # plt.yticks(np.arange(1,36,1),np.arange(-3.5, 3.5, step=0.2))
        

        plt.savefig(save_file_name+"_%04d.png"%i)

def test_on_file(model,file_names):
    for file_name in file_names:
        fname = param_path + 'feature_' + file_name
        feature = sio.loadmat(fname) #It is a dict with 2 keys: pos 936*2, pos_feature 936*600*9

        fname = param_path + 'pos_' + file_name
        gt = sio.loadmat(fname) #It is a dict with 1 key: pos_gt 6000*6

        labelMat = getLabel(xrange,yrange,gt['pos_gt']) # 0/1 matrix of size 936*600
        featureMat = feature['pos_feature']
        
        inputs, labels=[],[]

        inputs = np.transpose(featureMat,(1,0,2)).reshape((-1,36,26,9))
        labels = np.transpose(labelMat,(1,0)).reshape(-1,936)

        predict = model.predict(inputs)
        plot_map(file_name[:-4],predict,labels)
    
def test_on_array(model, X_test, Y_test):
    local_save_path=save_path+"test_array/"
    if not os.path.exists(local_save_path):
        os.makedirs(local_save_path)
    predict = model.predict(X_test)
    length = predict.shape[0]
    fig,axes = plt.subplots(1,2)
    save_file_name = local_save_path
    print("Now processing: "+str(save_file_name))
    for i in range(length):

        print("Step: %04d"%i)
        matrix_predict=predict[i].reshape(36,26)
        matrix_groundtruth=Y_test[i].reshape(36,26)
        
        axes[0].imshow(matrix_predict)
        # plt.sca(axes[0])
        # plt.xticks(np.arange(1,26,1),np.arange(-2.5, 2.5, step=0.2))
        # plt.yticks(np.arange(1,36,1),np.arange(-3.5, 3.5, step=0.2))
        
        axes[1].matshow(matrix_groundtruth)
        # plt.sca(axes[1])
        # plt.xticks(np.arange(1,26,1),np.arange(-2.5, 2.5, step=0.2))
        # plt.yticks(np.arange(1,36,1),np.arange(-3.5, 3.5, step=0.2))

        plt.savefig(save_file_name+"test_%04d.png"%i)


if __name__=='__main__':
    
    X_train,Y_train,X_test,Y_test = read_data()
    print(X_train.shape)
    print(Y_train.shape)

    model = get_model((36,26,9),(36,26))
    model.summary()

    optimizer= Adam(lr=0.00005)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    batch_size = 256
    iteration = 300
    record_every = 100
    best = -1
    need_train = True

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if need_train:

        print("Starting training process!")
        print("-------------------------------------")
        t_start = time.time()

        history = model.fit(X_train, Y_train, epochs=iteration, batch_size=batch_size, shuffle=True, validation_split=0.1)
        # val_acc = test_task(model, X_test, Y_test)
        # test_task(model, X_test, Y_test, mode=1)

        model.save(model_path+"autoencoder_model.h5")

        print(history.history.keys())
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.savefig("./train_loss_pic.png")
        plt.show(block=False)


    model.load_weights(model_path+"autoencoder_model.h5")
    model.summary()
    test_on_array(model,X_test,Y_test)


    # test_files=["jwj_hori2.mat","sh_walk1.mat","shjwj_vs1.mat"]
    # test_files=["sh_walk1.mat","jwj_rand2.mat","shjwj_vs1.mat"]
    test_files=["shjwj_ww6.mat","sh_static2.mat","shjwj_sw3.mat","sh_verti1.mat","shjwj_sw4.mat"]

    # test_on_file(model,test_files)