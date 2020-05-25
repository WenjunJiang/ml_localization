import scipy.io as sio
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


def getLabel(xrange,yrange, pos_gt):
    nSample = pos_gt.shape[0]//10
    nY = int((yrange[1]-yrange[0])/0.2)+1
    nX = int((xrange[1]-xrange[0])/0.2)+1
    ret = np.zeros((nX*nY, nSample))
    for i in range(pos_gt.shape[0]):
        x1,y1,x2,y2 = pos_gt[i,0],pos_gt[i,1],pos_gt[i,3],pos_gt[i,4]
        if x1 >= xrange[0] and y1>=yrange[0]:
            idx = int((x1-xrange[0])/0.2)*nY+int((y1-yrange[0])/0.2)
            ret[idx,i//10]=1
        if x2>=xrange[0] and y2>=yrange[0]:
            idx = int((x2 - xrange[0]) / 0.2) * nY + int((y2 - yrange[0]) / 0.2)
            ret[idx, i // 10] = 1
    return ret

def ml_clf(X_train,Y_train,X_test,Y_test):
    # clf = svm.SVC()
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(X_train,Y_train)
    Y_pred = clf.predict(X_test)
    FF,FT,TF,TT = 0,0,0,0
    for i in range(Y_pred.shape[0]):
        if Y_test[i]==0:
            if Y_pred[i]==0:
                FF+=1
            else:
                FT+=1
        else:
            if Y_pred[i] == 0:
                TF += 1
            else:
                TT += 1
    print('FF:{},FT:{},TF:{},TT:{}'.format(FF,FT,TF,TT))

if __name__=='__main__':
    param_path = '../../data/param_050220/'
    motion_files = ["jwj_hori1.mat","sh_verti2.mat","shjwj_vh1.mat","jwj_hori2.mat","sh_walk1.mat","shjwj_vs1.mat","jwj_rand1.mat","sh_walk2.mat","shjwj_vv1.mat","jwj_rand2.mat","sh_walk3.mat","shjwj_vw1.mat","jwj_static1.mat","sh_walk4.mat","shjwj_vw2.mat","jwj_static2.mat","shjwj_hh1.mat","shjwj_wh1.mat","jwj_verti1.mat","shjwj_hs1.mat","shjwj_wh2.mat","jwj_verti2.mat","shjwj_hv1.mat","shjwj_ws1.mat","jwj_walk1.mat","shjwj_hw1.mat","shjwj_ws2.mat","jwj_walk2.mat","shjwj_hw2.mat","shjwj_ws3.mat","jwj_walk3.mat","shjwj_rand1.mat","shjwj_ws4.mat","jwj_walk4.mat","shjwj_rand2.mat","shjwj_wv1.mat","shjwj_rand3.mat","shjwj_wv2.mat","shjwj_sh1.mat","shjwj_ww1.mat","sh_hori1.mat","shjwj_ss1.mat","shjwj_ww2.mat","sh_hori2.mat","shjwj_ss2.mat","shjwj_ww3.mat","sh_rand1.mat","shjwj_sv1.mat","shjwj_ww4.mat","sh_rand2.mat","shjwj_sw1.mat","shjwj_ww5.mat","sh_static1.mat","shjwj_sw2.mat","shjwj_ww6.mat","sh_static2.mat","shjwj_sw3.mat","sh_verti1.mat","shjwj_sw4.mat"]
    xrange = [-3.5,3.5]
    yrange = [-2.5,2.5]

    X_train, Y_train, X_test, Y_test = [],[],[],[]
    partTrain, partPercent = True, 0.7
    allTrain, allPercent = False, 0.7
    # for fi in range(len(motion_files)):
    for fi in range(1):
        fname = param_path+'feature_'+motion_files[fi]
        feature = sio.loadmat(fname) #It is a dict with 2 keys: pos 936*2, pos_feature 936*600*9

        fname = param_path+'pos_'+motion_files[fi]
        gt = sio.loadmat(fname) #It is a dict with 1 key: pos_gt 6000*6

        labelMat = getLabel(xrange,yrange,gt['pos_gt']) # 0/1 matrix of size 936*600
        featureMat = feature['pos_feature']

        if partTrain:
            X_train.append(featureMat[:,:int(featureMat.shape[1]*partPercent),:].reshape((-1,9)))
            Y_train.append(labelMat[:,:int(labelMat.shape[1]*partPercent)].flatten())
            X_test.append(featureMat[:,int(featureMat.shape[1]*partPercent):,:].reshape(-1,9))
            Y_test.append(labelMat[:,int(labelMat.shape[1] * partPercent):].flatten())
        elif allTrain:
            if fi<=len(motion_files)*allPercent:
                X_train.append(featureMat.reshape((-1,9)))
                Y_train.append(labelMat.flatten())
            else:
                X_test.append(featureMat.reshape((-1,9)))
                Y_test.append(labelMat.flatten())

    X_train = np.concatenate(X_train)
    Y_train = np.concatenate(Y_train)
    X_test = np.concatenate(X_test)
    Y_test = np.concatenate(Y_test)

    ml_clf(X_train,Y_train,X_test,Y_test)