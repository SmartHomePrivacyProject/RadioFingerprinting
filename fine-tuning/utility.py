#! /usr/bin/env python3.6

import os
import sys
import random
import pdb
import copy
import time

from collections import defaultdict
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.preprocessing.sequence import pad_sequences

RootDir = os.getenv('ROOT_DIR')
toolsDir = os.path.join(RootDir, 'tools')
sys.path.append(toolsDir)
import csv2npz
import mytools.fileUtils as fileUtils
import mytools.tools as mytools


def getSiteDict(inp, data_dim):
    wholePack = np.load(inp)
    x, y = wholePack['x'], wholePack['y']

    site_dict = defaultdict(list)
    for i in range(x.shape[0]):
        oneSample = x[i, :]
        oneSample = csv2npz.padData(oneSample, data_dim)
        oneLabel = y[i]
        site_dict[oneLabel].append(oneSample)

    sites = list(site_dict.keys())
    return site_dict, sites


def getSignatureDict(site_dict, n_shot, train_pool_size=20, test_size=70):
    sites = site_dict.keys()
    # create_signature and test_set
    signature_dict = defaultdict()
    test_dict = defaultdict()

    seedVal = int(time.time())
    random.seed(seedVal)

    tmp_train_pool_sz = train_pool_size
    tmp_n_instance = test_size
    for s in sites:
        data_set = site_dict[s]
        random.shuffle(data_set)

        train_pool = copy.deepcopy(data_set[:tmp_train_pool_sz])
        residual_pool = copy.deepcopy(data_set[tmp_train_pool_sz:])

        signature_dict[s] = random.sample(train_pool, n_shot)
        test_dict[s] = random.sample(residual_pool, tmp_n_instance)  # really wired, why in 7 hell they code like this?

    return signature_dict, test_dict


def getDataDict(droot, n_shot, data_dim, train_pool_size, test_size):
    site_dict, sites = getSiteDict(droot, data_dim)
    signature_dict, test_dict = getSignatureDict(site_dict, n_shot, train_pool_size, test_size)

    return signature_dict, test_dict, sites


def create_test_set_Wang_disjoint(signature_dict, test_dict, sites, features_model, type_exp):
    # Feed signature vector to the model to create embedded signature feature's vectors
    signature_vector_dict = {}
    for i in sites:
        signature_instance = signature_dict[i]
        signature_instance = np.array(signature_instance)
        signature_instance = signature_instance.astype('float32')
        signature_instance = signature_instance[:, :, np.newaxis]
        signature_vector = features_model.predict(signature_instance)
        if type_exp == "N-MEV":
            signature_vector = np.array([signature_vector.mean(axis=0)])
        signature_vector_dict[i] = signature_vector

    # Feed test vector to the model to create embedded test feature's vectors
    test_vector_dict = {}
    for i in sites:
        test_instance = test_dict[i]
        test_instance = np.array(test_instance)
        test_instance = test_instance.astype('float32')
        test_instance = test_instance[:, :, np.newaxis]
        test_vector = features_model.predict(test_instance)
        test_vector_dict[i] = test_vector

    return signature_vector_dict, test_vector_dict


def computeTopN(topN, knn, X_test, y_test):
    # Top 5
    count_correct = 0
    for s in range(len(X_test)):
        test_example = X_test[s]
        class_label = y_test[s]
        predict_prob = knn.predict_proba([test_example])
        best_n = np.argsort(predict_prob[0])[-topN:]
        class_mapping = knn.classes_
        top_n_list = []
        for p in best_n:
            top_n_list.append(class_mapping[p])
        if class_label in top_n_list:
            count_correct = count_correct + 1
    #print float(count_correct), float(len(X_test))
    acc_knn_top5 = float(count_correct) / float(len(X_test))
    acc_knn_top5 = float("{0:.15f}".format(round(acc_knn_top5, 6)))

    return acc_knn_top5


def kNN_train(signature_vector_dict, params):
    site_labels = list(signature_vector_dict.keys())
    random.shuffle(site_labels)
    X_train, y_train = mytools.datadict2data(signature_vector_dict)
    print('kNN training data shape: ', X_train.shape)

    knn = KNeighborsClassifier(n_neighbors=params['k'], weights=params['weights'], p=params['p'], metric=params['metric'], algorithm='brute')
    knn.fit(X_train, y_train)

    return knn, site_labels


def kNN_accuracy(signature_vector_dict, test_vector_dict, params):
    knnModel, tested_sites = kNN_train(signature_vector_dict, params)

    X_test, y_test = [], []
    for s in tested_sites:
        for i in range(len(test_vector_dict[s])):
            X_test.append(test_vector_dict[s][i])
            y_test.append(s)

    # Top-1
    acc_knn_top1 = accuracy_score(y_test, knnModel.predict(X_test))
    acc_knn_top1 = float("{0:.15f}".format(round(acc_knn_top1, 6)))
    # Top-5
    acc_knn_top5 = computeTopN(5, knnModel, X_test, y_test)
    print('KNN accuracy Top1 = ', acc_knn_top1, '\tKNN accuracy Top5 = ', acc_knn_top5)
    return acc_knn_top1, acc_knn_top5


def splitMonAndUnmon(test_vector_dict):
    X_test_mon, y_test_mon, X_test_unmon, y_test_unmon = [], [], [], []
    tested_sites = list(test_vector_dict.keys())
    maxLabel = max(tested_sites)
    for s in tested_sites:
        oneCls = test_vector_dict[s]
        num = len(oneCls)
        labels = np.ones(num, dtype=np.int) * s
        if s == maxLabel:
            X_test_unmon.extend(oneCls)
            y_test_unmon.extend(labels)
        else:
            X_test_mon.extend(oneCls)
            y_test_mon.extend(labels)

    X_test_mon, X_test_unmon = np.array(X_test_mon), np.array(X_test_unmon)
    return X_test_mon, y_test_mon, X_test_unmon, y_test_unmon, maxLabel


def calculatePrecAndRecAndTPRAndFPR(result_Mon, result_Unmon, y_test_mon, maxLabel, threshold_val):
    TP, FP, TN, FN = 0, 0, 0, 0
    monitored_label = list(set(y_test_mon))
    unmonitored_label = [maxLabel]

    # ==============================================================
    # Test with Monitored testing instances
    # evaluation
    for i in range(len(result_Mon)):
        sm_vector = result_Mon[i]
        predicted_class = np.argmax(sm_vector)
        max_prob = max(sm_vector)

        if predicted_class in monitored_label: # predicted as Monitored
            if max_prob >= threshold_val: # predicted as Monitored and actual site is Monitored
                TP = TP + 1
            else: # predicted as Unmonitored and actual site is Monitored
                FN = FN + 1
        elif predicted_class in unmonitored_label: # predicted as Unmonitored and actual site is Monitored
            FN = FN + 1

    # ==============================================================
    # Test with Unmonitored testing instances
    # evaluation
    for i in range(len(result_Unmon)):
        sm_vector = result_Unmon[i]
        predicted_class = np.argmax(sm_vector)
        max_prob = max(sm_vector)

        if predicted_class in monitored_label: # predicted as Monitored
            if max_prob >= threshold_val: # predicted as Monitored and actual site is Unmonitored
                FP = FP + 1
            else: # predicted as Unmonitored and actual site is Unmonitored
                TN = TN + 1
        elif predicted_class in unmonitored_label: # predicted as Unmonitored and actual site is Unmonitored
            TN = TN + 1

    print("TP : ", TP, "\tFP : ", FP, "\tTN : ", TN, "\tFN : ", FN)
    print("Total  : ", TP + FP + TN + FN)

    TPR = float(TP) / (TP + FN)
    FPR = float(FP) / (FP + TN)
    print("TPR : ", TPR, "\tFPR : ",  FPR)

    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    print("Precision : ", Precision, "\tRecall : ", Recall)
    return Precision, Recall, TPR, FPR


def kNN_precision_recall(signature_vector_dict, test_vector_dict, params, thresHold):
    # print("Size of problem : ", size_of_problem)
    print("Testing with threshold = ", thresHold)
    knnModel, tested_sites = kNN_train(signature_vector_dict, params)

    X_test_mon, y_test_mon, X_test_unmon, y_test_unmon, maxLabel = splitMonAndUnmon(test_vector_dict)

    result_Mon = knnModel.predict_proba(X_test_mon)
    result_Unmon = knnModel.predict_proba(X_test_unmon)

    Precision, Recall, TPR, FPR = calculatePrecAndRecAndTPRAndFPR(result_Mon, result_Unmon, y_test_mon, maxLabel, thresHold)
    return Precision, Recall, TPR, FPR


''' ----------------------------------------------------------
# ------------ 计算数据相似度，把大矩阵分成小矩阵 ------------
# ---------------------------------------------------------'''


def preprocessOneFile(fp):
    oneData = np.load(fp)
    return oneData


def loadData(fList, max_len, droot=''):
    if not isinstance(fList, list):
        fList = list(fList)
    if droot:
        new_fList = ['{}'.format(os.path.join(droot, fn)) for fn in fList]
        fList = new_fList
    allData = []
    for fp in fList:
        tmp = preprocessOneFile(fp)
        allData.append(tmp)

    X = pad_sequences(allData, maxlen=max_len,
                      padding='post', truncating='post')
    X = X[:, :, np.newaxis]
    return X


def computeSimMat(conv, block, item, max_len):
    dMat1 = loadData(block, max_len)
    dMat2 = loadData(item, max_len)
    embs1 = conv.predict(dMat1)
    embs2 = conv.predict(dMat2)

    embs1 = embs1 / np.linalg.norm(embs1, axis=-1, keepdims=True)
    embs2 = embs2 / np.linalg.norm(embs2, axis=-1, keepdims=True)

    all_sims = np.dot(embs1, embs2.T)
    return all_sims


def compute_one_row(conv, block, blockList, max_len):
    '''对每一个block, 计算他与其他block的值，然后连接起来返回'''
    for i, item in enumerate(blockList):
        if 0 == i:
            subMat = computeSimMat(conv, block, item, max_len)
        else:
            tmpMat = computeSimMat(conv, block, item, max_len)
            subMat = np.hstack((subMat, tmpMat))
    return subMat


def build_similarities(conv, droot, fnames, batch_size, max_len):
    '''传整个matrix可能会导致内存溢出，而我们只需要一个matrix
    所以我们可以分别算每个小的matrix，然后再把它们拼接起来
    尝试了下循环，根本写不出来，感觉应该用递归'''
    fList = ['{}'.format(os.path.join(droot, fn)) for fn in fnames]
    fNum = len(fList)
    if fNum <= batch_size:
        simMat = computeSimMat(conv, fList, fList, max_len)
    else:
        # cut the file list into block list with size of batch_size
        blockNum = fNum // batch_size + 1
        blockList = []
        start, end = 0, batch_size
        for i in range(blockNum):
            tmp = fList[start:end]
            blockList.append(tmp)
            start = end
            end = end + batch_size

        # now we get the block list, we can fill the mat row by row
        simMat = np.zeros(shape=(fNum, fNum))
        start, end = 0, batch_size
        for i, block in enumerate(blockList):
            one_row = compute_one_row(conv, block, blockList, max_len)
            simMat[start:end, :] = one_row
            start = end
            end = end + one_row.shape[0]

    return simMat
