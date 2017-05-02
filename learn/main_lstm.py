import numpy
import os
import time
import sys
import pandas
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score,confusion_matrix
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.core import Dense,Flatten,Activation
from keras.layers.recurrent import LSTM
from keras.callbacks import Callback,ModelCheckpoint
from layers import MyLayer


numpy.random.seed(6)

class MyCallback(Callback):
    def __init__(self,data,labels_patient,labels_class,m):
        super(MyCallback,self).__init__()
        self.data = data
        self.labels_patient = labels_patient
        self.labels_class = labels_class
        self.m = m

    def on_epoch_end(self, epoch, logs=None):
        labels_predict_total = []
        labels_true_total = []
        for j in xrange(4):
            labels_patient_test_i = self.labels_patient[numpy.argmax(self.labels_class, axis=1) == j]
            data_test_i = self.data[numpy.argmax(self.labels_class, axis=1) == j]
            labels_classes_test_i = self.labels_class[numpy.argmax(self.labels_class, axis=1) == j]

            labels_true = numpy.zeros(len(numpy.unique(labels_patient_test_i)))
            labels_predict = numpy.zeros(len(numpy.unique(labels_patient_test_i)))

            for i, label_patient in enumerate(numpy.unique(labels_patient_test_i)):
                # prendo tutte le sequenze
                data_test_t = data_test_i[labels_patient_test_i == label_patient]

                labels_predict[i] = numpy.argmax(numpy.sum(model.predict(data_test_t), axis=0))
                labels_true[i] = numpy.argmax(labels_classes_test_i[labels_patient_test_i == label_patient][0])

                # print "paziente {} della classe {} predetto {}, predizioni {} ".format(label_patient,j,labels_predict[i],p)
                # print numpy.argmax(numpy.sum(model.predict(data_test_t)[0],axis=0))
            labels_predict_total.extend(labels_predict)
            labels_true_total.extend(labels_true)
            #print "classe {} accuracy pazienti test {}".format(j, accuracy_score(labels_true, labels_predict))

        print confusion_matrix(labels_true_total, labels_predict_total)


#carico dati
BASE_FOLDER = "../processed/sequences/"
TRAIN_FOLDER_OUT = "train/"
TEST_FOLDER_OUT = "test/"
#train
l_train = [f for f in os.listdir(BASE_FOLDER+TRAIN_FOLDER_OUT) if "c3d" in f]
l_train= sorted(l_train,cmp=lambda x,y: 1 if int(x.split(".")[0]) > int(y.split(".")[0]) else -1)
#test
l_test = [f for f in os.listdir(BASE_FOLDER+TEST_FOLDER_OUT) if "c3d" in f]
l_test= sorted(l_test,cmp=lambda x,y: 1 if int(x.split(".")[0]) > int(y.split(".")[0]) else -1)
#parametri
SEQ_NUM = 45
SEQ_LEN = 75
NUM_PARAMS = 81
NUM_COORDS = 1

#TRAIN
data_train = numpy.zeros((len(l_train)*SEQ_NUM,SEQ_LEN,NUM_PARAMS,NUM_COORDS),dtype="float16")
index = 0
for f in l_train:
    arr = numpy.load(BASE_FOLDER+TRAIN_FOLDER_OUT+f)
    arr_0 = arr["arr_0"]
    data_train[index:index+len(arr_0)] = arr_0
    index+=len(arr_0)
#adesso so il numero di sequenze
data_train = data_train[0:index].copy()
labels_classes_train = numpy.load(BASE_FOLDER+TRAIN_FOLDER_OUT+"labels_classes.npy")
labels_patient_train = numpy.load(BASE_FOLDER+TRAIN_FOLDER_OUT+"labels_patients.npy")
#possibilita di eliminare una label
#labels_classes_train = numpy.where(labels_classes_train == 3,2,labels_classes_train)
for i in range(4):
    print "sequenze classe {} train {}  ".format(i,len(data_train[labels_classes_train==i]))
    print "pazienti classe {} train {} ".format(i,len(numpy.unique(labels_patient_train[labels_classes_train == i])))
print "---"
#classe 0 aumentiamo
data_train = numpy.vstack((data_train,numpy.repeat(data_train[labels_classes_train == 0],1,axis=0)))
labels_patient_train = numpy.hstack((labels_patient_train,numpy.repeat(labels_patient_train[labels_classes_train == 0],1)))
labels_classes_train = numpy.hstack((labels_classes_train,numpy.repeat(labels_classes_train[labels_classes_train == 0],1)))


#TEST
data_test = numpy.zeros((len(l_test)*SEQ_NUM,SEQ_LEN,NUM_PARAMS,NUM_COORDS),dtype="float16")
index = 0
for f in l_test:
    arr = numpy.load(BASE_FOLDER+TEST_FOLDER_OUT+f)
    arr_0 = arr["arr_0"]
    data_test[index:index+len(arr_0)] = arr_0
    index+=len(arr_0)
#adesso so il numero di sequenze
data_test = data_test[0:index].copy()
labels_classes_test = numpy.load(BASE_FOLDER+TEST_FOLDER_OUT+"labels_classes.npy")
labels_patient_test = numpy.load(BASE_FOLDER+TEST_FOLDER_OUT+"labels_patients.npy")
#possibilita di eliminare una label
#labels_classes_test = numpy.where(labels_classes_test == 3,2,labels_classes_test)

#distribuzione
for i in range(4):
    print "sequenze classe {} train {} test {} ".format(i,len(data_train[labels_classes_train==i]),len(data_test[labels_classes_test==i]))
    print "pazienti classe {} train {} test {}".format(i,len(numpy.unique(labels_patient_train[labels_classes_train == i])),len(numpy.unique(labels_patient_test[labels_classes_test == i])))

#POSTPROC
#pandas get dummies per avere le classi one-hot
labels_classes_train = pandas.get_dummies(labels_classes_train).values
labels_classes_test = pandas.get_dummies(labels_classes_test).values
#reshape
data_train = data_train.reshape(-1,SEQ_LEN,NUM_PARAMS*NUM_COORDS)
data_test = data_test.reshape(-1,SEQ_LEN,NUM_PARAMS*NUM_COORDS)

# from sklearn.preprocessing import Normalizer
# n = Normalizer()
# n.fit(data_train.reshape(-1,NUM_PARAMS*NUM_COORDS))
# data_train = n.transform(data_train.reshape(-1,NUM_PARAMS*NUM_COORDS)).reshape(-1,SEQ_LEN,NUM_PARAMS*NUM_COORDS)
# data_test = n.transform(data_test.reshape(-1,NUM_PARAMS*NUM_COORDS)).reshape(-1,SEQ_LEN,NUM_PARAMS*NUM_COORDS)

# #normalization
# maxs = numpy.max(data_train.reshape(-1, NUM_PARAMS * NUM_COORDS), axis=0)
# mins = numpy.min(data_train.reshape(-1, NUM_PARAMS * NUM_COORDS), axis=0)
# mins[mins == maxs] -= 1e-06
#
# data_train = (data_train - mins) / (maxs - mins)
# data_test = (data_test - mins) / (maxs - mins + 1e-8)



tic = time.clock()
model = Sequential()
# model.add(LSTM(units=64,input_shape=data_train.shape[1:],return_sequences=True,kernel_initializer="uniform",recurrent_initializer="uniform"))
# model.add(MyLayer(size=64))
# model.add(Activation("tanh"))
# model.add(LSTM(units=64,input_shape=data_train.shape[1:],return_sequences=True,kernel_initializer="uniform",recurrent_initializer="uniform"))
# model.add(Activation("tanh"))
# model.add(MyLayer(size=64))
model.add(LSTM(units=64,input_shape=data_train.shape[1:],return_sequences=True,kernel_initializer="uniform",recurrent_initializer="uniform"))
model.add(MyLayer(size=32))
model.add(Activation("tanh"))
model.add(Flatten())
model.add(Dense(1024,kernel_initializer="uniform"))
model.add(Activation("relu"))
model.add(Dense(496,kernel_initializer="uniform"))
model.add(Activation("relu"))
model.add(Dense(64,kernel_initializer="uniform"))
model.add(Activation("relu"))
model.add(Dense(32,kernel_initializer="uniform"))
model.add(Activation("relu"))
model.add(Dense(labels_classes_train.shape[1],kernel_initializer="uniform"))
model.add(Activation("softmax"))
model.compile(loss="binary_crossentropy",optimizer=Adam(),metrics=["acc"])



while True:
    epochs = raw_input("train? [epochs/]")
    if epochs == "":
        break
    tic = time.clock()

    model.fit(data_train, labels_classes_train, epochs=int(epochs), batch_size=100,
              validation_data=(data_test,labels_classes_test),verbose=2,callbacks=[MyCallback(data_test,labels_patient_test,labels_classes_test,model)])

    # divido per classi
    for j in xrange(4):
        labels_patient_train_i = labels_patient_train[numpy.argmax(labels_classes_train, axis=1) == j]
        data_train_i = data_train[numpy.argmax(labels_classes_train, axis=1) == j]
        labels_classes_train_i = labels_classes_train[numpy.argmax(labels_classes_train, axis=1) == j]

        labels_true = numpy.zeros(len(numpy.unique(labels_patient_train_i)))
        labels_predict = numpy.zeros(len(numpy.unique(labels_patient_train_i)))

        for i, label_patient in enumerate(numpy.unique(labels_patient_train_i)):
            # prendo tutte le sequenze
            data_train_t = data_train_i[labels_patient_train_i == label_patient]

            labels_predict[i] = numpy.argmax(numpy.sum(model.predict(data_train_t), axis=0))
            labels_true[i] = numpy.argmax(labels_classes_train_i[labels_patient_train_i == label_patient][0])
            #print "paziente {} della classe {} predetto {}, predizioni {} ".format(label_patient,j,labels_predict[i],p)
            # print numpy.argmax(numpy.sum(model.predict(data_test_t)[0],axis=0))
        print "classe {} accuracy pazienti train {}".format(j,accuracy_score(labels_true, labels_predict))
    #divido per classi
    for j in xrange(4):
        labels_patient_test_i = labels_patient_test[numpy.argmax(labels_classes_test, axis=1) == j]
        data_test_i = data_test[numpy.argmax(labels_classes_test,axis=1) == j]
        labels_classes_test_i = labels_classes_test[numpy.argmax(labels_classes_test,axis=1) == j]

        labels_true = numpy.zeros(len(numpy.unique(labels_patient_test_i)))
        labels_predict = numpy.zeros(len(numpy.unique(labels_patient_test_i)))

        for i,label_patient in enumerate(numpy.unique(labels_patient_test_i)):
            #prendo tutte le sequenze
            data_test_t = data_test_i[labels_patient_test_i == label_patient]
            p = numpy.sum(model.predict(data_test_t),axis=0)
            #top 2
            labels_top_2 = numpy.argsort(p)[-2:]
            labels_true_1 = numpy.argmax(labels_classes_test_i[labels_patient_test_i == label_patient][0])
            if labels_true_1 in labels_top_2:
                labels_predict[i] = labels_true_1
            else:
                labels_predict[i] = -1

            #labels_predict[i] = numpy.argmax(numpy.sum(model.predict(data_test_t)[-1],axis=0))
            labels_true[i] = numpy.argmax(labels_classes_test_i[labels_patient_test_i == label_patient][0])

            #print "paziente {} della classe {} predetto {}, predizioni {} ".format(label_patient,j,labels_predict[i],p)
            #print numpy.argmax(numpy.sum(model.predict(data_test_t)[0],axis=0))
        print "classe {} accuracy pazienti test {}".format(j,accuracy_score(labels_true,labels_predict))

    labels_predict_total = []
    labels_true_total = []
    for j in xrange(4):
        labels_patient_test_i = labels_patient_test[numpy.argmax(labels_classes_test, axis=1) == j]
        data_test_i = data_test[numpy.argmax(labels_classes_test,axis=1) == j]
        labels_classes_test_i = labels_classes_test[numpy.argmax(labels_classes_test,axis=1) == j]

        labels_true = numpy.zeros(len(numpy.unique(labels_patient_test_i)))
        labels_predict = numpy.zeros(len(numpy.unique(labels_patient_test_i)))

        for i,label_patient in enumerate(numpy.unique(labels_patient_test_i)):
            #prendo tutte le sequenze
            data_test_t = data_test_i[labels_patient_test_i == label_patient]

            labels_predict[i] = numpy.argmax(numpy.sum(model.predict(data_test_t),axis=0))
            labels_true[i] = numpy.argmax(labels_classes_test_i[labels_patient_test_i == label_patient][0])

            #print "paziente {} della classe {} predetto {}, predizioni {} ".format(label_patient,j,labels_predict[i],p)
            #print numpy.argmax(numpy.sum(model.predict(data_test_t)[0],axis=0))
        labels_predict_total.extend(labels_predict)
        labels_true_total.extend(labels_true)
        print "classe {} accuracy pazienti test {}".format(j,accuracy_score(labels_true,labels_predict))

    print confusion_matrix(labels_true_total,labels_predict_total)
    print "impiegato {}".format(time.clock() - tic)

