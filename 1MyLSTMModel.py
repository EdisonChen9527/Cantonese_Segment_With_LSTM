
import keras
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Bidirectional, TimeDistributed, Activation
from keras.models import Sequential

import numpy as np

'''
每句话最长截断为64个字，多退少补
onehot_length包含UNK
'''
onehot_length = 2542
n_timesteps = 64
batch_size = 128
'''分词的S、B、M、E四类'''
num_classes = 4
epochs = 12

'''读取字典'''
def read_dict(dict_path):
    dict_temp = open(dict_path,encoding='utf-8')
    character_dict = dict()
    count = 1
    for i in dict_temp:
        i = i.strip("\n")
        character_dict[i] = count
        count = count + 1
    character_dict["UNK"] = count
    return character_dict

'''读取指定文件，获取训练数据、测试数据'''
def open_data(X_train_orig_path,Y_train_orig_path,X_test_orig_path,Y_test_orig_path,dict_path):
    character_dict = read_dict(dict_path)

    count_train = 0
    X_train_orig_temp = open(X_train_orig_path,encoding='utf-8')
    Y_train_orig_temp = open(Y_train_orig_path,encoding='utf-8')
    X_train_orig = list()
    Y_train_orig = list()
    for i in X_train_orig_temp:
        i = i.strip("\n")
        j = Y_train_orig_temp.readline().strip("\n")
        X_train_orig_line = np.zeros((n_timesteps,1))
        Y_train_orig_line = np.zeros((n_timesteps,1))
        for index in range(min(len(i),n_timesteps)):
            X_train_orig_line[index,0] = character_dict.get(i[index],0)
            Y_train_orig_line[index,0] = int(j[index])
        X_train_orig.append(X_train_orig_line)
        Y_train_orig.append(Y_train_orig_line)
        count_train = count_train + 1

    count_test = 0
    X_test_orig_temp = open(X_test_orig_path,encoding='utf-8')
    Y_test_orig_temp = open(Y_test_orig_path,encoding='utf-8')
    X_test_orig = list()
    Y_test_orig = list()
    for i in X_test_orig_temp:
        i = i.strip("\n")
        j = Y_test_orig_temp.readline().strip("\n")
        X_test_orig_line = np.zeros((n_timesteps,1))
        Y_test_orig_line = np.zeros((n_timesteps,1))
        for index in range(min(len(i),n_timesteps)):
            X_test_orig_line[index,0] = character_dict.get(i[index],0)
            Y_test_orig_line[index,0] = int(j[index])
        X_test_orig.append(X_test_orig_line)
        Y_test_orig.append(Y_test_orig_line)
        count_test = count_test + 1

    return (X_train_orig,Y_train_orig,count_train,X_test_orig,Y_test_orig,count_test)

'''调整训练数据、测试数据格式'''
def load_data(X_train_orig,Y_train_orig,count_train,X_test_orig,Y_test_orig,count_test):
    X_train = np.zeros((count_train,n_timesteps))
    Y_train = np.zeros((count_train,n_timesteps))
    for m in range(count_train):
        for n in range(n_timesteps):
            if X_train_orig[m][n,0] != 0:
                X_train[m,n] = X_train_orig[m][n,0]
                Y_train[m,n] = int(Y_train_orig[m][n,0])
    Y_train.reshape((count_train,-1))

    X_test = np.zeros((count_test,n_timesteps))
    Y_test = np.zeros((count_test,n_timesteps))
    for m in range(count_test):
        for n in range(n_timesteps):
            if X_test_orig[m][n,0] != 0:
                X_test[m,n] = X_test_orig[m][n,0]
                Y_test[m,n] = int(Y_test_orig[m][n,0])
    Y_test.reshape((count_test,-1))

    return (X_train,Y_train,X_test,Y_test)

(X_train_orig,Y_train_orig,count_train,X_test_orig,Y_test_orig,count_test) = open_data("./X_train_orig","./Y_train_orig","./X_test_orig","./Y_test_orig","./characterDict")
(X_train,Y_train,X_test,Y_test) = load_data(X_train_orig,Y_train_orig,count_train,X_test_orig,Y_test_orig,count_test)

Y_train = keras.utils.to_categorical(Y_train, num_classes)
Y_test = keras.utils.to_categorical(Y_test, num_classes)

model = Sequential()
model.add(Embedding(input_dim=onehot_length+1, output_dim=256, input_length=n_timesteps, mask_zero=True))
model.add(Bidirectional(LSTM(128, activation='tanh', return_sequences=True)))
model.add(Bidirectional(LSTM(64, activation='tanh', return_sequences=True)))
model.add(Bidirectional(LSTM(32, activation='tanh', return_sequences=True)))

model.add(Dense(num_classes))
model.add(Activation("softmax"))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
print("X_train shape:",X_train.shape)
print("Y_train shape:",Y_train.shape)
print("X_test shape:",X_test.shape)
print("Y_test shape:",Y_test.shape)
model.fit(X_train, Y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_test, Y_test))
