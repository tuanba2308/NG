from __future__ import print_function
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import keras    
from keras.models import Sequential
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras import backend as K
import matplotlib

data=sys.argv[1] #dataset path: 'kddcup99.csv'
label=sys.argv[2] #label column name: 'label'

df=pd.read_csv(data)
list(df.columns)
X=df.drop(columns=[label])
columns=list(X)
obj_cols=[]
for i in columns:
	if X.dtypes[i].name == 'object':
		obj_cols.append(i)
for i in obj_cols:
	X[i]=X[i].astype('category').cat.codes

df['labels'] =df[label].astype('category').cat.codes
Y=df['labels']


x_train, x_test, y_train, y_test = train_test_split(np.asarray(X), np.asarray(Y), test_size=0.33, shuffle= True)
num_classes=Y.nunique()
input_shape =(X.columns.nunique(),)

y_train_binary = keras.utils.to_categorical(y_train, num_classes)
y_test_binary = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],1)

model = Sequential()
model.add(Conv1D(32, (3), input_shape=(X.columns.nunique(),1), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='softmax'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.summary()
batch_size = 128
epochs = 10
model = model.fit(x_train, y_train_binary,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test_binary))

