import numpy as np
import glob
import pandas as pd
import keras
import matplotlib.pyplot as plt
from random import randint

from keras.optimizers import Adam
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras import regularizers

#NN = Neural Network
#Number of dimensions for the adaf NN
input_dim = 4

#ADAF NN
#Building the NN

adaf = Sequential()
adaf.add(Dense(8*7, input_dim = input_dim , activation = 'sigmoid'))
adaf.add(Dense(60, activation = 'sigmoid'))
adaf.add(Dense(99, activation = 'sigmoid', kernel_regularizer=regularizers.l2(0.00001)))
adaf.add(Dense(99, activation = 'linear'))
adaf.load_weights('ADAF.h5')


#Number of dimensions for the jet NN
input_dim_jet = 5 #the input dim is the size of how many variables

#Jet NN
#Building the NN
jet = Sequential()
jet.add(Dense(10, input_dim = input_dim_jet , activation = 'relu'))
jet.add(Dense(44, activation = 'relu'))
jet.add(Dense(66, activation = 'relu'))
jet.add(Dense(99, activation = 'relu'))
jet.add(Dense(130, activation = 'linear'))
jet.compile(loss='mae', optimizer='Adam', metrics=['mape'])
#Loading the weights from external file
jet.load_weights('Jet.h5')
