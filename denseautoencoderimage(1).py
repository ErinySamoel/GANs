

import keras 
from keras import layers
import tensorflow as tf
# This is the size of our encoded representations 
encoding_dim = 30 # 32 floats -> compression of factor 24.5, assuming the input is 784 floats 
# This is our input image 
input_img = keras.Input(shape=(122,))
# "encoded" is the encoded representation of the input 
encoded = layers.Dense(encoding_dim, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(input_img)#relu
# "decoded" is the lossy reconstruction of the input
decoded = layers.Dense(122, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(encoded) #sigmoid
# This model maps an input to its reconstruction 
autoencoder = keras.Model(input_img, decoded) # This model maps an input to its encoded representation 
encoder = keras.Model(input_img, encoded) # This is our encoded (32-dimensional) input 
encoded_input = keras.Input(shape=(encoding_dim,))
# Retrieve the last layer of the autoencoder model 
decoder_layer = autoencoder.layers[-1] 
# Create the decoder model 
decoder = keras.Model(encoded_input, decoder_layer(encoded_input)) 
autoencoder.compile(optimizer='adam', loss='MSE') 
from keras.datasets import mnist 
import numpy as np 
import pandas as pd
# split a dataset into train and test sets
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
# reading csv files
gemo_data =  pd.read_csv("/content/Geom.csv")
#X_train, X_test, y_train, y_test = train_test_split(gemo_data)
x_train,x_test = train_test_split(gemo_data,test_size=0.2)
x_train = x_train.astype('float32') / 255. 
x_test = x_test.astype('float32') / 255. 

#x_train = x_train.values.reshape((len(x_train),np.prod(x_train.shape[1:])))
#x_test = x_test.values.reshape((len(x_test), np.prod(x_test.shape[1:]))) 
print(x_train.shape) 
print(x_test.shape)
autoencoder.fit(x_train, x_train, epochs=500, batch_size=256, shuffle=True, validation_data=(x_test, x_test)) 
# Encode and decode some digits # Note that we take them from the test set 
encoded_imgs = encoder.predict(x_test) 
decoded_imgs = decoder.predict(encoded_imgs) 
print(x_train.to_numpy())
tested=x_test.to_numpy()
import matplotlib.pyplot as plt 
n = 3# How many digits we will display 
k=0
#plt.figure(figsize=(20, 4)) 
for i in range(n): 
  # Display original 
  print(i)
  #ax = plt.subplot(2, n, i + 1) 
  while k <(122):
    plt.scatter(tested[i][k],tested[i][k+1])
    k+=2
  plt.show() 
  k=0

import matplotlib.pyplot as plt

n = 3  # How many digits we will display
#plt.figure(figsize=(20, 4))
k=0
for i in range(n):
    print(i)
    while k<(122) :
      plt.scatter(decoded_imgs[i][k],decoded_imgs[i][k+1])
      k+=2
    plt.show()  
    k=0