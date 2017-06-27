import keras
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.regularizers import l2
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Conv2D, MaxPooling2D, Flatten, AveragePooling2D
from keras.initializers import he_normal
from keras.utils import plot_model
from keras import backend as K
#from utils import save_obj, load_obj

import numpy as np

import matplotlib.pyplot as plt
plt.ion()

import tensorflow as tf
# Set Memory allocation in tf/keras to Growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

# Plot callback
class Plot_Callback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        return
    
    def on_train_end(self, logs={}):
        plt.ioff()
        plt.show()
 
    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_accuracies.append(logs.get('val_acc'))
        
        ax = plt.gca()
        ax.clear()
        ax.plot(range(1,len(self.losses)+1),self.losses,color='r',label='train_loss')
        ax.plot(range(1,len(self.losses)+1),self.val_losses,color='b',label='val_loss')
        ax.set_ylim([0,1])
        ax.legend(loc=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        plt.draw()
        plt.pause(0.01)
        return

if __name__ == "__main__":
    # Params
    use_datagen = True
    batch_size = 128
    epochs = 20
    
    # Load Data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    num_classes = int(max(y_train)) + 1
    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), padding='same', use_bias=False, input_shape=x_train.shape[1:]))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), strides=(2,2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, (3, 3), strides=(2,2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
#==============================================================================
#     model.add(AveragePooling2D(pool_size=(8,8), strides=(1,1)))
#==============================================================================
    
    model.add(Flatten())
    
    model.add(Dense(512, use_bias=False))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(num_classes, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    
#==============================================================================
#     model = Sequential()
#     
#     model.add(Conv2D(32, (3, 3), padding='same', use_bias=False, input_shape=x_train.shape[1:]))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(Conv2D(32, (3, 3), padding='same', use_bias=False))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#     
#     model.add(Conv2D(64, (3, 3), padding='same', use_bias=False))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(Conv2D(64, (3, 3), padding='same', use_bias=False))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#     
#     model.add(Conv2D(128, (3, 3), padding='same', use_bias=False))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(Conv2D(128, (3, 3), padding='same', use_bias=False))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#     
#     model.add(AveragePooling2D(pool_size=(4,4), strides=(1,1)))
#     
#     model.add(Flatten())
#     
# #==============================================================================
# #     model.add(Dense(512, use_bias=False))
# #     model.add(Activation('relu'))
# #     model.add(BatchNormalization())
# #     model.add(Dropout(0.5))
# #==============================================================================
#     
#     model.add(Dense(num_classes, use_bias=False))
#     model.add(BatchNormalization())
#     model.add(Activation('softmax'))
#==============================================================================
    
    # Compile model with optimizer
    #opt = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #opt = keras.optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    #Print/Plot Model
    model.summary()
    plot_model(model, show_shapes=True, show_layer_names=False)
    
    # Fit
    plot = Plot_Callback()
    if use_datagen:
        datagen = ImageDataGenerator(
               width_shift_range=0.1,
               height_shift_range=0.1,
               #zoom_range=0.2,
               horizontal_flip=True)
        datagen.fit(x_train)
        model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=x_train.shape[0] // batch_size, epochs=epochs, validation_data=(x_test, y_test), callbacks=[plot])
    else:
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True, callbacks=[plot])