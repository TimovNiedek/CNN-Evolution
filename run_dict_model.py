import warnings
warnings.filterwarnings("ignore")

import keras
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.regularizers import l2
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Conv2D, MaxPooling2D, Flatten, AveragePooling2D, Input
from keras.initializers import he_normal
from keras.utils import plot_model
from keras import backend as K

from initialization import initializeNetwork

from utils import save_obj, load_obj

import numpy as np

import pandas as pd

from operator import itemgetter

from glob import glob

import matplotlib.pyplot as plt
plt.ion()

import os

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
    
def conv_block(layers, last_name, conv_dict, block_id, stride=False):
    block_filters = int(32 * (32/block_id))
    
    # Implement ConvBlocks:
    for block in sorted(conv_dict['convblocks'],key=itemgetter('id')):
        # Name of Block
        block_name = '{}_{}'.format(block_id, block['id'])
        #print('CREATING', block_name)
        
        # Determine Block Input Layer Name
        inputs = []
        for i in block['input']:
            if i == -1:
                inputs.append(last_name)
            else:
                inputs.append('{}_{}'.format(block_id, i))
        if len(inputs) > 1:
            for idx in range(len(inputs)):
                inputs[idx] = layers[inputs[idx]]
            layers[block_name] = keras.layers.add(inputs)
            inputs = [block_name]
        input_name = inputs[0]
        
        use_stride = False
        if input_name == last_name and stride==True:
            use_stride = True
        
        for s in block['layers']:
            if s == 'c':
                if use_stride:
                    layers[block_name] = Conv2D(block_filters, (3,3), strides=(2,2), padding='same', use_bias=False)(layers[input_name])
                else:
                    layers[block_name] = Conv2D(block_filters, (3,3), padding='same', use_bias=False)(layers[input_name])
            elif s == 'b':
                layers[block_name] = BatchNormalization()(layers[input_name])
            elif s == 'r':
                layers[block_name] = Activation('relu')(layers[input_name])
            elif s == 'd':
                layers[block_name] = Dropout(0.25)(layers[input_name])
            input_name = block_name
        
    # Implement Output
    #print('CREATING {}_output'.format(block_id))
    inputs = []
    for i in conv_dict['output']:
        if i == -1:
            inputs.append(last_name)
        else:
            inputs.append('{}_{}'.format(block_id, i))
    if len(inputs) > 1:
        for idx in range(len(inputs)):
            inputs[idx] = layers[inputs[idx]]
        output_name = '{}_output'.format(block_id)
        layers[output_name] = keras.layers.add(inputs)
    else:
        output_name = inputs[0]
    return layers, output_name

def run_network(network_config, compile_mode=False, verbose=1):
    #Clear Session (see: https://github.com/fchollet/keras/issues/2102)
    K.clear_session()
    
    # Load Data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    num_classes = int(max(y_train)) + 1
    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes) 
    
    # Initialize
    layers = dict()
    layers['input'] = Input(shape=(32, 32, 3))
    last_name = 'input'
    
    # Create Model
    layers, last_name = conv_block(layers, last_name, network_config['block32'], 32)
    
    if network_config['subsample32'] == True:
        layers['32_pooling'] = MaxPooling2D(pool_size=(2, 2))(layers[last_name])
        last_name = '32_pooling'
    
    layers, last_name = conv_block(layers, last_name, network_config['block16'], 16, stride=1-network_config['subsample32'])
    
    if network_config['subsample16'] == True:
        layers['16_pooling'] = MaxPooling2D(pool_size=(2, 2))(layers[last_name])
        last_name = '16_pooling'
    
    layers, last_name = conv_block(layers, last_name, network_config['block8'], 8, stride=1-network_config['subsample16'])
    
    if network_config['subsample8'] == True:
        layers['8_pooling'] = MaxPooling2D(pool_size=(2, 2))(layers[last_name])
        last_name = '8_pooling'
    
    if network_config['final'] == True:
        layers['flatten'] = Flatten()(layers[last_name])
        layers['dense'] = Dense(512, use_bias=False)(layers['flatten'])
        layers['bnorm'] = BatchNormalization()(layers['dense'])
        layers['dropout'] = Dropout(0.5)(layers['bnorm'])
        last_name = 'dropout'
    else:
        pool_size = (4,4)
        if network_config['subsample8'] == False:
            pool_size = (8,8)
        layers['avg_pooling'] = AveragePooling2D(pool_size=pool_size, strides=(1,1))(layers[last_name])
        layers['flatten'] = Flatten()(layers['avg_pooling'])
        last_name = 'flatten'
    
    # Default Last Layer
    layers['softmax'] = Dense(num_classes, use_bias=False)(layers[last_name])
    layers['softmax'] = BatchNormalization()(layers['softmax'])
    layers['softmax'] = Activation('softmax')(layers['softmax'])
    
    # Create & Compile
    model = Model(inputs=layers['input'], outputs=layers['softmax'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Create/Load Results.csv
    results = pd.DataFrame(columns=['Json_Config','Val_Loss','Val_Acc','Fig_Name'])
    if os.path.isfile('results.csv'):
        results = pd.read_csv('results.csv')
    
    #Print/Plot Model
    #model.summary()
    #plot_model(model, show_shapes=True, show_layer_names=False)
    json_config = model.to_json()
    
    #Check if already existing in results file:
    for _, row in results.iterrows():
        if row['Json_Config'] == json_config:
            val_loss = row['Val_Loss']
            val_acc = row['Val_Acc']
            if verbose==1:
                print('LOADED : val_loss: {0:.3f},'.format(val_loss),'val_acc: {0:.3f}'.format(val_acc))
            return val_loss, val_acc
    
    if compile_mode:
        val_loss = np.random.random()
        val_acc = np.random.random()
        if verbose==1:
            print('RANDOM : val_loss: {0:.3f},'.format(val_loss),'val_acc: {0:.3f}'.format(val_acc))
        return val_loss, val_acc
    else:
        plot = Plot_Callback()
        datagen = ImageDataGenerator(
                   width_shift_range=0.1,
                   height_shift_range=0.1,
                   #zoom_range=0.2,
                   horizontal_flip=True)
        datagen.fit(x_train)
        history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=128), steps_per_epoch=x_train.shape[0] // 128, epochs=20, validation_data=(x_test, y_test), callbacks=[plot], verbose=0)
        val_loss = min(history.history['val_loss'])
        val_acc = max(history.history['val_acc'])
        fig_name = '{}.png'.format(len(glob('Model_Figures/*.png')))
        plot_model(model, show_shapes=True, show_layer_names=False, to_file='Model_Figures/{}'.format(fig_name))
        results = results.append(pd.DataFrame([[json_config, val_loss, val_acc, fig_name]], columns=['Json_Config','Val_Loss','Val_Acc','Fig_Name']))
        results.to_csv('results.csv', index=False)
        if verbose==1:
            print('TRAINED: val_loss: {0:.3f},'.format(val_loss),'val_acc: {0:.3f}'.format(val_acc))
        return val_loss, val_acc

def get_results(network_config):
    # Create/Load Results.csv
    results = pd.DataFrame(columns=['Json_Config','Val_Loss','Val_Acc'])
    if os.path.isfile('results.csv'):
        results = pd.read_csv('results.csv')
    
    # Check already existing
#==============================================================================
#     for index, row in results.iterrows():
#         bla = row['Val_Loss']
#==============================================================================
    
        
    val_loss, val_acc, json_config = run_network(network_config, compile_mode=True)    
    results = results.append(pd.DataFrame([[json_config, val_loss, val_acc]], columns=['Json_Config','Val_Loss','Val_Acc']))
    results.to_csv('results.csv', index=False)
    return val_loss, val_acc
 
if __name__ == "__main__":
#==============================================================================
#     sample_network = {'block32':
#                     {'convblocks':
#                         [
#                             {'id': 0, 'layers': ['c','r','b'], 'input': [-1]}
#                         ],
#                     'output': [0]
#                     },
#                   'subsample32': True,
#                   'block16':
#                     {'convblocks':
#                         [
#                             {'id': 0, 'layers': ['c','b','r'], 'input': [-1]},
#                             {'id': 1, 'layers': ['r','d','b','c'], 'input': [-1]}
#                         ],
#                     'output': [0,1]
#                     },
#                   'subsample16':False,
#                   'block8':
#                     {'convblocks':
#                         [
#                             {'id': 0, 'layers': ['b','r','c'], 'input': [-1]},
#                             {'id': 1, 'layers': ['c','b','r'], 'input': [0]},
#                             {'id': 2, 'layers': ['b','c','r'], 'input': [0]}
#                         ],
#                     'output': [1,2]
#                     },
#                   'subsample8': True,
#                   'final': False
#                  }
#==============================================================================
    network = {'block8': {'output': [0, 2], 'convblocks': [{'id': 0, 'input': [-1], 'layers': ['b', 'r', 'c', 'd']}, {'id': 1, 'input': [-1], 'layers': ['c', 'b', 'r']}, {'id': 2, 'input': [1], 'layers': ['c', 'b', 'd', 'r']}]}, 'block16': {'output': [0, 1], 'convblocks': [{'id': 1, 'input': [0], 'layers': ['b', 'c', 'r']}, {'id': 0, 'input': [-1], 'layers': ['d', 'c', 'b', 'r']}]}, 'final': False, 'block32': {'output': [0, 1], 'convblocks': [{'id': 0, 'input': [-1], 'layers': ['d', 'b', 'r', 'c']}, {'id': 1, 'input': [-1], 'layers': ['r', 'b', 'c', 'd']}]}, 'subsample32': False, 'subsample8': True, 'subsample16': True}
    run_network(network, compile_mode=True)