from keras.models import *
from keras.layers import *
from keras.optimizers import *

def AM_EEGNet(classes = 2, psd_frq = 90, psd_chans = 29, fc_conn = 406, fc_band = 6):
    """ 
    This model is the AM_EEGNet

    Inputs:
        
      classes    : int, number of classes to classify
      psd_frq    : int, number of frequency
      psd_chans  : int, number of channels
      fc_conn    : int, number of pairs of each channels
      fc_band    : int, number of frequency bands
      
    """
    
    # start the model
    Input_psd = Input(shape = (psd_frq, psd_chans, ))
    Input_fc = Input(shape = (fc_conn, fc_band, ))

    kernelSizePsd = (3)
    kernelSizefc = (2)

    # power spectrum density
    psd_block1 = Conv1D(filters = 128, kernel_size = kernelSizePsd)(Input_psd)
    psd_block1 = BatchNormalization()(psd_block1)
    psd_block1 = Activation('ReLU')(psd_block1)
    psd_block2 = Conv1D(filters = 256, kernel_size = kernelSizePsd)(psd_block1)
    psd_block2 = BatchNormalization()(psd_block2)
    psd_block2 = Activation('ReLU')(psd_block2)
    psd_block3 = Conv1D(filters = 128, kernel_size = kernelSizePsd)(psd_block2)
    psd_block3 = BatchNormalization()(psd_block3)
    psd_block3 = Activation('ReLU')(psd_block3)
    psd = Flatten()(psd_block3)
    psd = Dense(128, activation = 'ReLU')(psd)
    
    # functional connectivity
    fc_block1 = Conv1D(filters = 128, kernel_size = kernelSizefc)(Input_fc)
    fc_block1 = BatchNormalization()(fc_block1)
    fc_block1 = Activation('ReLU')(fc_block1)
    fc_block2 = Conv1D(filters = 256, kernel_size = kernelSizefc)(fc_block1)
    fc_block2 = BatchNormalization()(fc_block2)
    fc_block2 = Activation('ReLU')(fc_block2)
    fc_block3 = Conv1D(filters = 128, kernel_size = kernelSizefc)(fc_block2)
    fc_block3 = BatchNormalization()(fc_block3)
    fc_block3 = Activation('ReLU')(fc_block3)
    fc = Flatten()(fc_block3)
    fc = Dense(128, activation='ReLU')(fc)

    # concatenate
    target = concatenate([psd, fc])
    target = Dense(512, activation='ReLU')(target)
    output = Dense(classes, activation='softmax')(target)
    
    return Model(inputs = [Input_psd, Input_fc], outputs = output)