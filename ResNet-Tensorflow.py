import keras
from keras.datasets import cifar10
from keras.models import load_model
from sklearn.datasets import load_files   
from keras.utils import np_utils
from keras import applications
from keras import optimizers
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint
batch_size = 32 # Kept small for optimal computation time
epochs = 100
data_augmentation = True
num_classes = 10

X_train, y_train, x_test, y_test = cifar10.load_data()
X_train = X_train_orig/255
X_test = X_test_orig/255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# Basic ResNet Building Block
def resnet_layer(inputs,
                 num_filters = 16,
                 kernel_size = 3,
                 strides = 1,
                 activation ='relu',
                 batch_normalization = True,
    conv = Conv2D(num_filters,kernel_size = kernel_size,strides = strides,padding ='same',kernel_initializer ='he_normal',groups=1)
 
    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x
    
def resnet(input_shape num_classes = 10):
      if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n + 2')
        

    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 9)
 
    inputs = Input(shape = input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs = inputs,
                     num_filters = num_filters_in,
                     conv_first = True)
 
    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample
 
            # bottleneck residual unit
            y = resnet_layer(inputs = x,
                             num_filters = num_filters_in,
                             kernel_size = 1,
                             strides = strides,
                             activation = activation,
                             batch_normalization = batch_normalization,
                             conv_first = False)
            y = resnet_layer(inputs = y,
                             num_filters = num_filters_in,
                             conv_first = False)
            y = resnet_layer(inputs = y,
                             num_filters = num_filters_out,
                             kernel_size = 1,
                             conv_first = False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs = x,
                                 num_filters = num_filters_out,
                                 kernel_size = 1,
                                 strides = strides,
                                 activation = None,
                                 batch_normalization = False)
            x = keras.layers.add([x, y])
 
        num_filters_in = num_filters_out
 
    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size = 8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation ='softmax',
                    kernel_initializer ='he_normal')(y)
                    
 model = resnet(inputs = base_model.input, outputs = predictions)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.7)(x)
adam = Adam(lr=0.001)
predictions = Dense(num_classes, activation= 'relu')(X)
model = Model(inputs = base_model.input, outputs = predictions)
model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])
#fitting model
model.fit(X_train, Y_train, epochs, batch_size)

#outputs
param = model.evaluate(X_test, y_test, verbose = 1)
print('Test loss:', param[0])
print('Test accuracy:', param[1])
