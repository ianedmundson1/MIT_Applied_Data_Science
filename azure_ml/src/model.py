import tensorflow as tf

from keras.models import Sequential
# Importing all the different layers and optimizers
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation, LeakyReLU, Input,GlobalAveragePooling2D, Resizing
from keras.optimizers import Adam,SGD
from keras.preprocessing import image_dataset_from_directory
from keras.layers import Rescaling, RandomFlip, RandomRotation
from keras.callbacks import  EarlyStopping, ReduceLROnPlateau
from  keras_tuner import Hyperband,HyperModel

from keras.applications import VGG16

from keras.regularizers import l2

VGG16_model = VGG16(weights='imagenet', include_top=False)
VGG16_model.summary()
for layer in VGG16_model.layers:
    layer.trainable = False

# Unfreeze the top layers for fine-tuning
for layer in VGG16_model.layers[-3:]:
    layer.trainable = True


class cnn_model_color_VGG16_model(HyperModel):
    def build(self,hp):
        # Intializing a sequential model

        cnn_model = Sequential()
        # Adding the input layer
        cnn_model.add(Input(shape=(48, 48,3)))
        # Adding a random flip layer
        cnn_model.add(RandomFlip('horizontal'))
        # Adding a random rotation layer
        cnn_model.add(RandomRotation(0.2))
        
        # Add the Rescaling layer at the beginning
        cnn_model.add(Rescaling(1./255))
        
        #resize the image to 224x224
        cnn_model.add(Resizing(224, 224))
        # Add the VGG16 model
        cnn_model.add(VGG16_model)
        # Add convolutional layers
        
        cnn_model.add(GlobalAveragePooling2D())  

        # Adding a sequential layer with 300 neurons
        cnn_model.add(Dense(hp.Int('layer1',min_value=200,max_value=500,step=25), activation='relu', kernel_regularizer=l2(0.01)))
        cnn_model.add(BatchNormalization())
        # Adding a sequential layer with dropout of 0.2
        cnn_model.add(Dropout(0.2))
        # Adding a sequential layer with 200 neurons
        cnn_model.add(Dense(hp.Int('layer2',min_value=32,max_value=300,step=25), activation='relu', kernel_regularizer=l2(0.01)))
        cnn_model.add(BatchNormalization())
        cnn_model.add(LeakyReLU(alpha=0.1))
        cnn_model.add(Dropout(0.2))
        cnn_model.add(Dense(hp.Int('layer3',min_value=32,max_value=300,step=25), activation='relu', kernel_regularizer=l2(0.01)))
        cnn_model.add(BatchNormalization())
                    
        cnn_model.add(Dropout(0.2))
        # Adding a batch normalization layer
        
        # Adding the output layer with 10 neurons and activation functions as softmax since this is a multi-class classification problem
        cnn_model.add(Dense(4, activation='softmax'))


        
        # Using Adam Optimizer
        opt = Adam(learning_rate=0.001)

        # Compile model
        cnn_model.compile(optimizer=opt,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        
        return cnn_model