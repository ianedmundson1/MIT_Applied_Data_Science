from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Input, GlobalAveragePooling2D, Resizing
from keras.optimizers import Adam
from keras_tuner import HyperModel
from keras.metrics import AUC
from keras.applications.vgg16 import VGG16
from keras.regularizers import l2
import keras

# Move this OUTSIDE the HyperModel class - make it top-level
@keras.saving.register_keras_serializable()
class VGG16Preprocessing(keras.layers.Layer):
    """Custom preprocessing layer for VGG16 - converts RGB to BGR and zero-centers"""
    def call(self, inputs):
        # VGG16 expects BGR format with mean subtraction
        # RGB to BGR conversion
        x = inputs[..., ::-1]
        # Zero-center by ImageNet mean in BGR order
        x = x - [103.939, 116.779, 123.68]
        return x
    
    def get_config(self):
        return super().get_config()

# Load VGG16 once at module level
VGG16_model = VGG16(weights='imagenet', include_top=False)
for layer in VGG16_model.layers:
    layer.trainable = False
# Unfreeze the top layers for fine-tuning
for layer in VGG16_model.layers[-4:]:
    layer.trainable = True

class cnn_model_color_VGG16_model(HyperModel):
    def build(self, hp):
        # Initializing a sequential model
        cnn_model = Sequential()
        # Adding the input layer
        cnn_model.add(Input(shape=(48, 48, 3)))
        # Resize the image to 224x224
        cnn_model.add(Resizing(224, 224, interpolation='bilinear'))
        # Now reference the top-level class
        cnn_model.add(VGG16Preprocessing())
        # Add the VGG16 model
        cnn_model.add(VGG16_model)
        # Add convolutional layers
        cnn_model.add(GlobalAveragePooling2D())  
        # Adding a sequential layer with 300 neurons
        cnn_model.add(Dense(hp.Int('layer1', min_value=150, max_value=300, step=25), 
                           kernel_regularizer=l2(hp.Float('l2_reg', 1e-4, 1e-2, sampling='LOG'))))
        cnn_model.add(BatchNormalization())
        cnn_model.add(Activation('relu'))
        # Adding a sequential layer with dropout of 0.2
        cnn_model.add(Dropout(hp.Float('dropout1', 0.2, 0.5, step=0.1)))
        # Adding a sequential layer with 200 neurons
        cnn_model.add(Dense(hp.Int('layer2', min_value=15, max_value=60, step=25),
                           kernel_regularizer=l2(hp.Float('l2_reg', 1e-4, 1e-2, sampling='LOG'))))
        cnn_model.add(BatchNormalization())
        cnn_model.add(Activation('relu'))
        cnn_model.add(Dropout(hp.Float('dropout2', 0.2, 0.5, step=0.1)))
        # Adding the output layer with 10 neurons and activation functions as softmax
        cnn_model.add(Dense(4, activation='softmax'))
        # Using Adam Optimizer
        opt = Adam(learning_rate=hp.Float('learning_rate', min_value=3e-5, max_value=1e-4, sampling='LOG'))
        # Compile model
        cnn_model.compile(optimizer=opt,
                         loss='categorical_crossentropy',
                         metrics=['accuracy', AUC()])
        cnn_model.summary()
        return cnn_model