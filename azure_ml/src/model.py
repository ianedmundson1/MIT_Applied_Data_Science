from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Input, GlobalAveragePooling2D, Resizing
from keras.optimizers import Adam
from keras_tuner import HyperModel
from keras.metrics import AUC
from keras.applications.vgg16 import VGG16
from keras.regularizers import l2
import keras
from mlflow.pyfunc.model import PythonModel
import logging
import numpy as np
logger = logging.getLogger(__name__)


class EmotionClassifierWrapper(PythonModel):
    """
    Wrapper for emotion classifier that returns emotion labels and confidence
    instead of raw probability arrays
    """
    
    def __init__(self, emotions=None):
        """
        Args:
            emotions: List of emotion labels in order matching model output classes
        """
        self.emotions = emotions or ['happy', 'sad', 'surprise', 'neutral']
    
    def load_context(self, context):
        """Load the Keras model with custom preprocessing"""
        import keras
        import sys
        from pathlib import Path
        
        # Import custom preprocessing layer
        try:
            # Add model path to system path
            model_code_path = Path(context.artifacts.get("code_path", ""))
            if model_code_path.exists():
                sys.path.insert(0, str(model_code_path.parent))
            
            from model import VGG16Preprocessing
            custom_objects = {'VGG16Preprocessing': VGG16Preprocessing}
            logger.info("✅ Loaded VGG16Preprocessing")
        except ImportError as e:
            logger.warning(f"⚠️ Could not import VGG16Preprocessing: {e}")
            custom_objects = {}
        
        # Load the Keras model
        model_path = context.artifacts["model"]
        logger.info(f"Loading model from: {model_path}")
        
        self.model = keras.models.load_model(model_path, custom_objects=custom_objects)
        logger.info("✅ Model loaded successfully")
    
    def predict(self, context, model_input):
        """
        Predict emotions and return structured response with labels
        
        Args:
            model_input: Input images as numpy array (batch_size, 48, 48, 3)
        
        Returns:
            List of dictionaries with emotion, confidence, and all probabilities
        """
        # Convert input to numpy if needed
        if hasattr(model_input, 'values'):
            model_input = model_input.values
        
        model_input = np.asarray(model_input, dtype=np.float32)
        
        # Get raw predictions
        predictions = self.model.predict(model_input, verbose=0)
        
        # Format results with emotion labels
        results = []
        for pred in predictions:
            predicted_class = int(np.argmax(pred))
            confidence = float(pred[predicted_class])
            
            result = {
                'emotion': self.emotions[predicted_class],
                'confidence': confidence,
                'probabilities': {
                    emotion: float(prob)
                    for emotion, prob in zip(self.emotions, pred)
                }
            }
            results.append(result)
        
        return results
    
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
        cnn_model.add(Dense(
            hp.Int('units', 128, 384, step=128), 
            activation='relu',
            kernel_regularizer=l2(hp.Float('l2_reg', 1e-4, 1e-2, sampling='LOG'))  # HERE
            ))
        cnn_model.add(Dropout(hp.Float('dropout', 0.5, 0.7, step=0.1)))

        # Adding the output layer with 10 neurons and activation functions as softmax
        cnn_model.add(Dense(4, activation='softmax'))
        opt = Adam(learning_rate=hp.Float('lr', 1e-5, 1e-3, sampling='LOG'))
        
        # Compile model
        cnn_model.compile(optimizer=opt,
                         loss='categorical_crossentropy',
                         metrics=['accuracy', AUC()])
        cnn_model.summary()
        return cnn_model