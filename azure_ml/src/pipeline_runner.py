"""
Modern Azure ML SDK v2 Pipeline Runner with Type Safety
"""
import os
from pathlib import Path
from typing import Optional
import logging

from azure.ai.ml import MLClient, Input, Output, command, dsl
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Job
from azure.identity import DefaultAzureCredential

from config import AzureMLConfig

logger = logging.getLogger(__name__)


class EmotionClassificationPipeline:
    """
    Modern Azure ML SDK v2 pipeline with type safety and decorators
    """
    
    def __init__(self, ml_client: MLClient):
        self.ml_client = ml_client
        
    @dsl.pipeline(
        name="emotion_classification_v2",
        display_name="Emotion Classification Pipeline V2",
        description="Modern pipeline using SDK v2 decorators and typing",
    )
    def create_pipeline(
        self,
        training_data: Input(type=AssetTypes.URI_FOLDER),
        model_name: str = "emotion-classifier-v2",
        max_epochs: int = 30,
        learning_rate: float = 0.0001,
        batch_size: int = 128,
    ):
        """
        Create the training pipeline using modern SDK v2 patterns
        """
        
        # Data preparation component
        data_prep_job = self._create_data_prep_component()(
            raw_data=training_data,
            batch_size=batch_size,
            image_size=48,
            apply_augmentation=True,
        )
        data_prep_job.compute = "cpu-cluster"
        
        # Model training component  
        training_job = self._create_training_component()(
            training_data=data_prep_job.outputs.processed_data,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            model_name=model_name,
        )
        training_job.compute = "gpu-cluster"
        
        # Model evaluation component
        evaluation_job = self._create_evaluation_component()(
            trained_model=training_job.outputs.trained_model,
            test_data=data_prep_job.outputs.processed_data,
        )
        evaluation_job.compute = "cpu-cluster"
        
        # Return pipeline outputs
        return {
            "trained_model": training_job.outputs.trained_model,
            "evaluation_results": evaluation_job.outputs.evaluation_results,
            "model_card": evaluation_job.outputs.model_card,
        }
    
    def _create_data_prep_component(self):
        """Create data preparation component with modern typing"""
        
        @command(
            name="data_prep_v2",
            display_name="Data Preparation V2",
            environment="azureml:tensorflow-env@latest",
            code="./src",
            instance_count=1,
        )
        def data_preparation(
            raw_data: Input(type=AssetTypes.URI_FOLDER),
            processed_data: Output(type=AssetTypes.URI_FOLDER),
            data_stats: Output(type=AssetTypes.URI_FILE),
            batch_size: int = 128,
            image_size: int = 48,
            apply_augmentation: bool = True,
        ):
            """Data preparation with modern Azure ML SDK v2 patterns"""
            
            import json
            import sys
            import subprocess
            
            # Install required packages
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "tensorflow>=2.13.0", "pillow", "mlflow"
            ])
            
            import tensorflow as tf
            import mlflow
            from pathlib import Path
            
            # Start MLflow tracking
            with mlflow.start_run():
                mlflow.log_param("batch_size", batch_size)
                mlflow.log_param("image_size", image_size)
                mlflow.log_param("apply_augmentation", apply_augmentation)
                
                # Simple data preparation logic
                emotions = ['happy', 'sad', 'surprise', 'neutral']
                
                # Load datasets with TensorFlow
                train_ds = tf.keras.utils.image_dataset_from_directory(
                    Path(raw_data) / "train",
                    validation_split=0.2,
                    subset="training",
                    seed=42,
                    image_size=(image_size, image_size),
                    batch_size=batch_size,
                    class_names=emotions
                )
                
                val_ds = tf.keras.utils.image_dataset_from_directory(
                    Path(raw_data) / "train", 
                    validation_split=0.2,
                    subset="validation",
                    seed=42,
                    image_size=(image_size, image_size),
                    batch_size=batch_size,
                    class_names=emotions
                )
                
                test_ds = tf.keras.utils.image_dataset_from_directory(
                    Path(raw_data) / "test",
                    seed=42,
                    image_size=(image_size, image_size),
                    batch_size=batch_size,
                    class_names=emotions
                )
                
                # Apply data augmentation if requested
                if apply_augmentation:
                    data_augmentation = tf.keras.Sequential([
                        tf.keras.layers.RandomFlip("horizontal"),
                        tf.keras.layers.RandomRotation(0.1),
                    ])
                    train_ds = train_ds.map(
                        lambda x, y: (data_augmentation(x, training=True), y),
                        num_parallel_calls=tf.data.AUTOTUNE
                    )
                
                # Optimize performance
                AUTOTUNE = tf.data.AUTOTUNE
                train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
                val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
                test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
                
                # Save datasets
                Path(processed_data).mkdir(parents=True, exist_ok=True)
                tf.data.Dataset.save(train_ds, str(Path(processed_data) / "train"))
                tf.data.Dataset.save(val_ds, str(Path(processed_data) / "validation"))
                tf.data.Dataset.save(test_ds, str(Path(processed_data) / "test"))
                
                # Generate stats
                stats = {
                    "train_batches": len(train_ds),
                    "val_batches": len(val_ds), 
                    "test_batches": len(test_ds),
                    "batch_size": batch_size,
                    "image_size": image_size,
                    "num_classes": len(emotions),
                    "class_names": emotions
                }
                
                # Log metrics
                mlflow.log_metric("train_batches", len(train_ds))
                mlflow.log_metric("val_batches", len(val_ds))
                mlflow.log_metric("test_batches", len(test_ds))
                
                # Save stats
                with open(data_stats, 'w') as f:
                    json.dump(stats, f, indent=2)
                
                print("Data preparation completed successfully")
        
        return data_preparation
    
    def _create_training_component(self):
        """Create model training component"""
        
        @command(
            name="model_training_v2",
            display_name="Model Training V2", 
            environment="azureml:tensorflow-gpu-env@latest",
            code="./src",
            instance_count=1,
        )
        def model_training(
            training_data: Input(type=AssetTypes.URI_FOLDER),
            trained_model: Output(type=AssetTypes.URI_FOLDER),
            training_metrics: Output(type=AssetTypes.URI_FILE),
            max_epochs: int = 30,
            learning_rate: float = 0.0001,
            model_name: str = "emotion-classifier",
        ):
            """Model training with hyperparameter optimization"""
            
            import json
            import sys
            import subprocess
            from pathlib import Path
            
            # Install packages
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "tensorflow>=2.13.0", "keras-tuner", "mlflow"
            ])
            
            import tensorflow as tf
            import mlflow
            import mlflow.tensorflow
            from tensorflow import keras
            
            # Set seeds for reproducibility
            tf.random.set_seed(42)
            
            with mlflow.start_run():
                # Log parameters
                mlflow.log_param("max_epochs", max_epochs)
                mlflow.log_param("learning_rate", learning_rate)
                mlflow.log_param("model_name", model_name)
                
                # Load datasets
                train_ds = tf.data.Dataset.load(str(Path(training_data) / "train"))
                val_ds = tf.data.Dataset.load(str(Path(training_data) / "validation"))
                
                # Create model architecture
                model = keras.Sequential([
                    keras.layers.Rescaling(1./255),
                    keras.layers.Conv2D(32, 3, activation='relu'),
                    keras.layers.MaxPooling2D(),
                    keras.layers.Conv2D(32, 3, activation='relu'),
                    keras.layers.MaxPooling2D(),
                    keras.layers.Conv2D(32, 3, activation='relu'),
                    keras.layers.MaxPooling2D(),
                    keras.layers.Flatten(),
                    keras.layers.Dense(128, activation='relu'),
                    keras.layers.Dropout(0.5),
                    keras.layers.Dense(4, activation='softmax')  # 4 emotions
                ])
                
                # Compile model
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                # Create callbacks
                callbacks = [
                    keras.callbacks.EarlyStopping(
                        patience=5, 
                        restore_best_weights=True
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        factor=0.2, 
                        patience=3
                    )
                ]
                
                # Train model
                history = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=max_epochs,
                    callbacks=callbacks,
                    verbose=1
                )
                
                # Save model
                Path(trained_model).mkdir(parents=True, exist_ok=True)
                model_path = Path(trained_model) / "model.keras" 
                model.save(str(model_path))
                
                # Log model to MLflow
                mlflow.tensorflow.log_model(
                    model=model,
                    artifact_path="model",
                    registered_model_name=model_name
                )
                
                # Calculate final metrics
                final_metrics = {
                    "final_train_accuracy": float(history.history['accuracy'][-1]),
                    "final_val_accuracy": float(history.history['val_accuracy'][-1]),
                    "final_train_loss": float(history.history['loss'][-1]),
                    "final_val_loss": float(history.history['val_loss'][-1]),
                    "epochs_trained": len(history.history['accuracy'])
                }
                
                # Log metrics
                for metric_name, value in final_metrics.items():
                    mlflow.log_metric(metric_name, value)
                
                # Save metrics
                with open(training_metrics, 'w') as f:
                    json.dump(final_metrics, f, indent=2)
                
                print(f"Training completed. Final accuracy: {final_metrics['final_val_accuracy']:.4f}")
        
        return model_training
    
    def _create_evaluation_component(self):
        """Create model evaluation component"""
        
        @command(
            name="model_evaluation_v2",
            display_name="Model Evaluation V2",
            environment="azureml:tensorflow-env@latest", 
            code="./src",
            instance_count=1,
        )
        def model_evaluation(
            trained_model: Input(type=AssetTypes.URI_FOLDER),
            test_data: Input(type=AssetTypes.URI_FOLDER),
            evaluation_results: Output(type=AssetTypes.URI_FILE),
            model_card: Output(type=AssetTypes.URI_FILE),
        ):
            """Comprehensive model evaluation"""
            
            import json
            import sys
            import subprocess
            from pathlib import Path
            
            # Install packages
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "tensorflow>=2.13.0", "scikit-learn", "mlflow"
            ])
            
            import tensorflow as tf
            import mlflow
            import numpy as np
            from sklearn.metrics import classification_report, accuracy_score
            
            with mlflow.start_run():
                # Load model
                model_path = Path(trained_model) / "model.keras"
                model = tf.keras.models.load_model(str(model_path))
                
                # Load test dataset
                test_ds = tf.data.Dataset.load(str(Path(test_data) / "test"))
                
                # Evaluate model
                test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
                
                # Get predictions for detailed metrics
                y_true = []
                y_pred = []
                
                for images, labels in test_ds:
                    predictions = model.predict(images, verbose=0)
                    y_pred.extend(np.argmax(predictions, axis=1))
                    y_true.extend(labels.numpy())
                
                # Calculate detailed metrics
                emotions = ['happy', 'sad', 'surprise', 'neutral']
                class_report = classification_report(
                    y_true, y_pred, 
                    target_names=emotions, 
                    output_dict=True
                )
                
                results = {
                    "test_accuracy": float(test_accuracy),
                    "test_loss": float(test_loss),
                    "classification_report": class_report
                }
                
                # Log metrics
                mlflow.log_metric("test_accuracy", test_accuracy)
                mlflow.log_metric("test_loss", test_loss)
                
                # Save evaluation results
                with open(evaluation_results, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                
                # Create model card
                model_card_content = f"""# Emotion Classification Model Card

## Performance Summary
- **Test Accuracy**: {test_accuracy:.4f}
- **Test Loss**: {test_loss:.4f}

## Model Details
- **Architecture**: CNN with TensorFlow/Keras
- **Classes**: {', '.join(emotions)}
- **Input Size**: 48x48 RGB images

## Classification Report
{classification_report(y_true, y_pred, target_names=emotions)}
"""
                
                with open(model_card, 'w') as f:
                    f.write(model_card_content)
                
                print(f"Evaluation completed. Test accuracy: {test_accuracy:.4f}")
        
        return model_evaluation
    
    def submit_pipeline(
        self, 
        training_data_path: str,
        experiment_name: str = "emotion-classification-v2",
        **pipeline_params
    ) -> Job:
        """
        Submit the pipeline to Azure ML with proper typing
        """
        
        # Create pipeline
        pipeline_job = self.create_pipeline(
            training_data=Input(type=AssetTypes.URI_FOLDER, path=training_data_path),
            **pipeline_params
        )
        
        # Set pipeline properties
        pipeline_job.experiment_name = experiment_name
        pipeline_job.compute = "cpu-cluster"  # Default compute
        
        # Submit pipeline
        pipeline_job = self.ml_client.jobs.create_or_update(
            pipeline_job, 
            experiment_name=experiment_name
        )
        
        logger.info(f"Pipeline submitted: {pipeline_job.name}")
        logger.info(f"Studio URL: {pipeline_job.studio_url}")
        
        return pipeline_job


def main():
    """Main function to run the modern Azure ML pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run emotion classification pipeline")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--experiment_name", type=str, default="emotion-classification-v2")
    parser.add_argument("--model_name", type=str, default="emotion-classifier-v2")
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    
    args = parser.parse_args()
    
    # Initialize Azure ML client
    azure_config = AzureMLConfig()
    credential = DefaultAzureCredential()
    
    ml_client = MLClient(
        credential=credential,
        subscription_id=azure_config.subscription_id,
        resource_group_name=azure_config.resource_group,
        workspace_name=azure_config.workspace_name
    )
    
    # Create and submit pipeline
    pipeline = EmotionClassificationPipeline(ml_client)
    
    job = pipeline.submit_pipeline(
        training_data_path=args.data_path,
        experiment_name=args.experiment_name,
        model_name=args.model_name,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
    )
    
    print(f"Pipeline submitted successfully!")
    print(f"Job name: {job.name}")
    print(f"Studio URL: {job.studio_url}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()