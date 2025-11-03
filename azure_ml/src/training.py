"""
Azure ML training pipeline with MLOps best practices
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras_tuner import Hyperband
import numpy as np

from config import TrainingConfig, AzureMLConfig, get_ml_client
from model import cnn_model_color_VGG16_model
from preprocessing import prepare_datasets

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed_value: int = 42):
    """Set seeds for reproducibility across all random number generators"""
    # Set environment variable
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    # Clear Keras backend session
    tf.keras.backend.clear_session()
    
    # Set seeds
    import random
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    
    logger.info(f"Random seed set to {seed_value}")


def log_system_info():
    """Log system and environment information"""
    mlflow.log_param("tensorflow_version", tf.__version__)
    mlflow.log_param("python_version", sys.version)
    
    # Log GPU information
    gpus = tf.config.list_physical_devices('GPU')
    mlflow.log_param("gpu_available", len(gpus) > 0)
    mlflow.log_param("gpu_count", len(gpus))
    
    if gpus:
        for i, gpu in enumerate(gpus):
            mlflow.log_param(f"gpu_{i}_name", gpu.name)


def create_callbacks(config: TrainingConfig, model_dir: Path) -> list:
    """Create training callbacks"""
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=config.patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            verbose=1,
            min_delta=0.0001
        ),
        ModelCheckpoint(
            filepath=str(model_dir / "best_model_checkpoint.keras"),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    return callbacks


def run_hyperparameter_tuning(
    hypermodel, 
    train_dataset, 
    val_dataset, 
    config: TrainingConfig,
    output_dir: Path
) -> keras.Model:
    """Run hyperparameter tuning using Keras Tuner"""
    
    logger.info("Starting hyperparameter tuning...")
    
    tuner = Hyperband(
        hypermodel,
        objective='val_accuracy',
        max_epochs=config.max_epochs,
        factor=config.factor,
        seed=config.seed,
        directory=str(output_dir / 'tuning'),
        project_name=config.experiment_name,
        overwrite=True
    )
    
    # Log tuning parameters
    mlflow.log_param("max_epochs", config.max_epochs)
    mlflow.log_param("tuning_factor", config.factor)
    mlflow.log_param("tuning_objective", "val_accuracy")
    
    # Search for best hyperparameters
    tuner.search(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.max_epochs,
        verbose=1
    )
    
    # Get best model
    best_model = tuner.get_best_models(num_models=1)[0]
    
    # Log best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    for param in best_hps.space:
        param_value = best_hps.get(param.name)
        mlflow.log_param(f"best_{param.name}", param_value)
        logger.info(f"Best {param.name}: {param_value}")
    
    return best_model


def train_model(
    model: keras.Model,
    train_dataset,
    val_dataset,
    config: TrainingConfig,
    output_dir: Path
) -> keras.callbacks.History:
    """Train the model with proper logging"""
    
    logger.info("Starting model training...")
    
    # Create callbacks
    callbacks = create_callbacks(config, output_dir)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    # Log model parameters
    mlflow.log_param("learning_rate", config.learning_rate)
    mlflow.log_param("batch_size", config.batch_size)
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("loss_function", "categorical_crossentropy")
    
    # Train model with MLflow autologging
    mlflow.tensorflow.autolog(log_models=False)  # We'll log manually for better control
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.max_epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def evaluate_and_log_model(
    model: keras.Model,
    test_dataset,
    config: TrainingConfig,
    output_dir: Path
) -> Dict[str, float]:
    """Evaluate model and log metrics"""
    
    logger.info("Evaluating model...")
    
    # Evaluate on test set
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(
        test_dataset, 
        verbose=1
    )
    
    # Calculate F1 score
    test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)
    
    metrics = {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1
    }
    
    # Log metrics
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)
        logger.info(f"{metric_name}: {metric_value:.4f}")
    
    return metrics


def save_and_register_model(
    model: keras.Model,
    config: TrainingConfig,
    output_dir: Path,
    ml_client=None
):
    """Save model locally and register in Azure ML"""
    
    model_path = output_dir / "final_model.keras"
    model.save(str(model_path))
    logger.info(f"Model saved to {model_path}")
    
    # Log model to MLflow
    mlflow.tensorflow.log_model(
        model=model,
        artifact_path="model",
        registered_model_name=config.model_name
    )
    
    # Log model files as artifacts
    mlflow.log_artifact(str(model_path))
    
    logger.info("Model registered successfully")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train emotion classification model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--config_file", type=str, help="Path to config JSON file")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = TrainingConfig()
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # Set random seed for reproducibility
    set_seed(config.seed)
    
    # Initialize Azure ML client
    azure_config = AzureMLConfig()
    ml_client = None
    try:
        ml_client = get_ml_client(azure_config)
        logger.info("Connected to Azure ML workspace")
    except Exception as e:
        logger.warning(f"Could not connect to Azure ML: {e}")
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"emotion-training-{config.seed}") as run:
        
        # Log system info
        log_system_info()
        
        # Log configuration parameters
        mlflow.log_param("seed", config.seed)
        mlflow.log_param("image_size", config.image_size)
        mlflow.log_param("num_classes", config.num_classes)
        mlflow.log_param("emotions", str(config.emotions))
        
        # Prepare datasets
        logger.info("Preparing datasets...")
        train_dataset, val_dataset, test_dataset = prepare_datasets(
            data_path=args.data_path,
            config=config
        )
        
        # Create hypermodel
        hypermodel = cnn_model_color_VGG16_model()
        
        # Run hyperparameter tuning
        best_model = run_hyperparameter_tuning(
            hypermodel, train_dataset, val_dataset, config, output_dir
        )
        
        # Train final model
        history = train_model(
            best_model, train_dataset, val_dataset, config, output_dir
        )
        
        # Evaluate model
        metrics = evaluate_and_log_model(
            best_model, test_dataset, config, output_dir
        )
        
        # Save and register model
        save_and_register_model(
            best_model, config, output_dir, ml_client
        )
        
        logger.info(f"Training completed. Run ID: {run.info.run_id}")
        return best_model, metrics


if __name__ == "__main__":
    main()