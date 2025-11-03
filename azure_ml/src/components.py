"""
Azure ML SDK v2 Components with modern decorators and typing
"""
import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import logging

from azure.ai.ml import command, Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline
import mlflow

logger = logging.getLogger(__name__)


@command(
    name="data_preparation",
    display_name="Data Preparation Component", 
    description="Prepare and validate training data for emotion classification",
    environment="azureml:tensorflow-gpu-env@latest",
    code="./src",
    is_deterministic=True,
)
def data_preparation_component(
    # Inputs
    raw_data: Input(type=AssetTypes.URI_FOLDER, description="Raw training data folder"),
    config_file: Input(type=AssetTypes.URI_FILE, description="Training configuration JSON", optional=True),
    
    # Outputs  
    processed_data: Output(type=AssetTypes.URI_FOLDER, description="Processed training data"),
    data_stats: Output(type=AssetTypes.URI_FILE, description="Dataset statistics JSON"),
    
    # Parameters
    image_size: int = 48,
    batch_size: int = 128,
    apply_augmentation: bool = True,
    validation_split: float = 0.2,
) -> None:
    """
    Data preparation component using Azure ML SDK v2 decorators
    """
    import json
    import tensorflow as tf
    from preprocessing import prepare_datasets, get_dataset_info
    from config import TrainingConfig
    
    # Load configuration
    config = TrainingConfig()
    config.image_size = image_size
    config.batch_size = batch_size
    
    if config_file:
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # Start MLflow tracking for the component
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("image_size", image_size)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("apply_augmentation", apply_augmentation)
        
        # Prepare datasets
        train_ds, val_ds, test_ds = prepare_datasets(
            data_path=raw_data,
            config=config,
            apply_augmentation=apply_augmentation
        )
        
        # Save processed datasets
        train_path = Path(processed_data) / "train"
        val_path = Path(processed_data) / "validation" 
        test_path = Path(processed_data) / "test"
        
        # Create output directories
        for path in [train_path, val_path, test_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Save datasets in TensorFlow format
        tf.data.Dataset.save(train_ds, str(train_path))
        tf.data.Dataset.save(val_ds, str(val_path))
        tf.data.Dataset.save(test_ds, str(test_path))
        
        # Generate dataset statistics
        stats = {
            "train": get_dataset_info(train_ds, "train"),
            "validation": get_dataset_info(val_ds, "validation"),
            "test": get_dataset_info(test_ds, "test")
        }
        
        # Log dataset metrics
        for split, info in stats.items():
            mlflow.log_metric(f"{split}_samples", info.get("total_samples", 0))
            mlflow.log_metric(f"{split}_batches", info.get("num_batches", 0))
        
        # Save statistics
        with open(data_stats, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info("Data preparation completed successfully")


@command(
    name="model_training",
    display_name="Model Training Component",
    description="Train emotion classification model with hyperparameter tuning",
    environment="azureml:tensorflow-gpu-env@latest", 
    code="./src",
    is_deterministic=False,  # Due to hyperparameter tuning randomness
)
def model_training_component(
    # Inputs
    training_data: Input(type=AssetTypes.URI_FOLDER, description="Processed training data"),
    config_file: Input(type=AssetTypes.URI_FILE, description="Training configuration", optional=True),
    
    # Outputs
    trained_model: Output(type=AssetTypes.URI_FOLDER, description="Trained model artifacts"),
    training_metrics: Output(type=AssetTypes.URI_FILE, description="Training metrics JSON"),
    model_summary: Output(type=AssetTypes.URI_FILE, description="Model architecture summary"),
    
    # Parameters
    max_epochs: int = 30,
    learning_rate: float = 0.0001,
    patience: int = 12,
    max_trials: int = 30,
    seed: int = 42,
    model_name: str = "emotion-classifier",
) -> None:
    """
    Model training component with MLflow tracking
    """
    import json
    import tensorflow as tf
    from tensorflow import keras
    from keras_tuner import Hyperband
    import mlflow.tensorflow
    
    from config import TrainingConfig
    from model import cnn_model_color_VGG16_model
    from training import set_seed, log_system_info, create_callbacks
    
    # Set random seed
    set_seed(seed)
    
    # Load configuration
    config = TrainingConfig()
    config.max_epochs = max_epochs
    config.learning_rate = learning_rate
    config.patience = patience
    config.max_trials = max_trials
    config.seed = seed
    config.model_name = model_name
    
    if config_file:
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # Start MLflow run
    with mlflow.start_run():
        # Log system and configuration info
        log_system_info()
        mlflow.log_params({
            "max_epochs": max_epochs,
            "learning_rate": learning_rate,
            "patience": patience,
            "max_trials": max_trials,
            "seed": seed,
            "model_name": model_name
        })
        
        # Load datasets
        train_path = Path(training_data) / "train"
        val_path = Path(training_data) / "validation"
        
        train_ds = tf.data.Dataset.load(str(train_path))
        val_ds = tf.data.Dataset.load(str(val_path))
        
        # Create hypermodel
        hypermodel = cnn_model_color_VGG16_model()
        
        # Hyperparameter tuning
        tuner = Hyperband(
            hypermodel,
            objective='val_accuracy',
            max_epochs=max_epochs,
            factor=3,
            seed=seed,
            directory=str(Path(trained_model) / 'tuning'),
            project_name=f'{model_name}_tuning',
            overwrite=True
        )
        
        # Search for best hyperparameters
        tuner.search(
            train_ds,
            validation_data=val_ds,
            epochs=max_epochs,
            verbose=1
        )
        
        # Get best model
        best_model = tuner.get_best_models(num_models=1)[0]
        
        # Log best hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_hp_dict = {}
        for param in best_hps.space:
            param_value = best_hps.get(param.name)
            best_hp_dict[f"best_{param.name}"] = param_value
            mlflow.log_param(f"best_{param.name}", param_value)
        
        # Create callbacks
        output_path = Path(trained_model)
        output_path.mkdir(parents=True, exist_ok=True)
        callbacks = create_callbacks(config, output_path)
        
        # Compile and train final model
        best_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Train model
        history = best_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=max_epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save model
        model_path = output_path / "final_model.keras"
        best_model.save(str(model_path))
        
        # Register model with MLflow
        mlflow.tensorflow.log_model(
            model=best_model,
            artifact_path="model",
            registered_model_name=model_name
        )
        
        # Save training metrics
        final_metrics = {
            "final_train_accuracy": float(history.history['accuracy'][-1]),
            "final_val_accuracy": float(history.history['val_accuracy'][-1]),
            "final_train_loss": float(history.history['loss'][-1]),
            "final_val_loss": float(history.history['val_loss'][-1]),
            "epochs_trained": len(history.history['accuracy']),
            "best_hyperparameters": best_hp_dict
        }
        
        # Log final metrics
        for metric_name, value in final_metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(metric_name, value)
        
        # Save metrics to file
        with open(training_metrics, 'w') as f:
            json.dump(final_metrics, f, indent=2, default=str)
        
        # Save model summary
        with open(model_summary, 'w') as f:
            best_model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        logger.info(f"Training completed. Final validation accuracy: {final_metrics['final_val_accuracy']:.4f}")


@command(
    name="model_evaluation", 
    display_name="Model Evaluation Component",
    description="Comprehensive model evaluation with metrics and visualizations",
    environment="azureml:tensorflow-gpu-env@latest",
    code="./src",
    is_deterministic=True,
)
def model_evaluation_component(
    # Inputs
    trained_model: Input(type=AssetTypes.URI_FOLDER, description="Trained model artifacts"),
    test_data: Input(type=AssetTypes.URI_FOLDER, description="Test dataset"),
    config_file: Input(type=AssetTypes.URI_FILE, description="Configuration file", optional=True),
    
    # Outputs
    evaluation_results: Output(type=AssetTypes.URI_FILE, description="Evaluation metrics JSON"),
    evaluation_plots: Output(type=AssetTypes.URI_FOLDER, description="Evaluation plots and charts"),
    model_card: Output(type=AssetTypes.URI_FILE, description="Model card with performance summary"),
    
    # Parameters
    emotions: list = None,
) -> None:
    """
    Model evaluation component with comprehensive metrics
    """
    import json
    import tensorflow as tf
    from pathlib import Path
    import mlflow
    
    from config import TrainingConfig
    from evaluation import load_model, evaluate_model_detailed, create_evaluation_plots
    
    if emotions is None:
        emotions = ['happy', 'sad', 'surprise', 'neutral']
    
    # Load configuration
    config = TrainingConfig()
    config.emotions = emotions
    
    if config_file:
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    with mlflow.start_run():
        # Load model
        model = load_model(trained_model)
        
        # Load test dataset
        test_path = Path(test_data) / "test"
        test_ds = tf.data.Dataset.load(str(test_path))
        
        # Evaluate model
        results, y_true, y_pred, y_pred_proba = evaluate_model_detailed(
            model, test_ds, config
        )
        
        # Create plots directory
        plots_dir = Path(evaluation_plots)
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate evaluation plots
        create_evaluation_plots(
            y_true, y_pred, y_pred_proba, emotions, plots_dir
        )
        
        # Log metrics to MLflow
        for metric_name, value in results['overall_metrics'].items():
            mlflow.log_metric(f"eval_{metric_name}", value)
        
        # Log per-class metrics
        for emotion, metrics in results['per_class_metrics'].items():
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"eval_{emotion}_{metric_name}", value)
        
        # Save evaluation results
        with open(evaluation_results, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create model card
        model_card_content = f"""# Emotion Classification Model Card

## Model Overview
- **Model Name**: {config.model_name}
- **Model Type**: CNN with VGG16 backbone
- **Task**: Multi-class emotion classification
- **Classes**: {', '.join(emotions)}

## Performance Metrics
- **Overall Accuracy**: {results['overall_metrics']['accuracy']:.4f}
- **Weighted F1-Score**: {results['overall_metrics']['f1_weighted']:.4f}
- **Weighted Precision**: {results['overall_metrics']['precision_weighted']:.4f}
- **Weighted Recall**: {results['overall_metrics']['recall_weighted']:.4f}

## Per-Class Performance
"""
        for emotion, metrics in results['per_class_metrics'].items():
            model_card_content += f"\n### {emotion.title()}\n"
            model_card_content += f"- Precision: {metrics['precision']:.4f}\n"
            model_card_content += f"- Recall: {metrics['recall']:.4f}\n"
            model_card_content += f"- F1-Score: {metrics['f1_score']:.4f}\n"
            model_card_content += f"- Support: {metrics['support']}\n"
        
        with open(model_card, 'w') as f:
            f.write(model_card_content)
        
        logger.info(f"Evaluation completed. Overall accuracy: {results['overall_metrics']['accuracy']:.4f}")


@pipeline(
    name="emotion_classification_training_pipeline",
    display_name="Emotion Classification Training Pipeline",
    description="End-to-end pipeline for training emotion classification model",
    tags={"project": "facial-emotion-detection", "framework": "tensorflow"},
)
def emotion_classification_pipeline(
    # Pipeline inputs
    training_data: Input(type=AssetTypes.URI_FOLDER),
    config_file: Input(type=AssetTypes.URI_FILE, optional=True),
    
    # Pipeline parameters
    max_epochs: int = 30,
    learning_rate: float = 0.0001,
    batch_size: int = 128,
    model_name: str = "emotion-classifier",
) -> Dict[str, Output]:
    """
    Complete emotion classification training pipeline using Azure ML SDK v2 decorators
    """
    
    # Data preparation step
    data_prep = data_preparation_component(
        raw_data=training_data,
        config_file=config_file,
        batch_size=batch_size,
    )
    
    # Model training step
    training = model_training_component(
        training_data=data_prep.outputs.processed_data,
        config_file=config_file,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        model_name=model_name,
    )
    
    # Model evaluation step
    evaluation = model_evaluation_component(
        trained_model=training.outputs.trained_model,
        test_data=data_prep.outputs.processed_data,
        config_file=config_file,
    )
    
    return {
        "trained_model": training.outputs.trained_model,
        "training_metrics": training.outputs.training_metrics,
        "evaluation_results": evaluation.outputs.evaluation_results,
        "evaluation_plots": evaluation.outputs.evaluation_plots,
        "model_card": evaluation.outputs.model_card,
    }