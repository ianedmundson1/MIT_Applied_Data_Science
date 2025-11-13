"""
Local Pipeline Runner - Execute Azure ML components locally for development and testing
"""

import os
import sys

# ✅ Suppress TensorFlow logging BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF C++ logs
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
# os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Optional: Disable GPU if not needed

# Suppress TensorFlow warnings
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import os
import sys
import json
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# ✅ Set TensorFlow logging after import (belt and suspenders approach)
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)
logger = logging.getLogger(__name__)


@dataclass
class LocalAsset:
    """Local representation of Azure ML assets"""
    path: Union[str, Path]
    type: str
    description: str = ""
    
    def __post_init__(self):
        self.path = Path(self.path)

class LocalMLflowTracker:
    """Local MLflow tracking context manager"""
    
    def __init__(self, experiment_name: str = "local-emotion-classification"):
        self.experiment_name = experiment_name
        self.run_id = None
        
    def __enter__(self):
        try:
            import mlflow
            import mlflow.tensorflow
            
            # End any active run before starting a new one
            if mlflow.active_run():
                logger.warning(f"Found active MLflow run: {mlflow.active_run().info.run_id}. Ending it before starting new run.")
                mlflow.end_run()
            
            # Set tracking URI to use local file store
            mlflow.set_tracking_uri("file:///home/ian/Desktop/github/MIT_Applied_Data_Science/azure_ml/mlruns")
            
            # Set the experiment name
            mlflow.set_experiment(self.experiment_name)

            # Explicitly start a new run
            run = mlflow.start_run()
            self.run_id = run.info.run_id
            logger.info(f"Started MLflow run: {self.run_id} for experiment: {self.experiment_name}")

            # Enable autologging for TensorFlow
            mlflow.tensorflow.autolog(
                every_n_iter=1,
                log_models=True,
                disable=False,
                exclusive=False,
                disable_for_unsupported_versions=False,
                silent=False
            )
            
            logger.info(f"MLflow autologging enabled for experiment: {self.experiment_name}")
        except ImportError:
            logger.warning("MLflow not available, skipping tracking")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            import mlflow
            # Ensure the run is ended even if there's an exception
            if mlflow.active_run():
                mlflow.end_run()
                if self.run_id:
                    logger.info(f"Ended MLflow run: {self.run_id}")
        except Exception as e:
            logger.error(f"Failed to end MLflow run: {str(e)}")

class LocalPipelineRunner:
    """
    Local runner for Azure ML pipeline components
    Executes components locally for development and testing
    """
    
    def __init__(self, output_dir: str = "./local_outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{self.run_id}"
        self.run_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.run_dir / "pipeline.log"),
                logging.StreamHandler()
            ]
        )
        
        
    def create_local_asset(
        self, 
        name: str, 
        asset_type: str, 
        base_path: Optional[Path] = None
    ) -> LocalAsset:
        """Create a local asset (file or folder)"""
        
        if base_path is None:
            base_path = self.run_dir
            
        if asset_type == "URI_FOLDER":
            asset_path = base_path / name
            asset_path.mkdir(parents=True, exist_ok=True)
        else:  # URI_FILE
            asset_path = base_path / name
            asset_path.parent.mkdir(parents=True, exist_ok=True)
            
        return LocalAsset(path=asset_path, type=asset_type)
    
    def run_data_preparation_local(
        self,
        raw_data_path: str,
        config_file_path: Optional[str] = None,
        image_size: int = 48,
        batch_size: int = 128,
        apply_augmentation: bool = True
    ) -> Dict[str, LocalAsset]:
        """Run data preparation component locally"""
        
        logger.info("🔄 Starting local data preparation...")
        
        # Create output assets
        processed_data = self.create_local_asset("processed_data", "URI_FOLDER")
        data_stats = self.create_local_asset("data_stats.json", "URI_FILE")
        
        with LocalMLflowTracker("local-data-prep"):
            try:
                # Import required modules
                from config import TrainingConfig
                from preprocessing import prepare_datasets, get_dataset_info
                import mlflow
                
                # Setup configuration
                config = TrainingConfig()
                config.image_size = image_size
                config.batch_size = batch_size
                
                if config_file_path and Path(config_file_path).exists():
                    with open(config_file_path, 'r') as f:
                        config_dict = json.load(f)
                        for key, value in config_dict.items():
                            if hasattr(config, key):
                                setattr(config, key, value)
                
                # Log parameters
                mlflow.log_param("image_size", image_size)
                mlflow.log_param("batch_size", batch_size)
                mlflow.log_param("apply_augmentation", apply_augmentation)
                mlflow.log_param("raw_data_path", raw_data_path)
                
                # Prepare datasets
                train_ds, val_ds, test_ds = prepare_datasets(
                    data_path=raw_data_path,
                    config=config,
                    apply_augmentation=apply_augmentation
                )
                
                # Save datasets locally
                import tensorflow as tf
                
                train_path = processed_data.path / "train"
                val_path = processed_data.path / "validation"
                test_path = processed_data.path / "test"
                
                for path in [train_path, val_path, test_path]:
                    path.mkdir(parents=True, exist_ok=True)
                
                # Save as TensorFlow datasets
                tf.data.Dataset.save(train_ds, str(train_path))
                tf.data.Dataset.save(val_ds, str(val_path))
                tf.data.Dataset.save(test_ds, str(test_path))
                
                # Generate statistics
                stats = {
                    "train": get_dataset_info(train_ds, "train"),
                    "validation": get_dataset_info(val_ds, "validation"),
                    "test": get_dataset_info(test_ds, "test"),
                    "configuration": {
                        "image_size": image_size,
                        "batch_size": batch_size,
                        "apply_augmentation": apply_augmentation
                    }
                }
                
                # Log metrics
                for split, info in stats.items():
                    if isinstance(info, dict):
                        total_samples = info.get("total_samples", 0)
                        num_batches = info.get("num_batches", 0)
                        if isinstance(total_samples, (int, float)):
                            mlflow.log_metric(f"{split}_samples", total_samples)
                        if isinstance(num_batches, (int, float)):
                            mlflow.log_metric(f"{split}_batches", num_batches)
                
                # Save statistics
                with open(data_stats.path, 'w') as f:
                    json.dump(stats, f, indent=2, default=str)
                
                logger.info("✅ Data preparation completed successfully")
                
                return {
                    "processed_data": processed_data,
                    "data_stats": data_stats
                }
                
            except Exception as e:
                logger.error(f"❌ Data preparation failed: {str(e)}")
                raise
    
    def run_model_training_local(
        self,
        training_data_asset: LocalAsset,
        config_file_path: Optional[str] = None,
        max_epochs: int = 5,  # Reduced for local testing
        learning_rate: float = 0.0001,
        patience: int = 3,
        max_trials: int = 3,  # Reduced for local testing
        seed: int = 42,
        model_name: str = "emotion-classifier-local"
    ) -> Dict[str, LocalAsset]:
        """Run model training component locally"""
        
        logger.info("🔄 Starting local model training...")
        
        # Create output assets
        trained_model = self.create_local_asset("trained_model", "URI_FOLDER")
        training_metrics = self.create_local_asset("training_metrics.json", "URI_FILE")
        model_summary = self.create_local_asset("model_summary.txt", "URI_FILE")
        
        
        try:
            # Import required modules
            import tensorflow as tf
            from tensorflow import keras
            import mlflow
            import mlflow.tensorflow
            from sklearn.utils.class_weight import compute_class_weight
            from config import TrainingConfig
            from model import cnn_model_color_VGG16_model, EmotionClassifierWrapper
            from training import set_seed, log_system_info, create_callbacks
            import numpy as np
            # Set seed
            set_seed(seed)
            
            # Setup configuration
            config = TrainingConfig()
            config.max_epochs = max_epochs
            config.learning_rate = learning_rate
            config.patience = patience
            config.max_trials = max_trials
            config.seed = seed
            config.model_name = model_name
            
            if config_file_path and Path(config_file_path).exists():
                with open(config_file_path, 'r') as f:
                    config_dict = json.load(f)
                    for key, value in config_dict.items():
                        if hasattr(config, key):
                            setattr(config, key, value)

            max_epochs = config.max_epochs
            learning_rate = config.learning_rate
            patience = config.patience
            max_trials = config.max_trials
            seed = config.seed
            model_name = config.model_name
            
            # Log parameters
            
            # Log system info
            log_system_info()
            
            # Load datasets
            train_path = training_data_asset.path / "train"
            val_path = training_data_asset.path / "validation"
            
            train_ds = tf.data.Dataset.load(str(train_path))
            val_ds = tf.data.Dataset.load(str(val_path))
            
            # Compute class weights from training data
            logger.info("Computing class weights for imbalanced dataset...")
            
            # Extract all labels from the training dataset
            y_train = []
            for _, labels in train_ds:
                y_train.append(labels.numpy())
            
            y_train = np.concatenate(y_train, axis=0)
            y_train_labels = np.argmax(y_train, axis=1)  # Convert one-hot to class indices
            
            # Compute balanced class weights
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y_train_labels),
                y=y_train_labels
            )
            
            class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
            
            logger.info(f"Class weights computed: {class_weight_dict}")
        
            # # Log class weights to MLflow
            # for class_idx, weight in class_weight_dict.items():
            #     mlflow.log_param(f"class_weight_{class_idx}", weight)
                
            print("max trials:", max_trials)
            # For local testing, use a simpler model if hyperparameter tuning is too slow
            if max_trials == 0:
                logger.info("Using simple model for local testing (no hyperparameter tuning)")
                
                # Create simple model
                model = keras.Sequential([
                    keras.layers.Input(shape=(48, 48, 3)),  # Define input shape here
                    keras.layers.Rescaling(1./255),
                    keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
                    keras.layers.MaxPooling2D(),
                    keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
                    keras.layers.MaxPooling2D(),
                    keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
                    keras.layers.MaxPooling2D(),
                    keras.layers.Flatten(),
                    keras.layers.Dense(128, activation='relu'),
                    keras.layers.Dropout(0.5),
                    keras.layers.Dense(4, activation='softmax')
                ])
                                
                best_model = model
                
            else:
                # Use hyperparameter tuning
                from keras_tuner import BayesianOptimization
                
                hypermodel = cnn_model_color_VGG16_model()
                
                tuner = BayesianOptimization(
                    hypermodel,
                    objective='val_accuracy',
                    seed=seed,
                    max_trials=max_trials,
                    directory=str(trained_model.path / 'tuning'),
                    project_name=f'{model_name}_local_tuning',
                    overwrite=True,
                )

                # Search for best hyperparameters
                tuner.search(
                    train_ds,
                    validation_data=val_ds,
                    epochs=max_epochs,
                    verbose=1,
                    callbacks=create_callbacks(config, trained_model.path),
                    class_weight=class_weight_dict
                )
                
                best_model = tuner.get_best_models(num_models=1)[0]
                
                # ✅ Get training history from tuning results
                best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
                
                # Create a mock history object with the best trial's metrics
                class MockHistory:
                    def __init__(self, trial):
                        self.history = {
                            'accuracy': [trial.metrics.get_history('accuracy')[-1]] if trial.metrics.exists('accuracy') else [0],
                            'val_accuracy': [trial.score if trial.score else 0],
                            'loss': [trial.metrics.get_history('loss')[-1]] if trial.metrics.exists('loss') else [0],
                            'val_loss': [trial.metrics.get_history('val_loss')[-1]] if trial.metrics.exists('val_loss') else [0]
                        }
                
                history = MockHistory(best_trial)
                
                # Log best hyperparameters
                best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
                # Log best hyperparameters
                # best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
                # for param in best_hps.space:
                #     param_value = best_hps.get(param.name)
                #     mlflow.log_param(f"best_{param.name}", param_value)
            
            with LocalMLflowTracker("local-training"):
                # Create callbacks
                mlflow.log_params({
                    "max_epochs": max_epochs,
                    "learning_rate": learning_rate,
                    "patience": patience,
                    "max_trials": max_trials,
                    "seed": seed,
                    "model_name": model_name
                    })
                
                
                # Log class weights
                for class_idx, weight in class_weight_dict.items():
                    mlflow.log_param(f"class_weight_{class_idx}", weight)
            
                    # After training, save the base model
                model_save_path = trained_model.path / "final_model.keras"
                best_model.save(model_save_path)
                
                # Get emotions list from config
                emotions = getattr(config, 'emotions', ['happy', 'sad', 'surprise', 'neutral'])
                
                # Create input example for signature
                try:
                    # Get a sample batch
                    for batch_images, batch_labels in train_ds.take(1):
                        input_example = batch_images[:1].numpy()
                        break
                    
                    # ✅ Fix: Infer signature automatically
                    from mlflow.models.signature import infer_signature
                    sample_prediction = best_model.predict(input_example, verbose=0)
                    signature = infer_signature(input_example, sample_prediction)
                    
                except Exception as e:
                    logger.warning(f"Could not create signature: {e}")
                    input_example = None
                    signature = None
                
                # Log model with custom wrapper
                mlflow.pyfunc.log_model(
                    artifact_path="model",
                    python_model=EmotionClassifierWrapper(emotions=emotions),
                    artifacts={
                        "model": str(model_save_path),
                        #"code_path": str(Path(__file__).parent / "src")
                    },
                    code_paths=[str(Path(__file__).parent / "src" / "model.py")],
                    registered_model_name=model_name,
                    signature=signature,
                    input_example=input_example
                )
                
                logger.info(f"✅ Model logged with custom wrapper to MLflow registry: {model_name}")
                
                
                # # Also log as Keras model for compatibility
                # mlflow.keras.log_model(
                #     model=best_model,
                #     artifact_path="model", 
                #     registered_model_name=f"{model_name}_keras",
                #     signature=signature,
                #     input_example=serving_input_example
                # )
                
                
                # Calculate metrics
                final_metrics = {
                    "final_train_accuracy": float(history.history['accuracy'][-1]),
                    "final_val_accuracy": float(history.history['val_accuracy'][-1]),
                    "final_train_loss": float(history.history['loss'][-1]),
                    "final_val_loss": float(history.history['val_loss'][-1]),
                    "epochs_trained": len(history.history['accuracy']),
                    "local_run": True,
                    "run_id": self.run_id
                }
                
                # Log metrics
                for metric_name, value in final_metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(metric_name, value)
                
                logger.info(f"✅ Training completed. Final validation accuracy: {final_metrics['final_val_accuracy']:.4f}")
                
                return {
                    "trained_model": trained_model,
                    "training_metrics": training_metrics,
                    "model_summary": model_summary
                }
                
        except Exception as e:
            logger.error(f"❌ Model training failed: {str(e)}")
            raise
    
    def run_model_evaluation_local(
        self,
        trained_model_asset: LocalAsset,
        test_data_asset: LocalAsset,
        config_file_path: Optional[str] = None,
        emotions: Optional[list] = None
    ) -> Dict[str, LocalAsset]:
        """Run model evaluation component locally"""
        
        logger.info("🔄 Starting local model evaluation...")
        
        if emotions is None:
            emotions = ['happy', 'sad', 'surprise', 'neutral']
        
        # Create output assets
        evaluation_results = self.create_local_asset("evaluation_results.json", "URI_FILE")
        evaluation_plots = self.create_local_asset("evaluation_plots", "URI_FOLDER")
        model_card = self.create_local_asset("model_card.md", "URI_FILE")
        
        with LocalMLflowTracker("local-evaluation"):
            try:
                import tensorflow as tf
                import mlflow
                import json
                from pathlib import Path
                
                from config import TrainingConfig
                from evaluation import evaluate_model_detailed, create_evaluation_plots
                
                # Setup configuration
                config = TrainingConfig()
                config.emotions = emotions
                
                if config_file_path and Path(config_file_path).exists():
                    with open(config_file_path, 'r') as f:
                        config_dict = json.load(f)
                        for key, value in config_dict.items():
                            if hasattr(config, key):
                                setattr(config, key, value)
                
                emotions = config.emotions
                # Load model
                model_path = trained_model_asset.path / "final_model.keras"
                model = tf.keras.models.load_model(str(model_path))
                
                # Load test dataset
                test_path = test_data_asset.path / "test"
                test_ds = tf.data.Dataset.load(str(test_path))
                
                # Evaluate model
                results, y_true, y_pred, y_pred_proba = evaluate_model_detailed(
                    model, test_ds, config
                )
                
                # Create evaluation plots
                create_evaluation_plots(
                    y_true, y_pred, y_pred_proba, emotions, evaluation_plots.path
                )
                
                # Log metrics
                for metric_name, value in results['overall_metrics'].items():
                    mlflow.log_metric(f"eval_{metric_name}", value)
                
                # Log per-class metrics
                for emotion, metrics in results['per_class_metrics'].items():
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"eval_{emotion}_{metric_name}", value)
                
                # Save evaluation results
                results['local_run'] = True
                results['run_id'] = self.run_id
                
                with open(evaluation_results.path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                
                # Create model card
                model_card_content = f"""# Emotion Classification Model Card (Local Run)

## Run Information
- **Run ID**: {self.run_id}
- **Execution**: Local Development
- **Timestamp**: {datetime.now().isoformat()}

## Model Overview
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
                
                model_card_content += f"\n## Output Locations\n"
                model_card_content += f"- **Results**: {evaluation_results.path}\n"
                model_card_content += f"- **Plots**: {evaluation_plots.path}\n"
                model_card_content += f"- **Model**: {trained_model_asset.path}\n"
                
                with open(model_card.path, 'w') as f:
                    f.write(model_card_content)
                
                logger.info(f"✅ Evaluation completed. Overall accuracy: {results['overall_metrics']['accuracy']:.4f}")
                
                return {
                    "evaluation_results": evaluation_results,
                    "evaluation_plots": evaluation_plots,
                    "model_card": model_card
                }
                
            except Exception as e:
                logger.error(f"❌ Model evaluation failed: {str(e)}")
                raise
    
    def run_complete_pipeline_local(
        self,
        raw_data_path: str,
        config_file_path: Optional[str] = None,
        max_epochs: int = 5,
        learning_rate: float = 0.0001,
        batch_size: int = 128,
        model_name: str = "emotion-classifier-local"
    ) -> Dict[str, LocalAsset]:
        """Run the complete pipeline locally"""
        
        logger.info(f"🚀 Starting complete local pipeline run: {self.run_id}")
        logger.info(f"📁 Output directory: {self.run_dir}")
        
        try:
            # Step 1: Data Preparation
            data_outputs = self.run_data_preparation_local(
                raw_data_path=raw_data_path,
                config_file_path=config_file_path,
                batch_size=batch_size
            )
            
            # Step 2: Model Training
            training_outputs = self.run_model_training_local(
                training_data_asset=data_outputs["processed_data"],
                config_file_path=config_file_path,
                max_epochs=max_epochs,
                learning_rate=learning_rate,
                model_name=model_name
            )
            
            # Step 3: Model Evaluation
            evaluation_outputs = self.run_model_evaluation_local(
                trained_model_asset=training_outputs["trained_model"],
                test_data_asset=data_outputs["processed_data"],
                config_file_path=config_file_path
            )
            
            # Create pipeline summary
            pipeline_summary = {
                "pipeline_run_id": self.run_id,
                "execution_type": "local",
                "timestamp": datetime.now().isoformat(),
                "outputs": {
                    "data_preparation": {
                        "processed_data": str(data_outputs["processed_data"].path),
                        "data_stats": str(data_outputs["data_stats"].path)
                    },
                    "model_training": {
                        "trained_model": str(training_outputs["trained_model"].path),
                        "training_metrics": str(training_outputs["training_metrics"].path),
                        "model_summary": str(training_outputs["model_summary"].path)
                    },
                    "model_evaluation": {
                        "evaluation_results": str(evaluation_outputs["evaluation_results"].path),
                        "evaluation_plots": str(evaluation_outputs["evaluation_plots"].path),
                        "model_card": str(evaluation_outputs["model_card"].path)
                    }
                }
            }
            
            summary_path = self.run_dir / "pipeline_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(pipeline_summary, f, indent=2)
            
            logger.info("🎉 Complete pipeline execution finished successfully!")
            logger.info(f"📊 Pipeline summary: {summary_path}")
            logger.info(f"📈 Model card: {evaluation_outputs['model_card'].path}")
            
            # Combine all outputs
            all_outputs = {
                **data_outputs,
                **training_outputs, 
                **evaluation_outputs,
                "pipeline_summary": LocalAsset(summary_path, "URI_FILE")
            }
            
            return all_outputs
            
        except Exception as e:
            logger.error(f"❌ Pipeline execution failed: {str(e)}")
            raise


def main():
    """Main function for local pipeline execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run emotion classification pipeline locally")
    parser.add_argument("--data_path", type=str, required=True, 
                       help="Path to raw training data")
    parser.add_argument("--config_file", type=str, 
                       help="Path to configuration JSON file")
    parser.add_argument("--output_dir", type=str, default="./local_outputs",
                       help="Local output directory")
    parser.add_argument("--max_epochs", type=int, default=5,
                       help="Maximum training epochs (reduced for local testing)")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size (reduced for local testing)")
    parser.add_argument("--model_name", type=str, default="emotion-classifier-local",
                       help="Model name")
    
    args = parser.parse_args()
    
    # Validate input path
    if not Path(args.data_path).exists():
        print(f"❌ Error: Data path does not exist: {args.data_path}")
        sys.exit(1)
    
    # Create and run pipeline
    runner = LocalPipelineRunner(output_dir=args.output_dir)
    
    try:
        outputs = runner.run_complete_pipeline_local(
            raw_data_path=args.data_path,
            config_file_path=args.config_file,
            max_epochs=args.max_epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            model_name=args.model_name
        )
        
        print("\n🎉 Local pipeline execution completed successfully!")
        print(f"📁 Output directory: {runner.run_dir}")
        print(f"📊 Model card: {outputs['model_card'].path}")
        print(f"🔍 View MLflow UI with: mlflow ui")
        
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()