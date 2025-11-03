"""
Model evaluation script for Azure ML pipeline
"""
import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow import keras
import mlflow
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from config import TrainingConfig
from preprocessing import prepare_datasets

logger = logging.getLogger(__name__)


def load_model(model_path: str) -> keras.Model:
    """Load trained model from path"""
    model_file = Path(model_path) / "final_model.keras"
    if not model_file.exists():
        # Try alternative model file names
        alternatives = ["model.keras", "best_model_checkpoint.keras"]
        for alt in alternatives:
            alt_path = Path(model_path) / alt
            if alt_path.exists():
                model_file = alt_path
                break
        else:
            raise FileNotFoundError(f"No model file found in {model_path}")
    
    logger.info(f"Loading model from {model_file}")
    return keras.models.load_model(str(model_file))


def evaluate_model_detailed(
    model: keras.Model, 
    test_dataset: tf.data.Dataset,
    config: TrainingConfig
) -> tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Perform detailed model evaluation"""
    
    logger.info("Performing detailed model evaluation...")
    
    # Collect predictions and ground truth
    y_true = []
    y_pred = []
    y_pred_proba = []
    
    for batch_images, batch_labels in test_dataset:
        # Get predictions
        predictions = model.predict(batch_images, verbose=0)
        y_pred_proba.extend(predictions)
        
        # Convert to class indices
        batch_pred_classes = np.argmax(predictions, axis=1)
        batch_true_classes = np.argmax(batch_labels.numpy(), axis=1)
        
        y_pred.extend(batch_pred_classes)
        y_true.extend(batch_true_classes)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = \
        precision_recall_fscore_support(y_true, y_pred, average=None)
    
    # Create detailed results
    results = {
        'overall_metrics': {
            'accuracy': float(accuracy),
            'precision_weighted': float(precision),
            'recall_weighted': float(recall),
            'f1_weighted': float(f1)
        },
        'per_class_metrics': {}
    }
    
    emotions = config.emotions or ['happy', 'sad', 'surprise', 'neutral']
    for i, emotion in enumerate(emotions):
        if i < len(precision_per_class):
            results['per_class_metrics'][emotion] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]), 
                'f1_score': float(f1_per_class[i]),
                'support': int(support[i])
            }
    
    # Classification report
    class_report = classification_report(
        y_true, y_pred, 
        target_names=config.emotions,
        output_dict=True
    )
    results['classification_report'] = class_report
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    results['confusion_matrix'] = cm.tolist()
    
    return results, y_true, y_pred, y_pred_proba


def create_evaluation_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray, 
    y_pred_proba: np.ndarray,
    emotions: list,
    output_dir: Path
) -> None:
    """Create evaluation plots and save them"""
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=emotions, yticklabels=emotions)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    cm_path = output_dir / 'confusion_matrix.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Per-class accuracy
    plt.figure(figsize=(10, 6))
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    x = np.arange(len(emotions))
    width = 0.25
    
    plt.bar(x - width, precision, width, label='Precision', alpha=0.8)
    plt.bar(x, recall, width, label='Recall', alpha=0.8)
    plt.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    plt.xlabel('Emotions')
    plt.ylabel('Score')
    plt.title('Per-Class Performance Metrics')
    plt.xticks(x, emotions)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    metrics_path = output_dir / 'per_class_metrics.png'
    plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Evaluation plots saved to {output_dir}")


def log_evaluation_results(results: Dict[str, Any], plots_dir: Path) -> None:
    """Log evaluation results to MLflow"""
    
    # Log overall metrics
    for metric_name, value in results['overall_metrics'].items():
        mlflow.log_metric(f"eval_{metric_name}", value)
    
    # Log per-class metrics
    for emotion, metrics in results['per_class_metrics'].items():
        for metric_name, value in metrics.items():
            mlflow.log_metric(f"eval_{emotion}_{metric_name}", value)
    
    # Log plots as artifacts
    if plots_dir.exists():
        for plot_file in plots_dir.glob('*.png'):
            mlflow.log_artifact(str(plot_file))
    
    # Log detailed results as JSON
    results_json = json.dumps(results, indent=2, default=str)
    mlflow.log_text(results_json, "evaluation_results.json")


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate trained emotion classification model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data")
    parser.add_argument("--output_file", type=str, required=True, help="Output results file")
    parser.add_argument("--config_file", type=str, help="Path to config JSON file")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = TrainingConfig()
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # Start MLflow run for evaluation
    with mlflow.start_run(run_name="model-evaluation") as run:
        
        # Load model
        model = load_model(args.model_path)
        
        # Prepare test dataset
        logger.info("Preparing test dataset...")
        _, _, test_dataset = prepare_datasets(
            data_path=args.test_data,
            config=config,
            apply_augmentation=False  # No augmentation for evaluation
        )
        
        # Evaluate model
        results, y_true, y_pred, y_pred_proba = evaluate_model_detailed(
            model, test_dataset, config
        )
        
        # Create evaluation plots
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        emotions = config.emotions or ['happy', 'sad', 'surprise', 'neutral']
        create_evaluation_plots(
            y_true, y_pred, y_pred_proba, emotions, plots_dir
        )
        
        # Log results to MLflow
        log_evaluation_results(results, plots_dir)
        
        # Save results to file
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Evaluation completed. Results saved to {args.output_file}")
        logger.info(f"Overall Accuracy: {results['overall_metrics']['accuracy']:.4f}")
        logger.info(f"Weighted F1-Score: {results['overall_metrics']['f1_weighted']:.4f}")
        
        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()