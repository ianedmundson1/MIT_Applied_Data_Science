"""
Data preprocessing utilities for Azure ML training pipeline
"""
import os
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import tensorflow as tf
from keras.preprocessing import image_dataset_from_directory
import numpy as np

from config import TrainingConfig

logger = logging.getLogger(__name__)


def clean_corrupted_files(base_dir: str, files_to_remove: list) -> None:
    """Remove corrupted or unwanted files"""
    logger.info(f"Cleaning {len(files_to_remove)} files from {base_dir}")
    
    for file_path in files_to_remove:
        full_path = os.path.join(base_dir, file_path)
        try:
            os.remove(full_path)
            logger.debug(f"Removed: {full_path}")
        except FileNotFoundError:
            logger.warning(f"File not found: {full_path}")
        except Exception as e:
            logger.error(f"Error removing {full_path}: {e}")


def calculate_class_weights(dataset, num_classes: int) -> Dict[int, float]:
    """Calculate class weights for imbalanced dataset"""
    # Count samples per class
    class_counts = np.zeros(num_classes)
    
    for _, labels_batch in dataset:
        # Convert one-hot to class indices
        class_indices = tf.argmax(labels_batch, axis=1)
        for class_idx in class_indices:
            class_counts[class_idx.numpy()] += 1
    
    total_samples = np.sum(class_counts)
    
    # Calculate weights using sklearn's balanced approach
    class_weights = {}
    for i in range(num_classes):
        if class_counts[i] > 0:
            class_weights[i] = total_samples / (num_classes * class_counts[i])
        else:
            class_weights[i] = 1.0
    
    logger.info(f"Class distribution: {class_counts}")
    logger.info(f"Class weights: {class_weights}")
    
    return class_weights


def prepare_datasets(
    data_path: str, 
    config: TrainingConfig,
    apply_augmentation: bool = True
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Prepare training, validation, and test datasets with performance optimizations
    """
    logger.info(f"Preparing datasets from {data_path}")
    
    data_path_obj = Path(data_path)
    
    # Verify data structure
    required_dirs = ['train', 'validation', 'test']
    for dir_name in required_dirs:
        dir_path = data_path_obj / dir_name
        if not dir_path.exists():
            raise FileNotFoundError(f"Required directory not found: {dir_path}")
    
    # Load datasets
    datasets = {}
    for split in required_dirs:
        split_path = data_path_obj / split
        
        dataset = image_dataset_from_directory(
            str(split_path),
            image_size=(config.image_size, config.image_size),
            color_mode="rgb",  # Using RGB for VGG16
            batch_size=config.batch_size,
            label_mode='categorical',
            class_names=config.emotions,
            seed=config.seed,
            shuffle=(split == 'train')  # Only shuffle training data
        )
        
        # Apply performance optimizations
        AUTOTUNE = tf.data.AUTOTUNE
        dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
        
        # Apply data augmentation to training set
        if split == 'train' and apply_augmentation:
            dataset = apply_data_augmentation(dataset)
        
        datasets[split] = dataset
        logger.info(f"Loaded {split} dataset from {split_path}")
    
    # Calculate and log class weights for training set
    class_weights = calculate_class_weights(datasets['train'], config.num_classes)
    
    return datasets['train'], datasets['validation'], datasets['test']


def apply_data_augmentation(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Apply data augmentation to the training dataset"""
    
    def augment_fn(image, label):
        # Random horizontal flip
        image = tf.image.random_flip_left_right(image)
        
        # Random rotation (up to 20 degrees)
        image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
        
        # Random brightness adjustment
        image = tf.image.random_brightness(image, max_delta=0.1)
        
        # Random contrast adjustment
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        
        image = tf.image.random_saturation(image, 0.8, 1.2)
        # Ensure values are in [0, 255] range
        image = tf.clip_by_value(image, 0.0, 255.0)

        return image, label
    
    return dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)


def get_dataset_info(dataset: tf.data.Dataset, dataset_name: str) -> Dict[str, Any]:
    """Get information about a dataset"""
    info = {
        'name': dataset_name,
        'batch_size': None,
        'num_batches': 0,
        'total_samples': 0
    }
    
    for batch_images, batch_labels in dataset.take(1):
        info['batch_size'] = batch_images.shape[0]
        info['image_shape'] = batch_images.shape[1:]
        info['label_shape'] = batch_labels.shape[1:]
        break
    
    # Count total batches and samples
    for _ in dataset:
        info['num_batches'] += 1
    
    info['total_samples'] = info['num_batches'] * info['batch_size']
    
    return info


# Legacy code support - keeping original variable names for backward compatibility
def load_legacy_datasets(folder_path: str, picture_size: int, emotions: list, 
                        random_seed: int, batch_size: int = 128):
    """
    Legacy function to maintain compatibility with existing notebooks
    This should be replaced with prepare_datasets() in production
    """
    logger.warning("Using legacy dataset loading function. Consider migrating to prepare_datasets()")
    
    # Configure performance optimization
    AUTOTUNE = tf.data.AUTOTUNE
    
    # Load color datasets (RGB)
    train_set_color = image_dataset_from_directory(
        folder_path + 'train/',
        image_size=(picture_size, picture_size),
        color_mode="rgb",
        batch_size=batch_size,
        label_mode='categorical',
        seed=random_seed,
        class_names=emotions,
        shuffle=True
    )
    
    validation_set_color = image_dataset_from_directory(
        folder_path + 'validation/',
        image_size=(picture_size, picture_size),
        color_mode="rgb",
        batch_size=batch_size,
        label_mode='categorical',
        seed=random_seed,
        class_names=emotions,
        shuffle=True
    )
    
    test_set_color = image_dataset_from_directory(
        folder_path + 'test/',
        image_size=(picture_size, picture_size),
        color_mode="rgb",
        batch_size=batch_size,
        label_mode='categorical',
        seed=random_seed,
        class_names=emotions,
        shuffle=True
    )
    
    # Apply performance optimizations
    train_set_color = train_set_color.cache().prefetch(buffer_size=AUTOTUNE)
    validation_set_color = validation_set_color.cache().prefetch(buffer_size=AUTOTUNE)
    test_set_color = test_set_color.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_set_color, validation_set_color, test_set_color