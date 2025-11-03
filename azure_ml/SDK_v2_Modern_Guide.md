# Azure ML SDK v2 - Modern Decorators & Type Safety Guide

## Overview of Azure ML SDK v2 Modern Features

Azure ML SDK v2 introduces a **significantly more Pythonic** approach with:

### 🎯 **Key Modern Features:**

1. **Function Decorators** - `@command` and `@pipeline` decorators
2. **Strong Type Safety** - `Input()` and `Output()` with explicit types  
3. **Declarative Pipeline DSL** - `@dsl.pipeline` decorator
4. **Asset Type System** - `AssetTypes.URI_FOLDER`, `AssetTypes.URI_FILE`, etc.
5. **Automatic Dependency Injection** - Components automatically get inputs/outputs
6. **Built-in Parameterization** - Native support for pipeline parameters

## Comparison: Old vs New Approach

### ❌ **Original Approach (Your Code)**
```python
# Hardcoded training script
import mlflow
from model import cnn_model_color_VGG16_model

# Basic MLflow logging
if __name__ == "__main__":
    mlflow.tensorflow.log_model(best_model, "model")
```

### ✅ **Modern SDK v2 Approach**
```python
from azure.ai.ml import command, Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline

@command(
    name="model_training_v2",
    display_name="Model Training Component",
    environment="azureml:tensorflow-gpu-env@latest",
    code="./src",
)
def model_training_component(
    # Strongly typed inputs
    training_data: Input(type=AssetTypes.URI_FOLDER, description="Training data"),
    config_file: Input(type=AssetTypes.URI_FILE, optional=True),
    
    # Strongly typed outputs  
    trained_model: Output(type=AssetTypes.URI_FOLDER, description="Model artifacts"),
    training_metrics: Output(type=AssetTypes.URI_FILE, description="Metrics JSON"),
    
    # Native parameters
    max_epochs: int = 30,
    learning_rate: float = 0.0001,
) -> None:
    """Modern component with full type safety"""
    # Component logic here
```

## Key Advantages of SDK v2 Decorators

### 1. **Type Safety & IntelliSense**
```python
# SDK v2 provides full type hints
training_data: Input(type=AssetTypes.URI_FOLDER)  # IDE knows this is a folder path
max_epochs: int = 30  # IDE validates integer type
```

### 2. **Automatic Asset Management**
```python
# SDK v2 automatically handles:
# - Input/output validation
# - Path resolution  
# - Artifact tracking
# - Dependency management
```

### 3. **Declarative Pipeline Composition**
```python
@dsl.pipeline(name="emotion_pipeline")
def create_training_pipeline(
    training_data: Input(type=AssetTypes.URI_FOLDER),
    max_epochs: int = 30
):
    # Automatic dependency chaining
    data_prep = data_preparation_component(raw_data=training_data)
    training = model_training_component(
        training_data=data_prep.outputs.processed_data,
        max_epochs=max_epochs
    )
    evaluation = model_evaluation_component(
        trained_model=training.outputs.trained_model,
        test_data=data_prep.outputs.processed_data
    )
    
    return {
        "final_model": training.outputs.trained_model,
        "results": evaluation.outputs.evaluation_results
    }
```

### 4. **Built-in Parameterization**
```python
# Parameters are automatically exposed in Azure ML Studio UI
def training_component(
    learning_rate: float = 0.0001,  # Slider in UI
    max_epochs: int = 30,          # Number input in UI  
    model_name: str = "classifier", # Text input in UI
):
```

## Modern Component Architecture

### **Component Structure with SDK v2:**

```python
@command(
    name="data_prep",
    display_name="Data Preparation", 
    environment="azureml:tensorflow-env@latest",
    code="./src",
    instance_count=1,                    # Automatic scaling
    is_deterministic=True,               # Caching optimization
)
def data_preparation(
    # Inputs (automatically validated)
    raw_data: Input(
        type=AssetTypes.URI_FOLDER,
        description="Raw image dataset",
        mode="ro_mount"  # Read-only mount for performance
    ),
    
    # Outputs (automatically registered as assets)
    processed_data: Output(
        type=AssetTypes.URI_FOLDER,
        description="Preprocessed training data"
    ),
    
    # Parameters (automatically exposed in UI)
    batch_size: int = 128,
    image_size: int = 48,
    apply_augmentation: bool = True,
) -> None:
    """
    Modern data preparation component
    - Automatic input validation
    - Type-safe parameters  
    - Asset registration
    - Built-in caching
    """
    # Component implementation
```

## Pipeline Orchestration with Modern DSL

### **Pipeline Definition:**
```python
@dsl.pipeline(
    name="emotion_classification_v2",
    display_name="Emotion Classification Pipeline",
    description="End-to-end ML pipeline with modern SDK v2",
    tags={"framework": "tensorflow", "version": "v2"},
)
def emotion_pipeline(
    # Pipeline-level inputs (exposed in Studio UI)
    training_data: Input(type=AssetTypes.URI_FOLDER),
    model_name: str = "emotion-classifier",
    max_epochs: int = 30,
) -> Dict[str, Output]:
    """
    Modern pipeline with automatic:
    - Dependency resolution
    - Asset management  
    - Parameter validation
    - Error handling
    """
    
    # Component chaining with type safety
    prep = data_preparation(raw_data=training_data)
    prep.compute = "cpu-cluster"  # Resource assignment
    
    train = model_training(
        training_data=prep.outputs.processed_data,
        max_epochs=max_epochs,
        model_name=model_name
    )
    train.compute = "gpu-cluster"  # GPU for training
    
    eval = model_evaluation(
        trained_model=train.outputs.trained_model,
        test_data=prep.outputs.processed_data
    )
    eval.compute = "cpu-cluster"  # CPU for evaluation
    
    # Strongly typed return
    return {
        "model": train.outputs.trained_model,
        "metrics": eval.outputs.evaluation_results
    }
```

## Benefits Over Original Approach

| Feature | Original Code | Modern SDK v2 |
|---------|---------------|---------------|
| **Type Safety** | ❌ No typing | ✅ Full type hints |
| **Parameter Validation** | ❌ Manual | ✅ Automatic |
| **Asset Management** | ❌ Manual paths | ✅ Automatic tracking |
| **Pipeline Composition** | ❌ Monolithic | ✅ Modular components |
| **UI Integration** | ❌ Limited | ✅ Full Studio integration |
| **Caching** | ❌ None | ✅ Intelligent caching |
| **Scaling** | ❌ Manual | ✅ Automatic |
| **Error Handling** | ❌ Basic | ✅ Comprehensive |

## Usage Example

### **Running the Modern Pipeline:**

```python
from azure.ai.ml import MLClient, Input
from azure.ai.ml.constants import AssetTypes

# Initialize client
ml_client = MLClient.from_config()

# Create pipeline instance
pipeline_job = emotion_pipeline(
    training_data=Input(
        type=AssetTypes.URI_FOLDER, 
        path="azureml://datastores/workspaceblobstore/paths/emotion-data/"
    ),
    model_name="emotion-classifier-v2",
    max_epochs=50
)

# Submit with automatic validation
job = ml_client.jobs.create_or_update(
    pipeline_job,
    experiment_name="emotion-classification-modern"
)

print(f"Pipeline submitted: {job.studio_url}")
```

## Key Takeaways

1. **SDK v2 is dramatically more Pythonic** with decorators and type hints
2. **Automatic asset management** eliminates manual path handling
3. **Component-based architecture** enables better reusability
4. **Built-in parameterization** provides native UI integration  
5. **Declarative pipeline DSL** simplifies complex workflows
6. **Type safety** catches errors at development time
7. **Intelligent caching** improves performance and cost

The modern approach transforms Azure ML from a configuration-heavy platform into a **truly Pythonic ML development experience** with enterprise-grade capabilities.