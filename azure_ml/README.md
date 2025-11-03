# Emotion Classification Pipeline - Local & Azure ML

A modern MLOps pipeline for facial emotion classification that supports both **local development** and **Azure ML production** execution using Azure ML SDK v2.

## 🏗️ Architecture

```
azure_ml/
├── main.py                 # Main entry point (local/azure)
├── local_runner.py        # Local pipeline execution
├── setup_local.sh         # Development setup script
├── src/
│   ├── components.py      # Azure ML SDK v2 components  
│   ├── pipeline_runner.py # Azure ML pipeline runner
│   ├── config.py          # Configuration management
│   ├── training.py        # Enhanced training logic
│   ├── preprocessing.py   # Data processing utilities
│   ├── evaluation.py      # Model evaluation
│   ├── deployment.py      # MLOps deployment
│   └── model.py          # Model architecture
├── config/
│   ├── local_config.json  # Local development config
│   └── training_config.json # Production config
├── pipeline.yml           # Azure ML pipeline YAML
├── requirements.txt       # Dependencies
└── conda.yml             # Conda environment
```

## 🚀 Quick Start

### Local Development (Recommended for Testing)

1. **Setup Environment**
   ```bash
   ./setup_local.sh
   # or manually:
   pip install tensorflow keras scikit-learn matplotlib seaborn mlflow pandas numpy
   ```

2. **Run Complete Pipeline Locally**
   ```bash
   python main.py local --data_path ./path/to/your/data --max_epochs 5
   ```

3. **View Results**
   ```bash
   # MLflow UI for experiment tracking
   mlflow ui
   
   # Check outputs
   ls local_outputs/run_YYYYMMDD_HHMMSS/
   ```

### Azure ML Production

1. **Install Azure ML SDK**
   ```bash
   pip install azure-ai-ml azure-identity
   ```

2. **Configure Azure ML**
   ```bash
   # Set environment variables
   export AZURE_SUBSCRIPTION_ID="your-subscription-id"
   export AZURE_RESOURCE_GROUP="your-resource-group"  
   export AZURE_ML_WORKSPACE_NAME="your-workspace"
   ```

3. **Run on Azure ML**
   ```bash
   python main.py azure --data_path azureml://datastores/workspaceblobstore/paths/emotion-data/
   ```

## 💻 Local Execution Features

### Full Pipeline
```bash
python main.py local --data_path ./data --config_file config/local_config.json
```

### Individual Components
```bash
# Data preparation only
python main.py local --data_path ./data --component data_prep

# Note: Training and evaluation require previous steps
```

### Local Configuration
```json
{
  "learning_rate": 0.001,
  "batch_size": 32,
  "max_epochs": 5,
  "patience": 3,
  "image_size": 48,
  "num_classes": 4,
  "emotions": ["happy", "sad", "surprise", "neutral"],
  "local_development": true
}
```

## ☁️ Azure ML Execution Features

### Modern SDK v2 Components
- **Type-safe components** with `@command` decorators
- **Declarative pipelines** with `@dsl.pipeline` 
- **Automatic asset management** and tracking
- **Built-in parameterization** for Studio UI

### Pipeline Submission
```python
from src.pipeline_runner import EmotionClassificationPipeline
from azure.ai.ml import MLClient, Input
from azure.ai.ml.constants import AssetTypes

pipeline = EmotionClassificationPipeline(ml_client)
job = pipeline.submit_pipeline(
    training_data_path="azureml://datastores/workspaceblobstore/paths/data/",
    experiment_name="emotion-classification",
    max_epochs=30,
    learning_rate=0.0001
)
```

## 📊 Output Structure

### Local Outputs
```
local_outputs/
└── run_20241027_143022/
    ├── pipeline.log
    ├── pipeline_summary.json
    ├── processed_data/
    │   ├── train/
    │   ├── validation/
    │   └── test/
    ├── trained_model/
    │   ├── final_model.keras
    │   └── tuning/
    ├── training_metrics.json
    ├── evaluation_results.json
    ├── evaluation_plots/
    │   ├── confusion_matrix.png
    │   └── per_class_metrics.png
    └── model_card.md
```

### Azure ML Outputs
- **Registered Models** in Azure ML Model Registry
- **Experiment Tracking** in Azure ML Studio
- **Pipeline Artifacts** in Azure ML Storage
- **Deployment Endpoints** for inference

## 🔧 Configuration Options

### Local Development (`config/local_config.json`)
```json
{
  "learning_rate": 0.001,      // Lower for stability
  "batch_size": 32,            // Smaller for local resources
  "max_epochs": 5,             // Faster iteration
  "max_trials": 1,             // Skip hyperparameter tuning
  "local_development": true
}
```

### Production (`config/training_config.json`)
```json
{
  "learning_rate": 0.0001,
  "batch_size": 128,
  "max_epochs": 30,
  "max_trials": 30,
  "factor": 3,
  "compute_target": "gpu-cluster"
}
```

## 📈 MLflow Integration

Both local and Azure ML execution integrate with MLflow:

```bash
# View local experiments
mlflow ui

# Connect to Azure ML MLflow
mlflow server --backend-store-uri azureml://...
```

**Tracked Metrics:**
- Training/validation accuracy and loss
- Per-class precision, recall, F1-score
- Hyperparameter values
- System information
- Dataset statistics

## 🎯 Data Format

Expected directory structure:
```
data/
├── train/
│   ├── happy/          # Happy emotion images
│   ├── sad/            # Sad emotion images  
│   ├── surprise/       # Surprise emotion images
│   └── neutral/        # Neutral emotion images
├── validation/
│   ├── happy/
│   ├── sad/
│   ├── surprise/
│   └── neutral/
└── test/
    ├── happy/
    ├── sad/
    ├── surprise/
    └── neutral/
```

## 🔄 Development Workflow

1. **Local Development**
   ```bash
   # Quick iteration with small epochs
   python main.py local --data_path ./sample_data --max_epochs 2
   ```

2. **Local Validation**
   ```bash
   # Full local run with realistic settings
   python main.py local --data_path ./full_data --max_epochs 10
   ```

3. **Azure ML Production**
   ```bash
   # Production run with full hyperparameter tuning
   python main.py azure --data_path azureml://... --max_epochs 50
   ```

## 🐛 Troubleshooting

### Common Issues

**Local execution fails:**
```bash
# Check dependencies
pip install -r requirements.txt

# Verify data structure
ls -la data/train/happy/
```

**Azure ML connection fails:**
```bash
# Check authentication
az login
az account show

# Verify environment variables
echo $AZURE_SUBSCRIPTION_ID
```

**Memory issues locally:**
```bash
# Reduce batch size
python main.py local --data_path ./data --batch_size 16
```

### Debug Mode
```bash
# Enable detailed logging
export PYTHONPATH=./src
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
python main.py local --data_path ./data
```

## 🎛️ Advanced Usage

### Custom Model Architecture
Edit `src/model.py` to customize the CNN architecture.

### Custom Evaluation Metrics
Edit `src/evaluation.py` to add custom metrics.

### Deployment
```bash
# Deploy trained model to Azure ML endpoint
python src/deployment.py --model_name emotion-classifier --endpoint_name emotion-api
```

## 📝 Key Differences: Local vs Azure ML

| Feature | Local Execution | Azure ML Execution |
|---------|----------------|-------------------|
| **Resource Requirements** | Local CPU/GPU | Scalable cloud compute |
| **Execution Time** | Fast iteration | Production-scale training |
| **Experiment Tracking** | Local MLflow | Azure ML Studio |
| **Model Registry** | Local files | Azure ML Model Registry |
| **Deployment** | Manual | Automated endpoints |
| **Collaboration** | Individual | Team-wide |
| **Cost** | Free (local resources) | Pay per compute |

This setup provides the best of both worlds: **fast local development** with **production-ready Azure ML deployment**!
