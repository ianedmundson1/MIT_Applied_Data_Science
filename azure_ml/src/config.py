"""
Configuration management for Azure ML training pipeline
"""
import os
import json
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential


@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    # Model hyperparameters
    learning_rate: float = 0.0001
    batch_size: int = 128
    max_epochs: int = 30
    patience: int = 12
    
    # Data parameters
    image_size: int = 48
    num_classes: int = 4
    emotions: Optional[List[str]] = None
    
    # Hyperparameter tuning
    max_trials: int = 30
    factor: int = 3
    
    # Reproducibility
    seed: int = 42
    
    # Azure ML specific
    experiment_name: str = "facial-emotion-detection"
    model_name: str = "emotion-classifier"
    compute_target: str = "gpu-cluster"
    
    def __post_init__(self):
        if self.emotions is None:
            self.emotions = ['happy', 'sad', 'surprise', 'neutral']


@dataclass 
class AzureMLConfig:
    """Azure ML workspace configuration"""
    subscription_id: Optional[str] = None
    resource_group: Optional[str] = None
    workspace_name: Optional[str] = None
    
    def __post_init__(self):
        # Try to get from environment variables
        self.subscription_id = self.subscription_id or os.getenv('AZURE_SUBSCRIPTION_ID')
        self.resource_group = self.resource_group or os.getenv('AZURE_RESOURCE_GROUP')
        self.workspace_name = self.workspace_name or os.getenv('AZURE_ML_WORKSPACE_NAME')


def get_ml_client(config: AzureMLConfig) -> MLClient:
    """
    Get Azure ML client with managed identity authentication
    """
    try:
        # Use DefaultAzureCredential for secure authentication
        credential = DefaultAzureCredential()
        
        ml_client = MLClient(
            credential=credential,
            subscription_id=config.subscription_id,
            resource_group_name=config.resource_group,
            workspace_name=config.workspace_name
        )
        
        return ml_client
        
    except Exception as e:
        raise Exception(f"Failed to create ML client: {str(e)}")


def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Config file {config_path} not found, using defaults")
        return {}
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {str(e)}")