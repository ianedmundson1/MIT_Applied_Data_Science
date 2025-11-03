"""
Azure ML deployment and inference script
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment, 
    Model,
    Environment,
    CodeConfiguration
)
from azure.identity import DefaultAzureCredential

from config import AzureMLConfig, get_ml_client

logger = logging.getLogger(__name__)


def create_deployment_environment(ml_client: MLClient) -> Environment:
    """Create environment for model deployment"""
    
    env = Environment(
        name="emotion-classifier-env",
        description="Environment for emotion classification model",
        conda_file="./conda.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
    )
    
    # Register environment
    env = ml_client.environments.create_or_update(env)
    logger.info(f"Environment created: {env.name}:{env.version}")
    
    return env


def create_inference_endpoint(
    ml_client: MLClient,
    endpoint_name: str,
    description: str = "Emotion classification endpoint"
) -> ManagedOnlineEndpoint:
    """Create online endpoint for model inference"""
    
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description=description,
        auth_mode="key",  # or "aml_token" for token-based auth
        tags={"model": "emotion-classifier", "framework": "tensorflow"}
    )
    
    # Create endpoint
    endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    logger.info(f"Endpoint created: {endpoint.name}")
    
    return endpoint


def deploy_model_to_endpoint(
    ml_client: MLClient,
    endpoint_name: str,
    model_name: str,
    deployment_name: str = "blue",
    instance_type: str = "Standard_DS3_v2",
    instance_count: int = 1
) -> ManagedOnlineDeployment:
    """Deploy model to online endpoint"""
    
    # Get latest model version
    latest_model = ml_client.models.get(model_name, label="latest")
    
    # Create deployment
    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=latest_model,
        environment="emotion-classifier-env@latest",
        code_configuration=CodeConfiguration(
            code="./deployment", 
            scoring_script="score.py"
        ),
        instance_type=instance_type,
        instance_count=instance_count,
        request_settings={
            "request_timeout_ms": 60000,
            "max_concurrent_requests_per_instance": 1,
            "max_queue_wait_ms": 500
        },
        liveness_probe={
            "failure_threshold": 30,
            "success_threshold": 1,
            "timeout": 2,
            "period": 10,
            "initial_delay": 10
        },
        readiness_probe={
            "failure_threshold": 10,
            "success_threshold": 1, 
            "timeout": 10,
            "period": 10,
            "initial_delay": 10
        }
    )
    
    # Deploy model
    deployment = ml_client.online_deployments.begin_create_or_update(deployment).result()
    
    # Set deployment to receive 100% of traffic
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    endpoint.traffic = {deployment_name: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    
    logger.info(f"Model deployed to endpoint: {endpoint_name}/{deployment_name}")
    
    return deployment


def create_batch_endpoint(
    ml_client: MLClient,
    endpoint_name: str,
    model_name: str,
    compute_name: str = "cpu-cluster"
) -> None:
    """Create batch endpoint for batch inference"""
    
    from azure.ai.ml.entities import BatchEndpoint, BatchDeployment
    
    # Create batch endpoint
    endpoint = BatchEndpoint(
        name=endpoint_name,
        description="Batch inference endpoint for emotion classification"
    )
    
    endpoint = ml_client.batch_endpoints.begin_create_or_update(endpoint).result()
    
    # Get latest model
    latest_model = ml_client.models.get(model_name, label="latest")
    
    # Create batch deployment
    deployment = BatchDeployment(
        name="default",
        endpoint_name=endpoint_name,
        model=latest_model,
        environment="emotion-classifier-env@latest",
        code_configuration=CodeConfiguration(
            code="./deployment",
            scoring_script="batch_score.py"
        ),
        compute=compute_name,
        instance_count=2,
        max_concurrency_per_instance=2,
        mini_batch_size=10,
        retry_settings={
            "max_retries": 3,
            "timeout": 300
        },
        output_action="append_row",
        output_file_name="predictions.csv",
        logging_level="info"
    )
    
    deployment = ml_client.batch_deployments.begin_create_or_update(deployment).result()
    
    # Set as default deployment
    endpoint.defaults.deployment_name = deployment.name
    ml_client.batch_endpoints.begin_create_or_update(endpoint).result()
    
    logger.info(f"Batch endpoint created: {endpoint_name}")


def test_endpoint(ml_client: MLClient, endpoint_name: str, test_data: str) -> Dict[str, Any]:
    """Test the deployed endpoint"""
    
    # Load test data
    with open(test_data, 'r') as f:
        request_data = f.read()
    
    # Invoke endpoint
    response = ml_client.online_endpoints.invoke(
        endpoint_name=endpoint_name,
        request_file=test_data
    )
    
    logger.info(f"Endpoint test response: {response}")
    return {"status": "success", "response": response}


def main():
    """Main deployment function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy emotion classification model")
    parser.add_argument("--model_name", type=str, required=True, help="Name of registered model")
    parser.add_argument("--endpoint_name", type=str, required=True, help="Name of endpoint")
    parser.add_argument("--deployment_type", type=str, choices=["online", "batch"], 
                       default="online", help="Type of deployment")
    parser.add_argument("--instance_type", type=str, default="Standard_DS3_v2", 
                       help="VM instance type for deployment")
    parser.add_argument("--instance_count", type=int, default=1, 
                       help="Number of instances")
    
    args = parser.parse_args()
    
    # Initialize Azure ML client
    azure_config = AzureMLConfig()
    ml_client = get_ml_client(azure_config)
    
    try:
        if args.deployment_type == "online":
            # Create environment
            env = create_deployment_environment(ml_client)
            
            # Create endpoint
            endpoint = create_inference_endpoint(ml_client, args.endpoint_name)
            
            # Deploy model
            deployment = deploy_model_to_endpoint(
                ml_client=ml_client,
                endpoint_name=args.endpoint_name,
                model_name=args.model_name,
                instance_type=args.instance_type,
                instance_count=args.instance_count
            )
            
            logger.info("Online deployment completed successfully")
            
        elif args.deployment_type == "batch":
            # Create batch endpoint
            create_batch_endpoint(
                ml_client=ml_client,
                endpoint_name=args.endpoint_name,
                model_name=args.model_name
            )
            
            logger.info("Batch deployment completed successfully")
            
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()