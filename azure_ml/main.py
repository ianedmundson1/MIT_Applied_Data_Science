"""
Main entry point for Emotion Classification Pipeline
Supports both local and Azure ML execution
"""
import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Emotion Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run pipeline locally for development/testing
  python main.py local --data_path ./data --max_epochs 5

  # Run pipeline on Azure ML  
  python main.py azure --data_path azureml://datastores/workspaceblobstore/paths/emotion-data/

  # Run individual components locally
  python main.py local --data_path ./data --component data_prep
  python main.py local --data_path ./data --component training --max_epochs 10
        """
    )
    
    subparsers = parser.add_subparsers(dest='execution_mode', help='Execution mode')
    
    # Local execution subcommand
    local_parser = subparsers.add_parser('local', help='Run pipeline locally')
    local_parser.add_argument('--data_path', type=str, required=True,
                             help='Path to training data')
    local_parser.add_argument('--config_file', type=str,
                             help='Path to configuration JSON file')
    local_parser.add_argument('--output_dir', type=str, default='./local_outputs',
                             help='Local output directory')
    local_parser.add_argument('--max_epochs', type=int, default=5,
                             help='Maximum training epochs')
    local_parser.add_argument('--learning_rate', type=float, default=0.0001,
                             help='Learning rate')
    local_parser.add_argument('--batch_size', type=int, default=32,
                             help='Batch size')
    local_parser.add_argument('--model_name', type=str, default='emotion-classifier-local',
                             help='Model name')
    local_parser.add_argument('--component', type=str,
                             choices=['data_prep', 'training', 'evaluation', 'full'],
                             default='full',
                             help='Component to run (default: full pipeline)')
    
    # Azure ML execution subcommand
    azure_parser = subparsers.add_parser('azure', help='Run pipeline on Azure ML')
    azure_parser.add_argument('--data_path', type=str, required=True,
                             help='Azure ML data path (e.g., azureml://datastores/...)')
    azure_parser.add_argument('--experiment_name', type=str, default='emotion-classification',
                             help='Azure ML experiment name')
    azure_parser.add_argument('--model_name', type=str, default='emotion-classifier',
                             help='Model name')
    azure_parser.add_argument('--max_epochs', type=int, default=30,
                             help='Maximum training epochs')
    azure_parser.add_argument('--learning_rate', type=float, default=0.0001,
                             help='Learning rate')
    azure_parser.add_argument('--batch_size', type=int, default=128,
                             help='Batch size')
    azure_parser.add_argument('--compute_target', type=str, default='gpu-cluster',
                             help='Azure ML compute target')
    
    args = parser.parse_args()
    
    if not args.execution_mode:
        parser.print_help()
        print("\nPlease specify execution mode: 'local' or 'azure'")
        sys.exit(1)
    
    if args.execution_mode == 'local':
        run_local_pipeline(args)
    elif args.execution_mode == 'azure':
        run_azure_pipeline(args)


def run_local_pipeline(args):
    """Run pipeline locally using LocalPipelineRunner"""
    print("🏠 Running pipeline locally...")
    
    # Validate data path
    if not Path(args.data_path).exists():
        print(f"❌ Error: Data path does not exist: {args.data_path}")
        sys.exit(1)
    
    try:
        from local_runner import LocalPipelineRunner
        
        # Create runner
        runner = LocalPipelineRunner(output_dir=args.output_dir)
        
        if args.component == 'full':
            # Run complete pipeline
            outputs = runner.run_complete_pipeline_local(
                raw_data_path=args.data_path,
                config_file_path=args.config_file,
                max_epochs=args.max_epochs,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                model_name=args.model_name
            )
            
            print(f"\n🎉 Complete pipeline finished!")
            print(f"📁 Output directory: {runner.run_dir}")
            print(f"📊 Model card: {outputs['model_card'].path}")
            
        elif args.component == 'data_prep':
            # Run only data preparation
            outputs = runner.run_data_preparation_local(
                raw_data_path=args.data_path,
                config_file_path=args.config_file,
                batch_size=args.batch_size
            )
            print(f"✅ Data preparation completed: {outputs['processed_data'].path}")
            
        elif args.component == 'training':
            print("❌ Cannot run training alone - need processed data from data_prep step")
            print("Use 'full' pipeline or run data_prep first")
            sys.exit(1)
            
        elif args.component == 'evaluation':
            print("❌ Cannot run evaluation alone - need trained model from training step")
            print("Use 'full' pipeline or run previous steps first")
            sys.exit(1)
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required packages are installed: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Local execution failed: {str(e)}")
        sys.exit(1)


def run_azure_pipeline(args):
    """Run pipeline on Azure ML using modern SDK v2"""
    print("☁️  Running pipeline on Azure ML...")
    
    try:
        from src.pipeline_runner import EmotionClassificationPipeline
        from azure.ai.ml import MLClient, Input
        from azure.ai.ml.constants import AssetTypes
        from azure.identity import DefaultAzureCredential
        from src.config import AzureMLConfig
        
        # Initialize Azure ML client
        azure_config = AzureMLConfig()
        credential = DefaultAzureCredential()
        
        ml_client = MLClient(
            credential=credential,
            subscription_id=azure_config.subscription_id,
            resource_group_name=azure_config.resource_group,
            workspace_name=azure_config.workspace_name
        )
        
        print(f"✅ Connected to Azure ML workspace: {azure_config.workspace_name}")
        
        # Create and submit pipeline
        pipeline = EmotionClassificationPipeline(ml_client)
        
        job = pipeline.submit_pipeline(
            training_data_path=args.data_path,
            experiment_name=args.experiment_name,
            model_name=args.model_name,
            max_epochs=args.max_epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size
        )
        
        print(f"🚀 Pipeline submitted successfully!")
        print(f"📋 Job name: {job.name}")
        print(f"🌐 Studio URL: {job.studio_url}")
        print(f"📊 Monitor progress in Azure ML Studio")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure Azure ML SDK v2 is installed: pip install azure-ai-ml")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Azure ML execution failed: {str(e)}")
        print("Check your Azure ML configuration and credentials")
        sys.exit(1)


if __name__ == "__main__":
    main()
