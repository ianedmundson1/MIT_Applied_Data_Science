#!/bin/bash

# Quick setup script for local development
echo "🔧 Setting up local development environment..."

# Create necessary directories
mkdir -p local_outputs
mkdir -p data/sample

# Check Python environment
echo "🐍 Checking Python environment..."
python --version

# Install basic requirements (if not already installed)
echo "📦 Installing Python packages..."
pip install tensorflow keras scikit-learn matplotlib seaborn mlflow pandas numpy

# Optional: Install Azure ML SDK for cloud execution
read -p "Install Azure ML SDK v2 for cloud execution? (y/n): " install_azure
if [[ $install_azure == "y" || $install_azure == "Y" ]]; then
    pip install azure-ai-ml azure-identity
    echo "✅ Azure ML SDK installed"
fi

echo "✅ Setup complete!"
echo ""
echo "📋 Quick Start Commands:"
echo "  Local pipeline:  python main.py local --data_path ./path/to/data"
echo "  Azure pipeline:  python main.py azure --data_path azureml://datastores/..."
echo "  Help:           python main.py --help"
echo ""
echo "🔍 Example data structure expected:"
echo "  data/"
echo "  ├── train/"
echo "  │   ├── happy/"
echo "  │   ├── sad/"
echo "  │   ├── surprise/"
echo "  │   └── neutral/"
echo "  ├── validation/"
echo "  │   ├── happy/"
echo "  │   ├── sad/"
echo "  │   ├── surprise/"
echo "  │   └── neutral/"
echo "  └── test/"
echo "      ├── happy/"
echo "      ├── sad/"
echo "      ├── surprise/"
echo "      └── neutral/"