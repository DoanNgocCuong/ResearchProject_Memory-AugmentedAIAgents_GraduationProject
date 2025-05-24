#!/bin/bash

echo "Setting up Python virtual environment for FlashRAG..."

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

echo "Setup completed successfully!"
echo "To activate the environment, run: source venv/bin/activate" 