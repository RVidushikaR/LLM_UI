#!/bin/bash

echo "Creating conda environment..."
conda env create -f environment.yml

echo "Activating environment..."
source activate llama_ui

echo "Setting up Node project..."

# Only create package.json if it doesn't exist
if [ ! -f package.json ]; then
    npm init -y
fi

# Add ES module support (THIS FIXES YOUR WARNING)
npm pkg set type=module

echo "Installing Node dependencies..."
npm install express cors

echo "Setup complete!"
