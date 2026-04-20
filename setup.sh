#!/bin/bash

echo "Creating conda environment..."
conda env create -f environment.yml

echo "Activating environment..."
source activate llama_ui

echo "Installing Node dependencies..."
npm install express cors

echo "Setup complete!"
