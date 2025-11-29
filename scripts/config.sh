#!/bin/bash
# Setup script for pacman-agent with uv

set -e

echo "Setting up pacman-agent with uv..."

# Create virtual environment with uv
echo "Creating virtual environment..."
uv venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Initialize submodule if not already done
if [ ! -f "pacman-contest/.git" ]; then
    echo "Initializing pacman-contest submodule..."
    git submodule update --init --recursive
fi

# Install all dependencies
echo "Installing dependencies..."
uv pip install -e .

# Install the contest package
echo "Installing contest package..."
cd pacman-contest
uv pip install -e .
cd ..

echo "✓ Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source .venv/bin/activate"
