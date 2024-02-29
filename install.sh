#!/bin/bash

# Update package lists for upgrades and new package installations
sudo apt-get update -y

# Upgrade all installed packages
sudo apt-get upgrade -y

# Install Python3 and Python3 pip
sudo apt-get install -y python3 python3-pip

# Update pip
pip3 install --upgrade pip

# Install required Python libraries
pip3 install torch transformers torchvision

