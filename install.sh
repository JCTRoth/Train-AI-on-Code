#!/bin/bash

# Update package lists for upgrades and new package installations
sudo apt-get update -y

# Upgrade all installed packages
sudo apt-get upgrade -y

# Install Python3 and Python3 pip
sudo apt-get install -y python3 python3-pip

# Update pip
sudo pip3 install --upgrade pip

# Install required Python libraries
sudo pip3 install torch torchvision pandas
sudo pip3 install transformers[torch]
sudo pip3 install accelerate einops
sudo pip3 install -i https://pypi.org/simple/ bitsandbytes  # Install specific version for 8-bit quantization