# Train AI on Code

A flexible system for training and using diffrent models, for example Phi, on your own code repositories. This tool allows you to fine-tune models and interact with them through a conversational interface.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Features](#features)
- [Usage Guide](#usage-guide)
  - [Chat Mode](#chat-mode)
  - [Training Mode](#training-mode)
- [System Architecture](#system-architecture)
- [System Requirements](#system-requirements)

## Installation

### Requirements

- Python 3.8 or higher
- CUDA-compatible GPU recommended for faster inference and training
- At least 16GB RAM for quantized models, more for full-precision models

### Setup

Run the installation script for your platform:

```bash
# On Linux/Mac
chmod +x install.sh
./install.sh

# On Windows
.\install.ps1
```

## Run it

### Inference (Chat Mode)

Run the model in interactive chat mode:

```

## Quick Start

### Train a Model on Your Code

```bash
# Train using the quantized Phi-2 model for memory efficiency
python3 train_model.py --model phi-2-quantized --input-dir /home/user/myproject --relative-path-root /home/user
```

### Chat with Your Trained Model

```bash
# Use your trained model with code-optimized settings
python3 run_model.py --model local-trained --code-mode
```

### View Available Model Configurations

```bash
python3 run_model.py --show-config
```

## Features

- **Multiple Model Options**: Choose between Phi-2 (2.7B), Phi-1.5 (1.3B), or locally trained models
- **Memory Optimization**: 8-bit quantization for reduced memory usage
- **Code-Optimized Mode**: Special settings for better code generation
- **Interactive UI**: Colorized chat interface with dynamic parameter adjustment
- **Conversation Memory**: Maintains context for more coherent interactions
- **Robust Error Handling**: Graceful fallbacks and comprehensive error reporting
- **Flexible Training**: Train on any codebase with custom filtering options

## Usage Guide

### Chat Mode

![Chat Mode](.doc/chat_mode.png "Chat Mode")

Run the model in interactive chat mode:

```bash
python3 run_model.py --model local-trained
```

#### Command-line Options

- `--model`, `-m`: Model configuration to use (phi-2, phi-2-quantized, phi-1_5, local-trained)
- `--temperature`, `-t`: Temperature for text generation (0.1-1.0)
- `--max-length`, `-l`: Maximum token length for generated responses
- `--top-k`, `-k`: Top-k sampling parameter
- `--top-p`, `-p`: Top-p (nucleus) sampling parameter
- `--code-mode`, `-c`: Optimize settings for code generation
- `--fast-code`: Use fast code generation with shorter outputs
- `--debug`, `-d`: Enable debug logging
- `--show-config`, `-s`: Show all available model configurations

#### Interactive Commands

- Type `exit` to end the conversation
- Type `params` to view and change generation parameters
- Type `code-mode` to optimize settings for code generation
- Type `fast-code` for faster code generation with shorter outputs

### Training Mode

Train the model on your code repository:

```bash
python3 train_model.py --model phi-2 --input-dir /path/to/your/code
```

#### Command-line Options

- `--model`, `-m`: Model configuration to use for training
- `--input-dir`, `-i`: Input directory containing source code files
- `--relative-path-root`, `-r`: Path to remove to get relative paths
- `--limit-cpu`: Limit CPU usage to N-1 cores

## System Architecture

### Model Configurations

The system supports the following pre-defined model configurations:

| Configuration | Description | Parameters | Quantization |
|---------------|-------------|------------|-------------|
| `phi-2` | Original Phi-2 model | 2.7B | None |
| `phi-2-quantized` | Memory-efficient Phi-2 | 2.7B | 8-bit |
| `phi-1_5` | Smaller Phi model | 1.3B | None |
| `local-trained` | Your fine-tuned model | Varies | None |

### Core Components

#### Language Filter Module
- Provides file extension filters for different programming languages
- Used to selectively include files during training data collection

#### Training Module
- Built on Hugging Face Transformers library
- Supports multiple model configurations and quantization levels
- Manages the fine-tuning process with customizable parameters
- Automatically saves models and tokenizers to the output directory

#### Chat Interface
- Interactive command-line interface for model interaction
- Supports parameter adjustment during runtime
- Maintains conversation history for context
- Handles model loading and error recovery
- Optimized modes for code generation

## System Requirements

- **Python**: 3.8 or higher
- **RAM**: 16GB+ (32GB+ recommended for full models)
- **GPU**: CUDA-compatible GPU strongly recommended
- **Storage**: At least 10GB for model files

## Advanced Usage Examples

### Code-First Mode with Maximum Context

For optimal code generation with maximum context length:

```bash
python3 run_model.py --model phi-2 --code-mode
```

### Fast Code Generation

For faster response times with code:

```bash
python3 run_model.py --model phi-2 --fast-code
```

### Memory-Efficient Training

For systems with limited memory:

```bash
python3 train_model.py --model phi-2-quantized --input-dir /home/user/myproject --limit-cpu
```

### Debugging Issues

When encountering problems:

```bash
python3 run_model.py --model local-trained --debug
```

### List All Available Models

View configuration details:

```bash
python3 run_model.py --show-config
```


