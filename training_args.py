from transformers import TrainingArguments
from transformers import AutoConfig

# Define your training arguments

# Phi model configurations - allowing to switch between different models
model_configs = {
    'phi-2': {
        'name': 'microsoft/phi-2',
        'description': 'Original Phi-2 model (2.7B parameters)',
        'quantize': False
    },
    'phi-2-quantized': {
        'name': 'microsoft/phi-2',
        'description': 'Quantized Phi-2 model (8-bit for memory efficiency)',
        'quantize': True,
        'bits': 8
    },
    'phi-1_5': {
        'name': 'microsoft/phi-1_5',
        'description': 'Smaller Phi 1.5 model (1.3B parameters)',
        'quantize': False
    },
    'local-trained': {
        'name': './training_output',
        'description': 'Locally fine-tuned Phi model',
        'quantize': False
    }
}

# Default model to use
default_model = 'phi-2'
model_name_string = model_configs[default_model]['name']

# Training parameters
per_device_train_batch_size = 1
num_train_epochs = 10  # Adjust this based on your dataset size and convergence and hardware capabilities
logging_dir = './log'
output_dir = './training_output'

# Additional training parameters
learning_rate = 5e-5
weight_decay = 0.01
max_grad_norm = 1.0
max_seq_length = 512  # Maximum sequence length for tokenization
logging_steps = 10
save_steps = 50
warmup_ratio = 0.1  # Percentage of steps for warmup
fp16 = False  # Whether to use mixed precision training

# Advanced training options
gradient_accumulation_steps = 4  # Simulate larger batch sizes with limited GPU memory
evaluation_steps = 100  # How often to evaluate the model
early_stopping_patience = 3  # Stop training if no improvement for this many evaluations
early_stopping_threshold = 0.01  # Minimum change to qualify as improvement

# Configuration for Phi model training
Phi_config = AutoConfig.from_pretrained(
    model_name_string,
)