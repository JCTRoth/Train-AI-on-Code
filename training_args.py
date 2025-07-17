from transformers import TrainingArguments
from transformers import AutoConfig

# Define your training arguments

# Phi model - using Microsoft's Phi-2 model
model_name_string='microsoft/phi-2'

# Training parameters
per_device_train_batch_size = 1
num_train_epochs = 99
logging_dir = './log'
output_dir = './training_output'

# Configuration for Phi model training
Phi_config = AutoConfig.from_pretrained(
    model_name_string,
)