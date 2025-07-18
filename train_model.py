from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch.utils.data import DataLoader
from objects import ClassDataset
import logger
import language_filter_lists
import data_loader as data_loader
import psutil
import os
import torch
import argparse
from logger import get_logger
import training_args


def load_model_with_config(model_key):
    """
    Load a model based on the configuration key from training_args.model_configs
    
    Args:
        model_key: Key for the model configuration in training_args.model_configs
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Use default if provided key doesn't exist
    if model_key not in training_args.model_configs:
        get_logger().warning(f"Model '{model_key}' not found in configs, using default: {training_args.default_model}")
        model_key = training_args.default_model
    
    config = training_args.model_configs[model_key]
    model_name = config['name']
    
    get_logger().info(f"Loading model: {model_name} ({config['description']})")
    
    # Setup quantization if specified
    if config.get('quantize', False):
        get_logger().info(f"Using {config.get('bits', 8)}-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_n_bit=config.get('bits', 8),
            bnb_4bit_compute_dtype=torch.float16 if config.get('bits') == 4 else None,
            bnb_8bit_compute_dtype=torch.float16 if config.get('bits') == 8 else None,
        )
    else:
        quantization_config = None
    
    # Load the model with the appropriate configuration
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quantization_config
    )

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    
    # Set padding token to the eos token if pad token is not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer

def train(list_of_files, model_key=None):
    """
    Train the model on the provided list of files
    
    Args:
        list_of_files: List of file data to train on
        model_key: Key for model configuration in training_args.model_configs
    """
    # Load model and tokenizer using the specified configuration
    model, tokenizer = load_model_with_config(model_key or training_args.default_model)
    
    # Create dataset from input files
    classDataset = ClassDataset(inputDataList=list_of_files)

    # Create a DataLoader for the dataset
    dataloader = DataLoader(classDataset, batch_size=training_args.per_device_train_batch_size, shuffle=True)

    # Set your model to training mode
    model.train()

    # Iterate over epochs
    for epoch in range(training_args.num_train_epochs):
        get_logger().info(f"Epoch {epoch + 1}/{training_args.num_train_epochs}")
        get_logger().info(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

        # Iterate over batches in your DataLoader
        for batch_index, batch in enumerate(dataloader):
            print(f"\tBatch {batch_index + 1}/{len(dataloader)}")

            # Access batch data if needed
            # inputs, labels = batch

    # Create output directory if it doesn't exist
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
        get_logger().info(f"Created output directory: {training_args.output_dir}")
    
    # Save the trained model
    model.save_pretrained(training_args.output_dir)

    # Save the tokenizer
    tokenizer.save_pretrained(training_args.output_dir)
    
    get_logger().info(f"Model and tokenizer saved to {training_args.output_dir}")


def limit_cpu_usage():
    # Get the current process
    process = psutil.Process(os.getpid())

    # Get the number of CPU cores
    cpu_count = psutil.cpu_count()

    # Let one cpu core for the OS
    process.cpu_affinity([cpu_count - 1])

    print("Limit CPU number of cpus for pid: " + str(process.pid) + " to " + str(cpu_count - 1))

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train a Phi model on code')
    
    # Model selection
    parser.add_argument('--model', '-m', 
                       choices=list(training_args.model_configs.keys()),
                       default=training_args.default_model,
                       help='Model configuration to use for training')
    
    # Input directory
    parser.add_argument('--input-dir', '-i',
                       default="/home/jonas/Git/ShoppingListServer/",
                       help='Input directory containing source code files')
    
    # Relative path handling
    parser.add_argument('--relative-path-root', '-r',
                       default="/home/jonas/Git",
                       help='Path to remove to get relative paths')
    
    # CPU usage limitation
    parser.add_argument('--limit-cpu',
                       action='store_true',
                       help='Limit CPU usage to N-1 cores')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Configure logging
    logger.config_logger()

    # Limit CPU usage if requested
    if args.limit_cpu:
        limit_cpu_usage()

    # Display selected model info
    model_config = training_args.model_configs[args.model]
    get_logger().info(f"Selected model: {args.model} - {model_config['description']}")

    # Load dataset
    list_of_files = data_loader.load_dataset_as_list(
        input_dir=args.input_dir,
        removeToGetRelativePath=args.relative_path_root,
        listOfFilePostFixes=language_filter_lists.csharp_postfixes
    )
    
    # Train the model with the selected configuration
    train(list_of_files, model_key=args.model)
