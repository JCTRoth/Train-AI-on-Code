from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_scheduler
from torch.utils.data import DataLoader, random_split
from objects import ClassDataset
import logger
import language_filter_lists
import data_loader as data_loader
import psutil
import os
import sys
import torch
import argparse
import math
from tqdm.auto import tqdm
from logger import get_logger
import training_args
from training_summary import TrainingSummary
from model_evaluation import ModelEvaluator, EarlyStopping
from hardware_check import check_ram_requirements, print_system_info, limit_cpu_usage


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
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Load model and tokenizer using the specified configuration
    model, tokenizer = load_model_with_config(model_key or training_args.default_model)
    
    # Create dataset from input files
    full_dataset = ClassDataset(inputDataList=list_of_files)
    
    # Split dataset into training and validation sets (90% train, 10% validation)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    get_logger().info(f"Dataset split into {train_size} training samples and {val_size} validation samples")

    # Create a DataLoader for the training dataset
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=training_args.per_device_train_batch_size, 
        shuffle=True
    )

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay
    )
    
    # Initialize learning rate scheduler
    # Adjust total steps to account for gradient accumulation
    total_steps = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps) * training_args.num_train_epochs
    warmup_steps = int(total_steps * training_args.warmup_ratio)
    
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Set model to training mode
    model.train()
    
    # Initialize model evaluator
    evaluator = ModelEvaluator(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=val_dataset,
        max_length=training_args.max_seq_length,
        batch_size=training_args.per_device_train_batch_size
    )
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=training_args.early_stopping_patience,
        threshold=training_args.early_stopping_threshold
    )
    
    # Training metrics
    total_loss = 0
    logging_loss = 0
    global_step = 0
    optimization_step = 0
    best_eval_loss = float('inf')
    
    # Create output directory if it doesn't exist
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
        get_logger().info(f"Created output directory: {training_args.output_dir}")
    
    # Create checkpoints directory
    checkpoint_dir = os.path.join(training_args.output_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        get_logger().info(f"Created checkpoints directory: {checkpoint_dir}")
    
    # Training loop
    get_logger().info("Starting training...")
    progress_bar = tqdm(range(total_steps), desc="Training")
    
    # Initialize training summary tracker
    training_summary = TrainingSummary(training_args.output_dir)

    # Iterate over epochs
    for epoch in range(training_args.num_train_epochs):
        epoch_loss = 0
        get_logger().info(f"Epoch {epoch + 1}/{training_args.num_train_epochs}")
        get_logger().info(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

        # Iterate over batches in DataLoader
        for batch_index, batch in enumerate(train_dataloader):
            try:
                # Process the batch data - the batch is already a dictionary with tensors
                if isinstance(batch, dict):
                    # Move inputs to the same device as the model
                    device = next(model.parameters()).device
                    
                    # Handle the case where labels might be a list
                    processed_inputs = {}
                    for k, v in batch.items():
                        if k == 'labels' and isinstance(v, list):
                            # Convert list to tensor
                            processed_inputs[k] = torch.tensor(v).to(device)
                        else:
                            processed_inputs[k] = v.to(device)
                    
                    inputs = processed_inputs
                else:
                    # Handle individual batch items
                    input_texts = [item.content for item in batch]
                    
                    # Tokenize inputs
                    inputs = tokenizer(
                        input_texts, 
                        return_tensors="pt", 
                        padding="max_length",
                        truncation=True,
                        max_length=training_args.max_seq_length
                    )
                    
                    # Move inputs to the same device as the model
                    device = next(model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # Set labels for causal language modeling (input_ids serve as labels for next token prediction)
                    inputs["labels"] = inputs["input_ids"].clone()
                
                # Forward pass
                outputs = model(**inputs)
                loss = outputs.loss
                
                # Scale loss for gradient accumulation
                loss = loss / training_args.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Calculate loss for logging (scale back up for consistent reporting)
                current_loss = loss.item() * training_args.gradient_accumulation_steps
                epoch_loss += current_loss
                total_loss += current_loss
                
                # Gradient accumulation - only update weights after accumulating gradients
                if (batch_index + 1) % training_args.gradient_accumulation_steps == 0 or batch_index == len(train_dataloader) - 1:
                    # Clip gradients to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)
                    
                    # Update weights
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                    # Increment optimization step
                    optimization_step += 1
                    
                    # Update progress bar
                    progress_bar.update(1)
                
                # Increment global step
                global_step += 1
                
                # Logging
                if optimization_step > 0 and optimization_step % training_args.logging_steps == 0:
                    avg_loss = (total_loss - logging_loss) / training_args.logging_steps
                    current_lr = lr_scheduler.get_last_lr()[0]
                    get_logger().info(f"Step {optimization_step}: Average loss = {avg_loss:.4f}, LR = {current_lr:.2e}")
                    
                    # Add metrics to training summary
                    training_summary.add_metric(
                        step=optimization_step,
                        epoch=epoch + (batch_index / len(train_dataloader)),
                        loss=avg_loss,
                        learning_rate=current_lr
                    )
                    
                    logging_loss = total_loss
                
                # Model evaluation
                if optimization_step > 0 and optimization_step % training_args.evaluation_steps == 0:
                    # Evaluate model
                    eval_results = evaluator.evaluate()
                    eval_loss = eval_results["loss"]
                    
                    # Check for improvement and save best model
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        
                        # Save best model
                        best_model_dir = os.path.join(training_args.output_dir, "best_model")
                        if not os.path.exists(best_model_dir):
                            os.makedirs(best_model_dir)
                            
                        model.save_pretrained(best_model_dir)
                        tokenizer.save_pretrained(best_model_dir)
                        get_logger().info(f"New best model saved with validation loss: {best_eval_loss:.4f}")
                    
                    # Check for early stopping
                    if early_stopping(eval_loss):
                        get_logger().info("Early stopping triggered. Ending training.")
                        break
                
                # Save checkpoint
                if optimization_step > 0 and optimization_step % training_args.save_steps == 0:
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{optimization_step}")
                    
                    if not os.path.exists(checkpoint_path):
                        os.makedirs(checkpoint_path)
                    
                    # Save model checkpoint
                    model.save_pretrained(checkpoint_path)
                    tokenizer.save_pretrained(checkpoint_path)
                    
                    # Save optimizer and scheduler state
                    torch.save({
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'global_step': global_step,
                        'loss': avg_loss
                    }, os.path.join(checkpoint_path, "training_state.pt"))
                    
                    get_logger().info(f"Saved checkpoint at step {optimization_step}")
                
                # Print batch progress every 10 batches
                if batch_index % 10 == 0:
                    tqdm.write(f"Batch {batch_index + 1}/{len(train_dataloader)}, Loss: {current_loss:.4f}")
                
            except Exception as e:
                get_logger().error(f"Error processing batch {batch_index}: {str(e)}")
                continue
        
        # Check if early stopping was triggered
        if early_stopping.should_stop:
            break
            
        # End of epoch summary
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        get_logger().info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Save end-of-epoch model
        epoch_dir = os.path.join(training_args.output_dir, f"epoch-{epoch + 1}")
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)
        
        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)
        get_logger().info(f"Saved model for epoch {epoch + 1}")
    
    # Load the best model if available
    best_model_dir = os.path.join(training_args.output_dir, "best_model")
    if os.path.exists(best_model_dir):
        get_logger().info(f"Loading best model from {best_model_dir}")
        model = AutoModelForCausalLM.from_pretrained(best_model_dir)
    
    # Save the final model
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    
    # Generate and save training summary
    summary = training_summary.generate_training_summary()
    get_logger().info(f"Training completed in {summary.get('training_duration', 'unknown time')}")
    get_logger().info(f"Final model saved to {training_args.output_dir}")
    
    return model, tokenizer


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
    
    # Training parameters
    parser.add_argument('--learning-rate', '--lr',
                       type=float,
                       default=training_args.learning_rate,
                       help='Learning rate for training')
    
    parser.add_argument('--epochs', '-e',
                       type=int,
                       default=training_args.num_train_epochs,
                       help='Number of training epochs')
    
    parser.add_argument('--batch-size', '-b',
                       type=int,
                       default=training_args.per_device_train_batch_size,
                       help='Per device batch size')
    
    parser.add_argument('--gradient-accumulation', '-ga',
                       type=int,
                       default=training_args.gradient_accumulation_steps,
                       help='Number of steps to accumulate gradients (simulates larger batch sizes)')
    
    parser.add_argument('--max-seq-length', '-ml',
                       type=int,
                       default=training_args.max_seq_length,
                       help='Maximum sequence length for tokenization')
    
    parser.add_argument('--disable-early-stopping',
                       action='store_true',
                       help='Disable early stopping (train for all epochs)')
    
    # Resource usage
    parser.add_argument('--limit-cpu',
                       action='store_true',
                       help='Limit CPU usage to N-1 cores')
    
    # Advanced options
    parser.add_argument('--fp16',
                       action='store_true',
                       help='Use mixed precision training (fp16)')
    
    parser.add_argument('--force',
                       action='store_true',
                       help='Force training even if RAM check fails')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Configure logging
    logger.config_logger()

    # Print system information
    print_system_info()

    # Limit CPU usage if requested
    if args.limit_cpu:
        limit_cpu_usage()

    # Display selected model info
    model_config = training_args.model_configs[args.model]
    get_logger().info(f"Selected model: {args.model} - {model_config['description']}")
    
    # Update training arguments with command line arguments
    training_args.learning_rate = args.learning_rate
    training_args.num_train_epochs = args.epochs
    training_args.per_device_train_batch_size = args.batch_size
    training_args.gradient_accumulation_steps = args.gradient_accumulation
    training_args.max_seq_length = args.max_seq_length
    training_args.fp16 = args.fp16
    
    # Disable early stopping if requested
    if args.disable_early_stopping:
        training_args.early_stopping_patience = float('inf')
    
    # Display training configuration
    get_logger().info(f"Training configuration:")
    get_logger().info(f"  Learning rate: {training_args.learning_rate}")
    get_logger().info(f"  Epochs: {training_args.num_train_epochs}")
    get_logger().info(f"  Batch size: {training_args.per_device_train_batch_size}")
    get_logger().info(f"  Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    get_logger().info(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    get_logger().info(f"  Max sequence length: {training_args.max_seq_length}")
    get_logger().info(f"  Mixed precision (fp16): {training_args.fp16}")
    
    # Check if enough RAM is available
    has_enough_ram, ram_message = check_ram_requirements(args.model, training_args)
    get_logger().info(ram_message)
    
    if not has_enough_ram and not args.force:
        get_logger().error("Training aborted due to insufficient RAM. Use --force to override.")
        print("\nERROR: Insufficient RAM available for training.")
        print("You can use --force to run anyway, but this may cause system instability.")
        print("See the recommendations above to optimize memory usage.")
        sys.exit(1)
    elif not has_enough_ram and args.force:
        get_logger().warning("Training proceeding despite insufficient RAM (--force option used)")
        print("\nWARNING: Proceeding with training despite insufficient RAM (--force option used)")
        print("This may cause system instability or out-of-memory errors")

    # Load dataset
    list_of_files = data_loader.load_dataset_as_list(
        input_dir=args.input_dir,
        removeToGetRelativePath=args.relative_path_root,
        listOfFilePostFixes=language_filter_lists.csharp_postfixes
    )
    
    get_logger().info(f"Loaded {len(list_of_files)} code files for training")
    
    # Train the model with the selected configuration
    
    # Train the model with the selected configuration
    train(list_of_files, model_key=args.model)
