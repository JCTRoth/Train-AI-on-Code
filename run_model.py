from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import argparse
import training_args
from logger import get_logger
import os


def load_model_with_config(model_key):
    """
    Load a model based on the configuration key from training_args.model_configs
    
    Args:
        model_key: Key for the model configuration in training_args.model_configs
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Check if the key exists in our configurations
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
    
    # Check for local model first if using local-trained
    if model_key == 'local-trained':
        if not os.path.exists(model_name) or not os.path.isdir(model_name):
            get_logger().warning(f"Local trained model not found at {model_name}, falling back to default model")
            model_key = training_args.default_model
            config = training_args.model_configs[model_key]
            model_name = config['name']
        else:
            get_logger().info(f"Loading local trained model from {model_name}")
    
    # Load the model with the appropriate configuration
    try:
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
    except Exception as e:
        get_logger().error(f"Error loading model {model_name}: {str(e)}")
        if model_key == 'local-trained':
            get_logger().warning("Falling back to default model")
            model_key = training_args.default_model
            config = training_args.model_configs[model_key]
            model_name = config['name']
            
            # Try loading the default model instead
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map="auto",
                quantization_config=None
            )
            
            # Create tokenizer for default model
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
        else:
            raise
    
    # Set padding token to the eos token if pad token is not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer

# Function to generate a response
def generate_response(input_text, model, tokenizer, max_length=50, temperature=0.8, top_k=40, top_p=0.9):
    try:
        with torch.no_grad():
            # Create inputs with proper attention mask
            inputs = tokenizer(input_text, return_tensors="pt", padding=True)
            
            # Move inputs to the same device as the model
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate response
            output_ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                do_sample=True,
                max_length=min(max_length + len(inputs["input_ids"][0]), 2048),  # Respect model's max context size
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2
            )
            
        # Decode the response, only taking the newly generated tokens
        response_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response_text
    except Exception as e:
        get_logger().error(f"Error in generate_response: {str(e)}")
        return "Sorry, I had trouble generating a response. Let's try again."

# Start a chat loop
def start_conversation(model, tokenizer, generation_params=None):
    get_logger().info("Chat with the Phi model. Type 'exit' to end the conversation.")
    print("\n=== Phi Model Chat ===")
    print("Type 'exit' to end the conversation.")
    print("Type 'params' to view/change generation parameters.")
    print("Type 'code-mode' to optimize settings for code generation.")
    print("Type 'fast-code' for faster code generation with shorter outputs.")
    
    # Default generation parameters
    params = {
        'max_length': 512,      # Outputs
        'temperature': 0.5,     # Balanced creativity
        'top_k': 100,           # Diverse sampling
        'top_p': 0.95           # Broader nucleus sampling
    }
    
    # Code optimized parameters
    code_params = {
        'max_length': 512,
        'temperature': 0.3,
        'top_k': 40,
        'top_p': 0.2
    }
    
    # Fast code mode parameters (even faster response time)
    fast_code_params = {
        'max_length': 256,      # Shorter outputs for faster generation
        'temperature': 0.4,     # More balanced temperature
        'top_k': 30,            # More focused sampling
        'top_p': 0.85           # More focused sampling for speed
    }
    
    # Update with any provided parameters
    if generation_params:
        params.update(generation_params)
    
    # System instruction to encourage code output
    system_instruction = "You are a coding assistant that always provides complete, working code examples with explanations. When asked programming questions, respond with full implementation code in the appropriate programming language. For Java questions, include necessary imports and complete class definitions. Ensure your code is properly formatted within triple backticks."
    # Keep track of conversation history
    conversation_history = [f"System: {system_instruction}"]
    
    while True:
        user_input = input("\n\033[1mYou:\033[0m ")
        
        # Handle special commands
        if user_input.lower() == "exit":
            print("\nGoodbye!")
            break
        
        if user_input.lower() == "params":
            print("\nCurrent generation parameters:")
            for param, value in params.items():
                print(f"  {param}: {value}")
            
            # Allow changing parameters
            print("\nChange parameters? (yes/no)")
            if input().lower().startswith('y'):
                for param in params:
                    new_value = input(f"New value for {param} (current: {params[param]}, press Enter to keep): ")
                    if new_value:
                        try:
                            # Convert string to appropriate type
                            if isinstance(params[param], int):
                                params[param] = int(new_value)
                            elif isinstance(params[param], float):
                                params[param] = float(new_value)
                            else:
                                params[param] = new_value
                            print(f"  {param} updated to {params[param]}")
                        except ValueError:
                            print(f"  Invalid value for {param}, keeping {params[param]}")
            continue
        
        if user_input.lower() == "code-mode":
            # Switch to code-optimized parameters
            params.update(code_params)
            print("\n\033[1mSwitched to code generation mode\033[0m")
            print("Parameters updated for optimal code generation:")
            for param, value in params.items():
                print(f"  {param}: {value}")
            continue
            
        if user_input.lower() == "fast-code":
            # Switch to faster code-optimized parameters
            params.update(fast_code_params)
            print("\n\033[1mSwitched to fast code generation mode\033[0m")
            print("Parameters updated for faster code generation:")
            for param, value in params.items():
                print(f"  {param}: {value}")
            continue
        
        # Add to conversation history
        conversation_history.append(f"User: {user_input}")
        
        # Simple prompt format with limited history to fit in context window
        # Only include the last 5 exchanges to avoid context overflow
        recent_history = conversation_history[-10:]
        prompt = "\n".join(recent_history) + "\nAI:"
        
        # Generate response with specified parameters
        response = generate_response(
            prompt, 
            model, 
            tokenizer, 
            max_length=params['max_length'],
            temperature=params['temperature'],
            top_k=params['top_k'],
            top_p=params['top_p']
        )
        
        # Print response with styling
        print(f"\n\033[1mAI:\033[0m {response}")
        
        # Add response to history
        conversation_history.append(f"AI: {response}")

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Chat with a Phi model')
    
    # Model selection
    parser.add_argument('--model', '-m',
                       choices=list(training_args.model_configs.keys()),
                       default='local-trained',
                       help='Model configuration to use for conversation')
    
    # Generation parameters
    parser.add_argument('--temperature', '-t',
                       type=float,
                       default=0.8,
                       help='Temperature for text generation (0.1-1.0)')
    
    parser.add_argument('--max-length', '-l',
                       type=int,
                       default=50,
                       help='Maximum token length for generated responses')
    
    parser.add_argument('--top-k', '-k',
                       type=int,
                       default=40,
                       help='Top-k sampling parameter')
    
    parser.add_argument('--top-p', '-p',
                       type=float,
                       default=0.9,
                       help='Top-p (nucleus) sampling parameter')
    
    # Advanced parameters
    parser.add_argument('--debug', '-d',
                       action='store_true',
                       help='Enable debug logging')
    
    parser.add_argument('--show-config', '-s',
                       action='store_true',
                       help='Show all available model configurations')
    
    parser.add_argument('--code-mode', '-c',
                       action='store_true',
                       help='Start in code generation mode with optimized parameters')
                       
    parser.add_argument('--fast-code',
                       action='store_true',
                       help='Start in fast code generation mode with shorter outputs')
    
    return parser.parse_args()

if __name__ == "__main__":
    import logger
    # Configure logging
    logger.config_logger()
    
    # Parse command-line arguments
    args = parse_args()
    
    # Show available model configurations if requested
    if args.show_config:
        print("\nAvailable model configurations:")
        for key, config in training_args.model_configs.items():
            print(f"  {key}:")
            print(f"    Description: {config['description']}")
            print(f"    Path: {config['name']}")
            print(f"    Quantized: {config.get('quantize', False)}")
            if config.get('quantize', False):
                print(f"    Bits: {config.get('bits', 8)}")
            print()
        exit(0)
    
    # Enable debug logging if requested
    if args.debug:
        import logging
        logger.get_logger().setLevel(logging.DEBUG)
        get_logger().debug("Debug logging enabled")
    
    # Print selected model info
    model_config = training_args.model_configs[args.model]
    get_logger().info(f"Selected model: {args.model} - {model_config['description']}")
    
    try:
        # Load the model with the specified configuration
        model, tokenizer = load_model_with_config(args.model)
        
        # Set model to evaluation mode
        model.eval()
        
        # Setup generation parameters
        generation_params = {
            'max_length': args.max_length,
            'temperature': args.temperature,
            'top_k': args.top_k,
            'top_p': args.top_p
        }
        
        # Apply code mode if requested
        if args.code_mode:
            get_logger().info("Starting in code generation mode")
            # Apply code-optimized parameters
            generation_params.update({
                'max_length': 512,      # Reasonable length for code
                'temperature': 0.3,     # Deterministic but not too slow
                'top_k': 40,            # Balanced focused sampling
                'top_p': 0.9            # Slightly more focused for speed
            })
            
        # Apply fast code mode if requested
        if args.fast_code:
            get_logger().info("Starting in fast code generation mode")
            # Apply faster code-optimized parameters
            generation_params.update({
                'max_length': 256,      # Shorter outputs for faster generation
                'temperature': 0.4,     # More balanced temperature
                'top_k': 30,            # More focused sampling
                'top_p': 0.85           # More focused sampling for speed
            })
        
        # Start the conversation
        start_conversation(model, tokenizer, generation_params)
    except KeyboardInterrupt:
        print("\nChat session terminated by user.")
    except Exception as e:
        get_logger().error(f"Error during model operation: {str(e)}")
        import traceback
        if args.debug:
            traceback.print_exc()
