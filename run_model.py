from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import training_args
from logger import get_logger
import os


# Load the model and tokenizer from the trained output
model_path = "./training_output"
if os.path.exists(model_path):
    # Load model with basic configuration
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Load tokenizer and set proper padding
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    get_logger().info(f"Loaded model from {model_path}")
else:
    get_logger().info(f"Trained model not found, using default model")
    model = AutoModelForCausalLM.from_pretrained(training_args.model_name_string)
    tokenizer = AutoTokenizer.from_pretrained(training_args.model_name_string)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

# Set the model to evaluation mode
model.eval()

# Function to generate a response
def generate_response(input_text, model, tokenizer, max_length=50, temperature=0.8, top_k=40, top_p=0.9):
    with torch.no_grad():
        # Create inputs with proper attention mask
        inputs = tokenizer(input_text, return_tensors="pt", padding=True)
        
        # Generate response
        output_ids = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            do_sample=True,
            max_length=max_length + len(input_text),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )
        
    # Decode the response
    response_text = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response_text

# Start a chat loop
def start_conversation():
    get_logger().info("Chat with the trained Phi model. Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break
        
        # Simple prompt format
        prompt = f"Question: {user_input}\nAnswer:"
        
        # Generate response
        response = generate_response(prompt, model, tokenizer)
        
        print(f"\nAI: {response}")

if __name__ == "__main__":
    start_conversation()
