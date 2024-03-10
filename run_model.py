from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import training_args
from logger import get_logger


# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(training_args.model_name_string, is_decoder=True)
tokenizer = AutoTokenizer.from_pretrained(training_args.model_name_string,is_decoder=True)

# Set the model to evaluation mode
model.eval()

# Function to generate a response
def generate_response(input_text, model, tokenizer, max_length=50, temperature=0.7, top_k=90, top_p=1.95):
    with torch.no_grad():
        input_ids = torch.tensor([tokenizer.encode(input_text, add_special_tokens=True)])
        output_ids = model.generate(
            input_ids,
            do_sample=True,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
    response_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response_text

# Start a chat loop
while True:
    get_logger().info("Chat with the model. Type 'exit' to end the conversation.")
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    bot_response = generate_response(user_input, model, tokenizer)
    get_logger().info(f"Bot: {bot_response}")
