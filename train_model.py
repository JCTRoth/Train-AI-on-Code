from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from objects import ClassDataset
import logger
import language_filter_lists
import data_loader as data_loader
import psutil
import os
from logger import get_logger
import training_args


def train(list_of_files):
    # Load the pre-trained model without quantization
    get_logger().info(f"Loading model: {training_args.model_name_string}")
    model = AutoModelForCausalLM.from_pretrained(
        training_args.model_name_string,
        trust_remote_code=True,  # Required for Phi model
        device_map="auto",       # Use accelerate for optimal device placement
    )

    # Create tokenizer directly instead of through ClassDataset
    tokenizer = AutoTokenizer.from_pretrained(
        training_args.model_name_string,
        trust_remote_code=True,  # Required for Phi model
    )
    
    # Set padding token to the eos token if pad token is not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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



if __name__ == "__main__":
    # Input directory containing your files
    input_dir = "/home/jonas/Git/ShoppingListServer/"

    # Output CSV file path
    # output_csv = "/home/jonas/Schreibtisch/file.csv"

    removeToGetRelativePath = "/home/jonas/Git"

    logger.config_logger()

    # limit_cpu_usage()

    list_of_files = data_loader.load_dataset_as_list(input_dir=input_dir,
    removeToGetRelativePath=removeToGetRelativePath,
    listOfFilePostFixes=language_filter_lists.csharp_postfixes)
    
    train(list_of_files)

    # Convert files to CSV
    # convert_files_to_csv(input_dir, output_csv, removeToGetRelativePath)
