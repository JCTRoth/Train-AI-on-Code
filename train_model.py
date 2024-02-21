from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from objects import ClassDataset
import logger
import data_loader as data_loader

def train(list_of_files):
    # Load the pre-trained model
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)  # Adjust num_labels as needed
    classDataset = ClassDataset(inputDataList=list_of_files)

    # Define your training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        num_train_epochs=999,
        logging_dir='./log',
        output_dir='./training_output'
    )


    # Create a DataLoader for the dataset
    dataloader = DataLoader(classDataset, batch_size=training_args.per_device_train_batch_size, shuffle=True)

    # Set your model to training mode
    model.train()

    # Iterate over epochs
    for epoch in range(training_args.num_train_epochs):
        # Iterate over batches in your DataLoader
        for batch in dataloader:
            # Log training progress for the epoch
            print(f"Epoch {epoch + 1}/{training_args.num_train_epochs}:")

    # Save the trained model
    model.save_pretrained(training_args.output_dir)




if __name__ == "__main__":
    # Input directory containing your files
    input_dir = "/home/developer/Git/logiq-dao/"

    # Output CSV file path
    output_csv = "/home/developer/Schreibtisch/file.csv"

    removeToGetRelativePath = "/home/developer/Git"

    logger.config_logger()

    list_of_files = data_loader.load_dataset_as_list(input_dir=input_dir, removeToGetRelativePath=removeToGetRelativePath)

    train(list_of_files)

    # Convert files to CSV
    # convert_files_to_csv(input_dir, output_csv, removeToGetRelativePath)
