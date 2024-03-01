from transformers import TrainingArguments

# Define your training arguments


B1_E99 = TrainingArguments(
    per_device_train_batch_size=1,
    num_train_epochs=99,
    logging_dir='./log',
    output_dir='./training_output'
)