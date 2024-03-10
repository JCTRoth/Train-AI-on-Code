from transformers import TrainingArguments
from transformers import DistilBertForMaskedLM, DistilBertConfig

# Define your training arguments

# model_name_string="distilbert/distilbert-base-uncased"
model_name_string='distilbert/distilbert-base-uncased-distilled-squad'

# B1_E99 = TrainingArguments(
#     per_device_train_batch_size=1,
#    num_train_epochs=99,
#    logging_dir='./log',
#    output_dir='./training_output'
#)

B1_E99 = DistilBertConfig(
    vocab_size=30000,
    max_position_embeddings=514,
    per_device_train_batch_size=1,
    num_train_epochs=99,
    logging_dir='./log',
    output_dir='./training_output'
)