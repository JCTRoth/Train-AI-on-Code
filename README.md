# Train any AI Model on any Code base.

### Training Configuration Module:
   - Specifies configurations for the training process, such as batch size and number of epochs.

### Language Filter Lists Module:
- Contains lists of file postfixes for specific programming languages.
- Used for filtering files based on their postfixes during file listing and loading.

### Training Script:
   - Contains logic for training a machine learning model using the Hugging Face Transformers library.
   - Loads a pre-trained model, sets training arguments, and defines the training process.
   - Iterates over epochs and batches, saving the trained model and tokenizer.

### Interactive Chat Script:
   - Sets up a chat loop for interacting with a pre-trained language model.
   - Loads a pre-trained language model and tokenizer.
   - Defines a function to generate responses based on user input.
   - Continuously prompts the user for input and generates responses until the user types "exit".


