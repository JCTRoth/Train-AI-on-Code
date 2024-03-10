from transformers import AutoTokenizer
from torch.utils.data import Dataset
import training_args


class FileData:
    def __init__(self):
        return
        
    def init_with_data(self, file_name, relative_path, absolute_path, content):
        self._file_name = file_name
        self._relative_path = relative_path
        self._absolute_path = absolute_path
        self._content = content

    @property
    def file_name(self):
        return self._file_name

    @file_name.setter
    def file_name(self, value):
        self._file_name = value

    @property
    def relative_path(self):
        return self._relative_path

    @relative_path.setter
    def relative_path(self, value):
        self._relative_path = value

    @property
    def absolute_path(self):
        return self._absolute_path

    @absolute_path.setter
    def absolute_path(self, value):
        self._absolute_path = value

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        self._content = value

    def __str__(self):
        return f"File Name: {self.file_name}\n" \
               f"Relative Path: {self.relative_path}\n" \
               f"Absolute Path: {self.absolute_path}\n" \
               f"Content: {self.content}"


class ClassDataset(Dataset):
    inputDataList = []

    def __init__(self, inputDataList):
        self.inputDataList = inputDataList
        self.tokenizer = AutoTokenizer.from_pretrained(training_args.model_name_string)

    def __len__(self):
        return len(self.inputDataList)

    def __getitem__(self, idx):
        # Tokenize and encode text data
        encoding = self.tokenizer(str(self.inputDataList[idx]), return_tensors='pt', padding=True, truncation=True)

        # Use Classname as Label
        # Tokenize the string and encode it
        tokenized_input = self.tokenizer(self.inputDataList[idx].file_name, padding="max_length", truncation=True, max_length=128)
        # Your label will be the encoded tokens
        label = tokenized_input["input_ids"]

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label
        }

    def get_tokenizer(self):
        return self.tokenizer
