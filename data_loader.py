import os
import csv
from objects import FileData
import pandas as pd
from logger import get_logger
import language_filter_lists

def convert_files_to_csv(input_dir, output_csv, removeToGetRelativePath):
        
    try:
        # Open the CSV file in write mode
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            # Define the CSV writer
            csv_writer = csv.writer(csvfile)
            # Write the header row
            csv_writer.writerow(['Relative File Path', 'File Name', 'Content'])

            files_data_list = load_dataset_as_list(input_dir=input_dir, removeToGetRelativePath=removeToGetRelativePath)

            # Writes the list in a csv file
            for file_data in files_data_list:
                    
                    file_name = file_data.file_name
                    relative_path = file_data.relative_path
                    absolute_path = file_data.absolute_path
                    content = file_data.content

                    try:
                            # Write the file path, file name, and content to the CSV file
                            csv_writer.writerow([relative_path, file_name, content])
                            get_logger().info(f"Converted {relative_path} to CSV")
                    except Exception as e:
                        get_logger().error(f"Error writing file {absolute_path}: {str(e)}")
    except Exception as e:
        get_logger().error(f"Error writing to CSV file: {str(e)}")

    print(f"CSV file saved at: {output_csv}")



def load_dataset_as_list(input_dir, removeToGetRelativePath, listOfFilePostFixes):
    # Gets all files in sub folder every where
    # Collects the files in a list
    data_files = list_files(input_dir, removeToGetRelativePath, listOfFilePostFixes)
    data_files = load_content(data_files)

    number_of_rows = len(data_files)
    get_logger().info(f"Loaded {number_of_rows} files from {input_dir}")

    return data_files


def load_content(list_of_files):

    for file_data in list_of_files:
        # Read the content of the file
            try:
                with open(file_data.absolute_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    file_data.content = content
            except Exception as e:
                get_logger().error(f"Error reading file {file_data.absolute_path}: {str(e)}")

    return list_of_files


def list_files(folder_path, removeToGetRelativePath, listOfFilePostFixes):
    """
    List all files contains in the folder and sub-folders.
    :param folder_path:
    :return: list of {'path': list1, 'filename': list2}
    """  

    file_list: list[FileData] = []

    lenToGetRelative = len(removeToGetRelativePath)

    for current_dir, subdirs, files in os.walk(folder_path):
        for file in files:
            if not file.startswith('.'):
                if any(file.endswith(postfix) for postfix in listOfFilePostFixes):

                    relative_path = os.path.join(current_dir, file)
                    absolute_path = os.path.abspath(relative_path)
                    
                    data_file = FileData()

                    data_file.file_name = file
                    data_file.relative_path = absolute_path[lenToGetRelative:]
                    data_file.absolute_path = absolute_path
                    file_list.append(data_file)
    
    return file_list

def get_column_from_file(column_name, csv_file_loaded):
    # Access the specified column by its name
    column_data = csv_file_loaded[column_name].tolist()
    return column_data

def load_csv_file(csv_file):
    # Load and preprocess your CSV file
    csv_file_loaded = pd.read_csv(csv_file)
    return csv_file_loaded


if __name__ == "__main__":
    # Input directory containing your files
    input_dir = "/home/jonas/Git/ShoppingListApp/"

    # Output CSV file path
    output_csv = "/home/jonas/Schreibtisch/file.csv"

    removeToGetRelativePath = "/home/jonas/Git"

    file_list : list[FileData] = load_dataset_as_list(input_dir=input_dir,removeToGetRelativePath=removeToGetRelativePath,
                                                      listOfFilePostFixes=language_filter_lists.csharp_postfixes)


    print()
    # Convert files to CSV
    # convert_files_to_csv(input_dir, output_csv, removeToGetRelativePath)
