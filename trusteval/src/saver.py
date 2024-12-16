import os
import json
import csv
import yaml

class Saver:
    def __init__(self, base_folder_path):
        self.base_folder_path = base_folder_path

    def _get_full_path(self, file_path):
        if os.path.isabs(file_path):
            return file_path
        return os.path.join(self.base_folder_path, file_path)
    
    def list_files(self, directory):
        return os.listdir(self._get_full_path(directory))

    def exists(self, file_path):
        return os.path.exists(self._get_full_path(file_path))

    def ensure_directory_exists(self, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def save_json(self, data, file_path):
        """
        Save data to a JSON file.
        
        Args:
            data: Data to save.
            file_path: The relative or absolute path of the file.
        """
        full_path = self._get_full_path(file_path)
        self.ensure_directory_exists(full_path)
        
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    def save_csv(self, data, file_path, headers=None):
        """
        Save data to a CSV file.
        
        Args:
            data: Data to save.
            file_path: The relative or absolute path of the file.
            headers: Optional headers for CSV file.
        """
        full_path = self._get_full_path(file_path)
        self.ensure_directory_exists(full_path)
        
        with open(full_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if headers:
                writer.writerow(headers)
            writer.writerows(data)

    def save_yaml(self, data, file_path):
        """
        Save data to a YAML file.
        
        Args:
            data: Data to save.
            file_path: The relative or absolute path of the file.
        """
        full_path = self._get_full_path(file_path)
        self.ensure_directory_exists(full_path)
        
        with open(full_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True)

    def read_file(self, file_path):
        """
        Read a file based on its extension and return the content.
        
        Args:
            file_path: The relative or absolute path of the file.
        
        Returns:
            Parsed content of the file.
        """
        _, ext = os.path.splitext(file_path)
        file_path = self._get_full_path(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            if ext == '.json':
                return json.load(f)
            elif ext == '.csv':
                reader = csv.reader(f)
                return list(reader)
            elif ext in ('.yaml', '.yml'):
                return yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported file extension: {ext}")

    def copy_file(self, source_file_path, target_file_path):
        """
        Copy content from source file to target file.
        
        Args:
            source_file_path: The relative or absolute path of the source file.
            target_file_path: The relative or absolute path of the target file.
        """
        data = self.read_file(source_file_path)
        _, ext = os.path.splitext(target_file_path)
        
        if ext == '.json':
            self.save_json(data, target_file_path)
        elif ext == '.csv':
            self.save_csv(data, target_file_path)
        elif ext in ('.yaml', '.yml'):
            self.save_yaml(data, target_file_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def save_data(self, data, target_file_path):
        """
        Save data to the target file, determining format based on the file extension.
        
        Args:
            data: The data to save.
            target_file_path: The relative or absolute path of the file.
        """
        _, ext = os.path.splitext(target_file_path)
        print(f"Saving data to {target_file_path}")
        if ext == '.json':
            self.save_json(data, target_file_path)
        elif ext == '.csv':
            self.save_csv(data, target_file_path)
        elif ext in ('.yaml', '.yml'):
            self.save_yaml(data, target_file_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
