import os
import shutil


def clear_folder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        os.unlink(file_path)


def copy_files(source, destination):
    for file_name in os.listdir(source):
        full_file_name = os.path.join(source, file_name)
        if (os.path.isfile(full_file_name)):
            shutil.copy(full_file_name, destination)
