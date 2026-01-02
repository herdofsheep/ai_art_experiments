import os

def imagepaths_list_from_folder(folder_path):
    return [f"{folder_path}/{f}" for f in os.listdir(folder_path) if not f.startswith('.')]