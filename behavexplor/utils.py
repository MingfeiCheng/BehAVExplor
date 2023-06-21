import os
import shutil

def clear_and_create(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def create_path(output_path):
    clear_and_create(os.path.join(output_path, 'logs'))
    clear_and_create(os.path.join(output_path, 'models'))
    clear_and_create(os.path.join(output_path, 'simulation/records_lgsvl'))
    clear_and_create(os.path.join(output_path, 'simulation/records_apollo'))
    clear_and_create(os.path.join(output_path, 'simulation/scenarios'))