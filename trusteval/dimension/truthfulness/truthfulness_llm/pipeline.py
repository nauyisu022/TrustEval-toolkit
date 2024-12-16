import shutil
import os
from .dynamic import run as run_dynamic
from .dataset_pool import run as run_dataset_pool

def main(base_dir=None, mode='dataset_pool'):
    if mode == 'dataset_pool':
        print("Running dataset pool...")
        run_dataset_pool(base_dir=base_dir)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        shutil.copy(current_dir + '/file_config.json', base_dir + '/final')
    elif mode == 'dynamic':
        print("Running dynamic...")
        run_dynamic(base_dir=base_dir)