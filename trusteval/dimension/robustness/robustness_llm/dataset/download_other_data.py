from datasets import load_dataset
import os
from typing import Dict, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DatasetConfig:
    name: str
    subset: Optional[str] = None
    version: Optional[str] = None

class DatasetLoader:

    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            cache_dir = os.path.join(os.getcwd(), 'data')
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"Using cache directory: {self.cache_dir}")

    def load_single_dataset(self, config: DatasetConfig):
        try:
            kwargs = {"cache_dir": self.cache_dir}
            if config.subset:
                kwargs["name"] = config.subset
            if config.version:
                kwargs["version"] = config.version

            dataset = load_dataset(config.name, **kwargs)
            logger.info(f"Successfully loaded {config.name} dataset")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load {config.name} dataset: {e}")
            return None

    def load_all_datasets(self) -> Dict:
        dataset_configs = [
            DatasetConfig(name="imdb"),
            DatasetConfig(name="race", subset="all"),
            DatasetConfig(name="cnn_dailymail", version="3.0.0")
        ]

        datasets = {}
        for config in dataset_configs:
            dataset = self.load_single_dataset(config)
            if dataset:
                datasets[config.name] = dataset

        return datasets

def get_datasets(cache_dir: Optional[str] = None) -> Dict:
    loader = DatasetLoader(cache_dir)
    return loader.load_all_datasets()