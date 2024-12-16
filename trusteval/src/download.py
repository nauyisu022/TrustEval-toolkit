import gdown
import os
import zipfile
from .utils import colored_print as print


url_mapping = {
    'truthfulness_llm': 'https://drive.google.com/uc?id=10KfKoEv-wSw9Vwoq-AWJqsCVs0ABySrj',
    'truthfulness_vlm': 'https://drive.google.com/uc?id=1vBstxYHfc6QvtlkOtsAb-Ui_KNzuUc1U',
    'fairness_llm' : "https://drive.google.com/uc?id=1tgJcwTOtC-1B7KKB_SrOzJhaTAULzM9u",
    'ethics_llm':"https://drive.google.com/uc?id=1TEodgspBBPDhPifPInqS3vmABV0n2FlX",
    'fairness_t2i': 'https://drive.google.com/uc?id=1YrfzkHmqLsIJjeCg3mu2auf3lzqnU3nO',
    'privacy_t2i': 'https://drive.google.com/uc?id=1NMtGtbqbP_eWa3XOCUbBW2HAa_kETLSh',
    'robustness_t2i': 'https://drive.google.com/uc?id=1ffRa6c507SFQ-6jrRpKcwAa9tgfGcQx-',
    'safety_vlm': 'https://drive.google.com/uc?id=1cR5dEQbc0nTYkXGhksm_p_GWnNwr2QHJ',
    'safety_t2i': 'https://drive.google.com/uc?id=1LhnRbDI1M5kcCWLwjZdzt8PlCU0b9HCC',
    'advanced_ai_risk': 'https://drive.google.com/uc?id=118l7-vheNQljF3TfULmwMVaoWdDYRZSC',
    'privacy_vlm':'https://drive.google.com/uc?id=104lNneFNkuF_SHv-CpebBW5BtSxgffgo',
    'ethics_vlm':'https://drive.google.com/uc?id=1fHWR4a0fVwk9t9Z9yxOUydOQgfOHbN0H',
    'fairness_vlm':'https://drive.google.com/uc?id=1pfLLxQW_2slVhh8a_0vtygkz3XlNvGm8'
}

def download_dataset(section, output_path):
    url = url_mapping[section]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created directory: {output_path}")

    output_file = os.path.join(output_path, 'tmp.zip')

    print(f"Downloading dataset for section: {section}")
    gdown.download(url, output_file, quiet=False)

    print(f"Extracting dataset to: {output_path}")
    with zipfile.ZipFile(output_file, 'r') as zip_ref:
        zip_ref.extractall(output_path)

    print(f"Removing temporary zip file: {output_file}")
    os.remove(output_file)
    print(f"Dataset for section '{section}' has been downloaded and extracted to '{output_path}'",color="GREEN")