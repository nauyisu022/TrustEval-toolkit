import os
import requests
from tqdm import tqdm
import zipfile

urls = [
    "http://images.cocodataset.org/zips/train2014.zip",
    "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
    "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
    "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Train_mscoco.zip",
    "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip"
]

def download_file(url, folder):
    local_filename = os.path.join(folder, url.split('/')[-1])
    headers = {}
    mode = 'wb'

    if os.path.exists(local_filename):
        mode = 'ab'
        headers['Range'] = f'bytes={os.path.getsize(local_filename)}-'

    with requests.get(url, stream=True, headers=headers) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))

        with open(local_filename, mode) as f, tqdm(
            desc=local_filename,
            initial=os.path.getsize(local_filename) if mode == 'ab' else 0,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in r.iter_content(chunk_size=8192):
                size = f.write(chunk)
                progress_bar.update(size)
    return local_filename

def extract_zip(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file in zip_ref.namelist():
            if file.endswith('/'):  
                continue
            parts = file.split('/')
            new_path = os.path.join(extract_path, *parts)
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            with zip_ref.open(file) as source, open(new_path, "wb") as target:
                target.write(source.read())

    for root, dirs, files in os.walk(extract_path, topdown=False):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if not os.listdir(dir_path):  
                os.rmdir(dir_path)

def main(target_folder = "MSCOCO"):
    os.makedirs(target_folder, exist_ok=True)
    for url in urls:
        print(f"Downloading {url}")
        zip_file = download_file(url, target_folder)
        print(f"Extracting {zip_file}")
        extract_zip(zip_file, target_folder)
        os.remove(zip_file) 
        print(f"Finished processing {url}")

    print("All files have been downloaded and extracted.")

if __name__ == "__main__":
    main()