import kagglehub
import shutil
from pathlib import Path


Dataset_url = "adityajn105/flickr8k"
image_dir = "jpg"
captions_file = "captions.txt"

KAGGLE_image_dirs = ["Images", "jpg"]
KAGGLE_captions_dirs = ["captions.txt", "captions/captions.txt"]


def setup_dataset():
    data_dir = Path(__file__).resolve().parent
    image_folder_path = data_dir / image_dir
    labels_file_path = data_dir / captions_file

    if image_folder_path.exists() and labels_file_path.exists():
        print(f"Dataset already exists. Loading locally from '{data_dir}'.")
        return

    print(f"Dataset not found locally. Downloading to {data_dir}")
    data_dir.mkdir(parents=True, exist_ok=True)
    kaggle_cache_path = Path(kagglehub.dataset_download(Dataset_url))

    for src in KAGGLE_image_dirs:
        src_path = kaggle_cache_path / src
        if src_path.exists():
            shutil.copytree(src_path, image_folder_path, dirs_exist_ok=True)
            break
    else:
        raise FileNotFoundError(
            "Could not find the Flickr8k image folder in the Kaggle cache.")

    for src in KAGGLE_captions_dirs:
        src_path = kaggle_cache_path / src
        if src_path.exists():
            shutil.copy2(src_path, labels_file_path)
            break
    else:
        raise FileNotFoundError(
            f"Could not find {captions_file} in the Kaggle cache.")

    print("Flickr8k dataset is ready!")
    print("Images:", image_folder_path)
    print("Captions:", labels_file_path)


if __name__ == "__main__":
    setup_dataset()
