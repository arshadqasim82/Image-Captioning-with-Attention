import kagglehub
import os
import shutil

data_dir = os.path.dirname(os.path.abspath(__file__))
image_folder_path = os.path.join(data_dir, "jpg")
labels_file_path = os.path.join(data_dir, "captions.txt")

if os.path.exists(image_folder_path) and os.path.exists(labels_file_path):
    print(f"Dataset already exists. Loading locally from '{data_dir}'.")

else:
    print(f"Dataset not found locally. Downloading to {data_dir}")
    os.makedirs(data_dir, exist_ok=True)
    kaggle_path = kagglehub.dataset_download("adityajn105/flickr8k")

    image_dirs = [
        os.path.join(kaggle_path, "Images"),
        os.path.join(kaggle_path, "jpg"),
    ]

    for src in image_dirs:
        if os.path.exists(src):
            shutil.copytree(src, image_folder_path, dirs_exist_ok=True)
            break
    else:
        raise FileNotFoundError("Could not find the Flickr8k image folder.")

    caption_files = [
        os.path.join(kaggle_path, "captions.txt"),
        os.path.join(kaggle_path, "captions", "captions.txt"),
    ]

    for src in caption_files:
        if os.path.exists(src):
            shutil.copy2(src, labels_file_path)
            break
    else:
        raise FileNotFoundError("Could not find captions.txt.")

    print("Flickr8k dataset is ready!")
    print("Images:", image_folder_path)
    print("Captions:", labels_file_path)
