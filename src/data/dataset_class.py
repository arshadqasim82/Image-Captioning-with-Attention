import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image


class Flickr8kDataset(Dataset):
    def __init__(self, root_dir, captions_file, vocab, transform=None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(self.root_dir, "jpg")
        self.vocab = vocab
        self.transform = transform
        self.dataset_pairs = self.load_captions(captions_file)

    def load_captions(self, captions_file):
        pairs = []
        with open(captions_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            start_idx = 1 if "image" in lines[0].lower() else 0
            for line in lines[start_idx:]:
                parts = line.strip().split(',', 1)
                if len(parts) == 2:
                    img_name, caption = parts
                    pairs.append((img_name, caption))
        return pairs

    def len(self):
        return len(self.dataset_pairs)

    def get_item(self, idx):
        img_name, caption_text = self.dataset_pairs[idx]
        img_path = os.path.join(self.image_dir, img_name)
        with Image.open(img_path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        encoded_caption = self.vocab.encode(caption_text)
        caption_tensor = torch.tensor(encoded_caption, dtype=torch.long)
        return image, caption_tensor


class CollatePad:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def call(self, batch):
        images = [item[0].unsqueeze(0) for item in batch]
        images = torch.cat(images, dim=0)
        captions = [item[1] for item in batch]

        padded_captions = pad_sequence(
            captions,
            batch_first=True,
            padding_value=self.pad_idx
        )
        return images, padded_captions
