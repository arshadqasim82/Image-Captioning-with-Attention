import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class Flickr8kDataset(Dataset):
    def __init__(self, features_dir, captions_file, vocab):
        self.features_dir = features_dir
        self.vocab = vocab
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
                    feature_name = img_name.replace('.jpg', '.pt')
                    pairs.append((feature_name, caption))
        return pairs

    def __len__(self):
        return len(self.dataset_pairs)

    def __getitem__(self, idx):
        feature_name, caption_text = self.dataset_pairs[idx]
        feature_path = os.path.join(self.features_dir, feature_name)

        features = torch.load(feature_path, weights_only=True)

        encoded_caption = self.vocab.encode(caption_text)
        caption_tensor = torch.tensor(encoded_caption, dtype=torch.long)

        return features, caption_tensor


class CollatePad:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        features = [item[0].unsqueeze(0) for item in batch]
        features = torch.cat(features, dim=0)
        captions = [item[1] for item in batch]

        padded_captions = pad_sequence(
            captions,
            batch_first=True,
            padding_value=self.pad_idx
        )
        return features, padded_captions
