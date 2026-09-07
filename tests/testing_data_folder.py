import importlib.util
from pathlib import Path

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parents[1]


def load_module(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dataset_module = load_module(
    "dataset_class", project_root / "src" / "data" / "dataset_class.py"
)
tokenization_module = load_module(
    "tokenization", project_root / "src" / "data" / "tokenization.py"
)

CollatePad = dataset_module.CollatePad
Flickr8kDataset = dataset_module.Flickr8kDataset
Vocabulary = tokenization_module.Vocabulary

data_dir = project_root / "src" / "data"
captions_file = data_dir / "captions.txt"


class DatasetAdapter(Flickr8kDataset):
    def __len__(self):
        return self.len()

    def __getitem__(self, index):
        return self.get_item(index)


class CollatePadAdapter(CollatePad):
    def __call__(self, batch):
        return self.call(batch)


def load_vocabulary(captions_file):
    with captions_file.open("r", encoding="utf-8") as file:
        lines = file.readlines()

    captions = [line.split(",", 1)[1].strip()
                for line in lines[1:] if "," in line]
    vocabulary = Vocabulary(freq_threshold=1)
    vocabulary.build_vocab(captions)
    return vocabulary


vocab = load_vocabulary(captions_file)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = DatasetAdapter(
    root_dir=str(data_dir),
    captions_file=str(captions_file),
    vocab=vocab,
    transform=transform
)

pad_index = vocab.w2i["<pad>"]

data_loader = DataLoader(
    dataset=dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=CollatePadAdapter(pad_idx=pad_index)
)

images, captions = next(iter(data_loader))

print("--- Shape Check ---")
print(f"Images tensor shape: {images.shape}")
print(f"Captions tensor shape: {captions.shape}")

print("\n--- Decoding Check ---")
first_caption_tensor = captions[0].tolist()

decoded_text = vocab.decode(first_caption_tensor)
print(f"Decoded string: '{decoded_text}'")

first_image = images[0].permute(1, 2, 0).numpy()

plt.imshow(first_image)
plt.title(decoded_text)
plt.axis("off")
plt.show()
