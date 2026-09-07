import importlib.util
from pathlib import Path
from torch.utils.data import DataLoader

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
features_dir = data_dir / "features"


def load_vocabulary(captions_file):
    with captions_file.open("r", encoding="utf-8") as file:
        lines = file.readlines()

    captions = [line.split(",", 1)[1].strip()
                for line in lines[1:] if "," in line]
    vocabulary = Vocabulary(freq_threshold=1)
    vocabulary.build_vocab(captions)
    return vocabulary


vocab = load_vocabulary(captions_file)

dataset = Flickr8kDataset(
    features_dir=str(features_dir),
    captions_file=str(captions_file),
    vocab=vocab
)

pad_index = vocab.w2i["<pad>"]

data_loader = DataLoader(
    dataset=dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=CollatePad(pad_idx=pad_index)
)

features, captions = next(iter(data_loader))

print("--- Shape Check ---")
print(f"Features tensor shape: {features.shape}")
print(f"Captions tensor shape: {captions.shape}")

print("\n--- Decoding Check ---")
first_caption_tensor = captions[0].tolist()

decoded_text = vocab.decode(first_caption_tensor)
print(f"Decoded string: '{decoded_text}'")
