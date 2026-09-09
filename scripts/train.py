from src.model.decoder import DecoderRNN
from src.data.tokenization import Vocabulary
from src.data.dataset_class import Flickr8kDataset, CollatePad
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


def train():
    data_dir = project_root / "src" / "data"
    captions_file = data_dir / "captions.txt"
    features_dir = data_dir / "features"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    print("Building vocabulary...")
    with captions_file.open("r", encoding="utf-8") as file:
        lines = file.readlines()
    captions = [line.split(",", 1)[1].strip()
                for line in lines[1:] if "," in line]

    vocab = Vocabulary(freq_threshold=1)
    vocab.build_vocab(captions)
    pad_idx = vocab.w2i["<pad>"]

    print("Setting up DataLoader...")
    dataset = Flickr8kDataset(
        features_dir=str(features_dir),
        captions_file=str(captions_file),
        vocab=vocab
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=CollatePad(pad_idx=pad_idx)
    )

    embed_dim = 256
    decoder_dim = 512
    attention_dim = 256
    vocab_size = len(vocab.i2w)

    print("Initializing Model...")
    decoder = DecoderRNN(
        embed_dim=embed_dim,
        decoder_dim=decoder_dim,
        attention_dim=attention_dim,
        vocab_size=vocab_size
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(decoder.parameters(), lr=3e-4)
    num_epochs = 5
    print("Starting Training Loop...")

    for epoch in range(num_epochs):
        decoder.train()
        running_loss = 0.0

        for batch_idx, (features, captions) in enumerate(dataloader):
            features = features.to(device)
            captions = captions.to(device)
            optimizer.zero_grad()
            outputs, alphas = decoder(features, captions)
            targets = captions[:, 1:]
            outputs = outputs.reshape(-1, vocab_size)
            targets = targets.reshape(-1)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

        print(
            f"--- Epoch {epoch+1} completed. Average Loss: {running_loss/len(dataloader):.4f} ---")

        torch.save(decoder.state_dict(), str(
            project_root / f"decoder_epoch_{epoch+1}.pth"))


if __name__ == "__main__":
    train()
