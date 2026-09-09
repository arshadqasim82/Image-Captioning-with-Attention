from src.data.tokenization import Vocabulary
from src.model.decoder import DecoderRNN
from src.model.encoder import CNNEncoder
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


def generate_caption(image_path, decoder_path, captions_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(captions_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    captions = [line.split(",", 1)[1].strip()
                for line in lines[1:] if "," in line]
    vocab = Vocabulary(freq_threshold=1)
    vocab.build_vocab(captions)

    encoder = CNNEncoder().to(device)
    encoder.eval()

    decoder = DecoderRNN(
        embed_dim=256, decoder_dim=512, attention_dim=256, vocab_size=len(vocab.i2w)
    ).to(device)

    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    decoder.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(
        0).to(device)

    with torch.no_grad():
        features = encoder(image_tensor)
        word_index = vocab.w2i["<start>"]
        caption = []

        mean_features = features.mean(dim=1)
        h = decoder.init_h(mean_features)
        c = decoder.init_c(mean_features)

        for _ in range(20):
            word_tensor = torch.tensor([[word_index]]).to(device)
            embedding = decoder.embedding(word_tensor).squeeze(1)

            context, _ = decoder.attention(features, h)
            lstm_input = torch.cat((embedding, context), dim=1)
            h, c = decoder.lstm(lstm_input, (h, c))

            output = decoder.fc(h)
            word_index = output.argmax(dim=1).item()

            word = vocab.i2w[word_index]
            if word == "<end>":
                break
            caption.append(word)

    plt.imshow(image)
    plt.title(" ".join(caption))
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    test_image = project_root / "src" / "data" / \
        "jpg" / "1000268201_693b08cb0e.jpg"
    model_weights = project_root / "decoder_epoch_5.pth"
    text_file = project_root / "src" / "data" / "captions.txt"

    generate_caption(test_image, model_weights, text_file)
