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


def generate_caption_beam_search(image_path, decoder_path, captions_file, beam_size=3):
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
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = encoder(image_tensor)
        mean_features = features.mean(dim=1)

        h = decoder.init_h(mean_features)
        c = decoder.init_c(mean_features)

        start_word = vocab.w2i["<start>"]
        beams = [(0.0, [start_word], h, c)]
        completed_captions = []

        for step in range(20):
            new_beams = []
            for score, seq, h_state, c_state in beams:
                if seq[-1] == vocab.w2i["<end>"]:
                    completed_captions.append((score, seq))
                    continue

                word_tensor = torch.tensor([[seq[-1]]]).to(device)
                embedding = decoder.embedding(word_tensor).squeeze(1)

                context, _ = decoder.attention(features, h_state)
                lstm_input = torch.cat((embedding, context), dim=1)
                h_next, c_next = decoder.lstm(lstm_input, (h_state, c_state))

                output = torch.log_softmax(decoder.fc(h_next), dim=1)
                top_probs, top_idx = output.topk(beam_size, dim=1)

                for i in range(beam_size):
                    word_idx = top_idx[0][i].item()
                    word_prob = top_probs[0][i].item()
                    new_score = score + word_prob
                    new_beams.append(
                        (new_score, seq + [word_idx], h_next, c_next))

            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[
                :beam_size]

            if len(completed_captions) >= beam_size:
                break

    if not completed_captions:
        completed_captions = beams

    best_caption_indices = sorted(
        completed_captions, key=lambda x: x[0], reverse=True)[0][1]

    final_caption = []
    for idx in best_caption_indices:
        word = vocab.i2w[idx]
        if word not in ["<start>", "<end>", "<pad>"]:
            final_caption.append(word)

    plt.imshow(image)
    plt.title(" ".join(final_caption))
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    test_image = project_root / "src" / "data" / \
        "jpg" / "1000268201_693b08cb0e.jpg"
    model_weights = project_root / "decoder_epoch_5.pth"
    text_file = project_root / "src" / "data" / "captions.txt"

    generate_caption_beam_search(
        test_image, model_weights, text_file, beam_size=3)
