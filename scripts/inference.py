from src.data.tokenization import Vocabulary
from src.model.decoder import DecoderRNN
from src.model.encoder import CNNEncoder
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import math
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


def run_full_inference(image_path, decoder_path, captions_file, beam_size=3):
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

    original_image = Image.open(image_path).convert("RGB")
    original_image_resized = original_image.resize(
        (224, 224), Image.Resampling.LANCZOS)
    image_tensor = transform(original_image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = encoder(image_tensor)
        mean_features = features.mean(dim=1)

        h = decoder.init_h(mean_features)
        c = decoder.init_c(mean_features)

        start_word = vocab.w2i["<start>"]
        beams = [(0.0, [start_word], h, c, [])]
        completed_captions = []

        for step in range(20):
            new_beams = []
            for score, seq, h_state, c_state, alpha_seq in beams:
                if seq[-1] == vocab.w2i["<end>"]:
                    completed_captions.append((score, seq, alpha_seq))
                    continue

                word_tensor = torch.tensor([[seq[-1]]]).to(device)
                embedding = decoder.embedding(word_tensor).squeeze(1)

                context, alpha = decoder.attention(features, h_state)
                lstm_input = torch.cat((embedding, context), dim=1)
                h_next, c_next = decoder.lstm(lstm_input, (h_state, c_state))

                output = torch.log_softmax(decoder.fc(h_next), dim=1)
                top_probs, top_idx = output.topk(beam_size, dim=1)

                current_alpha = alpha.view(49).cpu().numpy()

                for i in range(beam_size):
                    word_idx = top_idx[0][i].item()
                    word_prob = top_probs[0][i].item()
                    new_beams.append((
                        score + word_prob,
                        seq + [word_idx],
                        h_next,
                        c_next,
                        alpha_seq + [current_alpha]
                    ))

            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[
                :beam_size]

            if len(completed_captions) >= beam_size:
                break

    if not completed_captions:
        completed_captions = beams

    best_timeline = sorted(
        completed_captions, key=lambda x: x[0], reverse=True)[0]
    best_indices = best_timeline[1]
    best_alphas = best_timeline[2]
    final_words = []
    final_alphas = []

    for idx, word_idx in enumerate(best_indices[1:]):
        word = vocab.i2w[word_idx]
        if word == "<end>":
            break
        final_words.append(word)
        final_alphas.append(best_alphas[idx])

    plot_heatmaps(original_image_resized, final_words, final_alphas)


def plot_heatmaps(image, words, alphas):
    num_words = len(words)
    cols = 5
    rows = math.ceil(num_words / cols)

    fig = plt.figure(figsize=(15, 3 * rows))

    for i in range(num_words):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(image)

        alpha_img = alphas[i].reshape(7, 7)
        ax.imshow(alpha_img, cmap='jet', alpha=0.5,
                  interpolation='bicubic', extent=(0, 224, 224, 0))

        ax.set_title(words[i], fontsize=14)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_image = project_root / "src" / "data" / "jpg" / "1000268201_693b08cb0e.jpg"
    model_weights = project_root / "decoder_epoch_5.pth"
    text_file = project_root / "src" / "data" / "captions.txt"

    run_full_inference(test_image, model_weights, text_file, beam_size=3)
