from src.data.tokenization import Vocabulary
from src.model.decoder import DecoderRNN
from src.model.encoder import CNNEncoder
import torch
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
import io
from pathlib import Path
import sys


project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))


app = FastAPI(title="Image Captioning API", version="1.0")

device = None
encoder = None
decoder = None
vocab = None
transform = None


@app.on_event("startup")
async def load_models():
    global device, encoder, decoder, vocab, transform
    print("Loading models into memory... This might take a moment.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    captions_file = project_root / "src" / "data" / "captions.txt"
    with open(captions_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    captions = [line.split(",", 1)[1].strip()
                for line in lines[1:] if "," in line]
    vocab = Vocabulary(freq_threshold=1)
    vocab.build_vocab(captions)

    encoder = CNNEncoder().to(device)
    encoder.eval()

    decoder_path = project_root / "decoder_epoch_5.pth"
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
    print("Models successfully loaded and ready for inference!")


@app.get("/health")
async def health_check():
    """Standard health check endpoint."""
    return {"status": "healthy", "model_loaded": encoder is not None}


@app.post("/caption")
async def generate_caption(file: UploadFile = File(...)):
    """Receives an image, passes it through the model, and returns a JSON caption."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = encoder(image_tensor)
        mean_features = features.mean(dim=1)

        h = decoder.init_h(mean_features)
        c = decoder.init_c(mean_features)

        start_word = vocab.w2i["<start>"]
        beam_size = 3
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
                    new_beams.append(
                        (score + word_prob, seq + [word_idx], h_next, c_next))

            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[
                :beam_size]

            if len(completed_captions) >= beam_size:
                break

        if not completed_captions:
            completed_captions = beams

        best_indices = sorted(completed_captions,
                              key=lambda x: x[0], reverse=True)[0][1]
        final_caption = [vocab.i2w[idx] for idx in best_indices if vocab.i2w[idx] not in [
            "<start>", "<end>", "<pad>"]]

    return {"filename": file.filename, "caption": " ".join(final_caption)}
