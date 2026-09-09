from src.api.main import app
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}


def test_caption_invalid_file_type(client):
    files = {"file": ("test.txt", io.BytesIO(
        b"This is a text file, not an image."), "text/plain")}
    response = client.post("/caption", files=files)

    assert response.status_code == 400
    assert response.json() == {"detail": "File must be an image."}


def test_caption_valid_image(client):
    img = Image.new('RGB', (224, 224), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    files = {"file": ("test_dummy.jpg", img_byte_arr, "image/jpeg")}
    response = client.post("/caption", files=files)

    assert response.status_code == 200

    json_data = response.json()
    assert "filename" in json_data
    assert "caption" in json_data
    assert json_data["filename"] == "test_dummy.jpg"
    assert type(json_data["caption"]) == str
