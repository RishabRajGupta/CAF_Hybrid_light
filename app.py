import io
import os
from typing import Dict

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
import torch
import torchvision.transforms as T

from model_definition import CAFHybridLight


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "CAFHybridLight_best.pth")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tiff", "webp"}
CLASS_LABELS = ["Fake", "Real"]  # PyTorch ImageFolder sorts alphabetically: Fake=0, Real=1
IMG_SIZE = 128


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model(model_path: str) -> CAFHybridLight:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu")
    model = CAFHybridLight(d=128, pretrained_tokenizers=True)

    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("model_state")
            or checkpoint.get("state_dict")
            or checkpoint
        )
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def build_preprocess_pipeline() -> T.Compose:
    return T.Compose(
        [
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def predict(
    model: CAFHybridLight, preprocess: T.Compose, image_bytes: bytes
) -> Dict[str, float]:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("Uploaded file is not a valid image.") from exc

    tensor = preprocess(image).unsqueeze(0)

    with torch.inference_mode():
        logits = model(tensor)

    # Normalize logits to probabilities regardless of output dimensionality.
    if logits.ndim == 0:
        fake_prob = torch.sigmoid(logits).item()
        probs = torch.tensor([1 - fake_prob, fake_prob])
    else:
        logits = logits.squeeze(0)
        if logits.numel() == 1:
            fake_prob = torch.sigmoid(logits).item()
            probs = torch.tensor([1 - fake_prob, fake_prob])
        else:
            probs = torch.softmax(logits, dim=0)

    probs = probs.tolist()
    return {
        CLASS_LABELS[idx]: float(prob) for idx, prob in enumerate(probs[: len(CLASS_LABELS)])
    }


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB upload limit
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "caf-hybrid-light")

detector = load_model(MODEL_PATH)
preprocess = build_preprocess_pipeline()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.post("/predict")
def handle_predict():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify({"error": "Please choose an image to upload."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type."}), 400

    try:
        predictions = predict(detector, preprocess, file.read())
    except ValueError as err:
        return jsonify({"error": str(err)}), 400
    except Exception as err:  # pragma: no cover - top-level guard
        return jsonify({"error": f"Inference failed: {err}"}), 500

    top_label, top_score = max(predictions.items(), key=lambda item: item[1])
    return jsonify(
        {
            "label": top_label,
            "confidence": round(top_score * 100, 2),
            "probabilities": predictions,
        }
    )


@app.get("/health")
def healthcheck():
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

