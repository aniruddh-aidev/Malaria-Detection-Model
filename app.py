import os
import io
import torch
import torch.nn as nn
from torch.nn import LeakyReLU
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "malaria_model.pt")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Transform — matches training exactly (no normalization) ───────────────────
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# ── Model definition — exact copy from Colab ─────────────────────────────────
class MLR_DTC(nn.Module):
    def __init__(self, input: int, hidden: int, output: int) -> None:
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input, hidden, 3),
            nn.BatchNorm2d(hidden),
            nn.LeakyReLU(),
            nn.Conv2d(hidden, hidden, 3),
            nn.BatchNorm2d(hidden),
            LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3),
            nn.BatchNorm2d(hidden),
            nn.LeakyReLU(),
            nn.Conv2d(hidden, hidden, 3),
            nn.BatchNorm2d(hidden),
            LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.Classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden * 29 * 29, output),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.Classifier(x)
        return x


# ── Load model ────────────────────────────────────────────────────────────────
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"\n\n  ❌  Model not found at: {MODEL_PATH}\n"
            "  → Run  python train.py  first to train the model.\n"
            "  → OR export from Colab:  torch.save(m0.state_dict(), 'malaria_model.pt')\n"
            "      then copy it into the model/ folder.\n"
        )

    checkpoint  = torch.load(MODEL_PATH, map_location=DEVICE)
    class_names = checkpoint.get("class_names", ["Parasitized", "Uninfected"])

    model = MLR_DTC(input=3, hidden=20, output=1)

    # Handles both plain state_dict and wrapped {"model_state_dict": ...}
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.to(DEVICE).eval()

    val_acc = checkpoint.get("val_accuracy", None)
    print(f"[INFO] MLR_DTC loaded | device={DEVICE} | classes={class_names}"
          + (f" | val_acc={val_acc*100:.2f}%" if val_acc else ""))
    return model, class_names


model, CLASS_NAMES = load_model()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file received."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    allowed = {"png", "jpg", "jpeg", "bmp", "tiff"}
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in allowed:
        return jsonify({"error": f"Unsupported file type: .{ext}"}), 400

    try:
        img    = Image.open(io.BytesIO(file.read())).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(DEVICE)   # [1, 3, 128, 128]

        with torch.inference_mode():
            logit = model(tensor)                          # [1, 1]
            prob  = torch.sigmoid(logit).item()            # P(Uninfected)

        # class index 0 = Parasitized, 1 = Uninfected (ImageFolder alphabetical)
        p_parasitized = round((1 - prob) * 100, 2)
        p_uninfected  = round(prob * 100, 2)

        label      = "Uninfected" if prob >= 0.5 else "Parasitized"
        confidence = p_uninfected if prob >= 0.5 else p_parasitized

        return jsonify({
            "label":      label,
            "confidence": confidence,
            "probs": {
                "Parasitized": p_parasitized,
                "Uninfected":  p_uninfected,
            },
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print(f"[INFO] Flask → http://localhost:5000")
    app.run(debug=True, port=5000)
