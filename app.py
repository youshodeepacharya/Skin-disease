import os
import time
import random
from io import BytesIO
from werkzeug.utils import secure_filename

from flask import Flask, render_template, request, redirect, url_for, jsonify
from PIL import Image
import numpy as np
import cv2

import torch
import torchvision.transforms as transforms
import timm

from openai import OpenAI

# ==============================
# 🔑 OpenAI Setup
# ==============================
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("⚠️ No API key found → Using fallback chatbot")

client = OpenAI(api_key=api_key) if api_key else None

# ==============================
# Config
# ==============================
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
UPLOAD_FOLDER = "static/uploads"
GRADCAM_FOLDER = "static/gradcam"
WEIGHTS_PATH = "efficientnet_b0_skin_finetuned.pth"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

# ==============================
# Flask
# ==============================
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["GRADCAM_FOLDER"] = GRADCAM_FOLDER

# ==============================
# Device
# ==============================
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

# ==============================
# Helpers
# ==============================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_name(name):
    try:
        name = name.split(". ", 1)[-1]
        if "-" in name:
            name = name.split("-")[0]
        return name.strip()
    except:
        return name

# ==============================
# Confidence Message
# ==============================
def confidence_message(conf):
    if conf > 90:
        return "Very high confidence prediction"
    elif conf > 70:
        return "Moderate confidence prediction"
    else:
        return "Low confidence — please verify with a doctor"

# ==============================
# Risk Indicator (Improved)
# ==============================
def get_risk_level(disease):
    if "Melanoma" in disease:
        return "High Risk", "🔴"
    elif "Carcinoma" in disease:
        return "Medium Risk", "🟠"
    elif "Viral" in disease or "Fungal" in disease:
        return "Low Risk", "🟢"
    else:
        return "Low Risk", "🟢"

# ==============================
# Next Steps
# ==============================
def get_next_steps(disease):
    if "Melanoma" in disease:
        return [
            "Seek immediate medical attention",
            "Avoid delaying consultation",
            "Monitor rapid changes"
        ]
    elif "Carcinoma" in disease:
        return [
            "Consult a dermatologist",
            "Monitor lesion growth",
            "Avoid sun exposure"
        ]
    else:
        return [
            "Keep the area clean",
            "Avoid scratching or irritation",
            "Monitor for any changes",
            "Consult a dermatologist if needed"
        ]

# ==============================
# Warning Signs (NEW 🔥)
# ==============================
def warning_signs():
    return [
        "Rapid growth",
        "Bleeding or pain",
        "Color change",
        "Irregular borders"
    ]

# ==============================
# Classes
# ==============================
CLASS_NAMES = [
    "1. Eczema 1677",
    "10. Warts Molluscum and other Viral Infections - 2103",
    "2. Melanoma 15.75k",
    "3. Atopic Dermatitis - 1.25k",
    "4. Basal Cell Carcinoma (BCC) 3323",
    "5. Melanocytic Nevi (NV) - 7970",
    "6. Benign Keratosis-like Lesions (BKL) 2624",
    "7. Psoriasis pictures Lichen Planus and related diseases - 2k",
    "8. Seborrheic Keratoses and other Benign Tumors - 1.8k",
    "9. Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k"
]
NUM_CLASSES = len(CLASS_NAMES)

# ==============================
# Model
# ==============================
model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=NUM_CLASSES
).to(device)

raw_state = torch.load(WEIGHTS_PATH, map_location=device)

clean_state = {}
for k, v in raw_state.items():
    if k.startswith("base_model."):
        clean_state[k.replace("base_model.", "")] = v
    else:
        clean_state[k] = v

model.load_state_dict(clean_state)
model.eval()

print("✅ Model loaded successfully")

# ==============================
# Grad-CAM
# ==============================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x, class_idx):
        self.model.zero_grad()
        output = self.model(x)
        output[:, class_idx].backward()

        acts = self.activations.cpu().numpy()[0]
        grads = self.gradients.cpu().numpy()[0]

        weights = grads.mean(axis=(1, 2))
        cam = np.zeros(acts.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam / (cam.max() + 1e-8)
        return cam

gradcam = GradCAM(model, model.conv_head)

# ==============================
# Transforms
# ==============================
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==============================
# Prediction
# ==============================
def predict_with_gradcam(pil_img, filename):
    tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    top3_prob, top3_idx = torch.topk(probs, 3)

    results = [
        {
            "class_name": clean_name(CLASS_NAMES[i.item()]),
            "probability": float(p.item())
        }
        for p, i in zip(top3_prob, top3_idx)
    ]

    cam = gradcam.generate(tensor, top3_idx[0].item())

    img = np.array(pil_img.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    cam_path = os.path.join(GRADCAM_FOLDER, filename)
    cv2.imwrite(cam_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return results[0], results, cam_path

# ==============================
# Routes
# ==============================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or not allowed_file(file.filename):
            return redirect(request.url)

        base, ext = os.path.splitext(secure_filename(file.filename))
        filename = f"{base}_{int(time.time())}{ext}"

        img = Image.open(BytesIO(file.read())).convert("RGB")
        img.save(os.path.join(UPLOAD_FOLDER, filename))

        main_pred, top3, cam_path = predict_with_gradcam(img, filename)

        conf = main_pred["probability"] * 100
        conf_msg = confidence_message(conf)

        risk_level, risk_icon = get_risk_level(main_pred["class_name"])
        next_steps = get_next_steps(main_pred["class_name"])
        warnings = warning_signs()

        return render_template(
            "index.html",
            uploaded_image=url_for("static", filename="uploads/" + filename),
            gradcam_image=url_for("static", filename="gradcam/" + filename),
            main_pred=main_pred,
            top3=top3,
            risk_level=risk_level,
            risk_icon=risk_icon,
            next_steps=next_steps,
            warnings=warnings,
            confidence_msg=conf_msg
        )

    return render_template("index.html")

# ==============================
# Chatbot (Smarter)
# ==============================
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    msg = data.get("message", "").lower()
    disease = data.get("disease", "this condition")

    if any(w in msg for w in ["die", "fatal"]):
        reply = random.choice([
            f"{disease} is generally not life-threatening.",
            f"You don’t need to panic — {disease} is usually manageable.",
            f"Most cases of {disease} are not dangerous."
        ])
    elif "cure" in msg or "treatment" in msg:
        reply = random.choice([
            f"For {disease}, basic skincare and medical consultation is recommended.",
            f"Treatment usually involves hygiene and dermatologist guidance."
        ])
    elif "cause" in msg:
        reply = f"{disease} may be caused by infections, immune responses, or environmental factors."
    else:
        reply = f"This appears to be {disease}. Please consult a dermatologist."

    return jsonify({"reply": reply})

# ==============================
# Run
# ==============================
if __name__ == "__main__":
    app.run(debug=True)