# 🔬 Malaria Cell Detector

A local Flask web app for malaria detection using a custom **MLR_DTC** CNN
trained on the [NIH Malaria Cell Images Dataset](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria).

---

## 📁 Project Structure

```
malaria-detector/
├── train.py                 ← Step 1: Train & save model
├── app.py                   ← Step 2: Run the web app
├── requirements.txt
├── model/
│   └── malaria_model.pt    ← created automatically by train.py
└── templates/
    └── index.html
```

---

## 🚀 Setup

### 1. Install dependencies
```bash
python -m venv venv
#Mac:
 source venv/bin/activate
# Windows:
 venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download the dataset
- Go to: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria
- Download & unzip → place it as:
```
malaria-detector/
└── data/
    └── cell_images/
        ├── Parasitized/
        └── Uninfected/
```

### 3. Train the model *(only once)*
```bash
python train.py
```
Saves `model/malaria_model.pt` automatically.


### 4. Run the app
```bash
python app.py
```
Open → **http://localhost:5000**

---

## 🧠 Model — MLR_DTC

| Property      | Value                              |
|---------------|------------------------------------|
| Architecture  | Custom 2-block CNN (MLR_DTC)       |
| Input size    | 128 × 128 × 3 (RGB)               |
| Output        | Single logit → sigmoid             |
| Loss          | BCEWithLogitsLoss                  |
| Optimizer     | Adam (lr=0.01)                     |
| Classes       | Parasitized (0) · Uninfected (1)  |
| Preprocessing | Resize(128,128) + ToTensor()       |


## Results
| Metric | Score |
|--------|-------|
| Train Accuracy | 92.8% |
| Validation Accuracy | 94.2% |


## Live Demo
Try it here: [Hugging Face Space](https://huggingface.co/spaces/annir241/Malaria-Detection-Model)
