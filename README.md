# IMDB Sentiment Analysis with BERT

Fine-tuning **BERT-base-uncased** for binary sentiment classification on the **IMDB 50K Movie Reviews** dataset. This notebook was prepared for training course.

---

## 📌 Overview

A complete end-to-end NLP pipeline that fine-tunes a pretrained BERT model to classify movie reviews as **positive** or **negative**. The notebook covers:

- Loading and exploring the IMDB dataset
- Label encoding and HuggingFace `Dataset` conversion
- BERT tokenization with truncation & padding (max_len = 256)
- Fine-tuning `bert-base-uncased` via HuggingFace `Trainer`
- Accuracy evaluation on a held-out test set

---

## 📂 Dataset

**IMDB Dataset of 50K Movie Reviews** — 50,000 highly polar reviews with balanced labels.

- Source: [lakshmi25npathi/imdb-dataset-of-50k-movie-reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) on Kaggle
- Columns: `review` (text), `sentiment` (`positive` / `negative`)
- Class distribution: **25,000 positive / 25,000 negative** (perfectly balanced)
- Split: **80% train / 20% test** → 40,000 / 10,000 reviews

---

## 🧠 Model

- **Base model:** `bert-base-uncased` (110M parameters)
- **Task head:** `BertForSequenceClassification` with `num_labels=2`
- **Framework:** HuggingFace `transformers`
- Only the classification head is newly initialized; the BERT encoder is fine-tuned end-to-end.

---

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Max sequence length | 256 |
| Per-device batch size | 8 |
| Epochs | 2 |
| Optimizer | AdamW (HuggingFace default) |
| Loss | CrossEntropyLoss |
| Metric | Accuracy |

### Tokenization
```python
tokenizer(
    review,
    truncation=True,
    padding="max_length",
    max_length=256
)
📈 Results
Metric	Value
Test Accuracy	93.06%
Test Loss	0.5493
Train Loss (final)	0.259
Training time	~40 min (Tesla T4)
Training loss steadily decreased across 5,000 steps, from 0.738 (step 500) to 0.259 (step 5000). Even with only 2 epochs, the model already reaches 93% accuracy — a strong result for a lightweight fine-tuning run.

Note: Fine-tuning for 3–4 epochs with a lower learning rate and linear decay typically pushes this to ~94–95% accuracy, matching the BERT paper's reported IMDB results.

🚀 Usage
1. Clone & install
bash
git clone https://github.com/behrouzyou/imdb-sentimental-bert.git
cd imdb-sentimental-bert
pip install transformers datasets scikit-learn torch pandas
2. Run the notebook
Open imdb-reviews-sentiment-analysis-bert-model.ipynb in Jupyter or Kaggle and execute cells sequentially. Update the CSV path if running locally:

python
df = pd.read_csv("IMDB Dataset.csv")
3. Inference (example)
python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "./results"  # or your saved checkpoint
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

text = "This movie was an absolute masterpiece. Brilliant acting and direction!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)

with torch.no_grad():
    logits = model(**inputs).logits
pred = torch.argmax(logits, dim=1).item()
print("Positive" if pred == 1 else "Negative")
🏋️ Environment
Framework: PyTorch + HuggingFace Transformers

Hardware: NVIDIA Tesla T4 (Kaggle)

Python: 3.12



📜 License
This project is intended for educational purposes. Feel free to use, modify, and share.

text

---

### ✅ Tips before uploading

1. **Add a `.gitignore`** to keep the repo clean:
   ```gitignore
   results/
   *.bin
   *.safetensors
   __pycache__/
   .ipynb_checkpoints/
   *.csv
Push commands:

bash
git init
git add README.md imdb-reviews-sentiment-analysis-bert-model.ipynb .gitignore
git commit -m "Add BERT fine-tuning on IMDB sentiment + README"
git branch -M main
git remote add origin https://github.com/behrouzyou/imdb-sentimental-bert.git
git push -u origin main
Optional — share the trained model: If you want to include the fine-tuned weights (checkpoint is ~440 MB), don't push it to GitHub directly. Instead:

Upload to Hugging Face Hub → your-username/imdb-bert-sentiment

Reference it in the README:

python
model = AutoModelForSequenceClassification.from_pretrained("your-username/imdb-bert-sentiment")
Optional — add badges to the top of the README:

markdown
![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-yellow)
![License](https://img.shields.io/badge/License-MIT-green)
