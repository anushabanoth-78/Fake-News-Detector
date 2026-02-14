# 📰 Fake News & Misinformation Detector

Detect fake vs real news articles using **Machine Learning**, **TF-IDF**, and **Logistic Regression**, complete with training scripts, evaluation charts, and an interactive **Streamlit web app**.

---
## Table of Contents

- [Overview](#overview)
- [Demo Video](#demo)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Training the Model](#training-the-model)
- [Evaluation & Charts](#evaluation--charts)
- [How It Works](#how-it-works)
- [Running the Streamlit App](#running-the-streamlit-app)
- [Code Modules](#code-modules)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)



## Overview

The **Fake News & Misinformation Detector** is a complete end-to-end **Natural Language Processing (NLP)** project that classifies news headlines and articles as **REAL** or **FAKE**.

It combines:

- **TF-IDF feature extraction**
- **Logistic Regression classifier**

This project also includes:

✓ Model evaluation with charts  
✓ Interactive Streamlit Web App  
✓ Modular and reusable project structure  
### Demo Video

▶️ Watch Full Project Demo:  
https://drive.google.com/file/d/1VVRmbsOYkUql4nt7nTk2N3dLETOrDUQS/view?usp=drive_link

## 📂 Project Structure

```bash
Fake-News-Detector/
│
├── data/
│   ├── Fake.csv
│   └── True.csv
│
├── outputs/
│   ├── charts/
│   │   ├── confusion_matrix.png
│   │   ├── pr_curve.png
│   │   ├── roc_curve.png
│   │   ├── nasa.png
│   │   ├── bakeing.png
│   │   ├── run command.png
│   │   └── run_fvs_cvs.png
│   │
│   ├── model.joblib
│   ├── pipeline.joblib
│   ├── vectorizer.joblib
│   └── metrics.json
│
├── src/
│   ├── detect_fake_news.py
│   ├── streamlit_app.py
│   ├── text_clean.py
│   ├── train_model.py
│   └── utils.py
│
├── test_news.py
├── test_multiple_news.py
├── requirements.txt
├── .gitignore
└── README.md

# Streamlit Web App Demo Screenshot

When launched, the app allows you to paste or type any news headline or paragraph and analyze its credibility in real time.

 **Features**
- Prediction: REAL / FAKE
- Probability visualization
- Adjustable FAKE detection threshold

### Confusion Matrix

<img width="1050" height="900" alt="confusion_matrix" src="https://raw.githubusercontent.com/anushabanoth-78/Fake-News-Detector/main/outputs/nasa.png" />

```

### Install Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install pandas numpy scikit-learn matplotlib streamlit joblib
```

---

## Dataset

| File | Type | Rows | Columns |
|------|------|------|----------|
| `True.csv` | Real news | 999 | `title`, `text`, `subject`, `date` |
| `Fake.csv` | Fake news | 999 | `title`, `text`, `subject`, `date` |

> **Dataset Source:**  
> This project uses and modifies the [*Fake and Real News Dataset*](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) by **Clément Bisaillon** (Kaggle).  
> Data was cleaned, header-fixed, and **downsampled to 999 REAL and 999 FAKE** news articles for balanced training and clear visualization.  
> Used purely for **educational and research** purposes.

---

## Training the Model

Run the following command from the project root:

```bash
python src/train_model.py --real data/True.csv --fake data/Fake.csv --text-col text --outdir outputs
```

This script will:
1. Load both datasets (real and fake).
2. Clean and merge them using `text_clean.py`.
3. Extract **TF-IDF** features.
4. Train a **Logistic Regression** classifier.
5. Save outputs:
   - `outputs/model.joblib`
   - `outputs/vectorizer.joblib`
   - `outputs/metrics.json`
   - Performance charts (`confusion_matrix.png`, `roc_curve.png`, `pr_curve.png`)

---

## Evaluation & Charts

After training, the model achieves **perfect classification accuracy** on this dataset.

### Confusion Matrix
<img width="1050" height="900" alt="confusion_matrix" src="https://github.com/user-attachments/assets/15735f60-4216-4aae-bf07-e265c034e5ef" />

| True Label | Predicted REAL | Predicted FAKE |
|-------------|----------------|----------------|
| REAL | 999 ✅ | 0 ❌ |
| FAKE | 0 ❌ | 999 ✅ |

The model correctly classified all 1,998 samples.

---

### ROC Curve
<img width="1050" height="900" alt="roc_curve" src="https://github.com/user-attachments/assets/a1b60b15-69f3-4440-ade5-92346731ffa9" />

The ROC curve touches the top-left corner  **AUC = 1.00**  
Perfect separability between classes.

---

### Precision–Recall Curve
<img width="1050" height="900" alt="pr_curve" src="https://github.com/user-attachments/assets/a0786968-55f4-40ad-92ca-6684e0312e50" />

Both precision and recall reach **1.00**, meaning zero false predictions.

---

### Key Metrics
| Metric | Value |
|---------|-------|
| Accuracy | 100 % |
| Precision (FAKE) | 1.00 |
| Recall (FAKE) | 1.00 |
| F1-Score | 1.00 |
| ROC-AUC | 1.00 |



## How It Works

### Pipeline Overview
1. **Text Cleaning** → Remove punctuation, URLs, emails, non-ASCII chars.  
2. **TF-IDF Vectorization** → Convert words into weighted numerical features.  
3. **Logistic Regression** → Predict probability of “FAKE” label.  
4. **Thresholding** → If `p(fake) ≥ 0.5` → FAKE, else REAL.

---



Output:
```
Label: FAKE | Fake probability: 0.560 | Threshold: 0.40
```

---

## Running the Streamlit App

### Launch the App
```bash
streamlit run src/streamlit_app.py
```

Then open the local web interface:
```
Local URL: http://localhost:8501
Network URL: http://10.196.40.168:8501

```

### App Features
- Paste any headline or paragraph  
- Analyze with one click  
- Adjust FAKE probability threshold  
- See model file locations and loaded status in sidebar  

---

## Code Modules

| Module | Purpose |
|---------|----------|
| `text_clean.py` | Handles text normalization (lowercasing, regex-based cleaning) |
| `utils.py` | Ensures output directories exist and handles JSON I/O |
| `train_model.py` | Loads data, trains the model, and generates metrics and plots |
| `detect_fake_news.py` | CLI script for predicting individual samples |
| `streamlit_app.py` | Streamlit web app for interactive user testing |

---

## Technologies Used

- **Python 3.10+**
- **scikit-learn** → TF-IDF Vectorizer, Logistic Regression  
- **pandas / numpy** → Data manipulation  
- **matplotlib** → Model visualization  
- **joblib** → Model persistence  
- **Streamlit** → Web interface  

---

## Future Improvements
- Integrate **BERT / DistilBERT** for contextual language understanding  
- Extend dataset for **multi-language** fake news detection  
- Add **Explainable AI** (LIME / SHAP) for model transparency  
- Deploy live on **Streamlit Cloud** or **Hugging Face Spaces**

## 👩‍💻 Project Author

**Banoth Anusha**
Final Year B.Tech — Computer Science & Engineering
Indian Institute of Technology Goa

Passionate about Machine Learning, AI, and building real-world data-driven applications.
This Fake News Detector project demonstrates end-to-end ML pipeline development including data preprocessing, model training, evaluation, and deployment with Streamlit.

🔗 GitHub: https://github.com/anushabanoth-78
Goa

📧 banoth.anusha.22031@iitgoa.ac.in

