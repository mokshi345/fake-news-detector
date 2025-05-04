# 📰 Fake-News Detector using DistilBERT

[Live Demo → Hugging Face Spaces 🚀](https://huggingface.co/spaces/mokshi4/ai-news-detector)

This is a professional-grade web application for real-time fake news detection, leveraging the power of **Transformers**. The model is built on **DistilBERT**, a distilled version of BERT, and fine-tuned on the **ISOT Fake News Dataset**. Users can input news headlines or short articles to receive predictions on whether the content is **Fake** or **Real**, along with confidence scores.

---

## ✨ Key Features

- 📝 **Input News Text** — Paste any news headline or paragraph.
- 🤖 **Real-time Classification** — Outputs prediction: *Fake* or *Real*.
- 📈 **Confidence Score** — Displays model’s certainty in percentage.
- 💾 **Downloadable Results** — Save predictions as a CSV.
- 🌐 **Deployed on Hugging Face Spaces** — Runs seamlessly in-browser with no installation.

---

## 🧠 Model Architecture

- **Base Model**: `distilbert-base-uncased` from Hugging Face Transformers
- **Fine-Tuned On**: ISOT Fake News Dataset (binary classification: Fake / Real)
- **Framework**: PyTorch
- **Deployment**: Streamlit on Hugging Face Spaces

---

## 📂 Tech Stack

| Component     | Library/Tool               |
|---------------|----------------------------|
| Language      | Python 3.10+               |
| Model         | Hugging Face Transformers  |
| Interface     | Streamlit                  |
| Deployment    | Hugging Face Spaces        |
| Data Handling | Pandas                     |
| Visualization| HTML + Streamlit widgets   |

---

## 📊 Dataset

- **Name**: ISOT Fake News Dataset
- **Classes**: Fake, Real
- **Source**: University of Victoria's ISOT Lab
- **Description**: A widely used benchmark dataset containing labeled news articles for fake news detection research.

---

## 🛠️ Setup Instructions

To run locally:

```bash
git clone https://github.com/mokshi345/ai-news-detector.git
cd ai-news-detector
pip install -r requirements.txt
streamlit run app.py
