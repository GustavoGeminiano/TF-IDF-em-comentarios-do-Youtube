# 🔍 YouTube Comment Similarity Search (NLP & TF-IDF)

A Python-based Natural Language Processing (NLP) tool that extracts comments from any YouTube video using the **YouTube Data API v3** and ranks them based on their semantic similarity to a custom search query using **TF-IDF Vectorization** and **Cosine Similarity**.

---

## 📌 Features

* 🤖 **Automated Data Extraction:** Fetches top-level comments and nested replies directly from YouTube.
* 🧹 **Text Preprocessing:** Cleans and normalizes text by removing language-specific stopwords using `nltk`.
* 📊 **TF-IDF Vectorization:** Transforms unstructured text into numerical feature vectors using `scikit-learn`.
* 🎯 **Cosine Similarity Ranking:** Ranks and filters the top most relevant comments based on user queries.
* 📄 **Technical Report:** Includes a complete LaTeX technical report detailing methodology, limitations, and results.

---

## 🛠️ Tech Stack

* **Language:** Python 3.8+
* **APIs:** Google API Client (`googleapiclient`)
* **Machine Learning / NLP:** `scikit-learn`, `nltk`

---

## 🚀 Getting Started

### 1. Prerequisites

Make sure you have a **YouTube Data API v3 Key**. You can get one for free from the [Google Cloud Console](https://console.cloud.google.com/).

### 2. Installation

Clone this repository and install the required dependencies:

```bash
git clone [https://github.com/GustavoGeminiano/TF-IDF-em-comentarios-do-Youtube.git](https://github.com/GustavoGeminiano/TF-IDF-em-comentarios-do-Youtube.git)
cd TF-IDF-em-comentarios-do-Youtube
pip install -r requirements.txt
