# 🚀 CAIC Summer of Technology 2025  
## 🧠 Machine Learning + Development Track: Week 1  
### 🔍 Problem Understanding, Dataset Exploration & Dev Planning  

---

## 🗒 Problem Statement

We're building a system to help marketers understand and optimize their tweets using machine learning. Two core tasks:

1. **Predict tweet likes** from metadata (text, timestamp, media)
2. **Generate tweet text** from metadata (brand, media, etc.)

You’ll work with ~300K tweets and eventually integrate your models into a user-facing app.

---

## 🎯 Week 1 Objectives

Before we model anything, we need to:
- Understand the dataset
- Clean and analyze it
- Plan how it will be used in an API
- Identify the features we’ll engineer for models
- Think ahead about frontend/backend integration

---

## 🧰 1. Environment Setup

Use **Google Colab** this week.

Pros:
- No setup required
- Integrated with Google Drive
- Great for collaboration

Optional: Setup Jupyter locally with `pandas`, `matplotlib`, `seaborn`, `textblob`.

---

## 📦 2. Load the Dataset

📎 **Training Dataset:**  
🔗 [Click here to open](https://docs.google.com/spreadsheets/d/1JcESl7qCCBvS6xpWMZplhCXunvmkcNU_/edit?usp=drive_link)


```python
import pandas as pd

df = pd.read_csv("path_to_data.csv")  # replace with your actual path

df.shape
df.head()
df.info()
df.isnull().sum()
```

---

## 🧼 3. Light Preprocessing (for EDA and Dev Readiness)

```python
df.dropna(subset=['content', 'username', 'company', 'likes'], inplace=True)
df['media'].fillna('no_media', inplace=True)
df['has_media'] = df['media'].apply(lambda x: x != 'no_media')
df['content'] = df['content'].astype(str).str.strip().str.lower()
df['datetime'] = pd.to_datetime(df['date'], errors='coerce')
```

This gives us:
- Clean features for EDA
- Fields we can send via API (in future)

---

## 📊 4. Exploratory Data Analysis

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df['likes'], kde=True)
sns.boxplot(x=df['likes'])
```

```python
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.day_name()
```

```python
df['word_count'] = df['content'].apply(lambda x: len(x.split()))
df['char_count'] = df['content'].apply(len)
```

---

### 🧠 What EDA Should Help You Plan

Think like a developer:
- What features can be passed to an ML model in an API request?
- Which fields should the frontend ask users for?
- What preprocessing should happen inside the API?
- Can we calculate features (e.g., word count, has_media) *inside* the backend?

---

## 🛠️ 5. Feature Plan for APIs

These are likely features to use:
- `company`, `username`
- `content` → `word_count`, `sentiment`
- `media` → `has_media`
- `datetime` → hour, weekday

Make a note of this mapping! You’ll use it in the backend.

---

## 🧠 Bonus Ideas

| Topic | Use |
|--|--|
| WordClouds | Visualize engaging keywords |
| TextBlob | Get sentiment polarity |
| TF-IDF | Later for text vectorization |
| t-SNE | Explore clustering of content |
| Named Entity Recognition | See what brands/places are mentioned |

---

## 📚 Resources

- [EDA Guide – Kaggle](https://www.kaggle.com/learn/data-cleaning)  
- [Feature Engineering – Kaggle](https://www.kaggle.com/learn/feature-engineering)  
- [Streamlit Docs](https://docs.streamlit.io/)  
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)  

---

## ✅ Week 1 Deliverables

- ✅ Cleaned dataset
- ✅ EDA notebook with plots & insights
- ✅ Feature list planned for modeling
- ✅ Plan for API input/output structure

---

## 🔮 What’s Next (Week 2)

- Build and train a model to predict likes (Task 1)
- Wrap it in an API using Flask or FastAPI
- Accept metadata and return predicted likes

---  
Build smart. Build useful. Let’s get started 🚀
