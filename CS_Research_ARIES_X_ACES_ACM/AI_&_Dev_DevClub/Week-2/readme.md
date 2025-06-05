# ⚙️ CAIC Summer of Technology 2025  
## 🔮 ML+Dev Track: Week 2  
### 🎯 Like Prediction Engine (Task 1)

---

## 🧠 Quick Recap

Last week, you:
- Explored and cleaned the tweet dataset  
- Engineered basic features (e.g., `word_count`, `has_media`, `hour`)  
- Understood what the API will look like  

Now, you're ready to **train your first ML model** 🎉  
This week is about **predicting tweet likes from metadata**.

---

## 🏁 Week 2 Goals

By the end of the week, you will:
1. Engineer meaningful features for predicting likes  
2. Train and evaluate an ML model  
3. Save the trained model  
4. Create a **REST API** that returns predicted likes from tweet metadata

✅ **Deliverable**: Trained model file + working Flask or FastAPI endpoint

---

## ⚒️ Step 1: Feature Engineering

Use the features you planned in Week 1:

| Metadata Field | Feature Name | Example |
|----------------|--------------|---------|
| `content`      | `word_count`, `char_count`, `sentiment` | 16 words, 78 characters, positive |
| `media`        | `has_media` | True / False |
| `datetime`     | `hour`, `day_of_week` | 14, Monday |
| `company`, `username` | One-hot or label encoding | toyota → 12 |

Note: Make sure all features are numeric before training.

---

## 🧠 Additional Feature Preparation (Required for Week 2)

To make your Week 2 modeling smooth, make sure the following features are included in your final dataset:

---

### 🧪 1. Sentiment Score (from `content`)

We'll use sentiment polarity as a numeric feature in the model. Use `TextBlob`:

```python
from textblob import TextBlob

df['sentiment'] = df['content'].apply(lambda x: TextBlob(x).sentiment.polarity)
```

This will give values between -1.0 (very negative) and 1.0 (very positive).

---

### 🔤 2. Character Count (from `content`)

Add total number of characters in each tweet:

```python
df['char_count'] = df['content'].apply(len)
```

---

### 🏷️ 3. Encode `company` or `username`

Since ML models need numeric inputs, we’ll encode company names:

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['company_encoded'] = le.fit_transform(df['inferred company'])
```

Use `company_encoded` (or `username_encoded`) instead of raw strings while training your model in Week 2.

---

📌 **Reminder:** All your final features (like `word_count`, `char_count`, `sentiment`, `has_media`, etc.) must be **numeric and ready** before Week 2!



---

## 🤖 Step 2: Train Your Model

Use **Scikit-learn** to train a regression model. Start simple — **LinearRegression** or **RandomForestRegressor**.

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

X = df[['word_count', 'char_count', 'has_media', 'hour', 'sentiment']]
y = df['likes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

preds = model.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)
print("RMSE:", rmse)
```

🎯 **Goal**: Lower RMSE → Better model

---

## 💾 Step 3: Save the Model

```python
import joblib
joblib.dump(model, 'like_predictor.pkl')
```

---

## 🌐 Step 4: Build an API (Flask or FastAPI)

### Flask Example
```python
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('like_predictor.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([
        data['word_count'],
        data['char_count'],
        data['has_media'],
        data['hour'],
        data['sentiment']
    ]).reshape(1, -1)
    
    prediction = model.predict(features)[0]
    return jsonify({'predicted_likes': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
```

Test using your browser or Postman 

---

## 🚀 Model Improvement Ideas

To improve your model, explore these:

### 🔁 Model Architectures

Note that this list is not exhaustive and is mean to just be a starting point for you to start exploring. We encourage you to go beyond these resources and make your own model architectures and try those out too!

| Model Type | Library | Notes |
|------------|---------|-------|
| Linear Regression | Scikit-learn | Good baseline |
| Random Forest | Scikit-learn | Handles non-linear relationships |
| Gradient Boosting | XGBoost, LightGBM | Often best performance |
| Neural Networks | PyTorch, TensorFlow | Try MLPRegressor or custom architectures |
| Tabular Transformers | PyTorch Tabular, HuggingFace | Advanced, experimental for structured data |

### 🧠 Feature Enhancements

- Add `emoji_count`, `has_hashtag`, or `has_url`
- TF-IDF score of content
- Company-wise average likes as a feature
- Historical tweet performance

### ⚙️ Training Improvements

- Use `GridSearchCV` or `RandomizedSearchCV` for hyperparameters
- Try cross-validation with `cross_val_score`
- Normalize/scale features with `StandardScaler` or `MinMaxScaler`

---

## ✅ Week 2 Deliverables

- ✅ Trained model (e.g., `like_predictor.pkl`)
- ✅ Working API (Flask/FastAPI)
- ✅ Feature explanation + model improvement notes (in notebook or markdown)

---

## 📚 Learning Resources

- [Sentdex YouTube Channel](https://www.youtube.com/@sentdex/)
- [FreeCodeCamp YouTube Channel](https://www.youtube.com/@freecodecamp)
- [Tech With Tim Flask Playlist](https://www.youtube.com/playlist?list=PLzMcBGfZo4-n4vJJybUVV3Un_NFS5EOgX)
- [Tech With Tim ML Playlist](https://www.youtube.com/playlist?list=PLzMcBGfZo4-mP7qA9cagf68V06sko5otr)
- [Nicholas Renotte YouTube Playlists](https://www.youtube.com/@NicholasRenotte/playlists)
- [Intro to ML - Kaggle](https://www.kaggle.com/learn/intro-to-machine-learning)

---

## 🔮 What’s Next (Week 3)

- Train a **text generation model** to write tweet content from metadata  
- Serve it via an API just like this one

---

Build smart. Predict smarter. Let’s go! 🚀
