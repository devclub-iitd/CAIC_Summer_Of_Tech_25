# 🤖 CAIC Summer of Technology 2025: ML+Dev Track

## Introduction

Welcome to the **Machine Learning + Development (ML+Dev)** track of CAIC Summer of Technology 2025!

This vertical is designed not only to teach you the basics of ML but to help you **build, integrate, and deploy full systems** that solve real-world problems. You’ll go from a raw dataset to a deployed web app that predicts and generates marketing content — a hands-on journey through **ML, backend, and frontend integration.**

---

## Why ML+Dev?

This track emphasizes **practical product building** using ML. Here's what you’ll gain:

- **Full-stack product thinking**: Connect data science to real-world use cases.
- **ML as a component**: Learn to embed ML models in real software systems.
- **Deployment and DevOps**: Deploy your own models online with interactive UIs.
- **Resume-worthy work**: Have a hosted link, codebase, and demo ready by the end.

---

## What Problem Are We Solving?

Inspired by Adobe Experience Cloud, you’ll build a content intelligence tool with two tasks:

### Task 1: Predict Likes (Behavior Simulation)
Given a tweet's metadata, predict how many likes it will receive. Marketers can use this to estimate engagement *before posting.*

### Task 2: Generate Tweet Text (Content Simulation)
Given metadata, generate tweet content that is engaging and aligned with the brand tone.

Together, these simulate and create content — powered by ML but designed as usable tools.

---

## Weekly Roadmap — CAIC Summer of Tech: ML+Dev Track

### Week 1: Problem Understanding & Dataset Familiarization
Focus on:
- Understanding the ML problem
- Loading & inspecting data
- Light preprocessing and EDA
- Planning integration (what features will go into the model? how will API accept inputs?)

✅ *Deliverable*: Cleaned dataset, insight-driven features, and structured Colab notebook

---

### Week 2: Build Like Prediction Engine (Task 1)
Focus on:
- Feature engineering
- ML model development for likes
- Save model using `joblib` or `pickle`
- Create an API that takes input and returns predicted likes

✅ *Deliverable*: Trained model + REST API (Flask/FastAPI)

---

### Week 3: Build Tweet Generation Engine (Task 2)
Focus on:
- Fine-tune or use a text generation model
- Wrap it into a callable function
- Serve it as an API

✅ *Deliverable*: Working generative model + API to return generated tweet

---

### Week 4: Frontend + System Integration
Focus on:
- Create UI using Streamlit, React or Flask
- Connect frontend → backend APIs → ML models
- Make it interactive (input metadata → get likes and/or tweet)

✅ *Deliverable*: Locally working end-to-end app

---

### Week 5: Deployment & Showcasing
Focus on:
- Containerization (optional)
- Deploy to Hugging Face Spaces, Render, Vercel, etc.
- Make a project video
- Write a devlog/blog

✅ *Deliverable*: Hosted working demo + GitHub repo + presentation

---

## Tools & Technologies

- **ML**: Scikit-learn, Transformers, XGBoost, HuggingFace
- **Dev**: Python, FastAPI / Flask, Streamlit / React
- **Deployment**: HuggingFace Spaces, Render, Docker
- **Data Analysis**: Pandas, Seaborn, Matplotlib

---

## Learning Resources

- [Kaggle: Intro to ML](https://www.kaggle.com/learn/intro-to-machine-learning)
- [ML Model Deployment Guide](https://mlops.community/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [HuggingFace Spaces Guide](https://huggingface.co/docs/hub/spaces)

---

## FAQ

**Q: Will I learn ML from scratch?**  
A: Yes — but even more, you’ll learn how to **use it in real apps**.

**Q: Is this dev or ML-focused?**  
A: It's ML-powered, but **product/dev-focused** — your main goal is to build something usable with ML inside it.

---

## Let’s Go!

Build it. Use it. Ship it. 🚀
