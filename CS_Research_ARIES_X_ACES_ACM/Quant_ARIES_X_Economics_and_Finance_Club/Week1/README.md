# 📈 CSOT Quantitative Finance & Research — Week 1

Welcome to Week 1 of our **CSOT on Quantitative Finance and Research**!  
Over the next five weeks, we’ll explore the exciting intersection of data and finance — helping you build a strong foundation in:

- 📊 Statistics & Probability
- 🧠 Machine Learning
- 📈 Financial & Mathematical Modeling
- 💻 Real-world Data Processing

Each week includes curated resources and a hands-on task. In the final week, you’ll complete a project that ties everything together, giving you a real taste of what it’s like to work in quantitative research.

---

## 🔍 Week 1: Getting Started with Tools & Data

### 🧰 Tools Setup

- **Download Data from Kaggle:**  
  📎 [Kaggle Datasets Guide](https://www.kaggle.com/docs/datasets)

- **Set up Jupyter Notebooks:**
  - [Official Installation Guide](https://docs.jupyter.org/en/latest/install/notebook-classic.html)
  - [Beginner Tutorial (Dataquest)](https://www.dataquest.io/blog/jupyter-notebook-tutorial/)

---

### 🧮 NumPy & Pandas Basics

- [NumPy for Absolute Beginners](https://numpy.org/devdocs/user/absolute_beginners.html)
- [NumPy Quickstart Guide](https://numpy.org/devdocs/user/quickstart.html)
- [Data Cleaning & Preprocessing (FreeCodeCamp)](https://www.freecodecamp.org/news/data-cleaning-and-preprocessing-with-pandasbdvhj/)
- [Data Preprocessing with Pandas (DataCamp)](https://www.datacamp.com/blog/data-preprocessing)

---

### 📘 Introduction to Probability

- [Introduction to Probability Course](https://www.probabilitycourse.com/)  
  🔹 For this week, focus on **Section 1** for foundational understanding.

---

## 💼 Stock Market Analytics: Week 1 Challenge

### 🎯 Objective

Dive into financial data processing and uncover market insights. You'll work with time-series data, perform multi-level indexing, calculate indicators, and analyze returns using **Pandas** and **NumPy**.

---

### 🪙 Step-by-Step Instructions

#### 📥 Step 1: Dataset Access

- Download the dataset: [US Stocks & ETFs Price-Volume Data](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs)
- Choose **5 large-cap US companies** (e.g., `AAPL`, `MSFT`, `AMZN`, `TSLA`, `GOOGL`)
- Load CSVs into Pandas and create a **MultiIndexed DataFrame**:
  - Outer Index: `Ticker`
  - Inner Index: `Date`

---

#### 🧹 Step 2: Data Cleaning

- Handle missing values:
  - Identify missing rows per ticker
  - Use interpolation or forward fill where appropriate
- Convert the `Date` column to `datetime` format
- Sort data by date within each ticker
- Filter the data to include only the **last 10 years**

---

#### 🔁 Step 3: Data Transformation

Add the following columns for each stock:

- `Daily Return`: % change in closing price
- `7-day Moving Average` of closing price
- `30-day Moving Average` of closing price
- `Rolling Volatility (30d)`: Standard deviation of returns over the last 30 days

---

#### 📊 Step 4: Exploratory Analysis

Answer the following:

- Which stock had the **highest average return** over the 10-year period?
- Which stock had the **most volatile month**, and when?

---

## 📌 Submission Guidelines

- Use a **Jupyter Notebook** with clean Markdown documentation
- Modularize your code using **functions**
- Use **MultiIndexed DataFrames** properly
- Avoid hardcoding — aim for **reusable and clean logic**

---

Let’s get started 🚀  
Build your first step into the world of **Quantitative Research** in Finance!
