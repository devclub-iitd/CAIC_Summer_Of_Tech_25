# WEEK-2

## Stock Market Prediction: Model Implementation and Evaluation

### Objective

Build, evaluate, and backtest predictive models for stock price forecasting using Linear Regression, ARIMA, and Random Forest. This week focuses on applying machine learning and time-series techniques to financial data, evaluating model performance, and simulating a trading strategy.

---

### Assignment Details

1. _Choose a Stock_  
   Select one stock from the dataset (e.g., AAPL).

2. _Data Splitting_  
   Split the data into training (first 80%) and testing (last 20%) sets, ensuring the split respects the time order.

3. _Model Implementation_  
   Implement the following models:

   - _Linear Regression_: Predict the next day's closing price using the past 5 days' closing prices as features.
   - _ARIMA_: Forecast the closing price using an ARIMA model.
   - _Random Forest_: Predict the closing price using technical indicators (e.g., 7-day and 30-day moving averages, RSI) as features.

4. _Model Evaluation_  
   Evaluate each model using Mean Absolute Error (MAE) or accuracy for direction prediction (whether the price goes up or down).

5. _Model Comparison_  
   Compare the models based on MAE and identify the best-performing one.

6. _Backtesting the Best Model_
   - Generate buy/sell signals: Buy if the predicted price > current price, sell otherwise.
   - Calculate the hypothetical profit/loss over the test period.

This assignment emphasizes applying machine learning concepts, evaluating model performance, and connecting to practical finance through a simple trading strategy.

---

### Resources

#### Random Forest

- [Random Forests Explained](https://youtu.be/J4Wdy0Wc_xQ?si=8OVWyvlE-ajtyVp4)
- [Random Forest Tutorial](https://youtu.be/sQ870aTKqiM?si=mu-ZeAH3v0AYRZ7E)
- [Random Forests from Scratch](https://carbonati.github.io/posts/random-forests-from-scratch/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

#### Linear Regression

- [Linear Regression Simple Explanation](https://pujappathak.medium.com/linear-regression-simple-explanation-with-example-fba51b2c181d)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

#### ARIMA

- [ARIMA Model Tutorial](https://youtu.be/-_2wOrEuFaM?si=xs2CKZycVD16QImp)
- [ARIMA for Time Series Forecasting](https://youtu.be/nK_jMFdSTZ8?si=NPvPWJa0_PbzztfH)

---

### Getting Started

- Use the dataset prepared in Week 1 (price-volume data for US stocks/ETFs).
- Ensure your Jupyter Notebook environment is set up with necessary libraries: pandas, numpy, scikit-learn, and statsmodels for ARIMA.
- Refer to Week 1 resources for data preprocessing steps if needed.

---

### Next Steps

- Experiment with hyperparameter tuning for Random Forest and ARIMA to improve performance.
- Explore additional features (e.g., volume, other technical indicators) to enhance model accuracy.
- Document your findings and trading strategy results in your notebook for submission.
