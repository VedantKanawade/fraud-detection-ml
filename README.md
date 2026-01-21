# Credit Card Fraud Detection
This project detects credit card fraud using a Random Forest classifier. 
It includes preprocessing, handling class imbalance, training, evaluation, and a Streamlit demo for interactive predictions.
Dataset: Credit Card Fraud Detection from Kaggle (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
Only a smaller subset (`creditcard_small.csv`) is included due to GitHub size limits.
Target: 'Class' (0 = non-fraud, 1 = fraud)
app.py             # Streamlit demo
notebooks/         # Model training scripts
models/            # Saved Random Forest model and scaler
data/creditcard_small.csv
README.md
1. Clone the repo
2. Create & activate a Python virtual environment
3. Install dependencies: pandas, numpy, scikit-learn, imbalanced-learn, joblib, streamlit
4. Run the app: `streamlit run app.py`
5. Upload CSV or enter manual transaction details to see predictions
Random Forest on subset:
ROC-AUC: 1.0
Classification report shows perfect precision, recall, and F1-score (demo subset)
- Train on full dataset with proper class weighting
- Compare different ML algorithms (XGBoost, LightGBM)
- Deploy fully hosted Streamlit app
