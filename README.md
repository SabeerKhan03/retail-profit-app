# 📦 Retail Profit Predictor

A machine learning web app that predicts whether a retail order
will result in a **Profit or Loss** — built and deployed live on AWS EC2.

🔗 **Live App:** http://YOUR_EC2_IP:8501 ← (we will fill this after AWS)

---

## 📌 About the Project

Retail companies often lose money on orders due to high discounts
and poor product-region combinations. This project analyzes 9,000+
Superstore retail transactions to identify what drives profit —
and deploys a live prediction tool anyone can use.

---

## 🔍 Key Business Insights Found

- High discounts (above 30%) strongly correlate with losses
- **Tables** sub-category has an average margin of -8.5% — most loss-making product
- **Furniture in the Central region** is consistently loss-making (-1.75% margin)
- Revenue grows steadily but profit margins are volatile — driven by discounting strategy

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| ML Model | Linear Regression (Scikit-learn) |
| Data Processing | Pandas, NumPy |
| API Backend | Flask |
| Web Frontend | Streamlit |
| Deployment | AWS EC2 t2.micro (Free Tier) |
| Version Control | Git + GitHub |

---

## 📁 Project Structure
```
retail-profit-app/
├── save_model.py        ← trains and saves model from raw CSV
├── app.py               ← Flask REST API (port 5000)
├── streamlit_app.py     ← Streamlit web UI (port 8501)
├── requirements.txt     ← all dependencies
├── retail_model.pkl     ← trained model
├── feature_columns.pkl  ← column order for prediction alignment
└── model_meta.pkl       ← dropdown options for the UI
```

---

## 🚀 Run Locally
```bash
# 1. Clone the repo
git clone https://github.com/SabeerKhan03/retail-profit-app.git
cd retail-profit-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place dataset in this folder
#    File: "Sample - Superstore.csv"

# 4. Train and save the model
python save_model.py

# 5. Terminal 1 — start Flask API
python app.py

# 6. Terminal 2 — start Streamlit UI
python -m streamlit run streamlit_app.py

# Open browser at: http://localhost:8501
```

---

## 👤 Author

**Sabeer Khan** — Final Year B.Tech (ECE), Gudlavalleru Engineering College  
[GitHub](https://github.com/SabeerKhan03)