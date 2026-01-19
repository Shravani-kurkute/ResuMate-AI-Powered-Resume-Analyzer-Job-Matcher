<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python" />
  <img src="https://img.shields.io/badge/Streamlit-App-red?logo=streamlit" />
  <img src="https://img.shields.io/github/repo-size/Shounak-Chavan/ResuMate" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

# 🤖 AI-Powered Resume Analyzer & Job Matching System
### *(Resume Parser • Job Matching)*

ResuMate is an intelligent AI-powered resume analysis & job recommendation system that extracts important information from resumes using NLP, predicts a student’s placement tier using an ML model, and matches candidates with top companies based on skills, experience, and academic performance.

The system combines **spaCy NLP parsing**, **SMOTE-balanced ML models**, and a **custom job matching engine** with an interactive **Streamlit** web app.

---

## 🚀 Features

- 📄 **Resume Upload** in PDF with automatic NLP extraction  
- 🧠 **Smart Data Extraction**: CGPA, 10th/12th marks, skills, projects, internships, certifications  
- 🎓 **Placement Tier Prediction** using ML (Premium/Standard/Basic/Not Placed)  
- 🏢 **Top Company Matches** based on skills + cutoff criteria  
- 📊 **Skill Gap Analysis** for every recommended company  
- 🎨 Fully responsive & animated **Streamlit UI**  
- ⚙️ **8 ML models trained**, best model selected with PCA & SMOTE  

---

## 🛠 Tech Stack

- Python 3.10  
- Streamlit  
- spaCy (en_core_web_md)  
- Scikit-Learn  
- Pandas & NumPy  
- SMOTE & PCA  
- Pickle  

---

## 📁 Project Structure
```
ResuMate/
│
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
├── Datasets/
│   ├── Eng_Dataset.csv
│   ├── Placement_Dataset_EDA.csv
│   ├── Placement_Dataset_Enhanced.csv
│   ├── Placement_Dataset_Preprocessed.csv
│   └── Placement_Model_Predictions.csv
│
├── Figures/
│
├── Models/
│   ├── Best_Placement_Model.pkl
│   └── Final_Scaler.pkl
│
├── Notebooks/
│   ├── 01_Project_Overview.ipynb
│   ├── 02_Exploratory_Data_Analysis.ipynb
│   ├── 03_Data_Preprocessing.ipynb
│   ├── 04_PCA_Dimensionality_Reduction.ipynb
│   └── 05_Machine_Learning_Models.ipynb
│
└── src/
    ├── Company_Database.json
    ├── Job_Matcher_06.py
    └── Resume_Parser_07.py
```

---

## ▶️ How to Run Locally

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/Shounak-Chavan/ResuMate.git
cd ResuMate
```

### **2️⃣ (Optional) Create Virtual Environment**
```bash
python -m venv venv
```

Activate the venv:

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### **3️⃣ Install Libraries**
```bash
pip install -r requirements.txt
```

### **4️⃣ Download spaCy Model**
```bash
python -m spacy download en_core_web_md
```

### **5️⃣ Run Streamlit App**
```bash
streamlit run app.py
```

---

## 🧠 Machine Learning Workflow

### **Models Trained (8 Total)**

- Logistic Regression ⭐ (Best — 91.36% Accuracy)
- SVM
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost

### **🏆 Final Model Selected: Logistic Regression**

Chosen for:
- Highest accuracy (91.36%)
- Best F1-score
- Best performance after PCA
- Best generalization with SMOTE-balanced classes
- Fast inference for Streamlit

### **🧩 Techniques Used**
- SMOTE Oversampling for class balancing
- PCA for dimensionality reduction
- StandardScaler for normalization
- Feature Engineering: Technical Skill Score, Communication Score, Internship/Project flags

---

## 🏢 Job Matching Engine

ResuMate uses a custom scoring algorithm that matches candidates with companies based on:

- Required skills vs. candidate skill set
- CGPA cutoff criteria
- Academic performance
- Experience indicators

Each company receives a **Match Score (0–100%)**, and the top 3 companies are shown with:
- Role & Location
- Package details
- Eligibility status
- Skill matches & gaps

---

## 📌 Future Enhancements

- Add resume scoring dashboard  
- Add ATS score analysis  
- Add LinkedIn job integration  
- Deploy on Streamlit Cloud / Render  
- Add user authentication  
- Add dark mode UI  

---

## ⚠️ Disclaimer

This project is **for educational and demonstration purposes only** and should not be used for real recruitment decisions.

---

## ⭐ If you found this project useful, please ⭐ the repository!
