# 🎓 Career Recommendation System using Machine Learning

## Project Overview
This project implements an AI-powered **Career Recommendation System** that predicts suitable career paths for students based on their:

- Skills  
- Interests  
- Education Level  
- Age
  
This project contains two user interfaces:
- **Tkinter Desktop GUI**
- **Streamlit Web App**

It also includes:
- A trained ML model (`RandomForestClassifier`)
- A preprocessed dataset
- A JSON file with detailed career information
- A feedback system that stores user responses into a CSV file

---

## 🧠 Input / Output Format

### **Input Format**
User enters the following via GUI / Web:

- **Age** (integer)
- **Education Level**  
  - 1 → Bachelor  
  - 2 → Master  
  - 3 → PhD  
- **Skills** (0 or 1)
- **Interests** (0 or 1)

### **Output Format**
System returns:

- 🎯 **Recommended Career**
- 📘 **Career Description**
- 🧰 Skills Required (optional)
- 💰 Salary Range (optional)
- 📚 Learning Resources (optional)
- 📝 Feedback saved to:  `feedback/feedback.csv` (optional)

---

## 📦 Required Python Libraries (Exact Versions)

Below is the complete list of dependencies used in this project: <br>
`matplotlib==3.10.7`
`numpy==2.3.4`
`packaging==25.0`
`pandas==2.3.3`
`pillow==12.0.0`
`pyparsing==3.2.5`
`python-dateutil==2.9.0.post0`
`pytz==2025.2`
`scikit-learn==1.7.2`
`scipy==1.16.3`
`seaborn==0.13.2`
`six==1.17.0`
`streamlit==1.40.2`

## Install all dependencies
```bash
pip install -r requirements.txt
```

---

## Steps to Execute the Project

- Clone your GitHub repo
```bash
git clone https://github.com/MahekRohitGor/career-recommendation-system
git clone https://github.com/MahekRohitGor/crs_gui
```

- Create and Activate Virtual Environment
```bash
cd career-recommendation-system
python -m venv venv
venv\Scripts\activate
```

- Install Required Libraries
```bash
pip install -r requirements.txt
```

- Train the Model (if needed)
```bash
cd model
python train_model.py
```

This generates:
```bash
model/career_model.pkl
```

- Run Tkinter Desktop Application
```bash
python gui\tk_gui.py
```

- Run Streamlit Web Application
```bash
streamlit run app/streamlit_app.py
```

- Career Recommendation System is deployed here: <br>
  **https://career-recommendation-system-snqvbzsqwwfspwnujbhfvb.streamlit.app/**



---
# Thank You
## Mahek Gor (202511044), Priyansee Soni (202511011), Anand Prakash (202511009)
