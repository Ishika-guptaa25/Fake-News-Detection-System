<img width="1692" height="759" alt="image" src="https://github.com/user-attachments/assets/efd8a7cb-59ea-4e7a-9aa1-caa5a1a5963a" /><h1 align="center">📰 Fake News Detection System</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-FC4C02?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Language-Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
</p>

<p align="center">
  <b>🔍 Detect Fake or Real News Instantly using AI & NLP</b><br>
  A Machine Learning + Streamlit web app that identifies whether a news article or headline is <b>Fake</b> 🟥 or <b>Real</b> 🟩 using TF-IDF vectorization and Logistic Regression.
</p>

---

## 🚀 Live Demo  
🎯 **[👉 Click Here to Try the App](https://fake-news-detection-system-wctktkqkeyaaavjp9ufsb9.streamlit.app/)**  


---

## ✨ Features

- 🔎 Detects **Fake** or **Real** news in seconds  
- 🧠 Built with **Logistic Regression** trained on **TF-IDF features**  
- 🖥️ Interactive **Streamlit interface** with modern UI  
- ⚡ Uses **compressed .pkl.gz** model files for faster loading  
- 📊 Achieves ~**97% Accuracy** on balanced dataset  

---

## 🧩 Project Structure  

Here’s how your project is organized:

Fake_News_Detection_Model/
│
├── app.py                                  # Streamlit web app (frontend)
├── Fake_News_Detection_Model.ipynb         # Jupyter notebook for training
├── model.pkl.gz                            # Trained Logistic Regression model
├── vectorizer.pkl.gz                       # TF-IDF vectorizer
├── requirements.txt                        # Dependencies
├── README.md                               # Project documentation
└── .gitignore                              # Ignored files/folders

---

## ⚙️ Technologies Used

| Category | Technology |
|-----------|-------------|
| 🐍 Programming | Python |
| 🧠 ML Algorithm | Logistic Regression |
| 📊 Feature Extraction | TF-IDF Vectorization |
| ⚙️ Framework | Streamlit |
| 📚 Libraries | Pandas, NumPy, Scikit-learn |
| 💾 Storage | Pickle + Gzip |

---

## 📊 Dataset Overview

| Type | Samples |
|------|----------|
| 🟥 Fake News | 23,481 |
| 🟩 True News | 21,417 |

Dataset was balanced using **upsampling** to ensure equal representation and fair model training.

---

## 🧮 Model Workflow  

1️⃣ **Combine and label** both Fake & True datasets  
2️⃣ **Preprocess** and clean text data  
3️⃣ **Vectorize** text using TF-IDF  
4️⃣ **Train** Logistic Regression model  
5️⃣ **Evaluate** and save using `pickle` + `gzip`

📈 **Accuracy:** ~97%  
⚖️ Balanced predictions with minimal bias  

---

## 🖥️ How It Works  

1️⃣ Enter or paste a news headline/article  
2️⃣ Click **“Predict”**  
3️⃣ The app displays whether it’s  
   - 🟥 **Fake News**, or  
   - 🟩 **Real News**  
4️⃣ Shows an approximate **confidence score**

---

## 💡 Sample Inputs  

| Input Example | Prediction |
|----------------|-------------|
| “The Ministry of Health announced today that a new vaccination drive will begin next month targeting rural districts. Officials say the program is backed by international partners and aims to reach over 10 million people in the first phase.” | 🟩 Real |
| “BREAKING government emails CONFIRM aliens living among us for 50 years!!! Deep state covering up meetings in secret bases! SHARE before they delete this!” | 🟥 Fake |
| “GENEVA (Reuters) - The UN Climate Summit concluded with 195 countries agreeing to reduce carbon emissions by 45 percent by 2030. Secretary-General António Guterres called it a significant step forward.” | 🟩 Real |
| “You won't BELIEVE what this celebrity did!!! Hollywood in PANIC MODE scrubbing evidence! What happens at 3:47 will SHOCK you! Watch before removed!” | 🟥 Fake |

---

## ⚙️ Run Locally (PyCharm / VS Code)

# Clone the repository
git clone https://github.com/Ishika-guptaa25/Fake-News-Detection-System.git

# Move into the directory
cd Fake-News-Detection-System

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
🖥️ Then open http://localhost:8501 in your browser.

---

## 🖼️ App Screenshots

<img width="700" height="499" alt="image" src="https://github.com/user-attachments/assets/8f463ff5-a34a-464d-b196-ad0285dfa384" />

---

## 👩‍💻 Author
Ishika Gupta
🎓 BCA 3rd Year Student | Aspiring Data Scientist
💻 Passionate about AI, ML, and Building Real-World Applications
📍 India

<p align="left"> <a href="https://github.com/Ishika-guptaa25"><img src="https://img.shields.io/badge/GitHub-Ishika--guptaa25-181717?style=for-the-badge&logo=github"/></a> <a href="https://www.linkedin.com/in/ishika-gupta-y25081402/"><img src="https://img.shields.io/badge/LinkedIn-Ishika%20Gupta-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"/></a> </p>

---

## 🌟 Acknowledgments
🧩 Dataset sourced from Kaggle’s Fake News Challenge

📚 Libraries: Scikit-Learn, Pandas, Streamlit

💡 Inspired by the OpenAI Community and real-world misinformation problems

---

## 🧾 License
This project is licensed under the MIT License –
Feel free to use, modify, and share for learning or research purposes.

<p align="center"> ⭐ If you like this project, give it a star on <a href="https://github.com/Ishika-guptaa25/Fake-News-Detection-System">GitHub</a> to support me! </p> ```
