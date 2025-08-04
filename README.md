# 👗 Sentiment-Driven Clothing Brand Recommendation System

This project addresses a common challenge in online shopping: recommending the most suitable **Pakistani clothing brand** based on **customer sentiments**. Given product features—like **fabric quality, price, customer service, allergic reactions, color theme**, etc.—our system predicts the best brand using **real-world review data** from experienced customers and a **BERT-Transformer-based sentiment classifier**.

---

## 📌 What’s Inside?

### ✅ Problem Statement
Pakistani customers often struggle with choosing the best clothing brand that meets specific preferences like:

- Good/Bad Color Theme  
- High/Low Fabric Quality  
- High/Low Price  
- Allergic/Non-Allergic Reactions  
- Good/Bad Customer Service  

---

### 🔍 Our Solution

We built a sentiment classification system that:

- Analyzes customer feedback across **Top 10 Pakistani Brands**:  
  **Maria.B, SanaSafinaz, Sapphire, Limelight, J., Bonanza Satrangi, Asim Jofa, Gul Ahmed, Khaadi**

- Trains a `bert-base-uncased` model on labeled sentiment data

- Deploys a Flask app for **real-time brand recommendation** based on user input

---

## 🛠️ Technologies Used

| Technology       | Purpose                                  |
|------------------|-------------------------------------------|
| Python 3.10+     | Core programming language                 |
| Pandas, NumPy    | Data handling and preprocessing           |
| Scikit-learn     | Evaluation metrics                        |
| Transformers     | BERT model for sentiment classification   |
| Flask            | Web app deployment                        |
| Matplotlib, Seaborn | Visualization and analysis             |

---

## 🚀 Getting Started

### ✅ Output on GitHub:

### 1. 📥 Clone the Repository

```
git clone https://github.com/Momna19Abid/Sentiment-Driven-Clothing-Brand-Recommendation-System.git
cd Sentiment-Driven-Clothing-Brand-Recommendation-System
```

### 2. 📦 Create a Virtual Environment (Optional but Recommended)

```
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate   # On Mac/Linux
```

### 3. 📦 Install Required Libraries

```
pip install -r requirements.txt
```

### 4. ▶️ Run the Application

```
python app.py
```

This will:
- Load the pre-trained BERT model.
- Accept input features through a Flask interface.
- Predict the best brand based on sentiment scores.

---

### 📊 Sample Output:

#### 🧾 Input Features:

High Fabric Quality, Good Customer Service, Non-Allergic Reaction

#### 🧠 Predicted Best Brand:

Maria.B, Sapphire, Limelight

---

### 📚 Key Concepts:
**Sentiment Analysis** – Determine positive or negative sentiment from textual reviews.  
**BERT** – Pre-trained language model fine-tuned on custom brand feedback data.  
**Flask** – Lightweight Python framework to deploy ML applications.  
**Real-World Dataset** – Authentic customer feedback from Top Pakistani clothing brands.  

### 💡 What You’ll Learn???
- How sentiment analysis works using BERT.
- How to train a custom classifier on labeled review data.
- How to deploy a recommendation engine using Flask.
- How real-world data improves online shopping experiences.

#### 🙌 Author:
👤 Momna Abid 🎓 | Computer Science Graduate | Machine Learning & AI Enthusiast | 🔗 LinkedIn: www.linkedin.com/in/momna-python-ml

### ⭐️ If you liked this project

##### ⭐ Star the repo.

##### 🍴 Fork and experiment.

##### 📢 Share your outputs on LinkedIn.

##### 🔧 Feel free to open issues or contribute via Pull Requests on LinkedIn.




---

