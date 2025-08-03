👗 Sentiment-Driven Clothing Brand Recommendation System
This  project addresses a common challenge in online shopping: recommending the most suitable Pakistani clothing brand based on customer sentiment.  Features—like fabric quality, price, customer service, allergic reactions,color theme etc —Our system predicts the best brand using real-world reviews data from customers who already have experienced different brands and BERT-Transformer based sentiment classification model is used .

📌 What’s Inside?
✅ Problem Statement
Pakistani customers often struggle with choosing the best clothing brand that meets specific preferences like:

Good/Bad Color Theme

High/Low Fabric Quality

High/Low Price

Allergic/Non-Allergic Reactions

Good/Bad Customer Service

🔍 Our Solution
Built a sentiment classification system that:

Analyzes customer feedback across Top 10 Pakistani Brands:
Maria.B, SanaSafinaz, Sapphire, Limelight, J., Bonanza Satrangi, Asim Jofa, Gul Ahmed, Khaadi

Trains a BERT-based model on labeled sentiment data

Deploys a Flask app for real-time brand recommendation based on user input

🛠️ Technologies Used:

Python 3.10+	
Pandas, NumPy
Scikit-learn	
Transformers (BERT)	Pretrained NLP model for classification
Flask	Web app deployment
Matplotlib, Seaborn

🚀 Getting Started
1. 📥 Clone the Repository

git clone https://github.com/yourusername/Sentiment-Driven-Clothing-Brand-Recommendation System.git
cd Sentiment-Driven-Clothing-Brand-Recommendation-System

3. 📦 Create a Virtual Environment (Optional but Recommended)
python -m venv venv
venv\Scripts\activate   # On Windows
# source venv/bin/activate   # On Mac/Linux

3. 📦 Install Required Libraries

pip install -r requirements.txt

5. ▶️ Run the Application

python app.py
This will:

Load the pre-trained BERT model

Accept input features through a Flask interface

Predict the best brand based on sentiment scores

📊 Sample Output
🧾 Input Features:
High Fabric Quality, Good Customer Service, Non-Allergic Reaction

🧠 Predicted Best Brand:
Maria.B

💬 Explanation:
The BERT model classifies customer sentiment (positive/negative) for each brand-feature pair, and the Flask app ranks the brands accordingly.

📂 Folder Structure
📁 Sentiment-Driven-Clothing-Brand-Recommendation-System/
│
├── app.py                     # Flask Web Interface
├── model/bert_model.pt        # Trained BERT model (to be added)
├── data/                      # Dataset (customer reviews)
├── utils.py                   # Utility functions (tokenization, prediction)
├── templates/                 # HTML templates for Flask app
├── static/brand_logos/        # Brand logo images
├── requirements.txt           # Python dependencies
├── README.md                  # This file

📚 Key Concepts
Concept	Description
Sentiment Analysis	Determine positive or negative sentiment from textual reviews
BERT	Pre-trained language model fine-tuned on custom data
Flask	Lightweight Python web framework for deployment
Real-World Dataset	Authentic customer feedback from top clothing brands in Pakistan

💡 What You’ll Learn
How sentiment analysis works using BERT

How to train a custom classifier on labeled review data

How to deploy a recommendation engine with Flask

How real-world data drives smarter e-commerce experiences

🙌 Author
👤 Momna Abid |Computer Science Graduate |Machine Learning & AI Enthusiast
🔗 LinkedIn: www.linkedin.com/in/momna-python-ml 

⭐️ If you liked this project
Star the repo ⭐

Fork it and experiment 💻

Share your outputs on LinkedIn!

Feel free to open issues or contribute via Pull Requests 🤝













