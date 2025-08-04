from flask import Flask, render_template, request
import pandas as pd
import torch
from transformers import BertTokenizer
from model import SentimentClassifier
from functools import lru_cache

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained("tokenizer")
model = SentimentClassifier().to(device)
model.load_state_dict(torch.load("sentiment_model.pt", map_location=device))
model.eval()

# Load data
df = pd.read_csv("Brand_data.csv")
unique_features = df["Feature"].unique().tolist()

# Cache model predictions
@lru_cache(maxsize=10000)
def predict_sentiment_cached(feature, brand):
    text = f"{feature} {tokenizer.sep_token} {brand}"
    encoding = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        prediction = torch.argmax(outputs, dim=1).item()
    return prediction

# Feature-wise brand recommendation
def get_best_brand_per_feature(selected_indexes):
    best_brands = {}

    for idx in selected_indexes:
        if 1 <= idx <= len(unique_features):
            feature = unique_features[idx - 1]
            feature_df = df[df["Feature"] == feature]

            if idx % 2 == 1:
                filtered_df = feature_df[feature_df["Response"] == 1]
            else:
                filtered_df = feature_df[feature_df["Response"] == 0]

            if not filtered_df.empty:
                top_brand = filtered_df["Brand"].value_counts().idxmax()
                best_brands[feature] = top_brand

    return best_brands

# Overall best brand
def get_overall_best_brand():
    brands = df["Brand"].unique()
    score_counts = {b: 0 for b in brands}
    for _, row in df.iterrows():
        if predict_sentiment_cached(row["Feature"], row["Brand"]) == 1:
            score_counts[row["Brand"]] += 1
    return max(score_counts.items(), key=lambda x: x[1])[0]

# Routes
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", features=unique_features)

@app.route("/results", methods=["POST"])
def results():
    selected = request.form.getlist("features")
    selected_indexes = [int(i) for i in selected]
    best_brands = get_best_brand_per_feature(selected_indexes)
    overall_brand = get_overall_best_brand()

    return render_template("results.html",
                           best_brands=best_brands,
                           overall=overall_brand)

@app.route("/overall", methods=["GET"])
def overall():
    overall_brand = get_overall_best_brand()
    return render_template("overall.html", overall=overall_brand)

if __name__ == "__main__":
    app.run(port=3000, debug=True)
