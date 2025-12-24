import pandas as pd
import re
from fastapi import FastAPI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="HS Code Classifier API")

# ==============================
# Helper Functions
# ==============================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def generate_reason(product, category_name, product_category=None):
    reasons = []

    for word in product.split():
        if word in category_name.lower():
            reasons.append(f"keyword '{word}' matches HS item name")

    if product_category:
        reasons.append(f"matches product type '{product_category}'")

    if not reasons:
        reasons.append("overall semantic similarity with HS item name")

    return "; ".join(reasons)

# ==============================
# Load HS Database (ONCE)
# ==============================

hs_df = pd.read_excel("hs_database.xlsx")
hs_df.columns = hs_df.columns.str.strip()
hs_df["Item English Name"] = hs_df["Item English Name"].apply(clean_text)
hs_df["combined_text"] = hs_df["Item English Name"]

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(hs_df["combined_text"])

# ==============================
# HS Matching Logic
# ==============================

def suggest_hs_codes(query_text, top_n=2):
    q_vec = vectorizer.transform([query_text])
    scores = cosine_similarity(q_vec, X)[0]
    top_indices = scores.argsort()[-top_n:][::-1]

    results = []
    for idx in top_indices:
        row = hs_df.iloc[idx]
        results.append({
            "item_seq": row["Item"],
            "hs_code": row["HS Code"],
            "reason": generate_reason(query_text, row["Item English Name"])
        })
    return results

# ==============================
# API Endpoints
# ==============================

@app.get("/")
def health_check():
    return {"status": "HS Classifier is running"}

@app.get("/classify")
def classify(product_name: str, category: str):
    product_name_clean = clean_text(product_name)
    category_clean = clean_text(category)

    query_text = product_name_clean + " " + category_clean
    hs_results = suggest_hs_codes(query_text, top_n=2)

    return {
        "product_name": product_name,
        "category": category,
        "results": hs_results
    }
