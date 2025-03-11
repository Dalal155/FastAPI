import os
import json
import pandas as pd
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Firebase
firebase_credentials = os.getenv("FIREBASE_CONFIG")
if not firebase_credentials:
    raise Exception("Firebase configuration not found in environment variables.")

try:
    cred = credentials.Certificate(json.loads(firebase_credentials))
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firebase initialized successfully!")
except Exception as e:
    print(f"Error initializing Firebase: {e}")
    db = None  # Set to None if initialization fails

# Create FastAPI application
app = FastAPI()

# Allow requests from other origins (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # This allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (POST, GET, etc)
    allow_headers=["*"],
)

# Incoming data model
class Initiative(BaseModel):
    mobadrahName: str
    briefDescription: str
    field: str  # Field for initiative category

# Load data from Firebase
def load_data_from_firebase():
    if db is None:
        return pd.DataFrame(columns=['id', 'mobadrahName', 'field', 'briefDescription'])

    try:
        # Adjust 'Mobadrat' to your collection name
        collection_ref = db.collection("Mobadrat")
        docs = collection_ref.stream()

        data = []
        for doc in docs:
            doc_data = doc.to_dict()
            # Assuming your Firestore documents have the fields you need
            data.append({
                "id": doc.id,  # Use document ID as Initiative ID
                "mobadrahName": doc_data.get("mobadrahName", ""),
                "field": doc_data.get("field", ""),
                "briefDescription": doc_data.get("briefDescription", "")
            })
        return pd.DataFrame(data)

    except Exception as e:
        print(f"Error loading data from Firebase: {e}")
        return pd.DataFrame(columns=['id', 'mobadrahName', 'field', 'briefDescription'])

# Load data when the app starts
data = load_data_from_firebase()

# Advanced text preprocessing function
def advanced_preprocess(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s\u0600-\u06FF]', '', text)  # Retain Arabic characters and words
    text = re.sub(r'\b(project|initiative|campaign|program|workshop|skills|مبادرة|مشروع|برنامج|ورشة)\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Prepare TF-IDF
tfidf_vectorizer = None
tfidf_matrix = None

if not data.empty:
    data['mobadrahName'] = data['mobadrahName'].fillna('').apply(advanced_preprocess)
    data['briefDescription'] = data['briefDescription'].fillna('').apply(advanced_preprocess)
    data['Combined Text'] = data['mobadrahName'] + " " + data['briefDescription']

    try:
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = tfidf_vectorizer.fit_transform(data['Combined Text'])
    except ValueError:
        print("⚠️ No data after cleaning! Similarity model not created.")
        tfidf_vectorizer = None
        tfidf_matrix = None
else:
    print("⚠️ No data loaded from Firebase. Similarity features will not work.")

# Similarity checking threshold
SIMILARITY_THRESHOLD = 0.7

# Function to check for duplicates
def check_for_duplicates(initiative: Initiative):
    if tfidf_vectorizer is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="⚠️ Database not ready for similarity analysis.")

    input_text = advanced_preprocess(initiative.mobadrahName.strip() + " " + initiative.briefDescription.strip())
    input_vector = tfidf_vectorizer.transform([input_text])
    similarity_scores = cosine_similarity(input_vector, tfidf_matrix)[0]

    duplicate_indices = [i for i, score in enumerate(similarity_scores) if score >= SIMILARITY_THRESHOLD]
    return duplicate_indices, similarity_scores

# Endpoint to register initiative
@app.post("/register_initiative")
async def register_initiative(initiative: Initiative):
    try:
        duplicate_indices, similarity_scores = check_for_duplicates(initiative)

        if duplicate_indices:
            duplicates = [
                {
                    "id": data.iloc[i]['id'],
                    "mobadrahName": data.iloc[i]['mobadrahName'],
                    "field": data.iloc[i]['field'],
                    "Similarity Score": round(similarity_scores[i], 2)
                }
                for i in duplicate_indices
            ]
            raise HTTPException(status_code=400, detail="The initiative can't be registered because there is another initiative with the same idea.")

        # Add to Firebase
        if db is None:
            raise HTTPException(status_code=500, detail="Firebase is not initialized. Cannot save data.")
        try:
            doc_ref = db.collection("Mobadrat").document()  # Automatically generate an ID
            doc_ref.set({
                "mobadrahName": initiative.mobadrahName,
                "briefDescription": initiative.briefDescription,
                "field": initiative.field
            })
            # Refresh the data from Firebase after adding a new item, and recompute TF-IDF
            global data, tfidf_vectorizer, tfidf_matrix  # Declare as global to modify them
            data = load_data_from_firebase()
            if not data.empty:
                data['mobadrahName'] = data['mobadrahName'].fillna('').apply(advanced_preprocess)
                data['briefDescription'] = data['briefDescription'].fillna('').apply(advanced_preprocess)
                data['Combined Text'] = data['mobadrahName'] + " " + data['briefDescription']
                tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
                tfidf_matrix = tfidf_vectorizer.fit_transform(data['Combined Text'])
            else:
                tfidf_vectorizer = None
                tfidf_matrix = None

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error saving to Firebase: {e}")

        return {"message": "✅ مبادرة تم تسجيلها بنجاح!"}

    except HTTPException as e:
        raise e  # Re-raise HTTPExceptions to propagate the error
    except Exception as e:
        print(f"Error: {e}")  # Log the error for debugging
        raise HTTPException(status_code=400, detail=str(e))

# Root endpoint
@app.get("/")
def root():
    return {"message": "🚀 API للكشف عن التكرار جاهزة للعمل!"}