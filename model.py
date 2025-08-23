#WangchanBERTa
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

#Random Forest
import joblib
import pandas as pd
import numpy as np

#Load Models
#- WangchanBERTa
content_model_dir = "putthaimastan/wangchanberta_thai_sms_content_classification"
wanchanberta_model = AutoModelForSequenceClassification.from_pretrained(content_model_dir,device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(content_model_dir)
wangchanberta_labels = ['safe', 'spam', 'scam']

#- Random Forest
rf_model_path = "models/link_classifier_73/best_sms_link_model.pkl"
tfidf_path = "models/link_classifier_73/tfidf_vectorizer.pkl"

rf_model = joblib.load(rf_model_path)
tfidf_vectorizer = joblib.load(tfidf_path)
rf_features = ['short_url', 'sus_tld', 'has_https', 'has_http', 'sus_redirect', 'ip_th', 'domain_age_lesser_than_three_month']

#Predict functions
#- SMS Content
def predict_sms_content(content_text):
    
    # Tokenize and encode the input text
    inputs = tokenizer(content_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        outputs = wanchanberta_model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1) # Convert logits to probabilities(0-1) using softmax function 
        predicted_class = torch.argmax(probs, dim=1).item()
    
    # Get Scam Class probability
    #scam_prob = round(probs.flatten().tolist()[wangchanberta_labels.index('scam')], 2)
        
    result = {
        "label": wangchanberta_labels[predicted_class],
        "probabilities": dict(zip(wangchanberta_labels, probs.flatten().tolist())),
        #"scam_probability": scam_prob
        }
    return result

#- SMS Link
def predict_sms_link(link_text,rf_features_dict):
    ''' 
    link_text: str
    rf_features_dict: dict
        {
            'short_url': bool 0/1,
            'sus_tld': bool 0/1,
            'has_https': bool 0/1,
            'has_http': bool 0/1,
            'sus_redirect': bool 0/1,
            'ip_th': bool 0/1,
            'domain_age_lesser_than_three_month': bool 0/1,
        }
    '''
    
    # Transform the input text using the TF-IDF vectorizer
    link_tfidf = tfidf_vectorizer.transform([link_text])
    
    # Convert rf_features dictionary to a DataFrame
    link_features_df = pd.DataFrame([rf_features_dict])[rf_features]
    
    # Convert Dataframe to NumPy Array
    link_features_arr = link_features_df.values
    
    # Convert sparse matrix to dense array for hstack
    link_tfidf_dense = link_tfidf.toarray()
    
    # Combine the features (Add Columns)
    model_input_features = np.hstack([link_features_arr, link_tfidf_dense])
    
    # Random Forest Prediction
    predict_class = rf_model.predict(model_input_features)[0]
    probs = rf_model.predict_proba(model_input_features)[0]
    
    # Map the predicted class to labels
    rf_labels_map = {0: 'safe', 1: 'spam', 2: 'scam'}
    predicted_label = rf_labels_map[predict_class]
    
    # Get Scam Class probabilities 
    #scam_prob = float(probs[2])
    
    result = {
        "label": predicted_label,
        "probabilities": {
            "safe": float(probs[0]),
            "spam": float(probs[1]),
            "scam": float(probs[2])
        },
        #"scam_probability": round(scam_prob, 2)
    }
    
    return result

#Calculate Risk Score (Majority Voting)
def compute_sms_risk_score(content_result, link_result):
    
    # Extract probabilities from WangchanBERTa content prediction (p(WangchanBERTa))
    content_probs = content_result['probabilities']
    content_safe = content_probs['safe']
    content_spam = content_probs['spam']
    content_scam = content_probs['scam']
   
    # Extract probabilities from Random Forest link prediction (p(Random Forest))
    link_probs = link_result['probabilities']
    link_safe = link_probs['safe']
    link_spam = link_probs['spam']  
    link_scam = link_probs['scam']
    
    # Declare variables using Majority Voting
    p_scam = max(content_scam, link_scam)
    p_spam = max(content_spam, link_spam)
    p_safe = max(content_safe, link_safe)
    
    # Calculate Risk Score
    risk_score = (p_scam/(p_scam + p_spam + p_safe)) * 100
    
    return int(risk_score)