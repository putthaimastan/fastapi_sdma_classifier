from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import predict_sms_content, predict_sms_link, compute_sms_risk_score
# For Data Preprocessing Functions
import re
from urllib.parse import urlparse
import requests
#API
from dotenv import load_dotenv
import os
from datetime import datetime
import whois
import socket


app = FastAPI()

app.add_middleware(CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load environment variables and define API key
load_dotenv(override=False)
API_KEY = os.getenv('API_NINJAS_KEY')

# Define the input data model
class SmsInput(BaseModel):
    sms_text: str

# Extract link from SMS text
def split_sms_text(sms_text):
    # Regular expression to find URLs in the text
    url_pattern = r'(https?://\S+|www\.\S+|\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:/\S*)?)'
    urls_list = re.findall(url_pattern, sms_text)
    
    #- url_text: str
    url_text = urls_list[0] if urls_list else None
    
    # Remove URLs from the original text
    #- content_text: str
    content_text = re.sub(url_pattern, '', sms_text).strip()
    
    return content_text, url_text

# Extract Features for Random Forest
def extract_rf_features(url: str) -> dict:
    if url is None:
        return {
            'short_url': 0,
            'sus_tld': 0,
            'has_https': 0,
            'has_http': 0,
            'sus_redirect': 0,
            'ip_th': 0,
            'domain_age_lesser_than_three_month': 0
        }
    #Features extraction logic
    # - Short URL
    short_url_list = ['bit.ly', 'tinyurl.com', 'goo.gl', 'shorturl.asia', 't.co', 'cutt.ly']
    parsed_url = urlparse(url) 
    domain = parsed_url.netloc.lower()
    final_domain = domain
    
    is_short_url = 1 if any(short_url in domain for short_url in short_url_list) else 0
    
    # - Suspicious TLD
    sus_tld_list = ['.xyz', '.me', '.club', '.cc']
    tld = domain.split('.')[-1]
    is_sus_tld = 1 if tld in sus_tld_list else 0
    
    # - Has HTTPS
    has_https = 1 if url.startswith('https://') else 0
    # - Has HTTP
    has_http = 1 if url.startswith('http://') and not has_https else 0
    
    # - Suspicious Redirect
    is_sus_redirect = 0
    try:
        if url.startswith('http://') or url.startswith('https://'):
            resp = requests.head(url, allow_redirects=True, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
            final_url = resp.url.lower()
            
            # Check Red flags in final URL
            red_flags = ['login', 'signin', 'account', 
                         'เข้าสู่ระบบ', 'สมัครสมาชิก', 'register', 
                         'line.me']
            
            original_domain = parsed_url.netloc.lower()
            final_domain = urlparse(final_url).netloc.lower()
            domain_changed = (original_domain != final_domain)
            
            # Check if final URL contains red flags
            sus_keywords = any(flag in final_url for flag in red_flags)
            
            if domain_changed or sus_keywords:
                is_sus_redirect = 1
    except:
        pass
    
    # - IP Location
    is_ip_th = 0
    try:
        url_api_url = 'https://api.api-ninjas.com/v1/urllookup?url={}'.format(final_domain) #
        url_response = requests.get(url_api_url, headers={'X-Api-Key': API_KEY }) #
        
        if url_response.status_code == requests.codes.ok: #
            url_data = url_response.json() #
            is_valid = url_data.get('is_valid', False)
            ip_location = url_data.get('country_code', '')
            
            if is_valid and ip_location.lower() == 'th':
                is_ip_th = 1
            else:
                is_ip_th = 0            
        else: #is_ip_th = 0
            print('Error:', url_response.status_code, url_response.text) #
            
    except Exception as e : #is_ip_th = 0
        print('Exception', e)
    
    # - Domain Age (ใช้ python-whois)
    is_domain_age_lesser_than_three_month = 0
    try:
        # ตั้ง timeout สำหรับ whois
        prev_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(5.0)  # 5 วินาที

        w = whois.whois(domain)
        creation_date = getattr(w, 'creation_date', None)

        # ถ้า creation_date เป็น list ให้ใช้ตัวแรกที่ไม่ None
        if isinstance(creation_date, list):
            creation_date = next((d for d in creation_date if d is not None), None)

        if creation_date:
            # ลบ timezone ถ้ามี
            if hasattr(creation_date, 'tzinfo') and creation_date.tzinfo is not None:
                creation_date = creation_date.replace(tzinfo=None)
            domain_age_days = (datetime.utcnow() - creation_date).days
            if domain_age_days <= 90:
                is_domain_age_lesser_than_three_month = 1
    except Exception:
        # ถ้า WHOIS fail → 0
        is_domain_age_lesser_than_three_month = 0
    finally:
        try:
            socket.setdefaulttimeout(prev_timeout)
        except Exception:
            pass
    
    return {
        'short_url': is_short_url,
        'sus_tld': is_sus_tld,
        'has_https': has_https,
        'has_http': has_http,
        'sus_redirect': is_sus_redirect,
        'ip_th': is_ip_th,
        'domain_age_lesser_than_three_month': is_domain_age_lesser_than_three_month
    }
        

@app.post("/predict_risk_score")
async def predict_risk_score(sms_input_data: SmsInput):
    try:
        content_text, url_text = split_sms_text(sms_input_data.sms_text)
        # Predict content risk using WangchanBERTa
        content_result = predict_sms_content(content_text)
        # Extract features for the link if it exists
        rf_features_dict = extract_rf_features(url_text)
        
        
        if url_text: #If there is a URL in the SMS text
            
            # Predict link risk using Random Forest
            link_result = predict_sms_link(url_text, rf_features_dict)
            
        else: #If there is no URL in the SMS text
            link_result = {
                "label": "safe",
                "probabilities": {
                    "safe": 1.0,
                    "spam": 0.0,
                    "scam": 0.0
                }
            }
        
        # Compute the risk score based on content and link predictions
        risk_score_pred = compute_sms_risk_score(content_result, link_result)
        return {'risk_score': risk_score_pred}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/")
async def root():
    return {"message": "This is Risk Scoring API"}
    