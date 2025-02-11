from fastapi import FastAPI, File, UploadFile
import os
import shutil
import fitz  # PyMuPDF for PDF parsing
import re
import spacy
import nltk
import httpx  # Added for sending data to ML backend
from collections import defaultdict
from fastapi.middleware.cors import CORSMiddleware

# Download necessary NLP models
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")

app = FastAPI()

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ML_BACKEND_URL = "http://ml-backend-url/predict"  # Replace with actual ML backend URL

# Extract text from PDF using PyMuPDF
def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text

# Extract details from resume
def extract_resume_details(text):
    details = defaultdict(list)

    # Extract Name (Assuming the first line contains name)
    lines = text.split("\n")
    if lines:
        details["name"] = lines[0].strip()

    # Extract Email
    email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    if email_match:
        details["email"] = email_match.group()

    # Extract Phone Number
    phone_match = re.search(r"\+?\d{10,15}", text)
    if phone_match:
        details["phone"] = phone_match.group()

    # Extract Education
    edu_keywords = ["education", "qualification", "degree", "university", "college", "bachelor", "master"]
    for line in lines:
        if any(word.lower() in line.lower() for word in edu_keywords):
            details["education"].append(line.strip())

    # Extract Work Experience
    work_keywords = ["experience", "intern", "worked", "employment", "company", "role"]
    for line in lines:
        if any(word.lower() in line.lower() for word in work_keywords):
            details["work_experience"].append(line.strip())

    # Extract Projects
    project_keywords = ["project", "developed", "implemented", "created"]
    for line in lines:
        if any(word.lower() in line.lower() for word in project_keywords):
            details["projects"].append(line.strip())

    # Extract Skills
    skills_keywords = ["skills", "technologies", "expertise", "programming"]
    for line in lines:
        if any(word.lower() in line.lower() for word in skills_keywords):
            details["skills"].append(line.strip())

    # Extract Extracurricular Activities
    extra_keywords = ["club", "volunteer", "organizer", "hackathon", "event"]
    for line in lines:
        if any(word.lower() in line.lower() for word in extra_keywords):
            details["extracurricular_activities"].append(line.strip())

    return details

# Check for Missing Fields
def check_missing_fields(details):
    required_sections = ["name", "email", "phone", "education", "work_experience", "projects", "skills"]
    missing_fields = [section for section in required_sections if not details.get(section)]
    return missing_fields

# Send resume data to ML backend
async def send_to_ml_backend(resume_data):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(ML_BACKEND_URL, json=resume_data)
            return response.json()
        except Exception as e:
            return {"error": str(e)}

@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = extract_text_from_pdf(file_location)
    extracted_details = extract_resume_details(text)
    missing_fields = check_missing_fields(extracted_details)

    # Send extracted data to ML backend
    ml_response = await send_to_ml_backend(extracted_details)

    return {
        "extracted_data": extracted_details,
        "missing_fields": missing_fields,
        "ml_response": ml_response  # Response from ML backend
    }
