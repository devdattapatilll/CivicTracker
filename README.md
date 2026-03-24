# CivicTrack — AI-Powered Civic Issue Detection Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=flat-square&logo=flask&logoColor=white)
![YOLOv12](https://img.shields.io/badge/YOLOv12-Object%20Detection-00B4D8?style=flat-square)
![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react&logoColor=black)
![Firebase](https://img.shields.io/badge/Firebase-Auth+Firestore-FFCA28?style=flat-square&logo=firebase&logoColor=black)
![Render](https://img.shields.io/badge/Render-Backend-46E3B7?style=flat-square&logo=render&logoColor=white)
![Vercel](https://img.shields.io/badge/Vercel-Frontend-000000?style=flat-square&logo=vercel&logoColor=white)

A community-driven platform for reporting civic issues — potholes, garbage, waterlogging, and road cracks — with AI-driven automatic detection and classification.

[Live Demo](https://capstone-project-vvlg.vercel.app) | [Backend API](https://civictrack-ml.onrender.com/health) | [Deployment Guide](docs/DEPLOYMENT.md)

</div>

---

## Overview

CivicTrack enables citizens to report infrastructure problems in their community using photos and text descriptions. The platform applies machine learning at two levels:

1. **Computer vision** (YOLOv12 / YOLOv8 object detection) scans uploaded images for potholes, garbage, waterlogging, and road cracks.
2. **Natural language processing** (TF-IDF + LinearSVC) classifies the text description into one of five categories.

Issues are stored in Firebase Firestore with real-time synchronization. Administrators can manage issue status from Pending to In Progress to Resolved.

---

## System Architecture

```
Frontend (Vercel)            ML Backend (Render)           Database (Firebase)
--------------------         -----------------------       ---------------------
| HTML5 / CSS3     |  REST   | Flask REST API       |      | Authentication    |
| React 18 (CDN)   | -----> | 4 YOLO detection     |      | Firestore (NoSQL) |
| Firebase SDK      |        | models               |      | Cloud Storage     |
|                   | -----> | NLP text classifier  |      |                   |
| SPA with sidebar  |        | Gunicorn server      |      | Real-time sync    |
|                   | <----- |                      |      |                   |
--------------------         -----------------------       ---------------------
        |                                                         ^
        |_________________________________________________________|
                          Firebase SDK (direct)
```

### Frontend

- Single-page application with a sidebar navigation layout
- Four pages: Home (dashboard), Report Issue, How It Works, Team
- React 18 loaded via CDN for dynamic components (stats, issue cards, category filters)
- Firebase JavaScript SDK for authentication and database operations
- Responsive design with mobile hamburger navigation

### ML Backend

- Flask REST API served by Gunicorn on Render (free tier)
- Four YOLO object detection models loaded lazily to conserve memory
- TF-IDF + LinearSVC text classifier for automatic category assignment
- CORS enabled for cross-origin requests from the Vercel frontend
- 16 MB maximum payload size for image uploads

### Database Layer

- Firebase Authentication with Google Sign-In and email/password
- Cloud Firestore for storing issues and help requests as documents
- Firebase Cloud Storage for uploaded images
- Server-side timestamps and real-time snapshot listeners

---

## Project Structure

```
CapstoneProject/
|
|-- app.py                     Flask ML backend (REST API, 6 endpoints)
|-- index.html                 Frontend single-page application
|-- styles.css                 Stylesheet (sidebar layout, responsive)
|-- requirements.txt           Python dependencies
|-- Procfile                   Render deployment start command
|-- README.md                  This file
|-- .gitignore                 Git exclusion rules
|
|-- models/                    Trained ML weights (~76 MB total)
|   |-- pothole.pt             YOLOv12s - pothole detection
|   |-- garbage.pt             YOLOv8n  - garbage/waste detection
|   |-- waterlog.pt            YOLOv8n  - waterlogging detection
|   |-- crack.pt               YOLOv8n  - road crack detection
|   |-- classifier.pkl         TF-IDF + LinearSVC text classifier
|
|-- training/                  Model training scripts
|   |-- train_colab.py         Google Colab training guide (T4 GPU)
|   |-- train_yolo.py          Local YOLO training (CPU, demo quality)
|   |-- train_nlp.py           NLP classifier training script
|
|-- assets/                    Static assets (diagrams, images)
|   |-- workflow.png           Workflow schematic diagram
|   |-- architecture.png       Architecture schematic diagram
|
|-- docs/                      Documentation
    |-- DEPLOYMENT.md          Step-by-step deployment instructions
```

---

## AI Models

### Computer Vision — YOLOv12 / YOLOv8

| Model File | Architecture | Detection Target | Size |
|------------|-------------|------------------|------|
| pothole.pt | YOLOv12s | Potholes, road surface damage | 18.1 MB |
| garbage.pt | YOLOv8n | Waste accumulation, overflowing bins | 18.1 MB |
| waterlog.pt | YOLOv8n | Flooding, waterlogged roads | 21.5 MB |
| crack.pt | YOLOv8n | Road cracks, surface fractures | 18.1 MB |

Models are loaded lazily on first request to conserve memory on free-tier hosting.

### Text Classification — NLP

| Component | Details |
|-----------|---------|
| Vectorizer | TF-IDF (unigrams + bigrams, 5000 features) |
| Classifier | Calibrated LinearSVC |
| Categories | Roads, Garbage, Water Leakage, Electricity, Other |
| Training samples | ~200 hand-annotated civic complaint texts |

The classifier outputs a category label and a confidence score. Issues with confidence below 65% are flagged for manual review.

---

## API Reference

Base URL: `https://civictrack-ml.onrender.com`

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Returns model load status and version info |
| POST | /classify-text | Classify text input into a category |
| POST | /detect | Run a specific YOLO model on an image |
| POST | /detect-all | Run all four YOLO models on one image |
| POST | /analyze | Full pipeline: text classification + image detection |
| POST | /annotate | Return annotated image with bounding boxes |

### Example: Text Classification

Request:
```bash
curl -X POST https://civictrack-ml.onrender.com/classify-text \
  -H "Content-Type: application/json" \
  -d '{"text": "Huge pothole near school causing accidents"}'
```

Response:
```json
{
  "category": "Roads",
  "confidence": 0.86,
  "needs_review": false
}
```

### Example: Full Analysis

Request body:
```json
{
  "text": "Garbage pile on main street",
  "image_base64": "data:image/jpeg;base64,..."
}
```

Response body:
```json
{
  "text_result": {
    "category": "Garbage",
    "confidence": 0.91,
    "needs_review": false
  },
  "image_result": {
    "has_detection": true,
    "category": "garbage",
    "confidence": 0.78,
    "count": 2
  }
}
```

---

## Setup and Installation

### Prerequisites

- Python 3.10 or later
- pip package manager

### 1. Clone the Repository

```bash
git clone https://github.com/devdattapatilll/CapstoneProject.git
cd CapstoneProject
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Backend Locally

```bash
python app.py
```

The API will start on `http://localhost:5000`. Verify with:
```bash
curl http://localhost:5000/health
```

### 4. Serve the Frontend

Open `index.html` directly in a browser, or use a local server:
```bash
python -m http.server 8080
```

Then visit `http://localhost:8080`.

Note: For local development, change `ML_SERVICE_URL` in `index.html` to `http://localhost:5000`.

---

## Model Training

### Option A: Google Colab (Recommended)

Upload `training/train_colab.py` to Google Colab with a T4 GPU runtime. This script downloads datasets from Roboflow and trains all four YOLO models with 30 epochs for production-quality results.

### Option B: Local CPU

```bash
python training/train_yolo.py
```

This runs a lightweight 3-epoch training cycle suitable for pipeline verification. Accuracy will be lower than Colab-trained models.

### NLP Classifier

```bash
python training/train_nlp.py
```

Generates `models/classifier.pkl` with ~200 training samples across five categories.

---

## Deployment

| Service | Purpose | URL | Tier |
|---------|---------|-----|------|
| Render | Flask ML backend | civictrack-ml.onrender.com | Free |
| Vercel | Static frontend | capstone-project-vvlg.vercel.app | Hobby (free) |
| Firebase | Auth + DB + Storage | — | Spark (free) |

Note: Render free-tier instances sleep after 15 minutes of inactivity. The first request after sleep takes approximately 50 seconds while the server cold-starts.

For detailed deployment instructions, see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

---

## Technology Stack

| Layer | Technologies |
|-------|-------------|
| Frontend | HTML5, CSS3, React 18 (CDN), Babel |
| Backend | Python, Flask, Flask-CORS, Gunicorn |
| Computer Vision | Ultralytics (YOLOv12s, YOLOv8n) |
| NLP | scikit-learn (TF-IDF, LinearSVC, CalibratedClassifierCV) |
| Database | Firebase Firestore |
| Authentication | Firebase Auth (Google OAuth, email/password) |
| File Storage | Firebase Cloud Storage |
| Geocoding | OpenStreetMap Nominatim API |
| Backend Hosting | Render.com |
| Frontend Hosting | Vercel |

---

## Team

| Name | Role | Responsibilities |
|------|------|-----------------|
| Sarthak Sant | Backend Developer, UI/UX Design | Flask API, interface design, design system |
| Mayank Rawat | Authentication and Admin | Firebase Auth, admin panel, access control |
| Rizul Pathania | Frontend Developer | React components, responsive UI, cross-browser testing |
| Devdatta Patil | ML Models and Deployment | YOLO training, NLP classifier, Render/Vercel deployment |

---

## License

This project was developed as an academic Capstone Project.

---

<div align="center">

CivicTrack — AI-Augmented Civic Reporting Platform

[Live Demo](https://capstone-project-vvlg.vercel.app)

</div>
