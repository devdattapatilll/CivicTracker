[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://devdatta-civictrack.vercel.app)
[![Backend](https://img.shields.io/badge/API-Render-blue)](https://civictrack-ml.onrender.com/health)

# CivicTrack — AI-Powered Community Issue Tracker

> Report civic issues like potholes, garbage, and road cracks. Our AI engine uses YOLO object detection and NLP text classification to automatically categorize issues from uploaded photos and descriptions.

**Live Demo**: [devdatta-civictrack.vercel.app](https://devdatta-civictrack.vercel.app)
**ML Backend**: Hosted on [Render](https://civictrack-ml.onrender.com/health)

---

## Features

- **AI Image Detection** — 3 YOLO models scan uploaded photos for potholes, garbage, and road cracks
- **NLP Text Classification** — TF-IDF + LinearSVC classifier categorizes text descriptions
- **Real-time Dashboard** — Live issue tracking with status updates (Pending → In Progress → Resolved)
- **Google Sign-In** — Firebase Authentication with admin panel
- **GPS Location** — Auto-detect location or search with OpenStreetMap
- **Responsive Design** — Works on desktop and mobile with sidebar navigation

---

## Issue Categories

| Category | AI Model | Description |
|---|---|---|
| Roads (Potholes) | `pothole.pt` (YOLO) | Detects potholes and road damage |
| Garbage | `garbage.pt` (YOLO) | Detects waste and overflowing bins |
| Road Cracks | `crack.pt` (YOLO) | Detects road cracks and surface damage |
| Other | NLP only | Catch-all for water, electricity, misc |

---

## Architecture Overview

The platform follows a client-server architecture:
- **Frontend** (Vercel) → static HTML/CSS/JS with React components
- **ML Backend** (Render) → Flask REST API with YOLO + NLP models
- **Database** (Firebase) → Firestore for issues, Authentication for users

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML5, CSS3, React 18, Babel |
| Backend (ML) | Python, Flask, Flask-CORS |
| AI Models | YOLO (Ultralytics), scikit-learn |
| Database | Firebase Firestore |
| Auth | Firebase Authentication |
| Hosting | Vercel (frontend), Render (backend) |

---

## Project Structure

```
CapstoneProject/
├── index.html          # Single-page application (React + vanilla JS)
├── styles.css          # Complete design system
├── app.py              # Flask ML service (3 YOLO + NLP)
├── requirements.txt    # Python dependencies
├── Procfile            # Render deployment config
├── models/
│   ├── pothole.pt      # YOLO pothole detection model
│   ├── garbage.pt      # YOLO garbage detection model
│   ├── crack.pt        # YOLO road crack detection model
│   └── classifier.pkl  # NLP text classifier (TF-IDF + LinearSVC)
├── training/
│   ├── train_nlp.py    # NLP classifier training script
│   ├── train_yolo.py   # YOLO training script (local)
│   └── train_colab.py  # YOLO training script (Google Colab)
├── docs/               # Documentation and reports
└── assets/             # Static assets
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check — shows model availability |
| POST | `/classify-text` | NLP text classification |
| POST | `/detect` | Single-category YOLO detection |
| POST | `/detect-all` | Run all 3 YOLO models on one image |
| POST | `/analyze` | Full analysis (text + image combined) |
| POST | `/annotate` | Returns annotated image with detections |

### Example: `/analyze`

```json
POST /analyze
{
  "text": "Large pothole near market road causing accidents",
  "image_base64": "data:image/jpeg;base64,..."
}

Response:
{
  "text_result": { "category": "Roads", "confidence": 0.92, "needs_review": false },
  "image_result": { "has_detection": true, "detected_category": "Roads", "confidence": 0.87 }
}
```

---


### Python Requirements
`
flask
flask-cors
ultralytics
scikit-learn
nltk
pillow
gunicorn
`
## Local Development

### Frontend
Open `index.html` in a browser or serve with any static server:
```bash
npx serve .
```

### Backend (ML Service)
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
python app.py                 # Starts on localhost:5000
```

### Retrain NLP Classifier
```bash
python training/train_nlp.py
# Output: models/classifier.pkl
```

---

## Deployment

- **Frontend** → Push to GitHub, deploy on Vercel (auto-deploys from `main` branch)
- **Backend** → Deploy on Render as a Web Service with `gunicorn app:app --timeout 120`

### Environment Variables (Render)
| Variable | Description |
|---|---|
| `MODEL_POTHOLE` | Path to pothole model (default: `models/pothole.pt`) |
| `MODEL_GARBAGE` | Path to garbage model (default: `models/garbage.pt`) |
| `MODEL_CRACK` | Path to crack model (default: `models/crack.pt`) |
| `NLP_MODEL` | Path to NLP classifier (default: `models/classifier.pkl`) |

---

## Team

| Name | Role |
|---|---|
| Sarthak Sant | Backend Developer, UI/UX Design |
| Mayank Rawat | Authentication and Admin Module |
| Rizul Pathania | Frontend Developer |
| Devdatta Patil | ML Models and Deployment |

---

## License

Capstone Project — © 2025–2026 CivicTrack
