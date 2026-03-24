# CivicTrack — Complete Deployment Guide

## Project Structure

```
CapstoneProject/
├── index.html              ← Frontend (deploy to Vercel)
├── styles.css              ← Styles
├── app.py                  ← Flask ML backend (deploy to Render)
├── train_nlp.py            ← NLP classifier training
├── TRAIN_IN_COLAB.py       ← YOLO training instructions
├── requirements.txt        ← Python dependencies
├── Procfile                ← Render start command
├── .gitignore
├── DEPLOYMENT.md           ← This file
└── models/                 ← Trained models (created during training)
    ├── pothole.pt
    ├── garbage.pt
    ├── waterlog.pt
    ├── crack.pt
    └── classifier.pkl
```

---

## Step 1: Set Up Firebase (5 minutes, FREE)

1. Go to [console.firebase.google.com](https://console.firebase.google.com)
2. Create a new project (or use existing `civictrack-484`)
3. Enable **Authentication**:
   - Go to Authentication → Sign-in method
   - Enable **Google** and **Email/Password**
4. Create **Firestore Database**:
   - Go to Firestore → Create database → Start in test mode
   - Create two collections: `issues` and `helpRequests`
5. Enable **Storage**:
   - Go to Storage → Get started
6. Update **Security Rules** (Firestore):
   ```
   rules_version = '2';
   service cloud.firestore {
     match /databases/{database}/documents {
       match /issues/{issueId} {
         allow read: if true;
         allow write: if request.auth != null;
       }
       match /helpRequests/{helpId} {
         allow read: if true;
         allow write: if request.auth != null;
       }
     }
   }
   ```
7. Firebase config is already in `index.html` — update if using a different project.

---

## Step 2: Train NLP Model (5 minutes, local)

```bash
pip install -r requirements.txt
python train_nlp.py
# Creates: models/classifier.pkl
```

---

## Step 3: Train YOLO Models in Google Colab (FREE T4 GPU)

1. Open [Google Colab](https://colab.research.google.com)
2. Go to Runtime → Change runtime type → **T4 GPU**
3. Upload `TRAIN_IN_COLAB.py` and `train_nlp.py` to Colab
4. Follow the cell-by-cell instructions in `TRAIN_IN_COLAB.py`
5. Download the final zip containing all `.pt` files
6. Place them in `models/` folder

**Datasets (all FREE):**

| Model | Dataset Source |
|-------|--------------|
| Pothole | Roboflow: `pothole-jujbl` |
| Garbage | Roboflow: search "garbage detection" |
| Waterlogging | Roboflow: search "water logging" or "flood detection" |
| Road Cracks | Roboflow: search "crack detection" |

---

## Step 4: Test Locally

```bash
# Terminal 1: ML backend
python app.py
# Runs on http://localhost:5000

# Terminal 2: Frontend
# Open index.html in browser (use VS Code Live Server)
```

Verify ML_SERVICE_URL in `index.html` is `http://localhost:5000` for local dev.

---

## Step 5: Deploy ML Backend to Render (FREE)

1. Push repo to **GitHub**
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo
4. Settings:
   - **Root directory**: `.` (root)
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120`
   - **Plan**: Free
5. Deploy

**Important notes:**
- Free tier sleeps after 15 min inactivity — first request after sleep takes ~30s
- If `.pt` files > 100 MB total, use **Git LFS**: `git lfs track "*.pt"`
- Models load lazily (first request for each model takes time, then fast)

---

## Step 6: Deploy Frontend to Vercel (FREE)

**Option A — GitHub Integration:**
1. Go to [vercel.com](https://vercel.com)
2. New Project → Import your GitHub repo
3. Framework Preset: **Other** (plain static)
4. Root Directory: `.` (root)
5. Deploy

**Option B — CLI:**
```bash
npm install -g vercel
vercel deploy
```

---

## Step 7: Update ML_SERVICE_URL

After Render deployment, update this line in `index.html`:

```javascript
// BEFORE
const ML_SERVICE_URL = "http://localhost:5000";

// AFTER
const ML_SERVICE_URL = "https://your-app-name.onrender.com";
```

Commit and push — Vercel auto-redeploys.

---

## 4 Detection Categories

| Category | YOLO Model | Detects |
|----------|-----------|---------|
| Roads | pothole.pt | Potholes |
| Garbage | garbage.pt | Waste, overflowing bins |
| Water Leakage | waterlog.pt | Flooding, waterlogging |
| Road Cracks | crack.pt | Cracks, road damage |
| Electricity | text-only | NLP classification only |
| Other | text-only | Catch-all |

---

## Free Resources

| Service | Purpose | Cost |
|---------|---------|------|
| Firebase Auth | Login (Google + Email) | Free (Spark) |
| Firebase Firestore | Issues + Help Requests | Free (1 GiB) |
| Firebase Storage | Uploaded images | Free (5 GB) |
| Render.com | Flask ML backend | Free (750 hrs/mo) |
| Vercel | Static frontend | Free (hobby) |
| Roboflow | Training datasets | Free (public) |
| Google Colab | YOLO training (T4 GPU) | Free |
| Nominatim | Location autocomplete | Free (OSM) |
| YOLOv12 | Object detection | Free (AGPL) |
| scikit-learn | NLP classifier | Free |

**Total cost: ₹0**

---

## Troubleshooting

| Issue | Solution |
|-------|---------|
| ML service returns 503 | Render free tier sleeping — wait 30s, retry |
| CORS error | Verify `CORS(app)` in app.py, check ML_SERVICE_URL |
| Firebase auth fails | Check auth providers are enabled in Firebase console |
| YOLO model not loading | Verify `.pt` file exists in `models/` folder |
| Image too large | Resize before upload, max 5 MB |
| NLP returns "Other" | Model may need more training data for that category |

---

## Team

| Name | Role |
|------|------|
| Sarthak Sant | Backend + UI/UX |
| Mayank Rawat | Authentication + Admin |
| Rizul Pathania | Frontend |
| Devdatta Patil | ML Models + Deployment |
