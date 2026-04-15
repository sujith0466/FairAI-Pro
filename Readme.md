# FairAI Pro

[![Flask](https://img.shields.io/badge/Backend-Flask-000000?logo=flask)](https://flask.palletsprojects.com/)
[![Pandas](https://img.shields.io/badge/Data-Pandas-150458?logo=pandas)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn)](https://scikit-learn.org/)
[![Frontend](https://img.shields.io/badge/Frontend-HTML%20%7C%20CSS%20%7C%20JS-1f6feb)](#tech-stack)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

FairAI Pro is a full-stack fairness auditing tool for binary classification models.  
It helps teams detect bias, evaluate mitigation impact, and generate concise AI explanations for stakeholders.

## Live Demo

- Frontend (Vercel): `https://fair-ai-pro.vercel.app/`
- Backend (Render): `https://fairai-pro.onrender.com/`

## Key Features

- Upload and inspect CSV datasets
- Analyze fairness with Logistic Regression
- View fairness metrics:
  - Fairness Score (0-100)
  - Statistical Parity Difference (SPD)
  - Disparate Impact Ratio (DIR)
  - Group-wise selection rates and accuracy
- Run mitigation analysis by removing sensitive feature influence
- Generate concise bias explanations via Gemini API
- Export analysis report as JSON

## Quick Demo Flow

1. Upload dataset (`.csv`) from the UI
2. Analyze bias by selecting target, sensitive column, and privileged group
3. View mitigation scores (before vs after mitigation)
4. Explain results with AI (`/api/explain`)
5. Export report for sharing and review


## Tech Stack

- Backend: Flask, pandas, scikit-learn
- Frontend: HTML, CSS, Vanilla JavaScript
- AI: Google Gemini API
- Deployment: Render (backend), Vercel (frontend)

## Project Structure

```text
Fair-AI/
|- backend/
|  |- app.py
|  |- bias_engine.py
|  |- requirements.txt
|  |- Procfile
|  `- uploads/
|- data/
|  `- sample_hiring.csv
|- frontend/
|  |- index.html
|  |- styles.css
|  `- app.js
|- .env.example
`- Readme.md
```

## Getting Started

### 1) Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2) Configure environment

```bash
# Linux / macOS
export GEMINI_API_KEY="your_api_key"

# Windows PowerShell
$env:GEMINI_API_KEY="your_api_key"
```

### 3) Run backend locally

```bash
cd backend
python app.py
```

Backend default: `http://localhost:5000`

## API Endpoints

- `GET /api/health`
- `POST /api/upload`
- `POST /api/sample`
- `POST /api/analyze`
- `POST /api/mitigation`
- `POST /api/explain`

## License

This project is licensed under the MIT License.  
Add a `LICENSE` file at the repository root if not added yet.

