"""
FairAI Pro — Backend Application
Bias Detection & Mitigation Engine for ML Models
"""

import os
import json
import traceback
import urllib.request
import urllib.error
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from bias_engine import analyze_bias, analyze_mitigation, get_dataset_info
import pandas as pd
import numpy as np

load_dotenv()

BASE_DIR = os.path.dirname(__file__)
FRONTEND_DIR = os.path.join(BASE_DIR, '..', 'frontend')
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_TIMEOUT_SECONDS = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "20"))
DEBUG_MODE = os.getenv("FLASK_DEBUG", "false").lower() == "true"
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
ALLOWED_UPLOAD_EXTENSIONS = {"csv"}
COLUMN_MAPPING = {
    "sex": "gender",
    "age_num": "age",
    "experience": "experience_years",
    "exp": "experience_years",
    "salary": "income",
    "earnings": "income",
    "target": "hired",
    "label": "hired"
}
REQUIRED_COLS = ["age", "gender", "education", "experience_years", "income", "hired"]

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
CORS(app, origins=[
    "http://localhost:3000",
    "http://localhost:5173",
    "https://fair-ai-pro.vercel.app"
])
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", os.path.join(BASE_DIR, 'uploads'))
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def _json_error(message, status_code):
    return jsonify({"error": message}), status_code


def to_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_serializable(v) for v in obj.tolist()]
    if pd.isna(obj):
        return None
    return obj


def _is_allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_UPLOAD_EXTENSIONS


def _normalize_col_name(name):
    return str(name).strip().lower()


def preprocess_dataframe(df):
    df = df.copy()
    column_mapping = {
        "sex": "gender",
        "age_num": "age",
        "experience": "experience_years",
        "exp": "experience_years",
        "salary": "income",
        "earnings": "income",
        "target": "hired",
        "label": "hired"
    }

    new_columns = []
    for col in df.columns:
        clean_col = col.strip().lower()
        mapped_col = column_mapping.get(clean_col, clean_col)
        new_columns.append(mapped_col)

    df.columns = new_columns
    print("FINAL COLUMNS:", df.columns.tolist())
    return df


def _normalize_and_validate_df(df, target_col=None, sensitive_col=None):
    df = df.copy()
    missing_columns = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns after mapping: {missing_columns}")

    if df.isnull().any().any():
        raise ValueError("Dataset contains missing values. Please clean your data.")

    normalized_target = _normalize_col_name(target_col) if target_col else None
    normalized_sensitive = _normalize_col_name(sensitive_col) if sensitive_col else None

    if normalized_target in COLUMN_MAPPING:
        normalized_target = COLUMN_MAPPING[normalized_target]
    if normalized_sensitive in COLUMN_MAPPING:
        normalized_sensitive = COLUMN_MAPPING[normalized_sensitive]

    if normalized_target and normalized_target not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    if normalized_sensitive and normalized_sensitive not in df.columns:
        raise ValueError(f"Sensitive column '{sensitive_col}' not found in dataset")

    if normalized_target:
        df[normalized_target] = pd.to_numeric(df[normalized_target], errors="coerce")
        if df[normalized_target].isnull().any():
            raise ValueError("Target column must contain only 0 and 1 values")

        values = set(df[normalized_target].unique())
        if not values.issubset({0, 1}):
            raise ValueError("Target column must contain only 0 and 1 values")

    return df, normalized_target, normalized_sensitive


def _build_explanation_prompt(fairness_score, sensitive_column, group_stats):
    group_lines = "\n".join([f"- {group}: {rate}" for group, rate in group_stats.items()])
    return (
        "Explain in simple terms why this machine learning model shows bias based on the fairness score "
        "and group statistics. Also suggest how to reduce the bias.\n\n"
        f"Fairness score: {fairness_score}\n"
        f"Sensitive column: {sensitive_column}\n"
        f"Group statistics (selection rates):\n{group_lines}\n\n"
        "Keep the response concise in 3 to 5 lines."
    )


def _generate_gemini_explanation(prompt_text):
    if not GEMINI_API_KEY:
        raise ValueError("Gemini API key is not configured. Set GEMINI_API_KEY in your environment.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 220
        }
    }
    body = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=GEMINI_TIMEOUT_SECONDS) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_details = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Gemini API request failed ({e.code}): {error_details}")
    except Exception as e:
        raise RuntimeError(f"Gemini API request failed: {str(e)}")

    try:
        text = response_data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError, TypeError):
        raise RuntimeError("Gemini API returned an unexpected response format.")

    # Enforce concise 3-5 lines max in backend output as a safety guard.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    concise_text = "\n".join(lines[:5]) if lines else "No explanation generated."
    return concise_text


@app.route('/')
def serve_index():
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "service": "FairAI Pro Engine"})


@app.errorhandler(413)
def file_too_large(_):
    return _json_error(f"Uploaded file is too large. Max size is {MAX_UPLOAD_MB} MB.", 413)


@app.route('/api/upload', methods=['POST'])
def upload_dataset():
    """Upload a CSV and return column info for configuration."""
    if 'dataset' not in request.files:
        return _json_error("No file uploaded", 400)

    file = request.files['dataset']
    if not file.filename or not _is_allowed_file(file.filename):
        return _json_error("Only CSV files are supported", 400)

    filepath = os.path.join(UPLOAD_FOLDER, 'current_dataset.csv')
    file.save(filepath)

    try:
        df = pd.read_csv(filepath)
        df = preprocess_dataframe(df)
        df, _, _ = _normalize_and_validate_df(df)
        df.to_csv(filepath, index=False)
        info = get_dataset_info(filepath)
        info = to_serializable(info)
        return jsonify(info)
    except ValueError as e:
        return _json_error(str(e), 400)
    except Exception as e:
        return _json_error(str(e), 500)


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Run bias analysis on the uploaded dataset."""
    data = request.get_json(silent=True)

    if not data:
        return _json_error("No configuration provided", 400)

    target_col = data.get('target_column')
    sensitive_col = data.get('sensitive_column')
    privileged_value = data.get('privileged_value')

    if not all([target_col, sensitive_col, privileged_value]):
        return _json_error("Missing required fields: target_column, sensitive_column, privileged_value", 400)

    filepath = os.path.join(UPLOAD_FOLDER, 'current_dataset.csv')
    if not os.path.exists(filepath):
        return _json_error("No dataset uploaded. Please upload a CSV first.", 400)

    try:
        import time
        start = time.time()
        
        df = pd.read_csv(filepath)
        df = preprocess_dataframe(df)
        df, normalized_target, normalized_sensitive = _normalize_and_validate_df(df, target_col, sensitive_col)
        
        if len(df) > 3000:
            df = df.sample(n=3000, random_state=42)
            
        results = analyze_bias(df, normalized_target, normalized_sensitive, privileged_value)
        results = {k: to_serializable(v) for k, v in results.items()}
        
        print("PROCESS TIME:", time.time() - start)
        return jsonify(results)
    except ValueError as e:
        return _json_error(str(e), 400)
    except Exception as e:
        traceback.print_exc()
        return _json_error(str(e), 500)


@app.route('/api/mitigation', methods=['POST'])
def mitigation():
    """Run simple mitigation analysis by removing sensitive attribute."""
    data = request.get_json(silent=True)

    if not data:
        return _json_error("No configuration provided", 400)

    target_col = data.get('target_column')
    sensitive_col = data.get('sensitive_column')

    if not all([target_col, sensitive_col]):
        return _json_error("Missing required fields: target_column, sensitive_column", 400)

    filepath = os.path.join(UPLOAD_FOLDER, 'current_dataset.csv')
    if not os.path.exists(filepath):
        return _json_error("No dataset uploaded. Please upload a CSV first.", 400)

    try:
        df = pd.read_csv(filepath)
        df = preprocess_dataframe(df)
        df, normalized_target, normalized_sensitive = _normalize_and_validate_df(df, target_col, sensitive_col)
        results = analyze_mitigation(df, normalized_target, normalized_sensitive)
        results = {k: to_serializable(v) for k, v in results.items()}
        return jsonify(results)
    except ValueError as e:
        return _json_error(str(e), 400)
    except Exception as e:
        traceback.print_exc()
        return _json_error(str(e), 500)


@app.route('/api/explain', methods=['POST'])
def explain():
    """Generate a concise AI explanation for fairness outcomes using Gemini."""
    data = request.get_json(silent=True)

    if not data:
        return _json_error("No input provided", 400)

    fairness_score = data.get("fairness_score")
    sensitive_column = data.get("sensitive_column")
    group_stats = data.get("group_stats")

    if fairness_score is None or not sensitive_column or not isinstance(group_stats, dict):
        return _json_error(
            "Missing/invalid fields. Required: fairness_score (number), sensitive_column (string), group_stats (object)",
            400
        )

    try:
        fairness_score = float(fairness_score)
    except (TypeError, ValueError):
        return _json_error("fairness_score must be a number", 400)

    prompt_text = _build_explanation_prompt(fairness_score, sensitive_column, group_stats)

    try:
        explanation = _generate_gemini_explanation(prompt_text)
        return jsonify({"explanation": explanation})
    except Exception as e:
        traceback.print_exc()
        return _json_error(f"Failed to generate explanation: {str(e)}", 502)


@app.route('/api/sample', methods=['POST'])
def use_sample():
    """Use the built-in sample hiring dataset."""
    sample_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_hiring.csv')
    if not os.path.exists(sample_path):
        return _json_error("Sample dataset not found", 404)

    import shutil
    dest = os.path.join(UPLOAD_FOLDER, 'current_dataset.csv')
    shutil.copy2(sample_path, dest)

    try:
        df = pd.read_csv(dest)
        df = preprocess_dataframe(df)
        df, _, _ = _normalize_and_validate_df(df)
        df.to_csv(dest, index=False)
        info = get_dataset_info(dest)
        info = to_serializable(info)
        return jsonify(info)
    except ValueError as e:
        return _json_error(str(e), 400)
    except Exception as e:
        return _json_error(str(e), 500)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=DEBUG_MODE)
