import requests
import json
import os
import time

LOCAL_API = "http://127.0.0.1:5000/api"
PROD_API = "https://fairai-pro.onrender.com/api"
DATA_DIR = r"D:\Fair-AI"

def run_tests(api_base, name):
    print(f"\n==========================================")
    print(f" TESTING ENVIRONMENT: {name}")
    print(f"==========================================")
    
    results = {}

    # STEP 1: Health Check
    try:
        r = requests.get(f"{api_base}/health")
        if r.status_code == 200:
            print(f"[PASS] STEP 1: Health Check (Status {r.status_code})")
            results["STEP 1"] = "PASS"
        else:
            print(f"[FAIL] STEP 1: Health Check returned {r.status_code}")
            results["STEP 1"] = "FAIL"
    except Exception as e:
        print(f"[FAIL] STEP 1: Health Check Failed - {e}")
        results["STEP 1"] = "FAIL"

    # STEP 3 & 8 & 9: Dataset Uploads
    datasets = {
        "valid_1": "valid_dataset_1.csv",
        "valid_2": "valid_dataset_2.csv",
        "valid_3": "valid_dataset_3.csv",
        "edge_case": "edge_case_dataset.csv",
        "invalid": "invalid_dataset.csv",
        "large": "large_test_dataset.csv"
    }

    upload_results = {}
    for key, filename in datasets.items():
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"[WARN] File {filename} not found.")
            upload_results[key] = "NOT_FOUND"
            continue
            
        try:
            start_time = time.time()
            with open(filepath, 'rb') as f:
                r = requests.post(f"{api_base}/upload", files={"dataset": f})
            elapsed = time.time() - start_time
            
            if key == "invalid":
                if r.status_code == 400:
                    print(f"[PASS] Upload '{key}': Correctly rejected (Status 400)")
                    upload_results[key] = "PASS"
                else:
                    print(f"[FAIL] Upload '{key}': Expected 400, got {r.status_code}")
                    upload_results[key] = "FAIL"
            elif key == "large":
                if r.status_code == 200 and elapsed < 10:
                    print(f"[PASS] Upload '{key}': Success in {elapsed:.2f}s")
                    upload_results[key] = "PASS"
                else:
                    print(f"[FAIL] Upload '{key}': Status {r.status_code}, Time {elapsed:.2f}s")
                    upload_results[key] = "FAIL"
            else:
                if r.status_code == 200:
                    print(f"[PASS] Upload '{key}': Success")
                    upload_results[key] = "PASS"
                else:
                    print(f"[FAIL] Upload '{key}': Status {r.status_code} - {r.text}")
                    upload_results[key] = "FAIL"
        except Exception as e:
            print(f"[FAIL] Upload '{key}' Failed: {e}")
            upload_results[key] = "FAIL"
            
    results["STEP 3"] = "PASS" if all(x == "PASS" for k, x in upload_results.items() if k.startswith("valid_")) and upload_results.get("edge_case") == "PASS" else "FAIL"
    results["STEP 8"] = upload_results.get("invalid", "FAIL")
    results["STEP 9"] = upload_results.get("large", "FAIL")

    # For following steps, use valid_1
    filepath = os.path.join(DATA_DIR, datasets["valid_1"])
    if not os.path.exists(filepath):
        print("Cannot proceed with steps 4-7 without valid_1")
        return results
        
    with open(filepath, 'rb') as f:
        r = requests.post(f"{api_base}/upload", files={"dataset": f})
    
    # We assume valid_1 has specific columns: let's fetch sample to guess columns or just use known ones
    # Typically valid_dataset_1 has 'hired' (target), 'gender' (sensitive), 'male' (privileged)
    
    # STEP 4: Bias Analysis
    try:
        payload = {
            "target_column": "hired",
            "sensitive_column": "gender",
            "privileged_value": "male"
        }
        r = requests.post(f"{api_base}/analyze", json=payload)
        if r.status_code == 200:
            data = r.json()
            if "fairness_score" in data["fairness"] and "spd" in data["fairness"] and "dir" in data["fairness"] and "details" in data["groups"]:
                print(f"[PASS] STEP 4: Bias Analysis successful.")
                results["STEP 4"] = "PASS"
            else:
                print(f"[FAIL] STEP 4: Missing fields in analysis payload. {data.keys()}")
                results["STEP 4"] = "FAIL"
        else:
            print(f"[FAIL] STEP 4: Analysis returned {r.status_code} - {r.text}")
            results["STEP 4"] = "FAIL"
    except Exception as e:
         print(f"[FAIL] STEP 4: Exception {e}")
         results["STEP 4"] = "FAIL"
         
    # STEP 5: Mitigation
    try:
        r = requests.post(f"{api_base}/mitigate", json=payload)
        if r.status_code == 200:
            data = r.json()
            if "before_score" in data and "after_score" in data and "improvement_score" in data:
                print(f"[PASS] STEP 5: Mitigation successful.")
                results["STEP 5"] = "PASS"
            else:
                print(f"[FAIL] STEP 5: Missing fields in mitigate payload. {data.keys()}")
                results["STEP 5"] = "FAIL"
        else:
             print(f"[FAIL] STEP 5: Mitigation returned {r.status_code} - {r.text}")
             results["STEP 5"] = "FAIL"
    except Exception as e:
         print(f"[FAIL] STEP 5: Exception {e}")
         results["STEP 5"] = "FAIL"
         
    # STEP 6: Explain with AI
    try:
        explain_payload = {
            "fairness_score": 60,
            "sensitive_column": "gender",
            "group_stats": {"male": 0.8, "female": 0.5}
        }
        r = requests.post(f"{api_base}/explain", json=explain_payload)
        if r.status_code == 200:
            data = r.json()
            exp = data.get("explanation", "")
            if exp and len(exp.split('n')) >= 3 or exp.count('-') >= 3 or exp.count('•') >= 3:
                print(f"[PASS] STEP 6: AI Explanation successful and has multiple bullets.")
                results["STEP 6"] = "PASS"
            else:
                print(f"[FAIL] STEP 6: Explanation missing or too short/single-line.")
                print(f"       Response: {exp}")
                results["STEP 6"] = "FAIL"
        else:
             print(f"[FAIL] STEP 6: AI Explanation returned {r.status_code} - {r.text}")
             results["STEP 6"] = "FAIL"
    except Exception as e:
         print(f"[FAIL] STEP 6: Exception {e}")
         results["STEP 6"] = "FAIL"

    return results

print("Starting tests...")
run_tests(LOCAL_API, "LOCAL BACKEND")
print("\n" + "="*50 + "\n")
run_tests(PROD_API, "PRODUCTION BACKEND")
