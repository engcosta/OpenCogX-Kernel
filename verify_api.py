
import requests
import json
import sys

BASE_URL = "http://127.0.0.1:8000"

def check_status():
    print("Checking status...")
    try:
        resp = requests.get(f"{BASE_URL}/status")
        resp.raise_for_status()
        data = resp.json()
        print(f"Memory stats: {data.get('memory', {}).get('stats', {})}")
        return data
    except Exception as e:
        print(f"Status check failed: {e}")
        return None

def ingest_knowledge():
    print("Ingesting knowledge...")
    # create a dummy file
    with open("test_knowledge.txt", "w") as f:
        f.write("The sky is blue. The grass is green.")
    
    try:
        resp = requests.post(f"{BASE_URL}/ingest", json={"path": "test_knowledge.txt"})
        resp.raise_for_status()
        print(f"Ingest result: {resp.json()}")
    except Exception as e:
        print(f"Ingest failed: {e}")

def ask_question():
    print("Asking question...")
    try:
        payload = {
            "question": "What color is the sky?",
            "strategy": "hybrid"
        }
        resp = requests.post(f"{BASE_URL}/ask", json=payload)
        resp.raise_for_status()
        data = resp.json()
        print("\nAsk Response:")
        print(json.dumps(data, indent=2))
        
        context = data.get("context_used", [])
        if context and len(context) > 0:
            print("\nSUCCESS: context_used is populated!")
        else:
            print("\nFAILURE: context_used is empty!")
            
    except Exception as e:
        print(f"Ask failed: {e}")

if __name__ == "__main__":
    status = check_status()
    if status:
        # If memory is empty, ingest something
        mem_count = status.get('memory', {}).get('stats', {}).get('semantic_count', 0)
        if mem_count == 0:
            ingest_knowledge()
            # check status again?
        
        # Ask
        ask_question()
