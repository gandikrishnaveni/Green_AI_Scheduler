import os
import json
import requests
from openai import OpenAI

# 1. Initialize Client
client = OpenAI(
    base_url=os.environ.get("API_BASE_URL", "https://api.openai.com/v1"),
    api_key=os.environ.get("API_KEY", "dummy-key")
)

BASE_URL = os.environ.get("ENV_URL", "http://localhost:7860")

def get_llm_action(state):
    """Calls LLM with a strict 10-second timeout."""
    prompt = f"State: {json.dumps(state)}. Respond with JSON: {{'command': 'run_job', 'job_id': '...'}} or {{'command': 'wait'}}"

    try:
        # TIMEOUT ADDED HERE
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" },
            timeout=10.0 
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"LLM Timeout or Error: {e}")
        return {"command": "wait", "job_id": None}

def run_evaluation(task="medium"):
    print(f"🚀 Starting Fast Evaluation: {task}")
    
    try:
        # TIMEOUT ADDED TO REQUESTS
        reset_resp = requests.post(f"{BASE_URL}/reset?task={task}", timeout=5)
        state = reset_resp.json()
    except: return

    done = False
    max_steps = 50  # HARD SAFETY LIMIT to prevent infinite loops
    step_count = 0

    while not done and step_count < max_steps:
        step_count += 1
        action = get_llm_action(state)
        
        try:
            # TIMEOUT ADDED TO STEP
            step_resp = requests.post(f"{BASE_URL}/step", json=action, timeout=5)
            response_data = step_resp.json()

            if isinstance(response_data, list):
                state, reward, done = response_data[0], response_data[1], response_data[2]
            else:
                state = response_data
                done = state.get("step", 0) >= 10
        except:
            break

    # Final Grade
    try:
        requests.post(f"{BASE_URL}/grade", timeout=5)
    except:
        pass
    print("✅ Evaluation finished within time limits.")

if __name__ == "__main__":
    run_evaluation()
