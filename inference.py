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
    """Calls LLM with strict JSON output."""
    prompt = f"Current State: {json.dumps(state)}. Respond with JSON: {{'command': 'run_job', 'job_id': '...'}} or {{'command': 'wait'}}"
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" },
            timeout=10.0 
        )
        return json.loads(response.choices[0].message.content)
    except:
        return {"command": "wait", "job_id": None}

def run_evaluation(task="medium"):
    # MANDATORY START TAG
    print(f"[START] task={task}", flush=True)
    
    try:
        reset_resp = requests.post(f"{BASE_URL}/reset?task={task}", timeout=5)
        state = reset_resp.json()
    except: return

    done = False
    step_count = 0
    total_reward = 0

    while not done and step_count < 50:
        step_count += 1
        action = get_llm_action(state)
        
        try:
            step_resp = requests.post(f"{BASE_URL}/step", json=action, timeout=5)
            response_data = step_resp.json()

            if isinstance(response_data, list):
                state, reward, done = response_data[0], response_data[1], response_data[2]
            else:
                state = response_data
                reward = 0
                done = state.get("step", 0) >= 10
            
            total_reward += reward
            # MANDATORY STEP TAG
            print(f"[STEP] step={step_count} reward={reward}", flush=True)
            
        except:
            break

    # Final Grade
    score = 0
    try:
        grade_resp = requests.post(f"{BASE_URL}/grade", timeout=5)
        score = grade_resp.json().get('score', 0)
    except:
        pass
    
    # MANDATORY END TAG
    print(f"[END] task={task} score={score} steps={step_count}", flush=True)

if __name__ == "__main__":
    run_evaluation()
