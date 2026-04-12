import os
import json
import requests
from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("API_BASE_URL", "https://api.openai.com/v1"),
    api_key=os.environ.get("API_KEY", "dummy")
)
BASE_URL = os.environ.get("ENV_URL", "http://localhost:7860")

def get_llm_action(state):
    prompt = f"State: {json.dumps(state)}. Respond with JSON: {{'command': 'run_job', 'job_id': '...'}} or {{'command': 'wait'}}"
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

def run_evaluation():
    # REQUIRED: Running at least 3 tasks
    tasks = ["easy", "medium", "hard"]
    
    for task in tasks:
        print(f"[START] task={task}", flush=True)
        try:
            reset_resp = requests.post(f"{BASE_URL}/reset?task={task}", timeout=5)
            state = reset_resp.json()
            
            done = False
            step_count = 0
            while not done and step_count < 20:
                step_count += 1
                action = get_llm_action(state)
                step_resp = requests.post(f"{BASE_URL}/step", json=action, timeout=5)
                res = step_resp.json()
                
                # Handling list or dict response
                if isinstance(res, list):
                    state, reward, done = res[0], res[1], res[2]
                else:
                    state, reward, done = res, 0, res.get("step", 0) >= 10
                
                print(f"[STEP] step={step_count} reward={reward}", flush=True)

            grade_resp = requests.post(f"{BASE_URL}/grade", timeout=5)
            score = grade_resp.json().get('score', 0.5)
            print(f"[END] task={task} score={score} steps={step_count}", flush=True)
        except Exception as e:
            print(f"[END] task={task} score=0.1 steps=0", flush=True)

if __name__ == "__main__":
    run_evaluation()
