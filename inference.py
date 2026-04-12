import os
import json
import requests
from openai import OpenAI

# 1. Initialize the client using the Meta/Scaler Proxy variables
# This satisfies the Phase 2 Deep Validation requirement
client = OpenAI(
    base_url=os.environ.get("API_BASE_URL", "https://api.openai.com/v1"),
    api_key=os.environ.get("API_KEY", "dummy-key-for-local")
)

# Your environment URL (Hugging Face Space)
BASE_URL = os.environ.get("ENV_URL", "http://localhost:7860")

def get_llm_action(state):
    """
    Calls the LLM via the provided proxy to decide the next scheduling action.
    """
    prompt = f"""
    You are a Green-AI Grid Architect. Minimize carbon footprint while completing jobs.
    Current State: {json.dumps(state)}
    
    Goal:
    - RUN jobs when carbon_intensity is low (<300).
    - WAIT when carbon_intensity is high (>400) UNLESS a deadline is immediate (step >= deadline - 1).
    
    Respond ONLY with valid JSON:
    {{"command": "run_job", "job_id": "string"}} or {{"command": "wait", "job_id": null}}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise scheduler. Output only JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"LLM Error: {e}")
        return {"command": "wait", "job_id": None}

def run_evaluation(task="medium"):
    """
    Runs a full episode. Includes defensive unpacking to prevent 'ValueError'.
    """
    print(f"🚀 Starting Evaluation for Task: {task}")
    
    # Reset Environment
    try:
        reset_resp = requests.post(f"{BASE_URL}/reset?task={task}")
        state = reset_resp.json()
    except Exception as e:
        print(f"Reset failed: {e}")
        return

    done = False
    while not done:
        action = get_llm_action(state)
        
        try:
            step_resp = requests.post(f"{BASE_URL}/step", json=action)
            response_data = step_resp.json()

            # DEFENSIVE UNPACKING: Handles both [obs, rew, done, info] and {obs}
            if isinstance(response_data, list) and len(response_data) >= 3:
                state = response_data[0]
                reward = response_data[1]
                done = response_data[2]
            elif isinstance(response_data, dict):
                state = response_data
                # Fallback if the server only sends the state back
                done = state.get("step", 0) >= 10
            else:
                print("Unknown response format from server.")
                break
                
            print(f"Step: {state.get('step')} | Carbon: {state.get('carbon_intensity')} | Action: {action['command']}")
            
        except Exception as e:
            print(f"Step failed: {e}")
            break

    # Final Grade
    try:
        grade_resp = requests.post(f"{BASE_URL}/grade")
        print(f"\n✅ Evaluation Complete! Final Score: {grade_resp.json().get('score')}")
    except:
        print("\nFinished, but could not retrieve final score.")

if __name__ == "__main__":
    run_evaluation()
