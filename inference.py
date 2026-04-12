import os
import json
import requests
from openai import OpenAI

# 1. Initialize the client using the Meta/Scaler Proxy variables
# The validator "injects" these into the environment during the run.
client = OpenAI(
    base_url=os.environ.get("API_BASE_URL", "https://api.openai.com/v1"),
    api_key=os.environ.get("API_KEY", "dummy-key-for-local-testing")
)

# Your environment URL (Hugging Face Space)
BASE_URL = os.environ.get("ENV_URL", "http://localhost:7860")

def get_llm_action(state):
    """
    Formulates a prompt for the LLM based on the current environment state
    and returns the structured JSON action.
    """
    prompt = f"""
    You are a Green-AI Grid Architect. Your goal is to complete computational jobs 
    while minimizing carbon footprint.
    
    Current Environment State:
    {json.dumps(state, indent=2)}
    
    Strategic Constraints:
    1. If carbon_intensity is HIGH (>400) and jobs have slack (deadline - step > 2), you should WAIT.
    2. If carbon_intensity is LOW (<250), RUN the job with the shortest duration.
    3. If a job's deadline is approaching (deadline - step <= 1), RUN it regardless of carbon cost.
    
    Respond ONLY with a valid JSON object:
    {{"command": "run_job", "job_id": "string"}} or {{"command": "wait", "job_id": null}}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise scheduling assistant. Output only JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error calling LLM Proxy: {e}")
        return {"command": "wait", "job_id": None}

def run_evaluation(task="medium"):
    """Runs a full episode and reports the final score."""
    print(f"🚀 Starting Evaluation for Task: {task}")
    
    # Reset Environment
    reset_resp = requests.post(f"{BASE_URL}/reset?task={task}")
    state = reset_resp.json()
    
    done = False
    while not done:
        # Get Action from LLM
        action = get_llm_action(state)
        
        # Step Environment
        step_resp = requests.post(f"{BASE_URL}/step", json=action)
        state, reward, done, info = step_resp.json()
        
        print(f"Step: {state['step']} | Carbon: {state['carbon_intensity']} | Action: {action['command']}")

    # Final Grade
    grade_resp = requests.post(f"{BASE_URL}/grade")
    print(f"\n✅ Evaluation Complete! Final Score: {grade_resp.json()['score']}")

if __name__ == "__main__":
    run_evaluation()
