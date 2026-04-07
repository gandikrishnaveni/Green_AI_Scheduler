import os
import json
import requests
from openai import OpenAI

# 1. Environment Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

ENV_URL = os.getenv("ENV_URL", "http://127.0.0.1:7860") 

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def get_llm_action(state):
    """Ask the LLM to decide the next move."""
    steps_left = state.get('steps_left', 10)
    pending_jobs = state.get('pending_jobs', [])
    pending_duration = sum(j.get('duration', 0) for j in pending_jobs)
    
    prompt = f"""
    Current State: {json.dumps(state)}
    CRITICAL RULES:
    1. Steps Remaining: {steps_left}
    2. Work Remaining: {pending_duration} steps.
    3. If Steps Remaining <= Work Remaining, you MUST return "run_job".
    4. If Carbon < 250, return "run_job".
    5. Return ONLY JSON: {{"command": "run_job", "job_id": "NLP_1"}} or {{"command": "wait", "job_id": null}}
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": "You are a precise JSON-only scheduler."},
                      {"role": "user", "content": prompt}],
            temperature=0.0
        )
        content = response.choices[0].message.content.strip()
        if "```" in content:
            content = content.split("```")[1].replace("json", "").strip()
        return json.loads(content)
    except Exception:
        # Fallback to run_job if we are low on time, otherwise wait
        command = "run_job" if steps_left <= pending_duration else "wait"
        return {"command": command, "job_id": "NLP_1" if command == "run_job" else None}

def run_task(task_id):
    print(f"[START] task={task_id} env=GreenAI-v1 model={MODEL_NAME}")
    
    try:
        reset_req = requests.post(f"{ENV_URL}/reset?task={task_id}")
        state = reset_req.json()
        
        step_idx, total_rewards, done = 0, [], False

        while not done and step_idx < 10:
            step_idx += 1
            action = get_llm_action(state)
            
            step_req = requests.post(f"{ENV_URL}/step", json=action)
            res_data = step_req.json()
            
            # STRICT META REQUIREMENT: Unpack 4 variables
            state, reward, done, info = res_data[0], res_data[1], res_data[2], res_data[3]
            total_rewards.append(f"{reward:.2f}")
            
            action_str = f"{action['command']}({action.get('job_id') or ''})"
            print(f"[STEP] step={step_idx} action={action_str} reward={reward:.2f} done={str(done).lower()} error=null")

        success = "true" if not state['pending_jobs'] else "false"
        
        # STRICT META REQUIREMENT: No 'score=' or 'error=' allowed on success line
        print(f"[END] success={success} steps={step_idx} rewards={','.join(total_rewards)}")

    except Exception as e:
        # STRICT META REQUIREMENT: Clean exit string on failure
        print(f"[END] success=false steps=0 rewards=")
        print(f"DEBUG ERROR: {str(e)}")

# This block right here is what makes the script actually run!
if __name__ == "__main__":
    target_task = os.getenv("TASK_NAME", "easy")
    run_task(target_task)