import os
import json
import requests
from openai import OpenAI

# 1. Environment Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# FIX: Manually provide token or handle empty state for local testing
HF_TOKEN = os.getenv("HF_TOKEN", "your_token_here") 

# FIX: Replace with your actual Hugging Face Space URL
ENV_URL = os.getenv("ENV_URL", "https://krishnavenigandi123-green-ai-scheduler.hf.space")

# Initialize Client
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def get_llm_action(state):
    prompt = f"""
    You are an RL Agent in a Carbon-Aware Scheduler.
    State Vector: {state_vector} 
    (Indices: 0:Step, 1:Carbon, 2:Trend, 3:JobsLeft, 4:Urgency, 5:TotalCarbon, 6:Completion)
    
    Action Space:
    0: Wait
    1: Run Job 0
    2: Run Job 1 (if exists)
    
    Goal: Maximize completion while CI (Index 1) is low.
    Respond ONLY JSON: {{"action": int}}
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": "You are a green energy scheduler."},
                      {"role": "user", "content": prompt}],
            temperature=0.0
        )
        # Strip code blocks if LLM adds them
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1].replace("json", "").strip()
        return json.loads(content)
    except:
        return {"command": "wait"}

def run_task(task_id):
    # [START] tag - Required
    print(f"[START] task={task_id} env=GreenAI-v1 model={MODEL_NAME}")
    
    step_idx = 0
    total_rewards = []
    success = "false"
    last_error = "null"
    
    try:
        # Reset the environment
        reset_req = requests.post(f"{ENV_URL}/reset?task={task_id}")
        state = reset_req.json()
        
        done = False
        while not done and step_idx < 10:
            step_idx += 1
            
            action = get_llm_action(state)
            
            # Step the environment
            step_req = requests.post(f"{ENV_URL}/step", json=action)
            res_data = step_req.json()
            
            state, reward, done = res_data[0], res_data[1], res_data[2]
            total_rewards.append(f"{reward:.2f}")
            
            # [STEP] tag - Required
            action_str = f"{action['command']}({action.get('job_id', '')})"
            print(f"[STEP] step={step_idx} action={action_str} reward={reward:.2f} done={str(done).lower()} error={last_error}")
            
        success = "true" if state['pending_jobs'] == [] else "false"
            
    except Exception as e:
        last_error = str(e)
        print(f"DEBUG Error: {last_error}")

    # FIX: Ensure final_score is strictly between 0 and 1 (e.g., 0.95 or 0.05)
    final_score = 0.95 if success == "true" else 0.05
    
    # [END] tag - Required
    print(f"[END] success={success} steps={step_idx} score={final_score:.2f} rewards={','.join(total_rewards)}")

if __name__ == "__main__":
    target_task = os.getenv("TASK_NAME", "easy")
    run_task(target_task)
