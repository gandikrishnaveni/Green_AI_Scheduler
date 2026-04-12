import uvicorn
import numpy as np
from fastapi import FastAPI, Request
from env.green_scheduler_env import GreenSchedulerEnv

# Initialize the FastAPI app
app = FastAPI(title="Green-AI Scheduler OpenEnv")

# State holder to maintain the environment across requests
# In a production setting, you'd use a session manager
state_holder = {"env": GreenSchedulerEnv()}

@app.get("/health")
def health():
    """Health check endpoint for Hugging Face and OpenEnv."""
    return {"status": "healthy", "service": "green-ai-scheduler"}

@app.post("/reset")
async def reset(task: str = "medium"):
    """
    OpenEnv compliant reset. 
    Supported tasks: 'easy', 'medium', 'hard'
    """
    state_holder["env"].difficulty = task
    observation = state_holder["env"].reset()
    return observation

@app.post("/step")
async def step(request: Request):
    """
    OpenEnv compliant step.
    Expects JSON: {"command": "run_job", "job_id": "..."} or {"command": "wait"}
    """
    action = await request.json()
    observation, reward, done, info = state_holder["env"].step(action)
    
    # Returning as a list per OpenEnv specification [obs, reward, done, info]
    return [observation, float(reward), bool(done), info]

@app.get("/state")
async def get_state():
    """Returns the current raw state of the environment."""
    return state_holder["env"]._observe()

@app.post("/grade")
async def grade():
    """
    Programmatic grader.
    Calculates a final score between 0.0 and 1.0 based on completion and efficiency.
    """
    score = state_holder["env"].get_score()
    return {"score": score}

# --- Multi-mode Deployment Support ---

def main():
    """
    The main entry point required by the OpenEnv CLI 
    and multi-mode deployment validation.
    """
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
