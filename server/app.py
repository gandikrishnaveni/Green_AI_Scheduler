import uvicorn
import numpy as np
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from env.green_scheduler_env import GreenSchedulerEnv

# Initialize the FastAPI app
app = FastAPI(title="Green-AI Scheduler OpenEnv")

# State holder to maintain the environment across requests
state_holder = {"env": GreenSchedulerEnv()}

@app.get("/health")
def health():
    """Health check endpoint for Hugging Face and OpenEnv."""
    return {"status": "healthy", "service": "green-ai-scheduler"}

@app.post("/reset")
async def reset(task: str = "medium"):
    """
    OpenEnv compliant reset. 
    Returns the initial observation as a clean dictionary.
    """
    state_holder["env"].difficulty = task
    observation = state_holder["env"].reset()
    # jsonable_encoder handles NumPy types that standard json library cannot
    return jsonable_encoder(observation)

@app.post("/step")
async def step(request: Request):
    """
    OpenEnv compliant step.
    Returns: [observation_dict, reward_float, done_bool, info_dict]
    """
    action = await request.json()
    observation, reward, done, info = state_holder["env"].step(action)
    
    # Pack into OpenEnv format and sanitize for JSON
    response = [
        observation,
        float(reward),
        bool(done),
        info
    ]
    return jsonable_encoder(response)

@app.get("/state")
async def get_state():
    """Returns the current raw state of the environment."""
    return jsonable_encoder(state_holder["env"]._observe())

@app.post("/grade")
async def grade():
    """
    Programmatic grader.
    Returns a final score between 0.0 and 1.0.
    """
    score = state_holder["env"].get_score()
    return {"score": float(score)}

# --- Multi-mode Deployment Support ---

def main():
    """
    The main entry point required by the OpenEnv CLI 
    and multi-mode deployment validation.
    """
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
