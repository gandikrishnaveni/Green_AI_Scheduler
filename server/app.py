from fastapi import FastAPI, Request
from env.green_scheduler_env import GreenSchedulerEnv
import numpy as np

app = FastAPI()
# Global store for the active session
session = {"env": None}

@app.post("/reset")
async def reset(task: str = "medium"):
    session["env"] = GreenSchedulerEnv(difficulty=task, seed=42)
    obs = session["env"].reset()
    # Convert numpy array to list for JSON response
    return {"observation": obs.tolist(), "state": session["env"]._observe().tolist()}

@app.post("/step")
async def step(request: Request):
    data = await request.json()
    action = data.get("action", 0) # 0=wait, 1+=job index
    
    obs, reward, done, info = session["env"].step(action)
    
    return [
        {"observation": obs.tolist()}, 
        float(reward), 
        bool(done), 
        info
    ]

@app.post("/grade")
async def grade(request: Request):
    if not session["env"]: return {"score": 0.45}
    metrics = session["env"].get_metrics()
    # Meta expects a single float score for the leaderboard
    return {"score": metrics["episode_return_normalized"]} # Ensure this is 0-1

@app.get("/health")
def health(): return {"status": "healthy"}
