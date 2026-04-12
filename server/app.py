import uvicorn
import numpy as np
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder

# IMPORT CHANGE: Looking in the same directory now
from .green_scheduler_env import GreenSchedulerEnv

app = FastAPI(title="Green-AI Scheduler")
state_holder = {"env": GreenSchedulerEnv()}

@app.get("/health")
def health():
    return {"status": "healthy", "service": "green-ai-scheduler"}

@app.post("/reset")
async def reset(task: str = "medium"):
    try:
        state_holder["env"].difficulty = task
        observation = state_holder["env"].reset()
        return jsonable_encoder(observation)
    except Exception as e:
        return {"error": str(e)}

@app.post("/step")
async def step(request: Request):
    try:
        action = await request.json()
        observation, reward, done, info = state_holder["env"].step(action)
        response = [observation, float(reward), bool(done), info]
        return jsonable_encoder(response)
    except Exception as e:
        return {"error": str(e)}

@app.post("/grade")
async def grade():
    score = state_holder["env"].get_score()
    return {"score": float(score)}

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
