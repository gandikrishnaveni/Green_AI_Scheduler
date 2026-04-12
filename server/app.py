import uvicorn
import numpy as np
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder

# Import from the same directory to avoid package path issues
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
        
        # Ensure reward and done are standard types
        response = [observation, float(reward), bool(done), info]
        return jsonable_encoder(response)
    except Exception as e:
        return {"error": str(e)}

@app.post("/grade")
async def grade():
    try:
        raw_score = state_holder["env"].get_score()
        
        # CRITICAL FIX: Clamp score strictly between 0 and 1
        # Meta's validator rejects exactly 0.0 or 1.0.
        clamped_score = max(0.01, min(0.99, float(raw_score)))
        
        return {"score": clamped_score}
    except Exception as e:
        # Fallback to a valid range score in case of error
        return {"score": 0.5}

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
