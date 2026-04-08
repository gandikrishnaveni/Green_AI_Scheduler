import os
import random
import uvicorn
from typing import List, Dict, Optional
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# --- Models ---
class State(BaseModel):
    current_step: int
    carbon_intensity: float
    pending_jobs: List[Dict]
    status: str
    steps_left: int

class Action(BaseModel):
    command: str
    job_id: Optional[str] = None

class GradeRequest(BaseModel):
    task: str
    state: dict
    reward_history: List[float]

# --- Environment Logic ---
class GreenAIEnv:
    def __init__(self):
        self.reset()

    def reset(self, task="easy"):
        self.step_count = 0
        self.jobs = [{"id": "NLP_1", "duration": 3, "deadline": 10}]
        
        if task == "medium":
            self.jobs.append({"id": "CV_1", "duration": 2, "deadline": 8})
        elif task == "hard":
            self.jobs.append({"id": "CV_1", "duration": 2, "deadline": 8})
            self.jobs.append({"id": "RL_1", "duration": 4, "deadline": 10})
            
        return self.get_state()

    def get_state(self):
        return {
            "current_step": self.step_count,
            "carbon_intensity": round(random.uniform(50, 400), 2),
            "pending_jobs": [j for j in self.jobs if j['duration'] > 0],
            "status": "Running",
            "steps_left": 10 - self.step_count
        }

    def step(self, action: Action):
        self.step_count += 1
        reward = 0.0
        
        if action.command == "run_job" and action.job_id:
            for job in self.jobs:
                if job['id'] == action.job_id and job['duration'] > 0:
                    job['duration'] -= 1
                    current_intensity = self.get_state()["carbon_intensity"]
                    # Rewarding greener runs
                    reward = 0.8 if current_intensity < 200 else 0.2
        
        elif action.command == "wait":
            current_intensity = self.get_state()["carbon_intensity"]
            # Rewarding waiting during high carbon peaks
            reward = 0.4 if current_intensity > 300 else 0.1

        state = self.get_state()
        done = self.step_count >= 10 or not state["pending_jobs"]
        
        return state, round(reward, 2), done

env = GreenAIEnv()

# --- API Endpoints ---
@app.get("/")
def home(): 
    return {"status": "GreenAI Environment Running"}

@app.post("/reset")
def reset(task: str = "easy"): 
    return env.reset(task)

@app.post("/step")
def step(action: Action):
    state, reward, done = env.step(action)
    info = {"status": "ok"}
    return [state, reward, done, info]

@app.post("/grade")
def grade(request: GradeRequest):
    """
    Phase 2 Scorer: 
    Meta requires score strictly between 0 and 1.
    """
    if not request.reward_history:
        return {"score": 0.01}
    
    # Calculate performance based on total rewards accumulated
    total_reward = sum(request.reward_history)
    max_possible = 8.0 # Rough estimate for 10 steps
    
    # Normalize score
    raw_score = total_reward / max_possible
    
    # CRITICAL: Clamp score between 0.01 and 0.99 (Never exactly 0 or 1)
    final_score = max(0.01, min(0.99, raw_score))
    
    return {"score": round(final_score, 2)}

# --- Entry Point ---
def main():
    """Entry point for the OpenEnv validator."""
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
