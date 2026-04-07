GreenAI-Scheduler: Carbon-Aware Reinforcement Learning Environment
1. Environment Overview & Motivation
As artificial intelligence models grow larger, their training and inference compute costs have a significant environmental impact. The GreenAI-Scheduler is a reinforcement learning environment built on the OpenEnv specification that simulates a real-world energy-aware task queue.

The motivation is to train an AI agent to dynamically schedule GPU workloads (like NLP and CV model training) based on the real-time carbon intensity of the power grid. The agent must balance strict job deadlines with the goal of minimizing overall carbon emissions, learning when to "wait" during high-carbon peaks and when to "run" during green-energy valleys.

2. Action and Observation Spaces
Observation Space (State)
The environment provides a structured JSON state at each step, containing:

current_step (int): The current timestep in the episode (0-10).
steps_left (int): The remaining steps before the absolute deadline.
carbon_intensity (float): A simulated real-time metric of grid carbon (50.0 to 400.0).
pending_jobs (List[Dict]): A queue of jobs, each containing an id and duration.
status (str): The current environment status ("Running").
Action Space
The agent can take one of two actions per step:

Wait: {"command": "wait", "job_id": null}
Run Job: {"command": "run_job", "job_id": "<ID>"}
3. Task Descriptions & Difficulty Levels
Task	Goal	Difficulty
Easy	Finish 1 NLP job (3 steps) in 10 steps.	Learning basic mechanics.
Medium	Balance 2 jobs (5 total steps) before deadline.	Priority & carbon avoidance.
Hard	Balance 3 jobs (9 total steps) in 10 steps.	Zero margin for error.
4. Setup and Usage Instructions
Running the Environment Locally
# Install dependencies
pip install fastapi uvicorn pydantic requests openai

# Start the environment server
python -m uvicorn main:app --host 0.0.0.0 --port 7860
