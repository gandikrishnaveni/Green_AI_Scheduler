# 🌍 GreenAI-Scheduler: Carbon-Aware Reinforcement Learning Environment

## 1. Environment Overview & Motivation
As artificial intelligence models grow larger, their training and inference compute costs have a significant environmental impact. The GreenAI-Scheduler is a reinforcement learning environment built on the OpenEnv specification that simulates a real-world energy-aware task queue.

The motivation is to train an AI agent to dynamically schedule GPU workloads (like NLP and CV model training) based on the real-time carbon intensity of the power grid. The agent must balance strict job deadlines with the goal of minimizing overall carbon emissions, learning when to "wait" during high-carbon peaks and when to "run" during green-energy valleys.

## 2. Action and Observation Spaces

### Observation Space (State)
The environment provides a structured JSON state at each step, containing:
* `current_step` (int): The current timestep in the episode (0-10). 
* `steps_left` (int): The remaining steps before the absolute deadline. 
* `carbon_intensity` (float): A simulated real-time metric of grid carbon (50.0 to 400.0). 
* `pending_jobs` (List[Dict]): A queue of jobs, each containing an id and duration. 
* `status` (str): The current environment status (e.g., "Running"). 

### Action Space
The agent can take one of two actions per step by returning a JSON object:
* **Wait:** `{"command": "wait", "job_id": null}` - Idles the cluster to avoid high carbon emissions.
* **Run Job:** `{"command": "run_job", "job_id": "<job_name>"}` - Executes a specific job.

## 3. Task Descriptions & Difficulty Levels

| Task | Goal | Difficulty & Mechanics |
| :--- | :--- | :--- |
| **Easy** | Finish 1 NLP job (3 steps) in 10 steps. | **Low:** Focuses on learning basic mechanics. Ample time to wait for green energy. |
| **Medium** | Balance 2 jobs (5 total steps) before deadline. | **Moderate:** Introduces job priority and restricts the number of "wait" actions. |
| **Hard** | Balance 3 jobs (9 total steps) in 10 steps. | **High:** Zero margin for error. Forces difficult trade-offs between deadlines and carbon usage. |

## 4. Setup and Usage Instructions

### Installation
The environment is packaged for multi-mode deployment via `pyproject.toml` and `uv.lock`.
```bash
git clone [https://github.com/gandikrishnaveni/Green_AI_Scheduler.git](https://github.com/gandikrishnaveni/Green_AI_Scheduler.git)
cd Green_AI_Scheduler
pip install .
