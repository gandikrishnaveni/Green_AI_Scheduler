import numpy as np
from env.green_scheduler_env import GreenSchedulerEnv

class CarbonAwareGreedyAgent:
    """
    Heuristic policy:
      1. If any job's deadline is within 2 steps, run it regardless of carbon.
      2. If carbon_intensity < threshold AND jobs remain, run the shortest job.
      3. Otherwise, wait for a cleaner window.
    
    This implements a simplified version of the Earliest-Deadline-First (EDF)
    scheduler with a carbon-aware gate, inspired by:
      Radovanovic et al. (2023) "Carbon-Aware Computing for Datacenters"
    """

    def __init__(self, carbon_threshold: float = 300.0):
        self.carbon_threshold = carbon_threshold

    def act(self, env: GreenSchedulerEnv) -> int:
        if not env.pending:
            return 0

        ci = env._carbon_intensity(env.step_count)
        
        # Rule 1: Deadline urgency overrides carbon preference
        urgent = [
            (i, j) for i, j in enumerate(env.pending)
            if j.deadline - env.step_count <= j.duration_remaining + 1
        ]
        if urgent:
            # Pick most urgent
            idx = min(urgent, key=lambda x: x[1].deadline)[0]
            return idx + 1

        # Rule 2: Run shortest job if grid is clean
        if ci < self.carbon_threshold:
            idx = min(range(len(env.pending)), key=lambda i: env.pending[i].duration_remaining)
            return idx + 1

        # Rule 3: Wait
        return 0