import math
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

@dataclass
class Job:
    id: str
    duration_remaining: int
    deadline: int
    original_duration: int

    def is_expired(self, step: int) -> bool:
        return step >= self.deadline

class GreenSchedulerEnv:
    """
    A carbon-aware job scheduling environment.
    
    Observation space (7-dim vector):
      [0] current_step (normalized 0-1)
      [1] carbon_intensity (normalized, 0-1 over [100, 600] range)
      [2] grid_trend (-1=rising, 0=stable, +1=falling)
      [3] jobs_remaining (normalized 0-1)
      [4] min_deadline_urgency (normalized 0-1, 1=job about to expire)
      [5] total_carbon_so_far (normalized)
      [6] completion_ratio (0-1)
    
    Action space:
      0 = wait (idle this step)
      1..N = run job N (index into pending_jobs)
    """

    MAX_CARBON = 600.0
    MIN_CARBON = 100.0
    MAX_STEPS = 12

    def __init__(self, difficulty: str = "medium", seed: Optional[int] = None):
        self.difficulty = difficulty
        self.seed = seed
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self.reset()

    def _make_jobs(self) -> List[Job]:
        configs = {
            "easy": [
                Job("NLP_Train", 3, 10, 3),
            ],
            "medium": [
                Job("NLP_Train", 3, 8, 3),
                Job("CV_Inference", 2, 10, 2),
            ],
            "hard": [
                Job("LLM_FineTune", 4, 9, 4),
                Job("CV_Batch", 3, 10, 3),
                Job("Data_Clean", 2, 10, 2),
            ],
        }
        return list(configs.get(self.difficulty, configs["medium"]))

    def _carbon_intensity(self, step: int) -> float:
        """
        Diurnal model: low carbon at solar peak (step ~5), high at night.
        Phase offset is randomized per episode for realism.
        """
        base = 350 + 200 * math.sin((step + self._phase_offset) * math.pi / 6)
        noise = self._rng.gauss(0, 15)
        return round(max(self.MIN_CARBON, min(self.MAX_CARBON, base + noise)), 2)

    def reset(self) -> np.ndarray:
        self._rng = random.Random(self.seed)
        self._np_rng = np.random.default_rng(self.seed)
        self._phase_offset = self._rng.uniform(0, 4)
        self.step_count = 0
        self.total_carbon = 0.0
        self.jobs_expired = 0
        self.pending: List[Job] = self._make_jobs()
        self.total_jobs = len(self.pending)
        self.completed_jobs = 0
        self.episode_log: List[Dict] = []
        return self._observe()

    def _observe(self) -> np.ndarray:
        ci = self._carbon_intensity(self.step_count)
        ci_next = self._carbon_intensity(self.step_count + 1)
        trend = np.sign(ci_next - ci)  # -1, 0, or +1

        urgency = 0.0
        if self.pending:
            min_slack = min(j.deadline - self.step_count for j in self.pending)
            urgency = max(0.0, 1.0 - min_slack / self.MAX_STEPS)

        obs = np.array([
            self.step_count / self.MAX_STEPS,
            (ci - self.MIN_CARBON) / (self.MAX_CARBON - self.MIN_CARBON),
            trend / 2.0 + 0.5,  # remap to [0,1] for neural net compat
            len(self.pending) / self.total_jobs,
            urgency,
            min(1.0, self.total_carbon / (self.total_jobs * self.MAX_STEPS * 350)),
            self.completed_jobs / self.total_jobs,
        ], dtype=np.float32)
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        action = 0: wait
        action = 1..N: run job at index action-1
        """
        assert not self._is_done(), "Cannot step a finished episode."

        ci = self._carbon_intensity(self.step_count)
        reward = 0.0
        info: Dict[str, Any] = {"step": self.step_count, "carbon_intensity": ci, "action": action}

        if action == 0:
            # Waiting is rewarded when carbon is high, penalized when urgent jobs exist
            urgency_penalty = sum(1 for j in self.pending if j.deadline - self.step_count <= 2)
            reward = (0.3 if ci > 350 else -0.1) - 0.15 * urgency_penalty
            info["event"] = "wait"
        else:
            job_idx = action - 1
            if job_idx < len(self.pending):
                job = self.pending[job_idx]
                self.total_carbon += ci
                job.duration_remaining -= 1
                reward = (self.MAX_CARBON - ci) / (self.MAX_CARBON - self.MIN_CARBON)  # [0,1]
                if job.duration_remaining <= 0:
                    self.pending.pop(job_idx)
                    self.completed_jobs += 1
                    # Deadline bonus: more reward for finishing well before deadline
                    slack = job.deadline - self.step_count
                    reward += 2.0 + 0.3 * slack
                    info["event"] = f"completed:{job.id}:slack={slack}"
                else:
                    info["event"] = f"progress:{job.id}"
            else:
                reward = -0.5  # invalid action
                info["event"] = "invalid_action"

        # Expire overdue jobs
        for job in [j for j in self.pending if j.is_expired(self.step_count + 1)]:
            self.pending.remove(job)
            self.jobs_expired += 1
            reward -= 3.0  # hard penalty for missing deadline
            info["event"] = info.get("event", "") + f"|expired:{job.id}"

        self.step_count += 1
        obs = self._observe()
        done = self._is_done()
        self.episode_log.append({**info, "reward": round(reward, 4), "done": done})
        return obs, round(reward, 4), done, info

    def _is_done(self) -> bool:
        return self.step_count >= self.MAX_STEPS or not self.pending

    def get_metrics(self) -> Dict[str, Any]:
        """Call after episode ends to get structured evaluation metrics."""
        carbon_per_job = (self.total_carbon / self.completed_jobs) if self.completed_jobs else float("inf")
        return {
            "completion_rate": round(self.completed_jobs / self.total_jobs, 3),
            "deadline_miss_rate": round(self.jobs_expired / self.total_jobs, 3),
            "total_carbon_gco2": round(self.total_carbon, 1),
            "carbon_per_job_gco2": round(carbon_per_job, 1),
            "steps_used": self.step_count,
            "episode_return": round(sum(e["reward"] for e in self.episode_log), 3),
        }
