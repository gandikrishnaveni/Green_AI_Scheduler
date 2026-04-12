import json
import argparse
from pathlib import Path
from env.green_scheduler_env import GreenSchedulerEnv
from agent.heuristic_agent import CarbonAwareGreedyAgent

def run_episode(env: GreenSchedulerEnv, agent: CarbonAwareGreedyAgent) -> dict:
    env.reset()
    done = False
    while not done:
        action = agent.act(env)
        _, _, done, _ = env.step(action)
    return env.get_metrics()

def run_experiment(n_episodes: int = 50, difficulty: str = "medium", carbon_threshold: float = 300.0):
    agent = CarbonAwareGreedyAgent(carbon_threshold=carbon_threshold)
    results = []

    for ep in range(n_episodes):
        # Different seed per episode = varied carbon cycles, but reproducible
        env = GreenSchedulerEnv(difficulty=difficulty, seed=ep * 42)
        metrics = run_episode(env, agent)
        metrics["episode"] = ep
        results.append(metrics)

    # Aggregate
    n = len(results)
    summary = {
        "config": {"n_episodes": n, "difficulty": difficulty, "carbon_threshold": carbon_threshold},
        "mean_completion_rate": round(sum(r["completion_rate"] for r in results) / n, 3),
        "mean_carbon_per_job": round(sum(r["carbon_per_job_gco2"] for r in results) / n, 1),
        "mean_episode_return": round(sum(r["episode_return"] for r in results) / n, 3),
        "deadline_miss_rate": round(sum(r["deadline_miss_rate"] for r in results) / n, 3),
        "episodes": results,
    }
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--difficulty", default="medium")
    parser.add_argument("--threshold", type=float, default=300.0)
    parser.add_argument("--out", default="results/experiment.json")
    args = parser.parse_args()

    summary = run_experiment(args.episodes, args.difficulty, args.threshold)
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    
    print(f"Completion rate:   {summary['mean_completion_rate']:.1%}")
    print(f"Carbon / job:      {summary['mean_carbon_per_job']} gCO2")
    print(f"Episode return:    {summary['mean_episode_return']}")
    print(f"Deadline miss:     {summary['deadline_miss_rate']:.1%}")