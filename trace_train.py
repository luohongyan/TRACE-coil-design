"""CUDA PyTorch implementation of the exact-state SeqTO Actor-Critic."""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import trace_environment as trace_env
from physics import initialize, make_problem


OUT = Path(__file__).resolve().parent / "outputs" / "trace"
OUT.mkdir(parents=True, exist_ok=True)


class TorchCNNActorCritic(nn.Module):
    def __init__(self, observation_shape, action_count=4):
        super().__init__()
        height, width, channels = observation_shape
        self.conv1 = nn.Conv2d(channels, 8, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            flat = self.conv2(F.relu(self.conv1(dummy))).numel()
        self.dense = nn.Linear(flat, 128)
        self.actor = nn.Linear(128, action_count)
        self.critic = nn.Linear(128, 1)
        nn.init.constant_(self.actor.bias, 0.0)
        with torch.no_grad():
            self.actor.bias[1] = 1.0

    def forward(self, observation):
        features = F.relu(self.conv1(observation))
        features = F.relu(self.conv2(features))
        features = torch.flatten(features, 1)
        features = F.relu(self.dense(features))
        return self.actor(features), self.critic(features).squeeze(-1)


class TorchAgent:
    def __init__(self, observation_shape, device, seed=0, learning_rate=1e-4,
                 gamma=0.995, critic_weight=0.2):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.device = torch.device(device)
        self.model = TorchCNNActorCritic(observation_shape).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=learning_rate)
        self.gamma = float(gamma)
        self.critic_weight = float(critic_weight)

    @staticmethod
    def _nchw(values):
        return np.transpose(np.asarray(values, dtype=np.float32), (0, 3, 1, 2))

    def act(self, observation, action_mask, deterministic=False):
        tensor = torch.from_numpy(
            np.transpose(observation[None, ...], (0, 3, 1, 2))).to(
                self.device, non_blocking=True)
        with torch.inference_mode():
            logits, value = self.model(tensor)
            mask = torch.as_tensor(action_mask, dtype=torch.bool,
                                   device=self.device).unsqueeze(0)
            logits = logits.masked_fill(~mask, -1e9)
            probability = torch.softmax(logits, dim=1)
            if deterministic:
                action = torch.argmax(probability, dim=1)
            else:
                action = torch.multinomial(probability, 1).squeeze(1)
        return int(action.item()), float(value.item()), probability[0].cpu().numpy()

    def update(self, trajectories, entropy_weight, elite_count=0,
               elite_bc_weight=0.15):
        observations, actions, returns, masks, elite_flags = [], [], [], [], []
        elite_start = max(0, len(trajectories)-int(elite_count))
        for trajectory_index, trajectory in enumerate(trajectories):
            rewards = trajectory["rewards"]
            episode_returns = np.empty(len(rewards), dtype=np.float32)
            value = 0.0
            for index in range(len(rewards)-1, -1, -1):
                value = rewards[index]+self.gamma*value
                episode_returns[index] = value
            observations.extend(trajectory["observations"])
            actions.extend(trajectory["actions"])
            returns.extend(episode_returns)
            masks.extend(trajectory["action_masks"])
            elite_flags.extend(
                [trajectory_index >= elite_start]*len(episode_returns))
        observations_t = torch.from_numpy(self._nchw(observations)).to(self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32,
                                    device=self.device)
        masks_t = torch.as_tensor(np.asarray(masks), dtype=torch.bool,
                                  device=self.device)
        elite_t = torch.as_tensor(elite_flags, dtype=torch.float32,
                                  device=self.device)
        logits, values = self.model(observations_t)
        logits = logits.masked_fill(~masks_t, -1e9)
        advantages = returns_t-values
        normalized = ((advantages-advantages.mean()) /
                      (advantages.std(unbiased=False)+1e-6)).detach()
        negative_log_policy = F.cross_entropy(
            logits, actions_t, reduction="none")
        actor_loss = (negative_log_policy*normalized).mean()
        bc_loss = ((negative_log_policy*elite_t).sum() /
                   (elite_t.sum()+1e-6))
        critic_loss = F.huber_loss(values, returns_t, delta=10.0)
        probability = torch.softmax(logits, dim=1)
        entropy = -(probability*torch.log_softmax(logits, dim=1)).sum(1).mean()
        loss = (actor_loss+self.critic_weight*critic_loss-
                float(entropy_weight)*entropy+
                float(elite_bc_weight)*bc_loss)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()
        return {"loss": float(loss.detach().cpu()),
                "actor_loss": float(actor_loss.detach().cpu()),
                "critic_loss": float(critic_loss.detach().cpu()),
                "behavior_cloning_loss": float(bc_loss.detach().cpu()),
                "entropy": float(entropy.detach().cpu()),
                "gradient_norm": float(norm.detach().cpu())}

    def behavior_clone(self, trajectory):
        observations = torch.from_numpy(
            self._nchw(trajectory["observations"])).to(self.device)
        actions = torch.as_tensor(trajectory["actions"], dtype=torch.long,
                                  device=self.device)
        masks = torch.as_tensor(np.asarray(trajectory["action_masks"]),
                                dtype=torch.bool, device=self.device)
        logits, _ = self.model(observations)
        logits = logits.masked_fill(~masks, -1e9)
        loss = F.cross_entropy(logits, actions)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()
        accuracy = (torch.argmax(logits, 1) == actions).float().mean()
        return {"loss": float(loss.detach().cpu()),
                "accuracy": float(accuracy.detach().cpu()),
                "gradient_norm": float(norm.detach().cpu())}


def run_episode(environment, agent, physics_weight=1.0, deterministic=False):
    observation = environment.reset()
    trajectory = {"observations": [], "actions": [],
                  "action_masks": [], "rewards": []}
    done, info = False, {}
    started = time.perf_counter()
    while not done:
        action_mask = environment.admissible_action_mask()
        action, _, _ = agent.act(observation, action_mask, deterministic)
        next_observation, reward, done, info = environment.step(
            action, physics_weight=physics_weight)
        trajectory["observations"].append(observation)
        trajectory["actions"].append(action)
        trajectory["action_masks"].append(action_mask)
        trajectory["rewards"].append(reward)
        observation = next_observation
    return {"trajectory": trajectory,
            "reward": float(sum(trajectory["rewards"])),
            "steps": len(trajectory["rewards"]),
            "rho": environment.mask.copy(), "info": info,
            "elapsed_s": time.perf_counter()-started}


def train(axis, args):
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    physics = argparse.Namespace(nl=args.nl, nz=args.nz, alpha=100.0,
        volume_fraction=args.volume_fraction, filter_radius=2.0)
    problem = make_problem(axis, physics)
    initialize(problem, np.full(problem.mesh.ne, args.volume_fraction))
    environment = trace_env.StabilizedSeqTOEnvironment(
        axis, problem, wire_width_mm=args.wire_width_mm,
        volume_fraction=args.volume_fraction,
        observation_shape=(args.observation_nl, args.observation_nz),
        resistance_weight=args.resistance_weight,
        resistance_limit_ohm=(args.resistance_limit_mohm * 1e-3
                              if args.resistance_limit_mohm else None),
        exact_state=True, roi_radius_mm=args.roi_radius_mm)
    first_observation = environment.reset()
    agent = TorchAgent(first_observation.shape, args.device,
        seed=args.seed+"xyz".index(axis), learning_rate=args.learning_rate,
        gamma=args.gamma)
    parameter_count = sum(p.numel() for p in agent.model.parameters())
    best, history, elite_buffer, batch, snapshots = None, [], [], [], {}
    started = time.perf_counter()
    for episode in range(1, args.episodes+1):
        physics_weight, entropy_weight = trace_env.curriculum_weights(
            episode, args.pretrain_episodes, args.physics_ramp_episodes,
            args.entropy_start, args.entropy_end, args.episodes)
        result = run_episode(environment, agent, physics_weight)
        batch.append(result["trajectory"])
        info = result["info"]
        error = info.get("error_percent", np.nan)
        if info.get("valid_topology") and np.isfinite(error):
            elite = trace_env.clone_trajectory(result["trajectory"])
            elite["rewards"][-1] -= (1.0-physics_weight)*min(error, 100.0)
            elite_buffer.append((float(error), elite))
            elite_buffer.sort(key=lambda item: item[0])
            del elite_buffer[args.elite_buffer_size:]
        optimization = {}
        if len(batch) >= args.batch_episodes or episode == args.episodes:
            replay = [item[1] for item in elite_buffer[:args.elite_replay_count]]
            optimization = agent.update(batch+replay, entropy_weight,
                elite_count=len(replay), elite_bc_weight=args.elite_bc_weight)
            batch = []
        history.append((episode, result["reward"], error,
                        float(info["reached_goal"]), info["occupied"],
                        result["steps"], physics_weight, entropy_weight))
        if info.get("valid_topology") and np.isfinite(error) and (
                best is None or error < best["error_percent"]):
            best = dict(info, rho=result["rho"], episode=episode)
        if episode in args.snapshot_episode_set:
            greedy = run_episode(environment, agent, 1.0, True)
            snapshots[episode] = {"episode": episode,
                "policy_rho": greedy["rho"], "policy_info": dict(greedy["info"]),
                "best_rho": None if best is None else best["rho"].copy(),
                "best_error_percent": None if best is None else best["error_percent"]}
        if episode % args.log_every == 0:
            print(f"{axis} ep {episode:5d} device={args.device} "
                  f"error={error:.3f}% best="
                  f"{best['error_percent'] if best else float('nan'):.3f}% "
                  f"entropy={optimization.get('entropy', float('nan')):.3f}",
                  flush=True)
    training_s = time.perf_counter()-started
    distillation = []
    if elite_buffer:
        for update in range(1, args.distill_updates+1):
            metrics = agent.behavior_clone(elite_buffer[0][1])
            if update == 1 or update % 100 == 0:
                distillation.append({"update": update, **metrics})
    greedy = run_episode(environment, agent, 1.0, True)
    inference = []
    for _ in range(args.inference_samples):
        inference.append(run_episode(environment, agent, 1.0, False))
    valid = [x for x in inference if x["info"].get("valid_topology")]
    best_sample = min(valid, key=lambda x:x["info"]["error_percent"]) if valid else None
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    record = {"axis": axis, "framework": "PyTorch", "device": str(agent.device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "episodes": args.episodes, "parameter_count": parameter_count,
        "training_s": training_s, "best": best, "deterministic": greedy,
        "best_of_k": best_sample, "history": np.asarray(history),
        "snapshots": snapshots, "distillation": distillation,
        "config": vars(args)}
    output = OUT/f"{axis}_seqto_torch_{args.run_tag}_{args.nl}x{args.nz}_{args.episodes}ep.pkl"
    with output.open("wb") as stream:
        pickle.dump(record, stream)
    torch.save({"model": agent.model.state_dict(),
                "optimizer": agent.optimizer.state_dict(),
                "observation_shape": first_observation.shape,
                "config": vars(args)}, output.with_suffix(".pt"))
    summary = {"output": str(output), "device": str(agent.device),
        "gpu": record["gpu_name"], "parameters": parameter_count,
        "training_s": training_s,
        "best_error_percent": None if best is None else best["error_percent"],
        "greedy_error_percent": greedy["info"].get("error_percent"),
        "greedy_s_including_FEM": greedy["elapsed_s"],
        "best_of_k_error_percent": None if best_sample is None else
            best_sample["info"]["error_percent"]}
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--axes", default="x")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--nl", type=int, default=128)
    parser.add_argument("--nz", type=int, default=80)
    parser.add_argument("--wire-width-mm", type=float, default=1.0)
    parser.add_argument("--volume-fraction", type=float, default=0.20)
    parser.add_argument("--roi-radius-mm", type=float, default=None,
                        help="ROI radius used by the terminal gradient reward; "
                             "default preserves the 2.5-mm baseline radius")
    parser.add_argument("--resistance-weight", type=float, default=0.0,
                        help="soft resistance-penalty weight (0 = off)")
    parser.add_argument("--resistance-limit-mohm", type=float, default=0.0,
                        help="resistance budget R_max in mOhm (0 = off)")
    parser.add_argument("--observation-nl", type=int, default=32)
    parser.add_argument("--observation-nz", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--batch-episodes", type=int, default=16)
    parser.add_argument("--elite-buffer-size", type=int, default=32)
    parser.add_argument("--elite-replay-count", type=int, default=8)
    parser.add_argument("--elite-bc-weight", type=float, default=1.0)
    parser.add_argument("--pretrain-episodes", type=int, default=500)
    parser.add_argument("--physics-ramp-episodes", type=int, default=1000)
    parser.add_argument("--entropy-start", type=float, default=0.02)
    parser.add_argument("--entropy-end", type=float, default=0.0002)
    parser.add_argument("--distill-updates", type=int, default=1500)
    parser.add_argument("--inference-samples", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--snapshot-episodes",
                        default="500,1000,2000,3000,5000")
    parser.add_argument("--run-tag", default="cuda_v1")
    args = parser.parse_args()
    args.snapshot_episode_set = {int(x) for x in
        args.snapshot_episodes.split(",") if x.strip()}
    for axis in args.axes:
        train(axis, args)


if __name__ == "__main__":
    main()
