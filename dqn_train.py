import argparse
import os
import numpy as np
import torch
from uav_dqn import MakeEnv
from dqn_agent import DQNAgent
import csv

def evaluate_policy(env, agent, n_episodes=5):
    total_reward = 0
    for _ in range(n_episodes):
        s, _ = env.reset()
        done = False
        ep_r = 0
        while not done:
            a = agent.select_action(s, evaluate=True)
            s_, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            ep_r += r
            s = s_
        total_reward += ep_r
    return total_reward / n_episodes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_episodes", type=int, default=500)
    parser.add_argument("--car_num", type=int, default=3)
    parser.add_argument("--car_speed", type=int, default=10)
    parser.add_argument("--target_rate", type=float, default=3)  # Gbps
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.01)
    parser.add_argument("--epsilon_decay", type=int, default=5000)
    parser.add_argument("--buffer_capacity", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--target_update", type=int, default=100)
    parser.add_argument("--learning_start", type=int, default=1000)
    parser.add_argument("--train_freq", type=int, default=4)
    parser.add_argument("--evaluate_freq", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    env = MakeEnv(set_num=args.car_num, car_speed=args.car_speed, target_rate=args.target_rate)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        target_update=args.target_update,
        device=args.device
    )

    os.makedirs("./output_dqn", exist_ok=True)
    total_steps = 0
    episode_rewards = []
    eval_rewards = []

    for episode in range(args.max_episodes):
        s, _ = env.reset()
        done = False
        ep_r = 0
        while not done:
            a = agent.select_action(s, evaluate=False)
            s_, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            agent.store_transition(s, a, r, s_, done)
            ep_r += r
            s = s_
            total_steps += 1

            if total_steps > args.learning_start and total_steps % args.train_freq == 0:
                loss, avg_q = agent.learn()
                agent.update_epsilon(total_steps)

        episode_rewards.append(ep_r)
        print(f"Episode {episode} | Reward: {ep_r:.2f} | Epsilon: {agent.epsilon:.3f}")

        if episode % args.evaluate_freq == 0:
            eval_r = evaluate_policy(env, agent, n_episodes=3)
            eval_rewards.append(eval_r)
            print(f"Eval reward: {eval_r:.2f}")
            # Lưu model nếu tốt nhất
            if len(eval_rewards) >= 3 and eval_r >= max(eval_rewards[-3:]):
                torch.save(agent.policy_net.state_dict(),
                           f"./output_dqn/best_dqn_ep{episode}.pth")

    # Lưu kết quả
    # Sau vòng lặp, lưu CSV
    with open("./output_dqn/episode_rewards.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        for ep, rew in enumerate(episode_rewards):
            writer.writerow([ep, rew])
    
    with open("./output_dqn/eval_rewards.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["eval_episode", "reward"])
        eval_episodes = list(range(0, len(eval_rewards) * args.evaluate_freq, args.evaluate_freq))
        for ep, rew in zip(eval_episodes, eval_rewards):
            writer.writerow([ep, rew])
    
    print("Training finished. CSV files saved.")

if __name__ == "__main__":
    main()