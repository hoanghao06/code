import os
import numpy as np
import pandas as pd
import torch
import csv
from torch.utils.tensorboard import SummaryWriter

from normalization import Normalization, RewardScaling
from ppo import PPO_continuous
from replaybuffer import ReplayBuffer
from uav import MakeEnv


def evaluate_policy(args, env, agent, state_norm):
    times = 8
    evaluate_reward = 0

    for _ in range(times):
        s, _ = env.reset()
        if args.use_state_norm:
            s = state_norm(s, update=False)

        done = False
        episode_reward = 0

        while not done:
            a = agent.evaluate(s)

            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action
            else:
                action = a

            s_, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if args.use_state_norm:
                s_ = state_norm(s_, update=False)

            episode_reward += r
            s = s_

        evaluate_reward += episode_reward

    return evaluate_reward / times


def main(args, seed, speed, target_rate, ROOT_PATH=None, load_path=None, s_mean_std=None):

    env = MakeEnv(set_num=args.car_num, car_speed=speed, target_rate=target_rate)
    env_evaluate = MakeEnv(set_num=args.car_num, car_speed=speed, target_rate=target_rate)

    env.seed(seed)
    env_evaluate.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space.high[0])

    args.max_episode_steps = env.max_episode_steps
    args.max_train_steps = args.max_episode_steps * args.max_train_episodes

    if not load_path:

        if not os.path.exists(ROOT_PATH):
            os.makedirs(ROOT_PATH)

        replay_buffer = ReplayBuffer(args)
        agent = PPO_continuous(args)

        writer = SummaryWriter(log_dir=ROOT_PATH + '/runs')

        state_norm = Normalization(shape=args.state_dim)

        total_steps = 0
        episode_num = 0
        evaluate_num = 0
        all_evaluate_rewards = []

        initial_entropy = args.entropy_coef

        while episode_num < args.max_train_episodes:

            s, _ = env.reset()
            if args.use_state_norm:
                s = state_norm(s)

            done = False
            ep_r = 0

            while not done:

                a, a_logprob = agent.choose_action(s)

                if args.policy_dist == "Beta":
                    action = 2 * (a - 0.5) * args.max_action
                else:
                    action = a

                s_, r, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_r += r

                if args.use_state_norm:
                    s_ = state_norm(s_)

                dw = True if (done and terminated) else False

                replay_buffer.store(s, a, a_logprob, r, s_, dw, done)

                s = s_
                total_steps += 1

                # 🔥 UPDATE
                if replay_buffer.count == args.batch_size:

                    progress = total_steps / args.max_train_steps
                    current_entropy = max(0.0001, initial_entropy * (1 - progress))

                    agent.update(replay_buffer, total_steps, writer, current_entropy)

                    writer.add_scalar('train/entropy_coef', current_entropy, total_steps)

                    replay_buffer.count = 0

            writer.add_scalar('train/reward_ep', ep_r, total_steps)
            episode_num += 1

            # ================== EVALUATE ==================
            if episode_num % args.evaluate_episode_freq == 0:
                evaluate_num += 1

                evaluate_reward = evaluate_policy(args, env_evaluate, agent, state_norm)
                all_evaluate_rewards.append(evaluate_reward)

                print(f"eval:{evaluate_num} reward:{evaluate_reward} episode:{episode_num}")

                writer.add_scalar('evaluate/reward_ep', evaluate_reward, total_steps)

                # ================== 🔥 SAVE ==================
                if (evaluate_num >= 5) and (evaluate_reward >= np.mean(all_evaluate_rewards[-5:])):

                    # save reward
                    path = ROOT_PATH + '/data_train'
                    os.makedirs(path, exist_ok=True)

                    np.save(
                        path + f'/reward_{evaluate_num}.npy',
                        np.array(all_evaluate_rewards)
                    )

                    # save model
                    agent.save_policy(
                        reward=evaluate_reward,
                        path=ROOT_PATH + '/model/',
                        episode_num=episode_num
                    )

                    # save fly data nếu có
                    if hasattr(env_evaluate, "buffer"):
                        env_evaluate.buffer.save(
                            path=ROOT_PATH + '/flydata/',
                            episode=episode_num,
                            target_rate=target_rate
                        )

        # ================== SAVE CSV ==================
        with open(ROOT_PATH + '/episode_rewards.csv', 'w', newline='') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(['Episode', 'Reward'])

            for i, r in enumerate(all_evaluate_rewards):
                writer_csv.writerow([i, r])

        writer.close()
