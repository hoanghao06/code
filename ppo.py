import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

device = 'cpu'


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


# ================= ACTOR =================
class Actor_Beta(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.alpha_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.beta_layer = nn.Linear(args.hidden_width, args.action_dim)

        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))

        alpha = F.softplus(self.alpha_layer(s)) + 1.0
        beta = F.softplus(self.beta_layer(s)) + 1.0

        return alpha, beta

    def get_dist(self, s):
        alpha, beta = self.forward(s)
        return Beta(alpha, beta)

    def mean(self, s):
        alpha, beta = self.forward(s)
        return alpha / (alpha + beta)


# ================= CRITIC =================
class Critic(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, 1)

        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        return self.fc3(s)


# ================= PPO =================
class PPO_continuous():
    def __init__(self, args):

        self.gamma = args.gamma
        self.lamda = args.lamda
        self.epsilon = args.epsilon
        self.K_epochs = args.K_epochs

        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size

        self.lr_a = args.lr_a
        self.lr_c = args.lr_c

        self.use_adv_norm = args.use_adv_norm
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay

        self.max_train_steps = args.max_train_steps

        self.actor = Actor_Beta(args).to(device)
        self.critic = Critic(args).to(device)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    # ================= EVAL =================
    def evaluate(self, s):
        s = torch.tensor(s, dtype=torch.float, device=device).unsqueeze(0)
        return self.actor.mean(s).detach().cpu().numpy().flatten()

    # ================= ACTION =================
    def choose_action(self, s):
        s = torch.tensor(s, dtype=torch.float, device=device).unsqueeze(0)

        with torch.no_grad():
            dist = self.actor.get_dist(s)
            a = dist.sample()
            a_logprob = dist.log_prob(a)

        return a.cpu().numpy().flatten(), a_logprob.cpu().numpy().flatten()

    # ================= UPDATE =================
    def update(self, replay_buffer, total_steps, writer, entropy_coef):

        s, a, a_logprob, r, s_, dw, done = replay_buffer.numpy_to_tensor(device=device)

        # ===== GAE =====
        with torch.no_grad():
            vs = self.critic(s)
            vs_ = self.critic(s_)

            deltas = r + self.gamma * (1 - dw) * vs_ - vs

            adv = []
            gae = 0

            for delta, d in zip(reversed(deltas.squeeze()), reversed(done.squeeze())):
                gae = delta + self.gamma * self.lamda * gae * (1 - d)
                adv.insert(0, gae)

            adv = torch.tensor(adv, dtype=torch.float, device=device).view(-1, 1)
            v_target = adv + vs

            if self.use_adv_norm:
                adv = (adv - adv.mean()) / (adv.std() + 1e-5)

        # ===== TRAIN =====
        for _ in range(self.K_epochs):

            for index in BatchSampler(
                SubsetRandomSampler(range(self.batch_size)),
                self.mini_batch_size,
                False
            ):

                dist_now = self.actor.get_dist(s[index])
                dist_entropy = dist_now.entropy().sum(1, keepdim=True)

                a_logprob_now = dist_now.log_prob(a[index])

                ratios = torch.exp(
                    a_logprob_now.sum(1, keepdim=True) -
                    a_logprob[index].sum(1, keepdim=True)
                )

                surr1 = ratios * adv[index]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]

                actor_loss = -torch.min(surr1, surr2) - entropy_coef * dist_entropy

                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()

                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)

                self.optimizer_actor.step()

                # ===== CRITIC =====
                v_s = self.critic(s[index])
                critic_loss = F.mse_loss(v_target[index], v_s)

                self.optimizer_critic.zero_grad()
                critic_loss.backward()

                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)

                self.optimizer_critic.step()

        # ===== LR DECAY =====
        if self.use_lr_decay:
            self.lr_decay(total_steps)

        # ===== LOG =====
        writer.add_scalar('loss/actor', actor_loss.mean(), total_steps)
        writer.add_scalar('loss/critic', critic_loss.mean(), total_steps)
        writer.add_scalar('entropy', dist_entropy.mean(), total_steps)

    # ================= LR DECAY =================
    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)

        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now

        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now

    # ================= SAVE MODEL =================
    def save_policy(self, reward, path, episode_num):

        actor_path = path + '/actor/'
        critic_path = path + '/critic/'

        os.makedirs(actor_path, exist_ok=True)
        os.makedirs(critic_path, exist_ok=True)

        torch.save(
            self.actor.state_dict(),
            actor_path + f'{episode_num}_{reward:.3f}.pth'
        )

        torch.save(
            self.critic.state_dict(),
            critic_path + f'{episode_num}_{reward:.3f}.pth'
        )
