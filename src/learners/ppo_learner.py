# code heavily adapted from https://github.com/AnujMahajanOxf/MAVEN
import copy

import torch as th
from torch.optim import Adam

from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from modules.critics import REGISTRY as critic_resigtry
from components.penalty import StuckPenaltyRewardShaper, OscillationPenaltyRewardShaper
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

class PPOLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger
        self.stuck_penalty = StuckPenaltyRewardShaper(
            max_lookback=20,
            base_penalty=0.0005,  # Adjust based on reward scale
            penalty_growth_rate=1.1
        )
        self.osc_penalty = OscillationPenaltyRewardShaper(
            lookback=20,
            osc_coeff=0.0005,
            growth_rate=1.1
        )

        self.mac = mac
        self.old_mac = copy.deepcopy(mac)
        self.agent_params = list(mac.parameters())
        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr)
        self.agent_scheduler = CosineAnnealingLR(
            self.agent_optimiser, T_max=self.args.t_max)

        self.critic = critic_resigtry[args.critic_type](scheme, args, context_shape=args.n_agents+2) 
        self.target_critic = copy.deepcopy(self.critic)

        self.critic_params = list(self.critic.parameters())
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.lr)
        self.critic_scheduler = CosineAnnealingLR(
            self.critic_optimiser, T_max=self.args.t_max)
        self.no_op_action = 0

        self.last_target_update_step = 0
        self.critic_training_steps = 0
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.segment_len = 1

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            rew_shape = (1,) if self.args.common_reward else (self.n_agents,)
            self.rew_ms = RunningMeanStd(shape=rew_shape, device=device)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        self.old_mac.agent.train()
        self.mac.agent.train()
        rewards = batch["reward"][:, :-1]
        pos_index = self.args.n_agents * 3 
        positions = batch["obs"][:, :, :, -pos_index:-pos_index+2]
        
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        actions = actions[:, :-1]
        no_op_mask = (actions == self.no_op_action).float().mean(dim=(0, 1))
        active_agents = (no_op_mask < 0.99).float()  # 1 for learning agents, 0 for no-op agents
        active_agents = active_agents.view(1, 1, -1)
        
        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        if self.args.common_reward:
            assert (
                rewards.size(2) == 1
            ), "Expected singular agent dimension for common rewards"
            # reshape rewards to be of shape (batch_size, episode_length, n_agents)
            rewards = rewards.expand(-1, -1, self.n_agents)

        rewards = self.stuck_penalty.shape_rewards(rewards, positions)
        rewards = self.osc_penalty.shape_rewards(rewards, positions)
        mask = mask.repeat(1, 1, self.n_agents)
        #mask = mask * active_agents
        critic_mask = mask.clone()

        old_mac_out = []
        self.old_mac.init_hidden(batch.batch_size)
        for t in range(0, batch.max_seq_length - 1, self.segment_len):
            t_end = min(t + self.segment_len, batch.max_seq_length - 1)
            agent_outs = self.old_mac.forward(batch, t=t, t_end=t_end, actions=actions, return_aux_losses=False)
            old_mac_out.append(agent_outs)
        old_mac_out = th.cat(old_mac_out, dim=1)  # Concat over time
        old_pi = old_mac_out
        old_pi[mask == 0] = 1.0

        old_pi_taken = th.gather(old_pi, dim=3, index=actions).squeeze(3)
        old_log_pi_taken = th.log(old_pi_taken + 1e-10)
        for k in range(self.args.epochs):
            mac_out = []
            aux_losses = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(0, batch.max_seq_length - 1, self.segment_len):
                t_end = min(t + self.segment_len, batch.max_seq_length - 1)
                agent_outs, aux_loss = self.mac.forward(batch, t=t, t_end=t_end, actions=actions, return_aux_losses=True)
                mac_out.append(agent_outs)
                aux_losses.append(aux_loss)
            mac_out = th.cat(mac_out, dim=1)  # Concat over time

            pi = mac_out
            faulty_indices = self.mac.agent.faulty_agent_indices
            if not self.mac.agent._faulty:
                faulty_indices = {}
            advantages, critic_train_stats = self.train_critic_sequential(
                self.critic, self.target_critic, batch, rewards, critic_mask, actions, faulty_indices=faulty_indices
            )
            advantages = advantages.detach()
            # Calculate policy grad with mask

            pi[mask == 0] = 1.0

            pi_taken = th.gather(pi, dim=3, index=actions).squeeze(3)
            log_pi_taken = th.log(pi_taken + 1e-10)

            ratios = th.exp(log_pi_taken - old_log_pi_taken.detach())
            surr1 = ratios * advantages
            surr2 = (
                th.clamp(ratios, 1 - self.args.eps_clip, 1 + self.args.eps_clip)
                * advantages
            )

            entropy = -th.sum(pi * th.log(pi + 1e-10), dim=-1)
              # Apply agent mask
            pg_loss = (
                -(
                    (th.min(surr1, surr2) + self.args.entropy_coef * entropy) * mask
                ).sum()
                / mask.sum()
            )
            total_loss = pg_loss + 0.05 * aux_loss
            
            self.agent_optimiser.zero_grad()
            total_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(
                self.agent_params, self.args.grad_norm_clip
            )
            self.agent_optimiser.step()
            self.agent_scheduler.step()


        self.old_mac.load_state(self.mac)

        self.critic_training_steps += 1
        if (
            self.args.target_update_interval_or_tau > 1
            and (self.critic_training_steps - self.last_target_update_step)
            / self.args.target_update_interval_or_tau
            >= 1.0
        ):
            self._update_targets_hard()
            self.last_target_update_step = self.critic_training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_train_stats["critic_loss"])
            for key in [
                "critic_loss",
                "critic_grad_norm",
                "td_error_abs",
                "q_taken_mean",
                "target_mean",
            ]:
                self.logger.log_stat(
                    key, sum(critic_train_stats[key]) / ts_logged, t_env
                )

            self.logger.log_stat(
                "advantage_mean",
                (advantages * mask).sum().item() / mask.sum().item(),
                t_env,
            )
            self.logger.log_stat("pg_loss", pg_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm.item(), t_env)
            self.logger.log_stat(
                "pi_max",
                (pi.max(dim=-1)[0] * mask).sum().item() / mask.sum().item(),
                t_env,
            )
            self.log_stats_t = t_env

    def train_critic_sequential(self, critic, target_critic, batch, rewards, mask, actions=None, faulty_indices={}):
        # Optimise critic
        MONOTONICITY_COEFF = 0.01 
        CONSERVATIVE_COEFF = 0.005 # NEW: Conservative Regularization Factor (Tune between 1e-4 and 1e-2)
        BETA = 5.0
        # --- Update 1: Retrieve flat_mask from critic forward pass ---
        with th.no_grad():
            target_mean_v, target_z_quantiles, _ = target_critic(batch, faulty_indices=faulty_indices)
            target_mean_v = target_mean_v.squeeze(3)

        if self.args.standardise_returns:
            target_mean_v = target_mean_v * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        target_returns = self.nstep_returns(
            rewards, mask, target_mean_v, self.args.q_nstep
        )
        
        if self.args.standardise_returns:
            self.ret_ms.update(target_returns)
            target_returns = (target_returns - self.ret_ms.mean) / th.sqrt(
                self.ret_ms.var
            )

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }

        # Identify faulty agents (agents that always take no-op)
        # --- Update 2: Retrieve flat_mask from critic forward pass ---
        mean_v, z_quantiles, fault_mask_reshaped = critic(batch, faulty_indices=faulty_indices) 
        mean_v = mean_v[:, :-1].squeeze(3) # [bs, max_t, n_agents]
        z_quantiles = z_quantiles[:, :-1] 
        fault_mask = fault_mask_reshaped[:, :-1] # [bs, max_t, n_agents, 1]

        # Target returns G_t are [bs, max_t, n_agents]
        target_returns_expanded = target_returns.detach().unsqueeze(-1) # [bs, max_t, n_agents, 1]
        tau_hat = critic.tau_hat.to(z_quantiles.device).squeeze(0) 

        # 1. Quantile Pinball Loss (L_pinball)
        td_error = target_returns_expanded - z_quantiles 
        error_indicator = (td_error < 0).float() 
        pinball_weight = th.abs(tau_hat - error_indicator)
        quantile_pinball_loss = th.abs(td_error) * pinball_weight
        
        # Apply fault weight factor 
        fault_weight_factor = 1.0 + fault_mask * (BETA - 1.0) 
        weighted_quantile_pinball_loss = quantile_pinball_loss * fault_weight_factor 
        mean_quantile_loss = weighted_quantile_pinball_loss.mean(dim=-1) # [bs, max_t, n_agents]
        masked_pinball_loss = mean_quantile_loss * mask 
        
        # 2. Monotonicity Regularization (R_mono)
        quantile_diffs = z_quantiles[:, :, :, :-1] - z_quantiles[:, :, :, 1:] 
        monotonicity_violation = F.relu(quantile_diffs)
        masked_violation = (monotonicity_violation ** 2) * mask.unsqueeze(-1)
        R_mono = masked_violation.sum() / (mask).sum()

        # --- NEW: 3. Conservative Distributional Regularization (R_conservative) ---
        # Goal: Penalize the predicted quantiles if they are too high relative to a known baseline (e.g., zero).
        # We penalize the sum of all predicted quantiles for being positive.
        # This acts as a structural lower bound regularization.
        
        # Only penalize positive predicted quantiles (Q-values)
        positive_quantiles = F.relu(z_quantiles) # [bs, max_t, n_agents, n_quantiles]
        
        # Apply mask and average over all dimensions
        # R_conservative aims to drive optimistic predictions down.
        masked_conservative_penalty = positive_quantiles * mask.unsqueeze(-1)
        
        R_conservative = masked_conservative_penalty.sum() / (mask).sum()

        # 4. Total Critic Loss
        L_pinball = masked_pinball_loss.sum() / (mask).sum()
        
        # Total loss now includes the regularization terms
        L_critic = L_pinball + (MONOTONICITY_COEFF * R_mono) + (CONSERVATIVE_COEFF * R_conservative) # Total loss

        # Backpropagation (Crucial Stability Check)
        self.critic_optimiser.zero_grad()
        L_critic.backward()
        
        # VITAL: Ensure aggressive Gradient Clipping (0.5 is a highly stable norm for PPO/Actor-Critic) [1]
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.critic_params, 0.5 
        )
        # Ensure step is only called once.
        self.critic_optimiser.step()
        self.critic_scheduler.step() # Re-add scheduler step if applicable

        #... (logging remains the same)...
        running_log["critic_loss"].append(L_critic.item())
        running_log["critic_grad_norm"].append(grad_norm.item())
        mask_elems = mask.sum().item()
        td_error_mean = target_returns.detach() - mean_v.detach() # Use the mean V for logging TD error
        running_log["td_error_abs"].append(
            (th.abs(td_error_mean) * mask).sum().item() / mask_elems
        )
        running_log["q_taken_mean"].append((mean_v * mask).sum().item() / mask_elems)
        running_log["target_mean"].append(
            (target_returns * mask).sum().item() / mask_elems
        )

        return td_error_mean, running_log

    def nstep_returns(self, rewards, mask, values, nsteps):
        nstep_values = th.zeros_like(values[:, :-1])
        for t_start in range(rewards.size(1)):
            nstep_return_t = th.zeros_like(values[:, 0])
            for step in range(nsteps + 1):
                t = t_start + step
                if t >= rewards.size(1):
                    break
                elif step == nsteps:
                    nstep_return_t += (
                        self.args.gamma ** (step) * values[:, t] * mask[:, t]
                    )
                elif t == rewards.size(1) - 1 and self.args.add_value_last_step:
                    nstep_return_t += (
                        self.args.gamma ** (step) * rewards[:, t] * mask[:, t]
                    )
                    nstep_return_t += self.args.gamma ** (step + 1) * values[:, t + 1]
                else:
                    nstep_return_t += (
                        self.args.gamma ** (step) * rewards[:, t] * mask[:, t]
                    )
            nstep_values[:, t_start, :] = nstep_return_t
        return nstep_values

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_hard(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.old_mac.cuda()
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(
            th.load(
                "{}/critic.th".format(path), map_location=lambda storage, loc: storage
            )
        )
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(
            th.load(
                "{}/agent_opt.th".format(path),
                map_location=lambda storage, loc: storage,
            )
        )
        self.critic_optimiser.load_state_dict(
            th.load(
                "{}/critic_opt.th".format(path),
                map_location=lambda storage, loc: storage,
            )
        )
