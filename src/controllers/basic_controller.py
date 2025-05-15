from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)


    def get_new_obs_with_faults(self, obs, bs):
        
        obs_faulty = th.zeros(obs.shape[0], obs.shape[1], obs.shape[2] + 9 ,device=obs.device)
        
        obs_t = obs.clone()
        
        faulty_indices = self.agent.faulty_agent_indices
        grid_index_map = {
            (-1, -1): 15,  # top-left
            (-1,  0): 23,  # top
            (-1,  1): 31,  # top-right
            ( 0, -1): 39,  # left
            ( 0,  0): 47,  # center (self)
            ( 0,  1): 55,  # right
            ( 1, -1): 63,  # bottom-left
            ( 1,  0): 71,  # bottom
            ( 1,  1): 79,  # bottom-right
        }

        agent_pos = obs_t[:, :, 0:2]
        lookup = th.full((3, 3), -1, dtype=th.long, device=obs.device)  # shape [3, 3]
        for (dy, dx), idx in grid_index_map.items():
            lookup[dy + 1, dx + 1] = idx  # shift -1:1 to 0:2
        for faulty_idx in faulty_indices:
    # Get positions of faulty agent across all envs → shape [10, 2]
            faulty_pos = agent_pos[:, faulty_idx, :]  # (envs, 2)
            
            # Expand to (10, 4, 2) to compare with all agents
            faulty_pos_expanded = faulty_pos.unsqueeze(1).expand(-1, 4, -1)
            
            # Get relative position dy, dx
            rel_pos = faulty_pos_expanded - agent_pos  # (10, 4, 2)
            rel_pos = faulty_pos_expanded - agent_pos  # shape (10, 4, 2)

            # Create visibility mask: only if both dy and dx ∈ [-1, 1]
            visible_mask = (rel_pos.abs() <= 1).all(dim=-1)  # shape (10, 4)

            # Use rel_pos only where visible
            dy = (rel_pos[..., 0] + 1).long()  # for indexing lookup
            dx = (rel_pos[..., 1] + 1).long()

            # For now set to -1, then overwrite visible positions
            obs_idx = th.full((bs, self.n_agents), -1, dtype=th.long, device=obs.device)
            obs_idx[visible_mask] = lookup[dy[visible_mask], dx[visible_mask]]  # shape (10, 4)

            # Prepare indexing to scatter add
            env_ids = th.arange(bs).unsqueeze(1).expand(-1, self.n_agents).flatten().to(obs.device)
            agent_ids = th.arange(self.n_agents).unsqueeze(0).expand(bs, -1).flatten().to(obs.device)
            obs_idx_flat = obs_idx.flatten()
            

            # Mask valid positions (within 3x3 grid)
            valid = obs_idx_flat != -1

            # Update: add 1 at the appropriate obs index
            obs_faulty[env_ids[valid], agent_ids[valid], obs_idx_flat[valid]] = 1
      
        obs_faulty[:, :, 0:15] = obs[:, :, 0:15]
        obs_faulty[:, :, 16:23] = obs[:, :, 15:22]
        obs_faulty[:, :, 24:31] = obs[:, :, 22:29]
        obs_faulty[:, :, 32:39] = obs[:, :, 29:36]
        obs_faulty[:, :, 40:47] = obs[:, :, 36:43]
        obs_faulty[:, :, 48:55] = obs[:, :, 43:50]
        obs_faulty[:, :, 56:63] = obs[:, :, 50:57]
        obs_faulty[:, :, 64:71] = obs[:, :, 57:64]
        obs_faulty[:, :, 72:79] = obs[:, :, 64:71]
        return obs_faulty       


    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        obs_t = batch["obs"][:, t]
        obs_faulty = self.get_new_obs_with_faults(obs_t, bs)
        inputs = []
        inputs.append(obs_faulty)  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"] + 9
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
