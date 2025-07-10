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
        self.comm = None
        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs, _ = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions


    def get_updated_obs_with_comm(self, obs):
        # If using communication, append the communication vector to the observations
        obs_faulty = th.zeros(obs.shape[0], obs.shape[1], obs.shape[2] + 9 , device=obs.device)
        
        bs = obs.shape[0]
        n_agents = obs.shape[1]
        if self.comm is None:
            self.comm = th.zeros((bs, self.args.n_agents, 1), device=obs.device)

        index_map = {8: 15, 15: 23, 22: 31, 29: 39, 36: 47,43: 55, 50: 63, 57: 71, 64: 79}
        # index_map = {
        #     (-1, -1): 15,  # top-left
        #     (-1,  0): 23,  # top
        #     (-1,  1): 31,  # top-right
        #     ( 0, -1): 39,  # left
        #     ( 0,  0): 47,  # center (self)
        #     ( 0,  1): 55,  # right
        #     ( 1, -1): 63,  # bottom-left
        #     ( 1,  0): 71,  # bottom
        #     ( 1,  1): 79,  # bottom-right
        # }

        for source_idx, target_idx in index_map.items():
            # Step 1: Mask where obs[:, :, source_idx] == 1
            mask = obs[:, :, source_idx] == 1

            # Step 2: Get one-hot vector from next 4 positions
            one_hot = obs[:, :, source_idx+1:source_idx+5]  # shape: (10, 4, 4)

            # Step 3: Get agent index from one-hot
            agent_idx = one_hot.argmax(dim=-1)  # shape: (10, 4)

            # Step 4: Use advanced indexing to fetch comm[batch, agent_idx]
            # Expand dimensions to match for gather
            comm_gather = self.comm.squeeze(-1).unsqueeze(1).expand(-1, n_agents, -1)  # (10, 4, 4)
            selected_comm = th.gather(comm_gather, dim=2, index=agent_idx.unsqueeze(-1)).squeeze(-1)  # (10, 4)

            # Step 5: Write the selected_comm to obs_new at target_idx
            obs_faulty[:, :, target_idx] = selected_comm * mask  # write only where mask is 1


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

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs_full = self.agent(agent_inputs, self.hidden_states)
        if self.args.use_comm:
            agent_outs, self.hidden_states, self.comm = agent_outs_full
        else:
            agent_outs, self.hidden_states = agent_outs_full
        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), self.comm

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

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        batch_obs = batch["obs"][:, t]  # b1av
        if self.args.use_comm:
            # If using communication, append the communication vector
            batch_obs = self.get_updated_obs_with_comm(batch_obs)
        inputs.append(batch_obs)  # b1av
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
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        if self.args.use_comm:
            input_shape += 9

        return input_shape
