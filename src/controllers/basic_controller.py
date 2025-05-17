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

        self.memory = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        self.agent.eval()
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def build_causal_mask(self, seq_len, mem_len, device):
        total_len = mem_len + seq_len
        # Allow attending to memory (mem_len), and to current and past in x
        mask = th.triu(th.ones(seq_len, total_len, device=device) * float('-inf'), diagonal=1)
        return mask 
    
    def forward(self, ep_batch, t, test_mode=False, t_end=None):
        agent_inputs = self._build_inputs(ep_batch, t, t_end=t_end)
        memory = self.memory
        avail_actions = ep_batch["avail_actions"]
        
        mem_len_now = 0 if memory is None else memory[0].size(1)
        mask = self.build_causal_mask(
            seq_len=agent_inputs.size(1), 
            mem_len=mem_len_now, device=agent_inputs.device)  # [1, mem_len + 1]
        mask = mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, total_len]
        mask = mask.expand(ep_batch.batch_size * self.n_agents, self.args.n_heads, -1, -1)
        agent_outs, hidden_states = self.agent(agent_inputs, memory=memory, attn_mask=mask)
        self.memory = self.agent.update_memory(memory, hidden_states)
        if t_end is not None:
            avail_actions = avail_actions[:, t:t_end]
            avail_actions = avail_actions.permute(0, 2, 1, 3)
            reshaped_avail_actions = avail_actions.reshape(
                ep_batch.batch_size * self.n_agents, *avail_actions.shape[2:])
        else:
            avail_actions = avail_actions[:, t]
            reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
            reshaped_avail_actions = reshaped_avail_actions.unsqueeze(1)
        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
        if t_end is not None:
            B, T, F = agent_outs.shape
            agent_outs_view = agent_outs.view(ep_batch.batch_size, T, self.n_agents, F)
        else:
            agent_outs_view = agent_outs.view(ep_batch.batch_size, self.n_agents, -1)
        return agent_outs_view

    def init_hidden(self, batch_size):
        #self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        self.memory = self.agent.init_memory(batch_size)
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

    def _build_inputs(self, batch, t, t_end=None):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        if t_end is None:
            bs = batch.batch_size
            inputs = []
            inputs.append(batch["obs"][:, t])  # b1av
            if self.args.obs_last_action:
                if t == 0:
                    inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
                else:
                    inputs.append(batch["actions_onehot"][:, t-1])
            if self.args.obs_agent_id:
                inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

            inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
            inputs = inputs.unsqueeze(1)  
            return inputs
        else:   
            bs = batch.batch_size
            all_inputs = []
            
            t_start = t
            for t_step in range(t_start, t_end):
                inputs = []
                inputs.append(batch["obs"][:, t_step])
                if self.args.obs_last_action:
                    if t_step == 0:
                        inputs.append(th.zeros_like(batch["actions_onehot"][:, t_step]))
                    else:
                        inputs.append(batch["actions_onehot"][:, t_step-1])
                if self.args.obs_agent_id:
                    inputs.append(
                        th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
                inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
                all_inputs.append(inputs)
            all_inputs = th.stack(all_inputs, dim=1)
            return all_inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
