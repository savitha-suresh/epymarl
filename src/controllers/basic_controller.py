from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
from modules.masks.masked_net import MaskNet


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
        agent_outputs  = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions
    
    def mask_forward(self, batch):
        # batch_obs [10, 500, 4, 71])

        bs = self.args.batch_size
        
        n_agents = self.n_agents
        batch_obs = batch["obs"][:, :-1]
        seq_len = batch_obs.shape[1]
        # Start with observation: [bs, seq_len, n_agents, obs_dim]
        inputs = [batch_obs]  # b s a v

        # Add last action if needed
        if self.args.obs_last_action:
            # Create shifted actions_onehot for last actions
            
            prev_actions = batch["actions_onehot"][:, :-1].clone()
            
            # At t=0, prev action = 0
            inputs.append(prev_actions)

        # Add agent ID if needed
        if self.args.obs_agent_id:
            agent_ids = th.eye(n_agents, device=batch.device).unsqueeze(0).unsqueeze(0)  # [1,1,n_agents,n_agents]
            agent_ids = agent_ids.expand(bs, seq_len-1, -1, -1)  # [bs, seq_len, n_agents, n_agents]
            inputs.append(agent_ids)

        # Concatenate on the last dimension
        # Result: [bs, seq_len, n_agents, input_dim]
        inputs = th.cat(inputs, dim=-1)
        inputs = inputs.permute(0, 2, 1, 3)
        inputs = inputs.flatten(0, 1) 
        mask_out = self.mask_net(inputs)
        return mask_out.view(self.args.batch_size, -1, self.n_agents)

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
    
    def mask_parameters(self):
        return self.mask_net.parameters()

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
        device = "cuda" if self.args.use_cuda else "cpu"
        self.mask_net = MaskNet(self.args, input_shape, self.args.hidden_dim).to(device)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
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
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
