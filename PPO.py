
from torch.distributions import Categorical
from models import *

################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]



class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim,netwotk_type, device):
        super(ActorCritic, self).__init__()


        if netwotk_type =="RNN_LSTM_onlyLastHidden":
            self.hidden_size = 256
            self.num_layers = 2
            self.actor = nn.Sequential(
                RNN_LSTM_onlyLastHidden(state_dim, self.hidden_size, self.num_layers, action_dim, device).to(device),
                nn.Softmax(dim=-1)
            )

            self.critic = RNN_LSTM_onlyLastHidden(state_dim, self.hidden_size, self.num_layers, 1, device).to(device)

        if netwotk_type == "MambaModel":
            self.hidden_size = 64
            self.num_layers = 2
            self.actor = nn.Sequential(MambaModel(state_dim, self.hidden_size, self.num_layers, action_dim, device).to(device),
                                       nn.Softmax(dim=-1))
            self.critic = MambaModel(state_dim, self.hidden_size, self.num_layers, 1, device).to(device)


    def forward(self):
        raise NotImplementedError

    def act(self, state):

        action_probs = self.actor(state) #outputs a probability distribution over the actions
        dist = Categorical(action_probs)

        action = dist.sample() #sample an action according to the probabilities -->stochasticity
        action_logprob = dist.log_prob(action) #log-probability of the sampled action--> for the ratio -->restricts large policy updates
        state_val = self.critic(state) # a baseline

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):

        action_probs = self.actor(state)
        # print("action_probs", action_probs.shape)

        dist = Categorical(action_probs)

        # print("dist", dist)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim,  lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                  device, netwotk_type , writer):



        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim,netwotk_type, device).to(device)


        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim,netwotk_type, device).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.writer = writer



    def select_action(self, state):

        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob, state_val = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.item()

    def update(self, n_episode):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                #the returns are computed for each episode independentl
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards - stabilize the learning process
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0), dim = 1).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0), dim = 1).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0), dim = 1).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0), dim =1).detach().to(self.device)

        # print("sizes ", old_states.shape, old_actions.shape, old_logprobs.shape, old_state_values.shape)
        # calculate advantages - how much better (or worse) the taken action was compared to the baseline provided by the critic
        advantages = rewards.detach() - old_state_values.detach()


        # Optimize policy for K epochs
        for i in range(self.K_epochs):
            # Evaluating old actions and values -for prob ratio
            # print("in k loop old_states, old_actions epoch: ", i, old_states.shape, old_actions.shape)
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values, dim =1)

            # Finding the ratio (pi_theta / pi_theta__old) -how much the new policy deviates from the old one for each action taken
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            policy_loss = -torch.min(surr1, surr2) #It improves the policy by increasing the likelihood of actions with positive advantages
            value_loss = 0.5 * self.MseLoss(state_values, rewards) #t trains the critic to accurately predict the returns
            entropy_loss = -0.01 * dist_entropy

            loss = policy_loss + value_loss + entropy_loss

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
        # Log loss onto Tensorboard
        self.writer.add_scalar("PPO Total Loss", loss.mean().item(), n_episode)
        self.writer.add_scalar("Loss/Policy", policy_loss.mean().item(), n_episode)
        self.writer.add_scalar("Loss/Value", value_loss.item(), n_episode)
        self.writer.add_scalar("Entropy", entropy_loss.mean().item(), n_episode)

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

