class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=LR):
            self.state_dim = state_dim
            self.action_dim = action_dim
    
            # Actor and Critic Networks
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, action_dim), nn.Softmax(dim=-1)
            )
            self.critic = nn.Sequential(
                nn.Linear(state_dim, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, 1)
            )
    
            self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
            self.buffer = []
    
    def select_action(self, state):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = self.actor(state_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            return action.item(), dist.log_prob(action)
    
    def store_transition(self, transition):
            self.buffer.append(transition)
    
    def train(self, gamma=0.99, epsilon=0.2):
            states, actions, rewards, log_probs, dones = zip(*self.buffer)
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            log_probs = torch.stack(log_probs)
    
            # Compute returns
            returns = []
            discounted_sum = 0
            for reward, done in zip(reversed(rewards), reversed(dones)):
                discounted_sum = reward + gamma * discounted_sum * (1 - done)
                returns.insert(0, discounted_sum)
            returns = torch.FloatTensor(returns)
    
            # Calculate advantages
            values = self.critic(states).squeeze()
            advantages = returns - values.detach()
    
            # Update Actor (Clipped PPO Loss)
            new_probs = self.actor(states)
            dist = torch.distributions.Categorical(new_probs)
            new_log_probs = dist.log_prob(actions)
    
            ratio = (new_log_probs - log_probs).exp()
            clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
            actor_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    
            # Update Critic (Value Loss)
            critic_loss = nn.MSELoss()(values, returns)
    
            # Backpropagation
            loss = actor_loss + 0.5 * critic_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Clear buffer
            self.buffer = []
