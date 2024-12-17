class ICM(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ICM, self).__init__()
        # Forward Model: Predict next state
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim)
        )
        # Inverse Model: Predict action from state transition
        self.inverse_model = nn.Sequential(
            nn.Linear(2 * state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state, next_state, action):
        # Predict action (inverse model)
        action_pred = self.inverse_model(torch.cat([state, next_state], dim=1))
        # Predict next state (forward model)
        state_action = torch.cat([state, action], dim=1)
        next_state_pred = self.forward_model(state_action)
        return action_pred, next_state_pred    

class ICM_Agent(DQNAgent):
    def __init__(self, state_dim, action_dim, icm=True, **kwargs):
        super().__init__(state_dim, action_dim, **kwargs)
        self.icm = ICM(state_dim, action_dim) if icm else None
        self.icm_optimizer = optim.Adam(self.icm.parameters(), lr=LR) if icm else None

    def compute_intrinsic_reward(self, state, next_state, action):
        if self.icm is None:
            return 0.0
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        action = torch.FloatTensor(np.eye(self.action_dim)[action]).unsqueeze(0)

        with torch.no_grad():
            _, next_state_pred = self.icm(state, next_state, action)
        intrinsic_reward = torch.norm(next_state - next_state_pred, p=2).item()
        return intrinsic_reward

    def train_with_icm(self, beta=0.2):
        if len(self.buffer) < BATCH_SIZE:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)
        intrinsic_rewards = [self.compute_intrinsic_reward(s, ns, a) for s, ns, a in zip(states, next_states, actions)]
        total_rewards = rewards + beta * np.array(intrinsic_rewards)

        # Standard DQN Training
        self.train_dqn(states, actions, total_rewards, next_states, dones)

        # Train ICM (Forward Model Loss)
        if self.icm:
            states = torch.FloatTensor(states)
            next_states = torch.FloatTensor(next_states)
            actions_onehot = torch.FloatTensor(np.eye(self.action_dim)[actions])
            _, next_state_pred = self.icm(states, next_states, actions_onehot)
            forward_loss = nn.MSELoss()(next_state_pred, next_states)

            self.icm_optimizer.zero_grad()
            forward_loss.backward()
            self.icm_optimizer.step()           
