class MAMLMetaLearner:
    def __init__(self, model, optimizer, meta_lr=1e-3):
        self.model = model  # A shared DQN or PPO model
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=meta_lr)
        self.fast_lr = 1e-2  # Learning rate for task-specific fine-tuning

    def meta_train(self, tasks, steps=5):
        meta_loss = 0
        task_losses = []

        for task in tasks:
            # Clone the model for task-specific updates
            task_model = deepcopy(self.model)
            task_optimizer = optim.Adam(task_model.parameters(), lr=self.fast_lr)

            # Task-specific training
            task_loss = 0
            for _ in range(steps):
                loss = self.compute_task_loss(task_model, task)
                task_optimizer.zero_grad()
                loss.backward()
                task_optimizer.step()
                task_loss += loss.item()

            task_losses.append(task_loss / steps)

            # Meta-optimization
            loss = self.compute_task_loss(task_model, task)
            meta_loss += loss

        meta_loss /= len(tasks)
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        print(f"Meta-Loss: {meta_loss.item()}, Task Losses: {task_losses}")

    def compute_task_loss(self, model, task):
        # Sample task-specific data
        states, actions, rewards, next_states, dones = task.sample_data(BATCH_SIZE)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = model(states).gather(1, actions).squeeze()
        next_q_values = model(next_states).max(1)[0].detach()
        targets = rewards + GAMMA * (1 - dones) * next_q_values

        loss = nn.MSELoss()(q_values, targets)
        return loss
