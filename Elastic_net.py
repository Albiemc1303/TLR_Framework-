def elastic_net_regularization(self, model, l1_ratio=0.5, alpha=1e-4):
        l1_penalty = sum(torch.sum(torch.abs(param)) for param in model.parameters())
        l2_penalty = sum(torch.sum(param ** 2) for param in model.parameters())
        return alpha * (l1_ratio * l1_penalty + (1 - l1_ratio) * l2_penalty)                  

 # Elastic Net Regularization
        elastic_loss = elastic_net_regularization(self.q_network, l1_ratio, alpha)

        # Final Loss
        loss = mse_loss + elastic_loss
