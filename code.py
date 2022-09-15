# Modification of gradients take place
# inside the forward function
def forward(self, x, labels):
    # x: Input tensor with shape [N, C, H, W]
    # labels: One hot ground truth with shape [N, C]
    logits = self.layers(x)

    def powerGradTransform(grad):
        alpha = 0.3
        # Recover the predicted probabilites
        # generated in the forward pass
        grad_temp = grad / grad.shape[0]
        pred = grad_temp + labels
        # Transform the predicted probabilities
        pred = pred**alpha
        pred /= pred.sum(-1, keepdim=True)
        # Modify the gradient
        modified_grad = pred - labels
        modified_grad = modified_grad * grad.shape[0]
        return modified_grad

    if self.training:
        logits.register_hook(powerGradTransform)
    return logits
