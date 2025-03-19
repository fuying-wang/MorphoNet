'''
Train sparse autoencoder for identify interpretable morphologies in WSIs.
'''
import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F


class SAE(nn.Module):
    '''
    Sparse AutoEncoder for extracting patterns from WSIs.
    '''
    def __init__(self, config, mode="sae", l1_coeff=1.):
        super(SAE, self).__init__()

        self.sparsity_lambda = l1_coeff
        self.sparsity_target = 0.05
        self.xavier_norm_init = True

        max_sample_size = 15000
        self.max_sample_size = max_sample_size
        expansion_factor = 8
        self.in_dim = config.in_dim
        self.n_learned_features = config.in_dim * expansion_factor
        """
        Map the original dimensions to a higher dimensional layer of features.
        Apply relu non-linearity to the linear transformation.
        """
        self.encoder = nn.Sequential(
            nn.Linear(self.in_dim, self.n_learned_features),
            nn.Sigmoid()
        )

        if self.xavier_norm_init:
            nn.init.xavier_uniform_(self.encoder[0].weight)
            nn.init.constant_(self.encoder[0].bias, 0)

        """
        Map back the features to the original input dimensions.
        Apply relu non-linearity to the linear transformation.
        """
        self.decoder = nn.Sequential(
            nn.Linear(self.n_learned_features, self.in_dim),
            nn.Tanh()
        )

        if self.xavier_norm_init:
            nn.init.xavier_uniform_(self.decoder[0].weight)
            nn.init.constant_(self.decoder[0].bias, 0)

    """
    We pass the original signal through the encoder. Then we pass
    that transformation to the decoder and return both results.
    """
    def forward(self, x, model_kwargs={}):
        assert x.size(2) == self.in_dim, f"Input size {x.size(2)} does not match the model input size {self.in_dim}"
        x = x.view(-1, x.size(2))
        # clip ensure the GPU memory is too large
        rand_indices = torch.randperm(x.size(0))[:self.max_sample_size]
        x = x[rand_indices, :]

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        rec_loss, sparsity_loss = self.loss_function(decoded, x, encoded)
        results_dict = {
            "loss": rec_loss + sparsity_loss,
            "encoded": encoded,
            "decoded": decoded,
        }
        log_dict = {
            "loss": results_dict["loss"].item(),
            "rec_loss": rec_loss.item(),
            "sparsity_loss": sparsity_loss.item(),
        }

        return results_dict, log_dict

    """
    This is the sparsity penalty we are going to use KL divergence
        - Encourage each hidden neuron to have an average activation (rho_hat) close to the target sparsity level (rho).

    Explanation:
        1. Compute the mean activation of each hidden neuron across the batch
            - We need the average activation to compare it with the target sparsity level. This tells us how active each neuron is on average.

        2. Retrieve the desired average activation level for the hidden neurons.
            - This is the sparsity level we want each neuron to achieve. 
            - Typically a small value like 0.05, meaning we want neurons to be active only 5% of the time.
        
        3.1. Set epsilon constant to prevent division by zero or taking the logarithm of zero.
        3.2. Use torch.clamp to ensure rho_hat stays within the range [epsilon, 1 - epsilon].
            - This is to avoid numerical issues like infinite or undefined values in subsequent calculations.

        4. Calculate the KL divergence between the target sparsity rho and the actual average activation rho_hat for each neuron.
            - rho * torch.log(rho / rho_hat) -> Measures the divergence when the neuron is active.
            - (1 - rho) * torch.log((1 - rho) / (1 - rho_hat)) -> Measures the divergence when the neuron is inactive.
            - The KL divergence quantifies how different the actual activation distribution is from the desired (target) distribution. 
            - A higher value means the neuron is deviating more from the target sparsity level.

        5. Aggregate the divergence values from all hidden neurons to compute a total penalty.
            - We want a single penalty value to add to the loss function, representing the overall sparsity deviation.

        6. Multiply the total KL divergence by a regularization parameter
            - sparsity_lambda controls the weight of the sparsity penalty in the loss function. 
            - A higher value means sparsity is more heavily enforced, while a lower value lessens its impact.
    """
    def sparsity_penalty(self, encoded, size_average=True):
        # the activation weight of each neuron
        rho_hat = torch.mean(encoded, dim=0)
        rho = self.sparsity_target
        epsilon = 1e-8
        rho_hat = torch.clamp(rho_hat, min=epsilon, max=1 - epsilon)
        kl_divergence = - rho * torch.log(rho_hat) - (1 - rho) * torch.log(1 - rho_hat) 
        if size_average:
            sparsity_penalty = torch.mean(kl_divergence)
        else:
            sparsity_penalty = torch.sum(kl_divergence)
        return self.sparsity_lambda * sparsity_penalty

    """
    Create a custom loss that combine mean squared error (MSE) loss 
    for reconstruction with the sparsity penalty.
    """
    def loss_function(self, x_hat, x, encoded):
        mse_loss = F.mse_loss(x_hat, x)
        sparsity_loss = self.sparsity_penalty(encoded)
        return mse_loss, sparsity_loss

