import torch
import torch.nn.functional as F
from .pca import pca_lowrank

def sample_tail_normal_distribution(sample_hidden, num_samples=1, max_try=1, epsilon=2):
    gaussian_mean = sample_hidden.mean(dim=0)
    gaussian_std = sample_hidden.std(dim=0)
    l2_norm = torch.sqrt(torch.sum((sample_hidden-gaussian_mean)**2, dim=(1, 2)))
    threshold = l2_norm[l2_norm.argsort()[-epsilon]]
    batch_size = num_samples * 100
    cont = 0
    neg_samples = torch.tensor([])
    while neg_samples.shape[0] < num_samples and cont < max_try:
        cont += 1
        samples = torch.randn(batch_size, *gaussian_mean.shape).to(gaussian_std.device) * gaussian_std + gaussian_mean
        l2_norm = torch.sqrt(torch.sum((samples-gaussian_mean)**2, dim=(1, 2)))
        tail_samples = samples[(l2_norm > threshold)]
        if tail_samples.shape[0] > 0:
            neg_samples = torch.concat([neg_samples, tail_samples], dim=0)
    if neg_samples.shape[0] > 0:
        return neg_samples[:num_samples]
    else:
        return None

def orthogonality_loss(neg_sample, pos_sample, k=10):
    neg_sample_flat = neg_sample[:,-1]
    pos_sample_flat = pos_sample[:,-1]
    
    # cov_matrix = torch.mm(neg_sample_flat.T, neg_sample_flat)  
    # _, _, V = torch.linalg.svd(cov_matrix.float())  
    # V_k = V[:, :k]  
    # projection = torch.mm(pos_sample_flat, V_k)  
    # loss = torch.norm(projection, p='fro')

    neg_sample_flat = neg_sample_flat.to(torch.float32)
    U, S, V_k = pca_lowrank(neg_sample_flat, q=k) 
    projection = torch.mm(pos_sample_flat, V_k)  
    loss = torch.norm(projection, p='fro')  / pos_sample_flat.size(0)

    return loss

def contrastive_loss(hidden_states, labels, temperature=0.5):
    batch_size = hidden_states.size(0)
    normal_hidden_states = F.normalize(hidden_states, p=2, dim=1)
    similarity_matrix = torch.matmul(normal_hidden_states, normal_hidden_states.T) / temperature
    positive_mask = labels.view(-1).unsqueeze(0) == labels.view(-1).unsqueeze(1)
    self_mask = torch.eye(batch_size, device=hidden_states.device, dtype=torch.bool)
    positive_mask = positive_mask & ~self_mask
    log_probs = F.log_softmax(similarity_matrix, dim=1)
    loss = (-torch.sum(log_probs * positive_mask.float(), dim=-1) / (torch.sum(positive_mask.float(), dim=-1) + 1e-24))
    
    
    loss_pos = loss.mean()
    return loss_pos