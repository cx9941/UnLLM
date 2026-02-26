import torch
import torch.linalg as LA


def orthogonal(X_correct,  eps = 1e-6):
    batch_size, N, D = X_correct.shape
    I = torch.eye(D).unsqueeze(0).repeat(batch_size, 1, 1).to(X_correct.dtype) 

    
    XtX_inv = torch.matmul(X_correct, X_correct.transpose(1, 2))

    # torch.save(X_correct, 'X_correct.pt')
    # torch.save(XtX_inv_reg, 'XtX_inv_reg.pt')

    
    XtX_inv_reg = XtX_inv + torch.eye(N, device=X_correct.device, dtype=X_correct.dtype).unsqueeze(0) * eps

    XtX_inv = LA.pinv(XtX_inv_reg.to(torch.float32)).to(XtX_inv.dtype)
    

    P_proj = I - torch.matmul(X_correct.transpose(1, 2), torch.matmul(XtX_inv, X_correct))
    return P_proj


def pca_first_component(X):
    X_centered = X - X.mean(dim=1, keepdim=True)  
    U, S, Vh = torch.svd(X_centered.to(torch.float32))  
    return Vh[:, :, 0].to(X_centered.dtype)  


def get_vector(X_correct, X_pos, X_neg, lm_head_weight, trend='pca', orthogonal_type='X_correct'):
    
    if orthogonal_type == 'X_correct':
        P_proj = orthogonal(X_correct)
    elif orthogonal_type == 'lm_weight':
        P_proj = orthogonal(lm_head_weight)
    else:
        assert False

    
    if trend == 'pca':
        T_pos = pca_first_component(X_pos)  
        T_neg = pca_first_component(X_neg)  
    elif trend == 'mean':
        T_pos = X_pos.mean(dim=1)  
        T_neg = X_neg.mean(dim=1)  
    else:
        raise ValueError("Unsupported trend type")

    
    P_proj = P_proj.to(X_correct.dtype)
    T_pos = T_pos.to(X_correct.dtype)
    T_neg = T_neg.to(X_correct.dtype)
    return P_proj, T_pos, T_neg

def interfere(P_proj, T_pos, T_neg, eta_1):
    delta_W_0 = torch.matmul(P_proj, (eta_1 * T_pos.unsqueeze(2) - eta_1 * T_neg.unsqueeze(2))).squeeze(2)
    return delta_W_0


if __name__ == '__main__':
    batch_size = 10
    N = 5
    Dimension = 3

    X_correct = torch.randn(batch_size, N, Dimension, dtype=torch.bfloat16)
    X_pos = torch.randn(batch_size, N, Dimension, dtype=torch.bfloat16)
    X_neg = torch.randn(batch_size, N, Dimension, dtype=torch.bfloat16)

    P_proj, T_pos, T_neg = get_vector(X_correct, X_pos, X_neg, trend='pca')

    
    delta_W_0 = interfere(P_proj, T_pos, T_neg, eta_1=0.01, eta_2=0.01)
    print(delta_W_0.shape)  # 输出的形状为 (batch_size, Dimension)