from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F

def _gram_matrix(feature):
    batch_size, n_feature_maps, height, width = feature.size()
    new_feature = feature.view(batch_size * n_feature_maps, height * width)
    return torch.mm(new_feature, new_feature.t())

def _compute_loss(generated_features, content_features, style_features, alpha, beta):
    content_loss = 0
    style_loss = 0
    w_l = 1 / len(generated_features)
    for gf, cf, sf in zip(generated_features, content_features, style_features):
        content_loss += F.mse_loss(gf, cf)
        G = _gram_matrix(gf)
        A = _gram_matrix(sf)
        style_loss += w_l * F.mse_loss(G, A)
    return alpha * content_loss + beta * style_loss

def inference(
    *,
    model,
    content_image,
    style_features,
    lr,
    iterations=100,
    optim_caller=optim.AdamW,
    alpha=1,
    beta=1
):
    generated_image = content_image.clone().requires_grad_(True)
    optimizer = optim_caller([generated_image], lr=lr)
    min_losses = [float('inf')] * iterations

    with torch.no_grad():
        content_features = model(content_image)
        
    def closure(iter):
        optimizer.zero_grad()
        generated_features = model(generated_image)
        total_loss = _compute_loss(generated_features, content_features, style_features, alpha, beta)
        total_loss.backward()
        min_losses[iter] = min(min_losses[iter], total_loss.item())
        return total_loss
    
    for iter in tqdm(range(iterations), desc='The magic is happening ✨'):
        optimizer.step(lambda: closure(iter))
        print(f'Loss ({iter+1}):', min_losses[iter])
    
    return generated_image