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
    iterations=35,
    alpha=1,
    beta=1
):
    generated_image = content_image.clone().requires_grad_(True)
    optimizer = optim.AdamW([generated_image], lr=lr)

    with torch.no_grad():
        content_features = model(content_image)
    
    for _ in tqdm(range(iterations), desc='The magic is happening ✨'):
        optimizer.zero_grad()

        generated_features = model(generated_image)
        total_loss = _compute_loss(generated_features, content_features, style_features, alpha, beta)

        total_loss.backward()
        optimizer.step()
    
    return generated_image