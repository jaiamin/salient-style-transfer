import torch
import torch.optim as optim
import torch.nn.functional as F

def gram_matrix(feature):
    b, c, h, w = feature.size()
    feature = feature.view(b * c, h * w)
    return feature @ feature.t()

def compute_loss(generated, content, style, bg_masks, alpha, beta):
    content_loss = sum(F.mse_loss(gf, cf) for gf, cf in zip(generated, content))
    style_loss = sum(
        F.mse_loss(
            gram_matrix(gf * bg) if bg is not None else gram_matrix(gf),
            gram_matrix(sf * bg) if bg is not None else gram_matrix(sf),
        ) / len(generated)
        for gf, sf, bg in zip(generated, style, bg_masks or [None] * len(generated))
    )
    return alpha * content_loss, beta * style_loss, alpha * content_loss + beta * style_loss

def inference(
    *,
    model,
    sod_model,
    content_image,
    content_image_norm,
    style_features,
    apply_to_background,
    lr=1.5e-2,
    iterations=51,
    optim_caller=optim.AdamW,
    alpha=1,
    beta=1,
):
    generated_image = content_image.clone().requires_grad_(True)
    optimizer = optim_caller([generated_image], lr=lr)

    with torch.no_grad():
        content_features = model(content_image)
        bg_masks = None
        
        if apply_to_background:
            seg_output = torch.sigmoid(sod_model(content_image_norm)[0])
            bg_mask = (seg_output <= 0.7).float()
            bg_masks = [
                F.interpolate(bg_mask.unsqueeze(1), size=cf.shape[2:], mode='bilinear', align_corners=False)
                for cf in content_features
            ]
        
    def closure():
        optimizer.zero_grad()
        generated_features = model(generated_image)
        content_loss, style_loss, total_loss = compute_loss(
            generated_features, content_features, style_features, bg_masks, alpha, beta
        )
        total_loss.backward()
        return total_loss
    
    for _ in range(iterations):
        optimizer.step(closure)
        if apply_to_background:
            with torch.no_grad():
                fg_mask = F.interpolate(1 - bg_masks[0], size=generated_image.shape[2:], mode='nearest')
                generated_image.data.mul_(1 - fg_mask).add_(content_image.data * fg_mask)
                
    return generated_image
