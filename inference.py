import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.transforms.functional import gaussian_blur

def save_mask(mask, title='mask'):
    plt.imshow(mask.cpu().numpy()[0], cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.savefig(f'{title}.png', bbox_inches='tight')
    plt.close()

def _gram_matrix(feature):
    batch_size, n_feature_maps, height, width = feature.size()
    new_feature = feature.view(batch_size * n_feature_maps, height * width)
    return torch.mm(new_feature, new_feature.t())

def _compute_loss(generated_features, content_features, style_features, resized_bg_masks, alpha, beta):
    content_loss = 0
    style_loss = 0
    w_l = 1 / len(generated_features)
    
    for i, (gf, cf, sf) in enumerate(zip(generated_features, content_features, style_features)):
        content_loss += F.mse_loss(gf, cf)
        
        if resized_bg_masks:
            blurred_bg_mask = gaussian_blur(resized_bg_masks[i], kernel_size=5)
            masked_gf = gf * blurred_bg_mask
            masked_sf = sf * blurred_bg_mask
            G = _gram_matrix(masked_gf)
            A = _gram_matrix(masked_sf)
        else:
            G = _gram_matrix(gf)
            A = _gram_matrix(sf)
        style_loss += w_l * F.mse_loss(G, A)
        
    return alpha * content_loss + beta * style_loss

def inference(
    *,
    model,
    segmentation_model,
    content_image,
    style_features,
    apply_to_background,
    lr,
    iterations=101,
    optim_caller=optim.AdamW,
    alpha=1,
    beta=1,
):
    generated_image = content_image.clone().requires_grad_(True)
    optimizer = optim_caller([generated_image], lr=lr)
    min_losses = [float('inf')] * iterations

    with torch.no_grad():
        content_features = model(content_image)

        resized_bg_masks = []
        background_ratio = None
        if apply_to_background:            
            segmentation_output = segmentation_model(content_image)['out']
            segmentation_mask = segmentation_output.argmax(dim=1)
            background_mask = (segmentation_mask == 0).float()
            foreground_mask = 1 - background_mask
            save_mask(background_mask, title='background-mask')
            
            background_pixel_count = background_mask.sum().item()
            total_pixel_count = segmentation_mask.numel()
            background_ratio = background_pixel_count / total_pixel_count
            print(f'Background Detected: {background_ratio * 100:.2f}%')

            for cf in content_features:
                _, _, h_i, w_i = cf.shape
                bg_mask = F.interpolate(background_mask.unsqueeze(1), size=(h_i, w_i), mode='bilinear', align_corners=False)
                resized_bg_masks.append(bg_mask)
        
    def closure(iter):
        optimizer.zero_grad()
        generated_features = model(generated_image)
        total_loss = _compute_loss(
            generated_features, content_features, style_features, resized_bg_masks, alpha, beta
        )
        total_loss.backward()
        min_losses[iter] = min(min_losses[iter], total_loss.item())
        return total_loss
    
    for iter in range(iterations):
        optimizer.step(lambda: closure(iter))

        if apply_to_background:
            with torch.no_grad():
                foreground_mask_resized = F.interpolate(foreground_mask.unsqueeze(1), size=generated_image.shape[2:], mode='nearest')
                generated_image.data = generated_image.data * (1 - foreground_mask_resized) + content_image.data * foreground_mask_resized

        if iter % 10 == 0: print(f'[{"Background" if apply_to_background else "Image"}] Loss ({iter}):', min_losses[iter])

    return generated_image, background_ratio
