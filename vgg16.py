import torch.nn as nn
import torchvision.models as models

class VGG_16(nn.Module):
    def __init__(self):
        super(VGG_16, self).__init__()
        self.model = models.vgg16(weights='DEFAULT').features[:30]
        
        for i, _ in enumerate(self.model):
            if i in [4, 9, 16, 23]:
                self.model[i] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
                
    def forward(self, x):
        features = []
        
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i in [0, 5, 10, 17, 24]:
                features.append(x)
        return features
    

if __name__ == '__main__':
    model = VGG_16()
    print(model)