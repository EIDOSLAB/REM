#  Copyright (c) 2021 EIDOSLab. All rights reserved.
#  This file is part of the EIDOSearch library.
#  See the LICENSE file for licensing terms (BSD-style).

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from EIDOSearch.datasets.transforms import ImageNet

if __name__ == '__main__':
    model = models.resnet18(pretrained=True)
    model = model.eval()
    
    img = Image.open("../dog.png")
    input = ImageNet[1](Image.open('../dog.png')).unsqueeze(0).to(next(model.parameters()).device)
    
    logits = model(input)
    probs = F.softmax(logits, dim=1)
    probs5 = probs.topk(5)
    
    
    def get_pil_transform():
        transf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224)
        ])
        
        return transf
    
    
    def get_preprocess_transform():
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transf = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        
        return transf
    
    
    pill_transf = get_pil_transform()
    preprocess_transform = get_preprocess_transform()
    
    
    def batch_predict(images):
        model.eval()
        batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        batch = batch.to(device)
        
        logits = model(batch)
        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()
    
    
    test_pred = batch_predict([pill_transf(img)])
    test_pred.squeeze().argmax()
    
    from lime import lime_image
    
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(np.array(pill_transf(img)),
                                             batch_predict,  # classification function
                                             top_labels=5,
                                             hide_color=0,
                                             num_samples=1000)  # number of images that will be sent to classification function
    
    from skimage.segmentation import mark_boundaries
    
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5,
                                                hide_rest=False)
    img_boundry1 = mark_boundaries(temp / 255.0, mask)
    plt.imshow(img_boundry1)
    plt.show()
    plt.clf()
    
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10,
                                                hide_rest=False)
    img_boundry2 = mark_boundaries(temp / 255.0, mask)
    plt.imshow(img_boundry2)
    plt.show()
