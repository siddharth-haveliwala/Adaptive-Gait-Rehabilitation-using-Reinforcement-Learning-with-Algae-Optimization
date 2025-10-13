import torch
import shap
import numpy as np
import matplotlib.pyplot as plt

def explain_with_shap(model, data_loader, device):
    data, labels = next(iter(data_loader))
    data = data.to(device)
    layer = model.conv2
    explainer = shap.GradientExplainer((model, layer), data)
    shap_values = explainer.shap_values(data, nsamples=50)
    
    if len(shap_values) > 0 and len(data) > 0:
        class_idx = 0
        max_display = min(len(data), 10)
        shap_numpy = np.array(shap_values[class_idx])
        display_data = -data.cpu().numpy()
        shap.image_plot(shap_numpy[:max_display], display_data[:max_display])
    else:
        print("No SHAP values to display")
