""" 
Grad-CAM

[1]: Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization 
     Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra
     arXiv:1610.02391
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


class GradCAM(nn.Module):
    def __init__(self, model, layer="layer4"):
        super(GradCAM, self).__init__()

        self.model = model
        self.device = next(
            self.model.parameters()
        ).device  # Getting device on which model is present
        try:
            self.layer = getattr(self.model, layer)
        except AttributeError:
            print(
                f"Couldn't find the the {layer} in {model}, using the Last Layer {model[-1]}"
            )
            self.layer = model[-1]

        self.gradients = dict()
        self.activations = dict()

        self.layer.register_forward_hook(self.forward_hook)
        self.layer.register_full_backward_hook(self.backward_hook)

    # Reference for Hooks:
    # https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html#forward-and-backward-function-hooks
    # The forward hook will be executed when a forward call is executed.
    # The backward hook will be executed in the backward phase.
    def backward_hook(self, module, grad_input, grad_output):
        """
        When passed the function to register_backward_hook, returns the
        module(Sequential), Backprop Input(tuple) and Backprop Output(tuple).
        """
        self.gradients["value"] = grad_output[0]

    def forward_hook(self, module, input, output):
        """
        When passed the function to register_backward_hook, returns the
        module(Sequential), Input(a tuple of packed inputs) and
        Output(a Tensor. output.data is the Tensor we are interested).
        """
        self.activations["value"] = output

    def heat_map_size(self, *input_size):
        """
        Calculates the Output size at the layer, at which hook is registered and
        runs the model through random data to get the size at that layer.
        When the model is ran, the forward_hook is process, and the output is saved in
        activations['value'], which is used to get size of the map.
        """
        self.model(torch.zeros(1, 3, *input_size, device=self.device))
        return self.activations["value"].shape[2:]

    def forward(self, input_images, class_idx=None):
        """
        When the GradCAM object is called, it runs this forward function, where
        the input images are run through the model and during that the forward
        and backward hooks are registered, which are used to get Saliency Maps
        for Grad CAM.
        """
        i_b, i_c, i_h, i_w = input_images.size()

        output = self.model(input_images.to(self.device))
        if class_idx is None:
            # Use the Class which model predicted with highest
            # probability to generate GradCAM Map.
            score = output.max(dim=1)  # Getting Maximum Class in each Row
            lab = score.indices
            score = score.values
        else:
            # Gathers values along an axis specified by dim.
            # Target must be same dimension as input.
            # So class_idx.unsqueeze(1) make it (batch_size, 1)
            # output.size -> (batch_size, num_classes)
            score = torch.gather(output, 1, class_idx.unsqueeze(0))
            lab = class_idx

        self.model.zero_grad()
        score.sum().backward()
        gradients = self.gradients["value"]
        activations = self.activations["value"]
        g_b, g_c, g_h, g_w = gradients.size()

        # Pooling Gradients -> (b,c,1,1)
        weights = F.adaptive_avg_pool2d(gradients, 1)

        # Weighted Sum of Activations
        # keepdim=True to keep shape as (batch_size,1, g_h, g_w)
        # Interpolation require it to have Channels
        mask = (weights * activations).sum(1, keepdim=True)

        # Interpolating Map to HxW of input Image.
        mask = F.interpolate(
            mask, size=(i_h, i_w), mode="bilinear", align_corners=False
        )

        # Normalizing
        mask_min, mask_max = mask.min(), mask.max()
        mask = (mask - mask_min).div(mask_max - mask_min).data

        return mask, lab


def _visualize_cam(mask, img, hm_lay=0.5, img_lay=0.5, alpha=1.0):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (BS, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (BS, 3, H, W) and each pixel value is in range [0, 1]
    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """
    heatmap = (255 * mask.squeeze(1)).type(torch.uint8).cpu().numpy()
    heatmap = [cv2.applyColorMap(i, cv2.COLORMAP_JET) for i in heatmap]
    heatmap = torch.tensor(heatmap).permute(0, 3, 1, 2).float().div(255)

    # BGR2RGB
    # b, g, r = heatmap.split(1, dim=1)
    # plt.imshow(b.numpy().squeeze())
    # heatmap = torch.cat([r, g, b], dim=1) * alpha

    result = heatmap * hm_lay + img.cpu() * img_lay

    result_min, result_max = result.min(), result.max()
    result = (result - result_min).div(result_max - result_min).data
    # result = result.div(result.max()).data

    return heatmap, result
