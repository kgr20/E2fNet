import torch
import torch.nn as nn
import torchmetrics

class ssim_mse_loss(nn.Module):
    def __init__(self, lambda_ssim=0.5, lambda_mes=0.5):
        super(ssim_mse_loss, self).__init__()
        self.ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0)
        self.mse = nn.MSELoss()
        self.lambda_ssim = lambda_ssim
        self.lambda_mse = lambda_mes

    def forward(self, img1, img2):
        ssim_loss = 1.0 - self.ssim(img1, img2)
        mse_loss = self.mse(img1, img2)
        return self.lambda_ssim*ssim_loss + self.lambda_mse*mse_loss

class GANLoss(nn.Module):
    def __init__(self, real_label=1.0, fake_label=0.0):
        """Initialize the GANLoss class.
        Inspired by GANLoss implementation from CycleGAN: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/
        
        Args:
            real_label: label for a real image
            fake_label: label of a fake image

        Note: DO NOT use sigmoid as the last layer of Discriminator. 
        Will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Args:
            prediction (tensor): tpyically the prediction from a discriminator
            target_is_real (bool): if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label

        return target_tensor.expand_as(prediction)

    def forward(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Args:
            prediction (tensor): tpyically the prediction output from a discriminator
            target_is_real (bool): if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        
        return loss