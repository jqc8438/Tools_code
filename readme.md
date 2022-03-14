> Implement of some tools in deep learning and computer vision.

## Loss Function 

**1. Dice Loss**
first used in the VNet (http://arxiv.org/abs/1606.04797)
dice = 2TP/(2TP + FP + FN)

Mostly used in medical image segmentation, when the foreground target is small and the positive and negative samples are unbalanced, the training is more inclined to tap the foreground region. And CE loss will handle the positive and negative samples fairly, and when the positive targets are too small, it will be swamped by more negative samples.

**2. Focal Loss**

![](http://latex.codecogs.com/svg.latex?\mathrm{FL}\left(p_{\mathrm{t}}\right)=-\alpha_{\mathrm{t}}\left(1-p_{\mathrm{t}}\right)^{\gamma} \log \left(p_{\mathrm{t}}\right))

first used in RetinaNet (https://arxiv.org/abs/1708.02002)

γ: focusing parameter, allows the model to focus more on the hard-to-classify samples during training by reducing the weights of the easy-to-classify samples.

a: control the weights to different categories.

in the RetinaNet, The best results were obtained in the authors' experiments with γ = 2 and a = 0.25.