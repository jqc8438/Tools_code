import torch
from torch import nn
from torch.nn import functional as F 

####################################################################
# Implement of Dice Loss
# first used in the VNet http://arxiv.org/abs/1606.04797
# dice = 2TP/(2TP + FP + FN)

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()
 
    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1e-8  # 为了防止除0的发生
        
        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)  
 
        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth) 
        #两个矩阵直接点乘，即逐点相乘相加，来表示分子部分
        # 图片的每个像素点直接相加代表|sum|和|target|

        score = 1 - score.sum() / num
        return score

####################################################################

####################################################################
# Implement of Focal Loss  
# first used in RetinaNet https://arxiv.org/abs/1708.02002
class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        """
        focal_loss损失函数, $-\alpha (1-\hat{y})^{\gamma} * CrossEntropyLoss(\hat{y}, y)$
        alpha: 类别权重. 当α是列表时, 为各类别权重, 当α为常数时, 类别权重为[α, 1-α, 1-α, ....]
        gamma: 难易样本调节参数.
        num_classes: 类别数量
        size_average: 损失计算方式, 默认取均值
        """
        super(focal_loss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes   # α可以以list方式输入, 用于对不同类别精细地赋予权重
            print(" --- Focal_loss alpha = {}, 将对每一类权重进行精细化赋值 --- ".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            print(" --- Focal_loss alpha = {} --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为[α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        preds: 预测类别. size:[B, C] or [B, S, C] B 批次, S长度, C类别数
        labels: 实际类别. size:[B] or [B, S] B批次, S长度
        """
        # assert preds.dim() == 2 and labels.dim()==1
        labels = labels.view(-1, 1) # [B * S, 1]
        preds = preds.view(-1, preds.size(-1)) # [B * S, C]
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1) # 先softmax, 然后取log
        preds_softmax = torch.exp(preds_logsoft)    # softmax

        print(labels)
        print(preds_softmax)
        preds_softmax = preds_softmax.gather(1, labels)   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels)
        alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

y_pred = torch.randn(3, 4, 5) # 3个样本, 5类
label = torch.tensor([[0, 3, 4, 1], [2, 1, 0, 0], [0, 0, 4, 1]]) # 0类特别多，应该给0类一个较大的$\alpha$权重
loss_fn = focal_loss(alpha = 0.75, num_classes=5)

loss = loss_fn(y_pred, label)

print(loss_fn(y_pred, label).item())
####################################################################