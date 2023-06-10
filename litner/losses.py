import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class Focal loss implementation"""

    def __init__(self, gamma=2, weight=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, inputs, target):
        """
        :param inputs: torch.Tensor, shape=[N, C]
        :param target: torch.Tensor, shape=[N, ]
        """
        logpt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        return F.nll_loss(logpt, target, self.weight, ignore_index=self.ignore_index)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        """
        :param output: torch.Tensor, shape=[N, C]
        :param target: torch.Tensor, shape=[N, ]
        """
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * F.nll_loss(
            log_preds, target, reduction=self.reduction, ignore_index=self.ignore_index
        )


class MultilabelCategoricalCrossentropy(nn.Module):
    """多标签分类的交叉熵；
    说明：y_true和y_pred的shape一致，y_true的元素非0即1， 1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred不用加激活函数，尤其是不能加sigmoid或者softmax！预测阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解本文。
    参考：https://kexue.fm/archives/7359
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_pred, y_true):
        """
        :param y_true: torch.Tensor, [..., num_classes]
        :param y_pred: torch.Tensor: [..., num_classes]
        """
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        y_pred_neg = y_pred - y_true * 1e12

        y_pred_pos = torch.cat([y_pred_pos, torch.zeros_like(y_pred_pos[..., :1])], dim=-1)
        y_pred_neg = torch.cat([y_pred_neg, torch.zeros_like(y_pred_neg[..., :1])], dim=-1)

        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)

        return (pos_loss + neg_loss).mean()


class SparseMultilabelCategoricalCrossentropy(nn.Module):
    """稀疏版多标签分类的交叉熵；
       请保证y_pred的值域是全体实数，换言之一般情况下y_pred不用加激活函数，尤其是不能加sigmoid或者softmax，预测阶段则输出y_pred大于0的类；
       详情请看：https://kexue.fm/archives/7359，https://kexue.fm/archives/8888
    """

    def __init__(self, mask_zero=False, epsilon=1e-7, **kwargs):
        super().__init__(**kwargs)
        self.mask_zero = mask_zero
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        """
        :param y_true: shape=[..., num_positive]
        :param y_pred: shape=[..., num_classes]
        """
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred = torch.cat([y_pred, zeros], dim=-1)

        if self.mask_zero:
            infs = zeros + float('inf')
            y_pred = torch.cat([infs, y_pred[..., 1:]], dim=-1)

        y_pos_2 = torch.gather(y_pred, dim=-1, index=y_true)  # [..., num_positive]
        y_pos_1 = torch.cat([y_pos_2, zeros], dim=-1)  # [..., num_positive+1]

        if self.mask_zero:
            y_pred = torch.cat([-infs, y_pred[..., 1:]], dim=-1)
            y_pos_2 = torch.gather(y_pred, dim=-1, index=y_true)

        pos_loss = torch.logsumexp(-y_pos_1, dim=-1)
        all_loss = torch.logsumexp(y_pred, dim=-1)  # a
        aux_loss = torch.logsumexp(y_pos_2, dim=-1) - all_loss  # b-a
        aux_loss = torch.clamp(1 - torch.exp(aux_loss), self.epsilon, 1)  # 1-exp(b-a)
        neg_loss = all_loss + torch.log(aux_loss)  # a + log[1-exp(b-a)]

        return pos_loss + neg_loss


class SpanLoss(nn.Module):
    def __init__(self, loss_type='cross_entropy', reduction='mean'):
        super().__init__()
        assert loss_type in ['cross_entropy', 'focal_loss', 'label_smoothing_ce']
        loss_fcts = {
            'cross_entropy': nn.CrossEntropyLoss(reduction=reduction),
            'focal_loss': FocalLoss(),
            'label_smoothing_ce': LabelSmoothingCrossEntropy(reduction=reduction)
        }
        self.loss_fct = loss_fcts[loss_type]

    def forward(self, preds, target, masks):
        # assert if inp and target has both start and end values
        assert len(preds) == 2, "start and end logits should be present for spn losses calc"
        assert len(target) == 2, "start and end logits should be present for spn losses calc"
        assert masks is not None, "masks should be provided."

        active_loss = masks.view(-1) == 1
        start_logits, end_logits = preds
        start_positions, end_positions = target

        start_logits = start_logits.view(-1, start_logits.size(-1))
        end_logits = end_logits.view(-1, start_logits.size(-1))

        active_start_logits = start_logits[active_loss]
        active_end_logits = end_logits[active_loss]
        active_start_labels = start_positions.view(-1)[active_loss]
        active_end_labels = end_positions.view(-1)[active_loss]

        start_loss = self.loss_fct(active_start_logits, active_start_labels)
        end_loss = self.loss_fct(active_end_logits, active_end_labels)

        return (start_loss + end_loss) / 2


class SpanLossForMultiLabel(nn.Module):
    def __init__(self, name='Span Binary Cross Entropy Loss'):
        super().__init__()
        self.name = name
        self.loss_fct = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, preds, target, masks, nested=False):
        assert masks is not None, "masks should be provided."
        if not nested:
            return self.flated_forward(preds, target, masks)

        start_logits, end_logits, span_logits = preds
        start_labels, end_labels, span_labels = target
        start_, end_ = start_logits > 0, end_logits > 0

        bs, seqlen, num_labels = start_logits.shape
        span_candidate = torch.logical_or(
            (start_.unsqueeze(-2).expand(-1, -1, seqlen, -1) & end_.unsqueeze(-3).expand(-1, seqlen, -1, -1)),
            (start_labels.unsqueeze(-2).expand(-1, -1, seqlen, -1).bool() & end_labels.unsqueeze(
                -3).expand(-1, seqlen, -1, -1).bool())
        )

        masks = masks[:, :, None].expand(-1, -1, num_labels)
        start_loss = self.loss_fct(start_logits.view(-1), start_labels.view(-1).float())
        start_loss = (start_loss * masks.reshape(-1)).view(-1, num_labels).sum(-1).sum() / (masks.sum() / num_labels)

        end_loss = self.loss_fct(end_logits.view(-1), end_labels.view(-1).float())
        end_loss = (end_loss * masks.reshape(-1)).view(-1, num_labels).sum(-1).sum() / (masks.sum() / num_labels)

        span_masks = masks.bool().unsqueeze(2).expand(-1, -1, seqlen, -1) & masks.bool().unsqueeze(
            1).expand(-1, seqlen, -1, -1)
        span_masks = torch.triu(span_masks.permute(0, 3, 1, 2), 0).permute(
            0, 2, 3, 1) * span_candidate  # start should be less equal to end
        span_loss = self.loss_fct(span_logits.view(bs, -1), span_labels.view(bs, -1).float())
        span_loss = span_loss.reshape(-1, num_labels).sum(-1).sum() / (
                    span_masks.view(-1, num_labels).sum() / num_labels)

        return start_loss + end_loss + span_loss

    def flated_forward(self, preds, target, masks):
        active_loss = masks.view(-1) == 1
        start_logits, end_logits = preds
        start_labels, end_labels = target

        active_start_logits = start_logits.view(-1, start_logits.size(-1))[active_loss]
        active_end_logits = end_logits.view(-1, start_logits.size(-1))[active_loss]

        active_start_labels = start_labels.view(-1, start_labels.size(-1))[active_loss].float()
        active_end_labels = end_labels.view(-1, end_labels.size(-1))[active_loss].float()

        start_loss = self.loss_fct(active_start_logits, active_start_labels).sum(1).mean()
        end_loss = self.loss_fct(active_end_logits, active_end_labels).sum(1).mean()

        return start_loss + end_loss


class RDropLoss(nn.Module):
    """R-Drop的Loss实现，官方项目：https://github.com/dropreg/R-Drop
    :param alpha: float, 控制rdrop的loss的比例
    :param rank: str, 指示y_pred的排列方式, 支持['adjacent', 'updown']
    """

    def __init__(self, alpha=4, rank='adjacent'):
        super().__init__()
        self.alpha = alpha

        # 支持两种方式，一种是奇偶相邻排列，一种是上下排列
        assert rank in {'adjacent', 'updown'}, "rank kwarg only support 'adjacent' and 'updown' "
        self.rank = rank

        self.loss_sup = nn.CrossEntropyLoss()
        self.loss_rdrop = nn.KLDivLoss(reduction='none')

    def forward(self, *args):
        """支持两种方式: 一种是y_pred, y_true, 另一种是y_pred1, y_pred2, y_true
        y_pred: torch.Tensor, 第一种方式的样本预测值, shape=[btz*2, num_labels]
        y_true: torch.Tensor, 样本真实值, 第一种方式shape=[btz*2,], 第二种方式shape=[btz,]
        y_pred1: torch.Tensor, 第二种方式的样本预测值, shape=[btz, num_labels]
        y_pred2: torch.Tensor, 第二种方式的样本预测值, shape=[btz, num_labels]
        """
        assert len(args) in {2, 3}, 'RDropLoss only support 2 or 3 input args'
        # y_pred是1个Tensor
        if len(args) == 2:
            y_pred, y_true = args
            loss_sup = self.loss_sup(y_pred, y_true)  # 两个都算

            if self.rank == 'adjacent':
                y_pred1 = y_pred[1::2]
                y_pred2 = y_pred[::2]
            elif self.rank == 'updown':
                half_btz = y_true.shape[0] // 2
                y_pred1 = y_pred[:half_btz]
                y_pred2 = y_pred[half_btz:]

        # y_pred是两个tensor
        else:
            y_pred1, y_pred2, y_true = args
            loss_sup = self.loss_sup(y_pred1, y_true)

        loss_rdrop1 = self.loss_rdrop(F.log_softmax(y_pred1, dim=-1), F.softmax(y_pred2, dim=-1))
        loss_rdrop2 = self.loss_rdrop(F.log_softmax(y_pred2, dim=-1), F.softmax(y_pred1, dim=-1))

        return loss_sup + torch.mean(loss_rdrop1 + loss_rdrop2) / 4 * self.alpha
