import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


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


class ContrastiveLoss(nn.Module):
    """对比损失：减小正例之间的距离，增大正例和反例之间的距离
    公式：labels * distance_matrix.pow(2) + (1-labels)*F.relu(margin-distance_matrix).pow(2)
    https://www.sbert.net/docs/package_reference/losses.html
    :param margin: float, 距离参数，distance>margin时候不参加梯度回传，默认为0.5
    :param size_average: bool, 是否对loss在样本维度上求均值，默认为True
    :param online: bool, 是否使用OnlineContrastiveLoss, 即仅计算困难样本的loss, 默认为False
    """

    def __init__(self, margin=0.5, size_average=True, online=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.size_average = size_average
        self.online = online

    def forward(self, distances, labels, pos_id=1, neg_id=0):
        """
        :param distances: torch.Tensor, 样本对之间的预测距离, shape=[btz]
        :param labels: torch.Tensor, 样本对之间的真实距离, shape=[btz]
        :param pos_id: int, 正样本的label
        :param neg_id: int, 负样本的label
        """
        if not self.online:
            losses = 0.5 * (
                labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(self.margin - distances).pow(2)
            )
            return losses.mean() if self.size_average else losses.sum()
        else:
            negs = distances[labels == neg_id]
            poss = distances[labels == pos_id]

            # select hard positive and hard negative pairs
            negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
            positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]

            positive_loss = positive_pairs.pow(2).sum()
            negative_loss = F.relu(self.margin - negative_pairs).pow(2).sum()

            return positive_loss + negative_loss


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
            loss_sup = (self.loss_sup(y_pred1, y_true) + self.loss_sup(y_pred2, y_true)) / 2

        loss_rdrop1 = self.loss_rdrop(F.log_softmax(y_pred1, dim=-1), F.softmax(y_pred2, dim=-1))
        loss_rdrop2 = self.loss_rdrop(F.log_softmax(y_pred2, dim=-1), F.softmax(y_pred1, dim=-1))

        return loss_sup + torch.mean(loss_rdrop1 + loss_rdrop2) / 4 * self.alpha


class SetCriterion(nn.Module):
    """ This class computes the losses for Set_RE.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class, subject position and object position)
    """

    def __init__(self, num_classes, loss_weight, na_coef, losses, matcher):
        """ Create the criterion.
        Parameters:
            num_classes: number of relation categories
            matcher: module able to compute a matching between targets and proposals
            loss_weight: dict containing as key the names of the losses and as values their relative weight.
            na_coef: list containg the relative classification weight applied to the NA category and positional classification weight applied to the [SEP]
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.matcher = HungarianMatcher(loss_weight, matcher)
        self.losses = losses
        rel_weight = torch.ones(self.num_classes + 1)
        rel_weight[-1] = na_coef
        self.register_buffer('rel_weight', rel_weight)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each losses' doc
        """
        # Retrieve the matching between the outputs of the last layers and the targets
        indices = self.matcher(outputs, targets)
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            if loss != "entity" or not self.empty_targets(targets):
                losses.update(self.get_loss(loss, outputs, targets, indices))
        losses = sum(losses[k] * self.loss_weight[k] for k in losses if k in self.loss_weight)

        return losses

    def relation_loss(self, outputs, targets, indices):
        """Classification losses (NLL)
        targets dicts must contain the key "relation" containing a tensor of dim [bsz]
        """
        src_logits = outputs['pred_rel_logits']  # [bsz, num_generated_triples, num_rel+1]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["relation"][i] for t, (_, i) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss = F.cross_entropy(src_logits.flatten(0, 1), target_classes.flatten(0, 1), weight=self.rel_weight)
        return {'relation': loss}

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty triples
        This is not really a losses, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_rel_logits = outputs['pred_rel_logits']
        device = pred_rel_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_rel_logits.argmax(-1) != pred_rel_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        return {'cardinality_error': card_err}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            'relation': self.relation_loss,
            'cardinality': self.loss_cardinality,
            'entity': self.entity_loss
        }
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def entity_loss(self, outputs, targets, indices):
        """Compute the losses related to the position of head entity or tail entity
        """
        idx = self._get_src_permutation_idx(indices)
        selected_pred_head_start = outputs["head_start_logits"][idx]
        selected_pred_head_end = outputs["head_end_logits"][idx]
        selected_pred_tail_start = outputs["tail_start_logits"][idx]
        selected_pred_tail_end = outputs["tail_end_logits"][idx]

        target_head_start = torch.cat([t["head_start_index"][i] for t, (_, i) in zip(targets, indices)])
        target_head_end = torch.cat([t["head_end_index"][i] for t, (_, i) in zip(targets, indices)])
        target_tail_start = torch.cat([t["tail_start_index"][i] for t, (_, i) in zip(targets, indices)])
        target_tail_end = torch.cat([t["tail_end_index"][i] for t, (_, i) in zip(targets, indices)])

        head_start_loss = F.cross_entropy(selected_pred_head_start, target_head_start)
        head_end_loss = F.cross_entropy(selected_pred_head_end, target_head_end)
        tail_start_loss = F.cross_entropy(selected_pred_tail_start, target_tail_start)
        tail_end_loss = F.cross_entropy(selected_pred_tail_end, target_tail_end)
        # print(losses)
        return {'head_entity': 1 / 2 * (head_start_loss + head_end_loss),
                "tail_entity": 1 / 2 * (tail_start_loss + tail_end_loss)}

    @staticmethod
    def empty_targets(targets):
        return all(len(target["relation"]) == 0 for target in targets)


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, loss_weight, matcher="avg"):
        super().__init__()
        self.cost_relation = loss_weight["relation"]
        self.cost_head = loss_weight["head_entity"]
        self.cost_tail = loss_weight["tail_entity"]
        self.matcher = matcher

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_rel_logits": Tensor of dim [batch_size, num_generated_triples, num_classes] with the classification logits
                 "{head, tail}_{start, end}_logits": Tensor of dim [batch_size, num_generated_triples, seq_len] with the predicted index logits
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_generated_triples, num_gold_triples)
        """
        with torch.no_grad():
            bsz, num_generated_triples = outputs["pred_rel_logits"].shape[:2]
            # We flatten to compute the cost matrices in a batch
            pred_rel = outputs["pred_rel_logits"].flatten(0, 1).softmax(
                -1)  # [bsz * num_generated_triples, num_classes]
            gold_rel = torch.cat([v["relation"] for v in targets])
            # after masking the pad token
            pred_head_start = outputs["head_start_logits"].flatten(0, 1).softmax(-1)
            pred_head_end = outputs["head_end_logits"].flatten(0, 1).softmax(-1)
            pred_tail_start = outputs["tail_start_logits"].flatten(0, 1).softmax(-1)
            pred_tail_end = outputs["tail_end_logits"].flatten(0, 1).softmax(-1)

            gold_head_start = torch.cat([v["head_start_index"] for v in targets])
            gold_head_end = torch.cat([v["head_end_index"] for v in targets])
            gold_tail_start = torch.cat([v["tail_start_index"] for v in targets])
            gold_tail_end = torch.cat([v["tail_end_index"] for v in targets])

            if self.matcher == "avg":
                cost = - self.cost_relation * pred_rel[:, gold_rel]
                cost -= self.cost_head * 1 / 2 * (pred_head_start[:, gold_head_start] + pred_head_end[:, gold_head_end])
                cost -= self.cost_tail * 1 / 2 * (pred_tail_start[:, gold_tail_start] + pred_tail_end[:, gold_tail_end])
            elif self.matcher == "min":
                cost = torch.cat([pred_head_start[:, gold_head_start].unsqueeze(1),
                                  pred_rel[:, gold_rel].unsqueeze(1),
                                  pred_head_end[:, gold_head_end].unsqueeze(1),
                                  pred_tail_start[:, gold_tail_start].unsqueeze(1),
                                  pred_tail_end[:, gold_tail_end].unsqueeze(1)], dim=1)
                cost = - torch.min(cost, dim=1)[0]
            else:
                raise ValueError("Wrong matcher")
            cost = cost.view(bsz, num_generated_triples, -1).cpu()
            num_gold_triples = [len(v["relation"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost.split(num_gold_triples, -1))]

            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
