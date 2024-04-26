import torch
import torch.nn as nn


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        # 目标序列和掩码序列截断到与模型输出相同的长度，以便与模型输出进行对齐。
        target = target[:, :input.size(1)] 
        mask = mask[:, :input.size(1)]
        # 负对数似然损失
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


def compute_loss(output, reports_ids, reports_masks):
    criterion = LanguageModelCriterion()
    # reports_ids[:, 1:] 表示目标序列中去掉起始标记 <bos>，reports_masks[:, 1:] 表示去掉起始标记后的掩码序列
    loss = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
    return loss