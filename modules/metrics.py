from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor import Meteor
from pycocoevalcap.rouge import Rouge


def compute_scores(gts, res):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

    gts：字典，包含图片的标识符及其对应的 参考 答案。
    res：字典，包含图片的标识符及其 生成 的答案。
    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """

    # Set up scorers
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L")
    ]
    eval_res = {}
    # Compute score for each metric
    for scorer, method in scorers:
        try:# 对应BLEU，需要verbose=0
            score, scores = scorer.compute_score(gts, res, verbose=0)
        except TypeError:# 不需要verbose=0
            score, scores = scorer.compute_score(gts, res)
        if type(method) == list: # 对应BLEU的列表
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:# 其他方法
            eval_res[method] = score
    return eval_res
