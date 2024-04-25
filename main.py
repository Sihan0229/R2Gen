import torch
import argparse # 解析命令行参数
import numpy as np
# 自定义
from modules.tokenizers import Tokenizer # 分词器模块
from modules.dataloaders import R2DataLoader # 数据加载器模块
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler # 构建优化器和学习调度器的函数
from modules.trainer import Trainer
from modules.loss import compute_loss
from models.r2gen import R2GenModel


def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='data/iu_xray/images/', help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='data/iu_xray/annotation.json', help='the path to the directory containing the data.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'], help='the dataset to be used.')
    # 文本最大序列长度
    parser.add_argument('--max_seq_length', type=int, default=60, help='the maximum sequence length of the reports.')
    # 词频截断阈值，用于过滤掉低频词
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    # 数据加载器使用的工作进程数
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')

    # Model settings (for visual extractor)
    # 可视化特征提取器的类型
    parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')
    # 是否加载预训练的可视化特征提取器
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True, help='whether to load the pretrained visual extractor')

    # Model settings (for Transformer)
    # Transformer 模型的隐藏层维度
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    # Transformer 模型中的前馈神经网络的隐藏层维度
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    # 图像补丁特征的维度
    parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features.')
    # 注意力头数
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    # 起始符号bos，结束符号eos，填充符号pad的索引
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    # 输出层的 dropout 率
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')
    # for Relational Memory
    # 关系记忆中的记忆槽数量
    parser.add_argument('--rm_num_slots', type=int, default=3, help='the number of memory slots.')
    # 关系记忆中的注意力头数
    parser.add_argument('--rm_num_heads', type=int, default=8, help='the numebr of heads in rm.')
    # 关系记忆中的隐藏层维度
    parser.add_argument('--rm_d_model', type=int, default=512, help='the dimension of rm.')

    # Sample related
    # 报告生成的采样方法
    parser.add_argument('--sample_method', type=str, default='beam_search', help='the sample methods to sample a report.')
    # 使用束搜索时的束大小
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/iu_xray', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/', help='the patch to save the results of experiments')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period.')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr_ve', type=float, default=5e-5, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=1e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=50, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.1, help='the gamma of the learning rate scheduler.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')

    args = parser.parse_args()
    return args
'''
--temperature：采样时的温度。
--sample_n：每个图像的采样数量。
--group_size：分组大小。
--output_logsoftmax：是否输出概率。
--decoding_constraint：解码约束。
--block_trigrams：是否使用块三元组。
--n_gpu：要使用的 GPU 数量。
--epochs：训练的时期数量。
--save_dir：保存模型的路径。
--record_dir：实验结果记录的路径。
--save_period：保存模型的周期。
--monitor_mode：监视指标的模式，可以是 'min' 或 'max'。
--monitor_metric：要监视的指标。
--early_stop：训练的早停参数。
--optim：优化器的类型。
--lr_ve：可视化特征提取器的学习率。
--lr_ed：其余参数的学习率。
--weight_decay：权重衰减。
--amsgrad：是否使用 AMSGrad 优化器。
--lr_scheduler：学习率调度器的类型。
--step_size：学习率调度器的步长。
--gamma：学习率调度器的 gamma 参数。
--seed：随机种子。
--resume：是否从现有检查点恢复训练。
'''

def main():
    # parse arguments
    args = parse_agrs() # 解析命令行参数，并将解析后的参数存储在 args 变量中。
    # fix random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # create tokenizer
    tokenizer = Tokenizer(args)

    # create data loader
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True) # shuffle=True 表示在每个 epoch 开始时对数据进行洗牌
    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

    # build model architecture
    model = R2GenModel(args, tokenizer)

    # get function handles of loss and metrics
    criterion = compute_loss
    metrics = compute_scores

    # build optimizer, learning rate scheduler
    optimizer = build_optimizer(args, model)# 优化器
    lr_scheduler = build_lr_scheduler(args, optimizer)# 学习率调度器

    # build trainer and start to train
    trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader)
    trainer.train()


if __name__ == '__main__':
    main()
