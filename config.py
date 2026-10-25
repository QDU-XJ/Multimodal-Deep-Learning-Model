import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(
        description="医学图像多模态融合项目配置"
    )

    # ───── General Options ─────
    general = parser.add_argument_group('General Options')
    general.add_argument('--log-interval', type=int, default=200,help='每隔多少迭代打印一次日志 ')
    general.add_argument('--epochs', type=int, default=60,help='训练总轮数')
    general.add_argument('--seed', type=int, default=42,help='随机种子，保证可复现 ')
    general.add_argument('--config-file', type=str, default='',help='可选的配置文件路径，若指定则优先读取')
    general.add_argument('--save-dir', type=str, default=r"",
                         help='训练结果保存目录 ')
    general.add_argument('--cam-save-dir', type=str, default=r"", help='cam结果保存目录 ')

    # ───── Optimizer Options ─────
    optim = parser.add_argument_group('Optimizer Options')
    optim.add_argument('--optim', default='adamw', type=str,help="优化器类型")
    optim.add_argument('--lr', default=5e-5, type=float, help='初始学习率 ')
    optim.add_argument('--weight-decay', default=1e-4, type=float,help='权重衰减 ')
    optim.add_argument('--warmup-epochs', type=int, default=5, help='Warmup epochs')
    optim.add_argument('--lr-min', type=float, default=5e-6, help='最低学习率')

    # ───── Learning Rate Scheduler Options ─────
    sched = parser.add_argument_group('Learning Rate Scheduler')
    sched.add_argument('--scheduler', default='cos', type=str,help="学习率调度策略")
    sched.add_argument('--lr-decay', default=0.9, type=float,help='StepLR 衰减因子')

    # ───── Dataset Options ─────
    data = parser.add_argument_group('Dataset Options')
    data.add_argument('--dataset', type=str, default='yingxiang',help='数据集名称')
    data.add_argument('--train-file', type=str,
                      default=r'',
                      help='训练数据文件路径')
    data.add_argument('--val-file', type=str, default=r'',
                      help='验证数据文件路径')
    data.add_argument('--test-file', type=str, default=r'',
                      help='测试数据文件路径')

    data.add_argument('--yx-img-dir', type=str,default=r'',help='影像图像目录')
    data.add_argument('--yx-rad-dir', type=str,default=r'',help='影像放射组学特征目录')
    data.add_argument('--yx-img-extn', type=str, default='jpg',help='影像图像扩展名 (default: jpg)')
    data.add_argument('--yx-num-lesions', type=int, default=1,help='影像病灶数量 ')
    data.add_argument('--yx-lesion-size', type=int, default=128,help='影像病灶大小 ')

    data.add_argument('--bl-img-dir', type=str, default=r'',
                      help='Dataset location')
    data.add_argument('--bl-rad-dir', type=str, default=r'', help='Dataset location')
    data.add_argument('--bl-img-extn', type=str, default='jpg', help='Extension of WSIs. Default is tiff')
    data.add_argument('--bl-num-bags', type=int, default=6, help='Number of bags for running')
    data.add_argument('--bl-bag-size', type=int, default=512, help='Bag size.')
    data.add_argument('--bl-word-size', type=int, default=256, help='Word size.')

    data.add_argument('--split-file', type=str,default=r'',help='数据划分文件')
    data.add_argument('--batch-size', type=int, default=16,help='批量大小')
    data.add_argument('--data-workers', type=int, default=1,help='数据加载进程数 ')

    # ───── Criterion (Loss) Options ─────
    crit = parser.add_argument_group('Criterion Options')
    crit.add_argument('--loss-fn', default='YXLoss', type=str,help='损失函数名称 ')
    crit.add_argument('--w-ce', default='1.5', type=float, help='')
    crit.add_argument('--w-auc', default='1.4', type=float, help='')
    crit.add_argument('--w-surv', default='0.8', type=float, help='')
    crit.add_argument('--w-cindex', default='0.1', type=float, help='')
    crit.add_argument('--w-align', default='0.2', type=float, help='')
    crit.add_argument('--w-cmc', default='0.05', type=float, help='')
    crit.add_argument('--align-temperature', default='0.07', type=float, help='')
    crit.add_argument('--cmc-temperature', default='0.1', type=float, help='')
    crit.add_argument('--gamma-focal', default='2.0', type=float, help='')

    # ───── Model Options ─────
    model = parser.add_argument_group('Medical Imaging Model Details')
    model.add_argument('--model', default='blyx', type=str,help='整体模型名称 ')

    # —— 影像分支 (YingXiang) ——
    model.add_argument('--yx-model',    default='YXModel', type=str, help='影像模型类名')
    model.add_argument('--yx-pretrained', default='', type=str, help='影像预训练权重路径')
    model.add_argument('--yx-cnn-name', default='resnet18', type=str, help='影像CNN骨干')
    model.add_argument('--yx-cnn-pretrained', action='store_true', default=True, help='是否使用CNN预训练权重')
    model.add_argument('--yx-cnn-features', default=512, type=int, help='CNN 输出特征维度')
    model.add_argument('--yx-out-features', default=64, type=int,
                       help='影像分支融合后特征维度')
    model.add_argument('--yx-attn-heads', default=2, type=int, help='影像分支注意力头数')
    model.add_argument('--yx-vit-layers', default=2, type=int, help='影像ViT层数')
    model.add_argument('--yx-drop-path-rate', default=0.0, type=float, help='影像DropPath率')
    model.add_argument('--yx-dropout', default=0.2, type=float,    help='影像分支Dropout率')
    model.add_argument('--yx-attn-dropout', default=0.2, type=float, help='影像注意力Dropout率')
    model.add_argument('--yx-attn-fn', default='softmax', choices=['tanh','sigmoid','softmax'],
                       help='影像注意力函数')
    model.add_argument('--yx-num-way', default=2, type=int,     help='影像分支类别数')

    # —— 病理分支 (BingLing) ——
    model.add_argument('--bl-model', default="BLModel", type=str, help='Name of BingLiModel')
    model.add_argument('--bl-pretrained', default='', type=str)
    model.add_argument('--bl-cnn-name', default="resnet18", type=str, help='Name of backbone')
    model.add_argument('--bl-cnn-s', type=float, default=2.0, help='Factor by which channels will be scaled. Default is 2.0 for espnetv2')
    model.add_argument('--bl-cnn-pretrained', action='store_true', default=True)
    model.add_argument('--bl-cnn-weight', default=False, type=str, help='Pretrained model')
    model.add_argument('--bl-cnn-features', type=int, default=512,help='Number of cnn features extracted by the backbone')
    model.add_argument('--bl-out-features', type=int, default=64, help='Number of output features after merging bags and words')
    model.add_argument('--bl-attn-heads', default=2, type=int, help='Number of attention heads')
    model.add_argument('--bl-vit-layers', default=2, type=int, help='')
    model.add_argument('--bl-drop-path-rate', default=0.0, type=int, help='')
    model.add_argument('--bl-dropout', default=0.2, type=float, help='Dropout value')
    model.add_argument('--bl-max-bsz-cnn-gpu0', type=int, default=32, help='Max. batch size on GPU0')
    model.add_argument('--bl-attn-dropout', type=float, default=0.2, help='Proability to drop bag and word attention weights')
    model.add_argument('--bl-attn-fn', type=str, default='softmax', choices=['tanh', 'sigmoid', 'softmax'],help='Proability to drop bag and word attention weights')
    model.add_argument('--bl-num-way', type=int, default=2)

    model.add_argument('--blyx-model', default="BLYXModel", type=str, help='Name of BingLiYingXiangModel')
    model.add_argument('--blyx-out-features', type=int, default=64,help='Number of output features after merging bags and words')
    model.add_argument('--blyx-dropout', default=0.2, type=float, help='Dropout value')

    # 恢复训练 / 检查点
    model.add_argument('--resume', action='store_true', default=False,help='是否从检查点恢复训练')
    model.add_argument('--yx-checkpoint',   default='', type=str, help='影像模型检查点路径')
    model.add_argument('--bl-checkpoint', default="", type=str, help='Checkpoint for resuming')
    model.add_argument('--BEST-MODEL-PATH', default=r'',
                       type=str, help='Checkpoint for resuming')
    # 分类总类数
    model.add_argument('--n-classes', default=2, type=int,help='最终分类任务类别数 (default: 2)')
    # ───── Return ─────
    args = parser.parse_args()

    # 如果指定了 config-file，可在这里加载并覆盖参数
    if args.config_file:
        # TODO: 读取 JSON/YAML 并更新 args
        pass

    # 确保保存目录存在
    os.makedirs(args.save_dir, exist_ok=True)
    return args

if __name__ == "__main__":
    args = get_args()
    print(args)
