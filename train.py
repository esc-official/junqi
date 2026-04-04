"""
军棋 AI 训练入口。
使用方法: python train.py [--config configs/default.yaml]
"""
import argparse
import yaml
import os
import sys

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from junqi.training.trainer import Trainer


DEFAULT_CONFIG = {
    # 设备
    'device': 'cuda',  # 'cuda' or 'cpu'

    # 环境
    'max_turns': 500,
    'reward_config': {
        'alpha_piece': 0.3,
        'beta_info': 0.01,
        'gamma_discount': 0.99,
        'pbrs_weight': 0.1,
        'terminal_reward': 1.0,
    },

    # 网络
    'network': {
        'hidden_dim': 128,
        'num_res_blocks': 12,
    },

    # PPO
    'ppo': {
        'lr': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_eps': 0.2,
        'value_coeff': 0.5,
        'entropy_coeff': 0.01,
        'belief_coeff': 0.1,
        'max_grad_norm': 0.5,
        'num_epochs': 4,
        'mini_batch_size': 64,
    },

    # 自博弈
    'self_play': {
        'pool_size': 20,
    },

    # 训练
    'episodes_per_update': 16,
    'snapshot_interval': 100,
    'checkpoint_interval': 1000,
    'total_episodes': 100_000,
    'checkpoint_dir': './checkpoints',
}


def main():
    parser = argparse.ArgumentParser(description='军棋 AI RL 训练')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径 (YAML)')
    parser.add_argument('--device', type=str, default=None,
                        help='训练设备 (cuda/cpu)')
    parser.add_argument('--episodes', type=int, default=None,
                        help='总训练 episode 数')
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复训练')
    args = parser.parse_args()

    # 加载配置
    config = dict(DEFAULT_CONFIG)
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            user_config = yaml.safe_load(f)
        if user_config:
            _deep_update(config, user_config)

    # 命令行覆盖
    if args.device:
        config['device'] = args.device
    if args.episodes:
        config['total_episodes'] = args.episodes

    print("=" * 60)
    print("  军棋 AI 强化学习训练系统")
    print("=" * 60)
    print(f"\n配置:")
    _print_config(config)

    # 创建训练器
    trainer = Trainer(config)

    # 从检查点恢复
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # 开始训练
    trainer.train()


def _deep_update(base: dict, override: dict):
    """递归合并字典"""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def _print_config(config: dict, indent: int = 2):
    """打印配置"""
    for key, value in config.items():
        if isinstance(value, dict):
            print(f"{' ' * indent}{key}:")
            _print_config(value, indent + 2)
        else:
            print(f"{' ' * indent}{key}: {value}")


if __name__ == '__main__':
    main()
