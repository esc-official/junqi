"""
训练编排器。
管理完整的训练流程：自博弈数据收集 → PPO 更新 → 对手池维护 → 日志记录。
"""
import os
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from ..env.junqi_env import JunqiEnv
from ..env.game import GameResult
from ..env.pieces import NUM_PIECE_TYPES
from ..env.board import NUM_POSITIONS, ROWS, COLS, row_col
from ..model.network import JunqiNetwork
from ..belief.belief_state import BeliefState, ALL_TYPES, TYPE_TO_IDX
from .ppo import PPOAgent, RolloutBuffer
from .self_play import SelfPlayManager


class Trainer:
    """
    军棋 AI 训练器。
    
    训练流程:
    1. 使用当前模型 vs 对手池中的对手进行自博弈
    2. 收集双方的经验数据
    3. 使用 PPO 更新当前模型
    4. 定期将当前模型加入对手池
    5. 记录训练统计
    """

    def __init__(self, config: dict):
        self.config = config

        # 设备
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Trainer] Using device: {self.device}")

        # 环境
        self.env = JunqiEnv(
            max_turns=config.get('max_turns', 500),
            reward_config=config.get('reward_config', {}),
        )

        # 网络
        net_config = config.get('network', {})
        self.network = JunqiNetwork(
            hidden_dim=net_config.get('hidden_dim', 128),
            num_res_blocks=net_config.get('num_res_blocks', 12),
        )
        print(f"[Trainer] Network parameters: {self.network.count_parameters():,}")

        # PPO
        ppo_config = config.get('ppo', {})
        self.agent = PPOAgent(
            network=self.network,
            lr=ppo_config.get('lr', 3e-4),
            gamma=ppo_config.get('gamma', 0.99),
            gae_lambda=ppo_config.get('gae_lambda', 0.95),
            clip_eps=ppo_config.get('clip_eps', 0.2),
            value_coeff=ppo_config.get('value_coeff', 0.5),
            entropy_coeff=ppo_config.get('entropy_coeff', 0.01),
            belief_coeff=ppo_config.get('belief_coeff', 0.1),
            max_grad_norm=ppo_config.get('max_grad_norm', 0.5),
            num_epochs=ppo_config.get('num_epochs', 4),
            mini_batch_size=ppo_config.get('mini_batch_size', 64),
            device=self.device,
        )

        # 自博弈
        sp_config = config.get('self_play', {})
        self.self_play = SelfPlayManager(
            network_class=JunqiNetwork,
            network_kwargs={
                'hidden_dim': net_config.get('hidden_dim', 128),
                'num_res_blocks': net_config.get('num_res_blocks', 12),
            },
            pool_size=sp_config.get('pool_size', 20),
            device=self.device,
            save_dir=config.get('checkpoint_dir', './checkpoints/opponents'),
        )

        # 训练参数
        self.episodes_per_update = config.get('episodes_per_update', 16)
        self.snapshot_interval = config.get('snapshot_interval', 100)
        self.checkpoint_interval = config.get('checkpoint_interval', 1000)
        self.total_episodes = config.get('total_episodes', 100_000)
        self.checkpoint_dir = config.get('checkpoint_dir', './checkpoints')

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # 统计
        self.episode_count = 0
        self.update_count = 0
        self.current_elo = 1000.0
        self.stats_history: List[Dict] = []

    def train(self):
        """主训练循环"""
        print(f"\n{'='*60}")
        print(f"  军棋 AI 训练开始")
        print(f"  总 Episodes: {self.total_episodes:,}")
        print(f"  每次更新 Episodes: {self.episodes_per_update}")
        print(f"  设备: {self.device}")
        print(f"{'='*60}\n")

        start_time = time.time()

        while self.episode_count < self.total_episodes:
            # 1. 收集经验
            buffers, episode_stats = self._collect_episodes(self.episodes_per_update)

            # 2. PPO 更新
            train_stats = self.agent.update(buffers)

            self.update_count += 1

            # 3. 定期添加到对手池
            if self.episode_count % self.snapshot_interval < self.episodes_per_update:
                self.self_play.add_snapshot(self.network, self.current_elo)

            # 4. 定期保存检查点
            if self.episode_count % self.checkpoint_interval < self.episodes_per_update:
                self._save_checkpoint()

            # 5. 记录统计
            elapsed = time.time() - start_time
            stats = {
                'episode': self.episode_count,
                'update': self.update_count,
                'elo': self.current_elo,
                'elapsed_min': elapsed / 60,
                **episode_stats,
                **{f'train/{k}': v for k, v in train_stats.items()},
            }
            self.stats_history.append(stats)

            # 打印日志
            if self.update_count % 10 == 0:
                self._print_stats(stats)

        print(f"\n训练完成! 总用时: {(time.time()-start_time)/3600:.2f} 小时")
        self._save_checkpoint(final=True)

    def _collect_episodes(
        self,
        num_episodes: int,
    ) -> Tuple[List[RolloutBuffer], Dict]:
        """
        收集自博弈经验。
        
        Returns:
            (buffers, stats)
        """
        all_buffers = []
        total_rewards = {0: [], 1: []}
        results = {r.name: 0 for r in GameResult}
        game_lengths = []

        for _ in range(num_episodes):
            # 选择对手
            opp_type, opponent = self.self_play.select_opponent('mixed')

            # 对弈
            buf_p0, buf_p1, result, length = self._play_one_game(
                opp_type, opponent
            )

            all_buffers.append(buf_p0)
            total_rewards[0].append(sum(buf_p0.rewards))
            total_rewards[1].append(sum(buf_p1.rewards))
            results[result.name] += 1
            game_lengths.append(length)

            self.episode_count += 1

        # 汇总统计
        stats = {
            'avg_reward_p0': np.mean(total_rewards[0]) if total_rewards[0] else 0,
            'avg_reward_p1': np.mean(total_rewards[1]) if total_rewards[1] else 0,
            'avg_game_length': np.mean(game_lengths) if game_lengths else 0,
            'win_rate': results.get('PLAYER0_WIN', 0) / max(num_episodes, 1),
            'loss_rate': results.get('PLAYER1_WIN', 0) / max(num_episodes, 1),
            'draw_rate': results.get('DRAW', 0) / max(num_episodes, 1),
        }

        return all_buffers, stats

    def _play_one_game(
        self,
        opp_type: str,
        opponent: object,
    ) -> Tuple[RolloutBuffer, RolloutBuffer, GameResult, int]:
        """
        进行一局完整的自博弈。
        
        Player 0: 当前训练中的模型
        Player 1: 对手 (网络或规则AI)
        
        Returns:
            (buffer_p0, buffer_p1, result, game_length)
        """
        # 随机选择布阵模板
        t0 = random.randint(0, 49)
        t1 = random.randint(0, 49)

        obs, info = self.env.reset(
            template_idx_p0=t0,
            template_idx_p1=t1,
            seed=None,
        )

        buf_p0 = RolloutBuffer()
        buf_p1 = RolloutBuffer()

        done = False

        while not done:
            player = self.env.get_current_player()
            action_mask = self.env.get_action_mask(player)

            # 获取对方棋子真实类型标签（辅助任务 label）
            belief_label = self._get_belief_label(player)

            if player == 0:
                # 当前模型选择动作
                current_obs = self.env._get_obs(player)
                action, log_prob, value = self.agent.select_action(
                    current_obs, action_mask
                )

                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                buf_p0.add(
                    state=current_obs,
                    action=action,
                    action_mask=action_mask,
                    log_prob=log_prob,
                    reward=reward,
                    value=value,
                    done=done,
                    belief_label=belief_label,
                )
            else:
                # 对手选择动作
                if opp_type == 'network':
                    current_obs = self.env._get_obs(player)
                    state_t = torch.FloatTensor(current_obs).unsqueeze(0).to(self.device)
                    mask_t = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        action, log_prob, _, value = opponent.get_action_and_value(
                            state_t, mask_t
                        )
                    action = action.item()
                    log_prob = log_prob.item()
                    value = value.item()
                else:
                    # 规则 AI
                    action = opponent.select_action(self.env.game, action_mask)
                    log_prob = 0.0
                    value = 0.0
                    current_obs = np.zeros(self.env.observation_shape, dtype=np.float32)

                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                buf_p1.add(
                    state=current_obs,
                    action=action,
                    action_mask=action_mask,
                    log_prob=log_prob,
                    reward=-reward,  # 对手视角的奖励取反
                    value=value,
                    done=done,
                )

        result = self.env.game.result
        length = self.env.game.turn_count

        return buf_p0, buf_p1, result, length

    def _get_belief_label(self, player: int) -> np.ndarray:
        """
        获取信念预测的真实标签。
        
        在训练时，我们有上帝视角，可以知道对方棋子的真实类型。
        标签格式: (12, 5) — 每个位置上对方棋子的类型 index，无棋子处为 -1。
        """
        opponent = 1 - player
        label = np.full((ROWS, COLS), -1, dtype=np.int64)

        for pid, piece in self.env.game.pieces[opponent].items():
            if piece.alive and pid in self.env.game.position_of[opponent]:
                p = self.env.game.position_of[opponent][pid]
                r, c = row_col(p)
                # 如果是 player 1 的视角，需要翻转
                if player == 1:
                    r = ROWS - 1 - r
                label[r, c] = TYPE_TO_IDX[piece.piece_type]

        return label

    def _save_checkpoint(self, final: bool = False):
        """保存检查点"""
        suffix = 'final' if final else f'ep{self.episode_count}'
        path = os.path.join(self.checkpoint_dir, f'junqi_{suffix}.pt')
        self.agent.save(path)
        self.self_play.save_pool()
        print(f"  [Checkpoint] Saved: {path}")

    def _print_stats(self, stats: Dict):
        """打印训练统计"""
        print(
            f"  Ep {stats['episode']:>7,} | "
            f"ELO {stats['elo']:.0f} | "
            f"WR {stats.get('win_rate', 0):.1%} | "
            f"Len {stats.get('avg_game_length', 0):.0f} | "
            f"R {stats.get('avg_reward_p0', 0):.3f} | "
            f"PL {stats.get('train/policy_loss', 0):.4f} | "
            f"VL {stats.get('train/value_loss', 0):.4f} | "
            f"Ent {stats.get('train/entropy', 0):.4f} | "
            f"{stats.get('elapsed_min', 0):.1f}min"
        )

    def load_checkpoint(self, path: str):
        """加载检查点"""
        self.agent.load(path)
        self.self_play.load_pool()
        print(f"  [Checkpoint] Loaded: {path}")
