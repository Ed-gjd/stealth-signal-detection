# 隐身目标噪声检测实验（Stealth Signal Detection）

> 2026-08-19 信号检测实验：**隐身目标（小 RCS）在噪声里还能不能检测到？**——不靠嘴说，直接算给数据看。
>
> 流程：雷达方程算 SNR → 理论检测概率（Marcum Q 函数）→ 蒙特卡洛实测验证 → 距离剖面图。

## 内容

| 文件 | 说明 |
|---|---|
| [detect_stealth_in_noise.py](detect_stealth_in_noise.py) | 实验脚本（X 波段多功能雷达场景参数，np.random.seed=42） |
| [detect_in_noise.png](detect_in_noise.png) | 输出图：距离剖面 + 理论/实测检测概率对比 |

## 运行

```bash
pip install numpy scipy matplotlib
python detect_stealth_in_noise.py   # 生成 detect_in_noise.png
```

依赖仅 numpy / scipy / matplotlib，脚本自包含、无外部数据。

## 说明

私有归档实验。与「相控阵雷达」学习线同源（见学习盘点），单独成仓便于回看。
