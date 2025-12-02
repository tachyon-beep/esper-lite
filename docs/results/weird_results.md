(.venv) john@nyx:~/esper-lite$ uv run python -m esper.scripts.train ppo     --vectorized     --n-envs 2    --devices cuda:0 cuda:1     --episodes 200     --entropy-coef-start 0.2     --entropy-coef-end 0.01     --entropy-anneal-episodes 150 --num-workers 4 --max-epochs 75
============================================================

PPO Vectorized Training (INVERTED CONTROL FLOW + CUDA STREAMS)
============================================================

Task: cifar10 (topology=cnn, type=classification)
Episodes: 200 (across 2 parallel envs)
Max epochs per episode: 75
Policy device: cuda:0
Env devices: ['cuda:0', 'cuda:1'] (1 envs per device)
Random seed: 42
Entropy annealing: 0.2 -> 0.01 over 150 episodes
Learning rate: 0.0003
Telemetry features: ENABLED

Loading cifar10 (2 independent DataLoaders)...
[11:09:37] env1_seed_0 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_0' (conv_enhance, 74.0K params)
[11:09:37] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[11:09:39] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[11:09:39] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[11:09:45] env1_seed_0 | Stage transition: TRAINING → BLENDING
[11:09:45] env1_seed_0 | Stage transition: BLENDING → CULLED
[11:09:45] env1_seed_0 | Culled (conv_enhance, Δacc +7.72%)
    [env1] Culled 'env1_seed_0' (conv_enhance, Δacc +7.72%)
[11:09:46] env0_seed_0 | Stage transition: TRAINING → BLENDING
[11:09:46] env0_seed_0 | Stage transition: BLENDING → CULLED
[11:09:46] env0_seed_0 | Culled (norm, Δacc +4.70%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +4.70%)
[11:09:48] env1_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_1' (conv_enhance, 74.0K params)
[11:09:48] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[11:09:50] env0_seed_1 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_1' (depthwise, 4.8K params)
[11:09:50] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[11:09:53] env1_seed_1 | Stage transition: TRAINING → CULLED
[11:09:53] env1_seed_1 | Culled (conv_enhance, Δacc -1.35%)
    [env1] Culled 'env1_seed_1' (conv_enhance, Δacc -1.35%)
[11:09:55] env1_seed_2 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_2' (attention, 2.0K params)
[11:09:55] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[11:10:02] env0_seed_1 | Stage transition: TRAINING → BLENDING
[11:10:05] env1_seed_2 | Stage transition: TRAINING → BLENDING
[11:10:07] env0_seed_1 | Stage transition: BLENDING → CULLED
[11:10:07] env0_seed_1 | Culled (depthwise, Δacc +6.77%)
    [env0] Culled 'env0_seed_1' (depthwise, Δacc +6.77%)
[11:10:12] env0_seed_2 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_2' (conv_enhance, 74.0K params)
[11:10:12] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[11:10:13] env1_seed_2 | Stage transition: BLENDING → SHADOWING
[11:10:13] env0_seed_2 | Stage transition: TRAINING → CULLED
[11:10:13] env0_seed_2 | Culled (conv_enhance, Δacc +0.00%)
    [env0] Culled 'env0_seed_2' (conv_enhance, Δacc +0.00%)
[11:10:15] env0_seed_3 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_3' (depthwise, 4.8K params)
[11:10:15] env0_seed_3 | Stage transition: GERMINATED → TRAINING
[11:10:16] env1_seed_2 | Stage transition: SHADOWING → PROBATIONARY
[11:10:25] env1_seed_2 | Stage transition: PROBATIONARY → CULLED
[11:10:25] env1_seed_2 | Culled (attention, Δacc +8.00%)
    [env1] Culled 'env1_seed_2' (attention, Δacc +8.00%)
[11:10:26] env1_seed_3 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_3' (conv_enhance, 74.0K params)
[11:10:26] env1_seed_3 | Stage transition: GERMINATED → TRAINING
[11:10:30] env0_seed_3 | Stage transition: TRAINING → BLENDING
[11:10:34] env1_seed_3 | Stage transition: TRAINING → BLENDING
[11:10:34] env0_seed_3 | Stage transition: BLENDING → CULLED
[11:10:34] env0_seed_3 | Culled (depthwise, Δacc +5.54%)
    [env0] Culled 'env0_seed_3' (depthwise, Δacc +5.54%)
[11:10:34] env1_seed_3 | Stage transition: BLENDING → CULLED
[11:10:34] env1_seed_3 | Culled (conv_enhance, Δacc -10.17%)
    [env1] Culled 'env1_seed_3' (conv_enhance, Δacc -10.17%)
[11:10:37] env0_seed_4 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_4' (norm, 0.1K params)
[11:10:37] env1_seed_4 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_4' (attention, 2.0K params)
[11:10:37] env0_seed_4 | Stage transition: GERMINATED → TRAINING
[11:10:37] env1_seed_4 | Stage transition: GERMINATED → TRAINING
[11:10:39] env1_seed_4 | Stage transition: TRAINING → CULLED
[11:10:39] env1_seed_4 | Culled (attention, Δacc +0.00%)
    [env1] Culled 'env1_seed_4' (attention, Δacc +0.00%)
[11:10:40] env1_seed_5 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_5' (conv_enhance, 74.0K params)
[11:10:40] env1_seed_5 | Stage transition: GERMINATED → TRAINING
[11:10:46] env0_seed_4 | Stage transition: TRAINING → BLENDING
[11:10:47] env1_seed_5 | Stage transition: TRAINING → BLENDING
[11:10:51] env0_seed_4 | Stage transition: BLENDING → CULLED
[11:10:51] env0_seed_4 | Culled (norm, Δacc +4.09%)
    [env0] Culled 'env0_seed_4' (norm, Δacc +4.09%)
[11:10:53] env1_seed_5 | Stage transition: BLENDING → CULLED
[11:10:53] env1_seed_5 | Culled (conv_enhance, Δacc -3.59%)
    [env1] Culled 'env1_seed_5' (conv_enhance, Δacc -3.59%)
[11:10:54] env1_seed_6 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_6' (attention, 2.0K params)
[11:10:54] env1_seed_6 | Stage transition: GERMINATED → TRAINING
[11:10:56] env0_seed_5 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_5' (norm, 0.1K params)
[11:10:56] env0_seed_5 | Stage transition: GERMINATED → TRAINING
[11:11:01] env0_seed_5 | Stage transition: TRAINING → CULLED
[11:11:01] env0_seed_5 | Culled (norm, Δacc -1.97%)
    [env0] Culled 'env0_seed_5' (norm, Δacc -1.97%)
[11:11:03] env0_seed_6 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_6' (depthwise, 4.8K params)
[11:11:03] env0_seed_6 | Stage transition: GERMINATED → TRAINING
[11:11:08] env1_seed_6 | Stage transition: TRAINING → BLENDING
[11:11:09] env0_seed_6 | Stage transition: TRAINING → BLENDING
[11:11:09] env1_seed_6 | Stage transition: BLENDING → CULLED
[11:11:09] env1_seed_6 | Culled (attention, Δacc -0.20%)
    [env1] Culled 'env1_seed_6' (attention, Δacc -0.20%)
[11:11:13] env0_seed_6 | Stage transition: BLENDING → CULLED
[11:11:13] env0_seed_6 | Culled (depthwise, Δacc -0.63%)
    [env0] Culled 'env0_seed_6' (depthwise, Δacc -0.63%)
[11:11:14] env0_seed_7 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_7' (attention, 2.0K params)
[11:11:14] env0_seed_7 | Stage transition: GERMINATED → TRAINING
[11:11:16] env1_seed_7 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_7' (conv_enhance, 74.0K params)
[11:11:16] env1_seed_7 | Stage transition: GERMINATED → TRAINING
[11:11:25] env1_seed_7 | Stage transition: TRAINING → BLENDING
[11:11:27] env1_seed_7 | Stage transition: BLENDING → CULLED
[11:11:27] env1_seed_7 | Culled (conv_enhance, Δacc +1.51%)
    [env1] Culled 'env1_seed_7' (conv_enhance, Δacc +1.51%)
[11:11:28] env1_seed_8 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_8' (depthwise, 4.8K params)
[11:11:28] env1_seed_8 | Stage transition: GERMINATED → TRAINING
[11:11:32] env0_seed_7 | Stage transition: TRAINING → CULLED
[11:11:32] env0_seed_7 | Culled (attention, Δacc -0.71%)
    [env0] Culled 'env0_seed_7' (attention, Δacc -0.71%)
[11:11:33] env1_seed_8 | Stage transition: TRAINING → CULLED
[11:11:33] env1_seed_8 | Culled (depthwise, Δacc -2.50%)
    [env1] Culled 'env1_seed_8' (depthwise, Δacc -2.50%)
[11:11:35] env0_seed_8 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_8' (depthwise, 4.8K params)
[11:11:35] env1_seed_9 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_9' (conv_enhance, 74.0K params)
[11:11:35] env0_seed_8 | Stage transition: GERMINATED → TRAINING
[11:11:35] env1_seed_9 | Stage transition: GERMINATED → TRAINING
[11:11:39] env0_seed_8 | Stage transition: TRAINING → CULLED
[11:11:39] env0_seed_8 | Culled (depthwise, Δacc +1.79%)
    [env0] Culled 'env0_seed_8' (depthwise, Δacc +1.79%)
[11:11:42] env1_seed_9 | Stage transition: TRAINING → BLENDING
[11:11:42] env0_seed_9 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_9' (conv_enhance, 74.0K params)
[11:11:42] env1_seed_9 | Stage transition: BLENDING → CULLED
[11:11:42] env1_seed_9 | Culled (conv_enhance, Δacc +1.54%)
    [env1] Culled 'env1_seed_9' (conv_enhance, Δacc +1.54%)
[11:11:42] env0_seed_9 | Stage transition: GERMINATED → TRAINING
[11:11:44] env1_seed_10 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_10' (attention, 2.0K params)
Batch 1: Episodes 2/200
  Env accuracies: ['71.9%', '72.3%']
  Avg acc: 72.1% (rolling: 72.1%)
  Avg reward: 120.9
  Actions: {'WAIT': 16, 'GERMINATE_NORM': 23, 'GERMINATE_ATTENTION': 23, 'GERMINATE_DEPTHWISE': 20, 'GERMINATE_CONV_ENHANCE': 27, 'FOSSILIZE': 16, 'CULL': 25}
  Successful: {'WAIT': 16, 'GERMINATE_NORM': 3, 'GERMINATE_ATTENTION': 5, 'GERMINATE_DEPTHWISE': 5, 'GERMINATE_CONV_ENHANCE': 8, 'FOSSILIZE': 0, 'CULL': 19}
  Policy loss: -0.0127, Value loss: 318.7407, Entropy: 1.9432, Entropy coef: 0.1975
[11:11:46] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[11:11:46] env1_seed_0 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_0' (depthwise, 4.8K params)
[11:11:46] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[11:11:46] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[11:11:53] env0_seed_0 | Stage transition: TRAINING → BLENDING
[11:11:53] env1_seed_0 | Stage transition: TRAINING → BLENDING
[11:11:58] env1_seed_0 | Stage transition: BLENDING → CULLED
[11:11:58] env1_seed_0 | Culled (depthwise, Δacc +16.45%)
    [env1] Culled 'env1_seed_0' (depthwise, Δacc +16.45%)
[11:11:59] env1_seed_1 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_1' (norm, 0.1K params)
[11:12:00] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[11:12:01] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[11:12:04] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[11:12:04] env0_seed_0 | Stage transition: PROBATIONARY → CULLED
[11:12:04] env0_seed_0 | Culled (norm, Δacc +17.18%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +17.18%)
[11:12:06] env1_seed_1 | Stage transition: TRAINING → BLENDING
[11:12:06] env0_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_1' (conv_enhance, 74.0K params)
[11:12:06] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[11:12:11] env0_seed_1 | Stage transition: TRAINING → CULLED
[11:12:11] env0_seed_1 | Culled (conv_enhance, Δacc -9.66%)
    [env0] Culled 'env0_seed_1' (conv_enhance, Δacc -9.66%)
[11:12:14] env1_seed_1 | Stage transition: BLENDING → SHADOWING
[11:12:14] env0_seed_2 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_2' (norm, 0.1K params)
[11:12:15] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[11:12:18] env1_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[11:12:19] env1_seed_1 | Stage transition: PROBATIONARY → CULLED
[11:12:19] env1_seed_1 | Culled (norm, Δacc +7.46%)
    [env1] Culled 'env1_seed_1' (norm, Δacc +7.46%)
[11:12:23] env0_seed_2 | Stage transition: TRAINING → BLENDING
[11:12:23] env1_seed_2 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_2' (norm, 0.1K params)
[11:12:23] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[11:12:29] env1_seed_2 | Stage transition: TRAINING → BLENDING
[11:12:31] env0_seed_2 | Stage transition: BLENDING → SHADOWING
[11:12:34] env0_seed_2 | Stage transition: SHADOWING → PROBATIONARY
[11:12:36] env0_seed_2 | Stage transition: PROBATIONARY → CULLED
[11:12:36] env0_seed_2 | Culled (norm, Δacc +5.53%)
    [env0] Culled 'env0_seed_2' (norm, Δacc +5.53%)
[11:12:38] env1_seed_2 | Stage transition: BLENDING → SHADOWING
[11:12:38] env0_seed_3 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_3' (conv_enhance, 74.0K params)
[11:12:38] env0_seed_3 | Stage transition: GERMINATED → TRAINING
[11:12:41] env1_seed_2 | Stage transition: SHADOWING → PROBATIONARY
[11:12:43] env1_seed_2 | Stage transition: PROBATIONARY → FOSSILIZED
[11:12:43] env1_seed_2 | Fossilized (norm, Δacc +9.17%)
    [env1] Fossilized 'env1_seed_2' (norm, Δacc +9.17%)
[11:12:49] env0_seed_3 | Stage transition: TRAINING → BLENDING
[11:12:58] env0_seed_3 | Stage transition: BLENDING → SHADOWING
[11:13:01] env0_seed_3 | Stage transition: SHADOWING → PROBATIONARY
[11:13:11] env0_seed_3 | Stage transition: PROBATIONARY → CULLED
[11:13:11] env0_seed_3 | Culled (conv_enhance, Δacc -9.51%)
    [env0] Culled 'env0_seed_3' (conv_enhance, Δacc -9.51%)
[11:13:13] env0_seed_4 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_4' (norm, 0.1K params)
[11:13:13] env0_seed_4 | Stage transition: GERMINATED → TRAINING
[11:13:15] env0_seed_4 | Stage transition: TRAINING → CULLED
[11:13:15] env0_seed_4 | Culled (norm, Δacc +0.00%)
    [env0] Culled 'env0_seed_4' (norm, Δacc +0.00%)
[11:13:16] env0_seed_5 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_5' (conv_enhance, 74.0K params)
[11:13:16] env0_seed_5 | Stage transition: GERMINATED → TRAINING
[11:13:23] env0_seed_5 | Stage transition: TRAINING → BLENDING
[11:13:31] env0_seed_5 | Stage transition: BLENDING → SHADOWING
[11:13:35] env0_seed_5 | Stage transition: SHADOWING → PROBATIONARY
Batch 2: Episodes 4/200
  Env accuracies: ['59.5%', '75.2%']
  Avg acc: 67.3% (rolling: 69.7%)
  Avg reward: 118.5
  Actions: {'WAIT': 26, 'GERMINATE_NORM': 28, 'GERMINATE_ATTENTION': 15, 'GERMINATE_DEPTHWISE': 30, 'GERMINATE_CONV_ENHANCE': 20, 'FOSSILIZE': 20, 'CULL': 11}
  Successful: {'WAIT': 26, 'GERMINATE_NORM': 5, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 1, 'GERMINATE_CONV_ENHANCE': 3, 'FOSSILIZE': 1, 'CULL': 11}
  Policy loss: -0.0355, Value loss: 371.1922, Entropy: 1.9275, Entropy coef: 0.1949
[11:13:51] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[11:13:51] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[11:13:55] env0_seed_0 | Stage transition: TRAINING → CULLED
[11:13:55] env0_seed_0 | Culled (norm, Δacc +3.83%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +3.83%)
[11:13:55] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[11:13:55] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[11:13:58] env1_seed_0 | Stage transition: TRAINING → CULLED
[11:13:58] env1_seed_0 | Culled (norm, Δacc +2.70%)
    [env1] Culled 'env1_seed_0' (norm, Δacc +2.70%)
[11:14:00] env0_seed_1 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_1' (norm, 0.1K params)
[11:14:00] env1_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_1' (conv_enhance, 74.0K params)
[11:14:00] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[11:14:00] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[11:14:03] env0_seed_1 | Stage transition: TRAINING → CULLED
[11:14:03] env0_seed_1 | Culled (norm, Δacc +6.45%)
    [env0] Culled 'env0_seed_1' (norm, Δacc +6.45%)
[11:14:09] env0_seed_2 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_2' (norm, 0.1K params)
[11:14:09] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[11:14:10] env0_seed_2 | Stage transition: TRAINING → CULLED
[11:14:10] env0_seed_2 | Culled (norm, Δacc +0.00%)
    [env0] Culled 'env0_seed_2' (norm, Δacc +0.00%)
[11:14:14] env1_seed_1 | Stage transition: TRAINING → BLENDING
[11:14:18] env0_seed_3 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_3' (norm, 0.1K params)
[11:14:18] env1_seed_1 | Stage transition: BLENDING → CULLED
[11:14:18] env1_seed_1 | Culled (conv_enhance, Δacc +5.85%)
    [env1] Culled 'env1_seed_1' (conv_enhance, Δacc +5.85%)
[11:14:18] env0_seed_3 | Stage transition: GERMINATED → TRAINING
[11:14:19] env1_seed_2 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_2' (attention, 2.0K params)
[11:14:19] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[11:14:24] env0_seed_3 | Stage transition: TRAINING → BLENDING
[11:14:24] env0_seed_3 | Stage transition: BLENDING → CULLED
[11:14:24] env0_seed_3 | Culled (norm, Δacc +3.45%)
    [env0] Culled 'env0_seed_3' (norm, Δacc +3.45%)
[11:14:29] env1_seed_2 | Stage transition: TRAINING → BLENDING
[11:14:29] env0_seed_4 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_4' (attention, 2.0K params)
[11:14:29] env1_seed_2 | Stage transition: BLENDING → CULLED
[11:14:29] env1_seed_2 | Culled (attention, Δacc -1.58%)
    [env1] Culled 'env1_seed_2' (attention, Δacc -1.58%)
[11:14:29] env0_seed_4 | Stage transition: GERMINATED → TRAINING
[11:14:31] env1_seed_3 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_3' (attention, 2.0K params)
[11:14:31] env1_seed_3 | Stage transition: GERMINATED → TRAINING
[11:14:36] env0_seed_4 | Stage transition: TRAINING → BLENDING
[11:14:36] env1_seed_3 | Stage transition: TRAINING → CULLED
[11:14:36] env1_seed_3 | Culled (attention, Δacc -1.23%)
    [env1] Culled 'env1_seed_3' (attention, Δacc -1.23%)
[11:14:39] env1_seed_4 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_4' (conv_enhance, 74.0K params)
[11:14:39] env1_seed_4 | Stage transition: GERMINATED → TRAINING
[11:14:41] env1_seed_4 | Stage transition: TRAINING → CULLED
[11:14:41] env1_seed_4 | Culled (conv_enhance, Δacc +0.00%)
    [env1] Culled 'env1_seed_4' (conv_enhance, Δacc +0.00%)
[11:14:42] env1_seed_5 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_5' (norm, 0.1K params)
[11:14:43] env1_seed_5 | Stage transition: GERMINATED → TRAINING
[11:14:44] env0_seed_4 | Stage transition: BLENDING → SHADOWING
[11:14:47] env0_seed_4 | Stage transition: SHADOWING → PROBATIONARY
[11:14:49] env1_seed_5 | Stage transition: TRAINING → BLENDING
[11:14:56] env0_seed_4 | Stage transition: PROBATIONARY → FOSSILIZED
[11:14:56] env0_seed_4 | Fossilized (attention, Δacc +1.55%)
    [env0] Fossilized 'env0_seed_4' (attention, Δacc +1.55%)
[11:14:57] env1_seed_5 | Stage transition: BLENDING → SHADOWING
[11:15:01] env1_seed_5 | Stage transition: SHADOWING → PROBATIONARY
[11:15:06] env1_seed_5 | Stage transition: PROBATIONARY → FOSSILIZED
[11:15:06] env1_seed_5 | Fossilized (norm, Δacc +7.69%)
    [env1] Fossilized 'env1_seed_5' (norm, Δacc +7.69%)
Batch 3: Episodes 6/200
  Env accuracies: ['72.2%', '77.1%']
  Avg acc: 74.7% (rolling: 71.4%)
  Avg reward: 150.8
  Actions: {'WAIT': 21, 'GERMINATE_NORM': 19, 'GERMINATE_ATTENTION': 21, 'GERMINATE_DEPTHWISE': 23, 'GERMINATE_CONV_ENHANCE': 14, 'FOSSILIZE': 27, 'CULL': 25}
  Successful: {'WAIT': 21, 'GERMINATE_NORM': 6, 'GERMINATE_ATTENTION': 3, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 2, 'FOSSILIZE': 2, 'CULL': 19}
  Policy loss: -0.0033, Value loss: 266.5861, Entropy: 1.9286, Entropy coef: 0.1924
[11:15:59] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[11:15:59] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[11:16:03] env0_seed_0 | Stage transition: TRAINING → CULLED
[11:16:03] env0_seed_0 | Culled (norm, Δacc +4.31%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +4.31%)
[11:16:04] env0_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_1' (conv_enhance, 74.0K params)
[11:16:04] env1_seed_0 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_0' (depthwise, 4.8K params)
[11:16:04] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[11:16:04] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[11:16:08] env1_seed_0 | Stage transition: TRAINING → CULLED
[11:16:08] env1_seed_0 | Culled (depthwise, Δacc -2.07%)
    [env1] Culled 'env1_seed_0' (depthwise, Δacc -2.07%)
[11:16:09] env0_seed_1 | Stage transition: TRAINING → CULLED
[11:16:09] env0_seed_1 | Culled (conv_enhance, Δacc +4.01%)
    [env0] Culled 'env0_seed_1' (conv_enhance, Δacc +4.01%)
[11:16:09] env1_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_1' (conv_enhance, 74.0K params)
[11:16:09] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[11:16:11] env0_seed_2 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_2' (depthwise, 4.8K params)
[11:16:11] env1_seed_1 | Stage transition: TRAINING → CULLED
[11:16:11] env1_seed_1 | Culled (conv_enhance, Δacc +0.00%)
    [env1] Culled 'env1_seed_1' (conv_enhance, Δacc +0.00%)
[11:16:11] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[11:16:13] env1_seed_2 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_2' (depthwise, 4.8K params)
[11:16:13] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[11:16:18] env0_seed_2 | Stage transition: TRAINING → CULLED
[11:16:18] env0_seed_2 | Culled (depthwise, Δacc -3.18%)
    [env0] Culled 'env0_seed_2' (depthwise, Δacc -3.18%)
[11:16:23] env0_seed_3 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_3' (depthwise, 4.8K params)
[11:16:23] env1_seed_2 | Stage transition: TRAINING → CULLED
[11:16:23] env1_seed_2 | Culled (depthwise, Δacc -3.74%)
    [env1] Culled 'env1_seed_2' (depthwise, Δacc -3.74%)
[11:16:23] env0_seed_3 | Stage transition: GERMINATED → TRAINING
[11:16:25] env1_seed_3 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_3' (norm, 0.1K params)
[11:16:25] env1_seed_3 | Stage transition: GERMINATED → TRAINING
[11:16:32] env0_seed_3 | Stage transition: TRAINING → BLENDING
[11:16:33] env0_seed_3 | Stage transition: BLENDING → CULLED
[11:16:33] env0_seed_3 | Culled (depthwise, Δacc +4.34%)
    [env0] Culled 'env0_seed_3' (depthwise, Δacc +4.34%)
[11:16:37] env1_seed_3 | Stage transition: TRAINING → BLENDING
[11:16:40] env0_seed_4 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_4' (norm, 0.1K params)
[11:16:40] env1_seed_3 | Stage transition: BLENDING → CULLED
[11:16:40] env1_seed_3 | Culled (norm, Δacc +3.37%)
    [env1] Culled 'env1_seed_3' (norm, Δacc +3.37%)
[11:16:40] env0_seed_4 | Stage transition: GERMINATED → TRAINING
[11:16:42] env0_seed_4 | Stage transition: TRAINING → CULLED
[11:16:42] env0_seed_4 | Culled (norm, Δacc +0.00%)
    [env0] Culled 'env0_seed_4' (norm, Δacc +0.00%)
[11:16:42] env1_seed_4 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_4' (norm, 0.1K params)
[11:16:42] env1_seed_4 | Stage transition: GERMINATED → TRAINING
[11:16:45] env0_seed_5 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_5' (norm, 0.1K params)
[11:16:45] env0_seed_5 | Stage transition: GERMINATED → TRAINING
[11:16:48] env1_seed_4 | Stage transition: TRAINING → BLENDING
[11:16:52] env0_seed_5 | Stage transition: TRAINING → BLENDING
[11:16:52] env1_seed_4 | Stage transition: BLENDING → CULLED
[11:16:52] env1_seed_4 | Culled (norm, Δacc +6.81%)
    [env1] Culled 'env1_seed_4' (norm, Δacc +6.81%)
[11:16:53] env0_seed_5 | Stage transition: BLENDING → CULLED
[11:16:53] env0_seed_5 | Culled (norm, Δacc +5.85%)
    [env0] Culled 'env0_seed_5' (norm, Δacc +5.85%)
[11:16:53] env1_seed_5 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_5' (norm, 0.1K params)
[11:16:53] env1_seed_5 | Stage transition: GERMINATED → TRAINING
[11:16:56] env0_seed_6 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_6' (norm, 0.1K params)
[11:16:56] env1_seed_5 | Stage transition: TRAINING → CULLED
[11:16:56] env1_seed_5 | Culled (norm, Δacc -0.22%)
    [env1] Culled 'env1_seed_5' (norm, Δacc -0.22%)
[11:16:57] env0_seed_6 | Stage transition: GERMINATED → TRAINING
[11:16:58] env0_seed_6 | Stage transition: TRAINING → CULLED
[11:16:58] env0_seed_6 | Culled (norm, Δacc +0.00%)
    [env0] Culled 'env0_seed_6' (norm, Δacc +0.00%)
[11:16:58] env1_seed_6 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_6' (norm, 0.1K params)
[11:16:58] env1_seed_6 | Stage transition: GERMINATED → TRAINING
[11:17:00] env0_seed_7 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_7' (attention, 2.0K params)
[11:17:00] env0_seed_7 | Stage transition: GERMINATED → TRAINING
[11:17:01] env0_seed_7 | Stage transition: TRAINING → CULLED
[11:17:01] env0_seed_7 | Culled (attention, Δacc +0.00%)
    [env0] Culled 'env0_seed_7' (attention, Δacc +0.00%)
[11:17:01] env1_seed_6 | Stage transition: TRAINING → CULLED
[11:17:01] env1_seed_6 | Culled (norm, Δacc -1.43%)
    [env1] Culled 'env1_seed_6' (norm, Δacc -1.43%)
[11:17:03] env0_seed_8 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_8' (conv_enhance, 74.0K params)
[11:17:03] env1_seed_7 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_7' (norm, 0.1K params)
[11:17:03] env0_seed_8 | Stage transition: GERMINATED → TRAINING
[11:17:03] env1_seed_7 | Stage transition: GERMINATED → TRAINING
[11:17:08] env1_seed_7 | Stage transition: TRAINING → CULLED
[11:17:08] env1_seed_7 | Culled (norm, Δacc -1.37%)
    [env1] Culled 'env1_seed_7' (norm, Δacc -1.37%)
[11:17:15] env1_seed_8 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_8' (attention, 2.0K params)
[11:17:15] env1_seed_8 | Stage transition: GERMINATED → TRAINING
[11:17:23] env1_seed_8 | Stage transition: TRAINING → CULLED
[11:17:23] env1_seed_8 | Culled (attention, Δacc -4.80%)
    [env1] Culled 'env1_seed_8' (attention, Δacc -4.80%)
[11:17:25] env1_seed_9 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_9' (attention, 2.0K params)
[11:17:25] env1_seed_9 | Stage transition: GERMINATED → TRAINING
[11:17:30] env1_seed_9 | Stage transition: TRAINING → CULLED
[11:17:30] env1_seed_9 | Culled (attention, Δacc -5.17%)
    [env1] Culled 'env1_seed_9' (attention, Δacc -5.17%)
[11:17:31] env0_seed_8 | Stage transition: TRAINING → CULLED
[11:17:31] env0_seed_8 | Culled (conv_enhance, Δacc -1.22%)
    [env0] Culled 'env0_seed_8' (conv_enhance, Δacc -1.22%)
[11:17:33] env0_seed_9 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_9' (depthwise, 4.8K params)
[11:17:33] env1_seed_10 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_10' (conv_enhance, 74.0K params)
[11:17:33] env0_seed_9 | Stage transition: GERMINATED → TRAINING
[11:17:33] env1_seed_10 | Stage transition: GERMINATED → TRAINING
[11:17:40] env1_seed_10 | Stage transition: TRAINING → BLENDING
[11:17:44] env0_seed_9 | Stage transition: TRAINING → BLENDING
[11:17:44] env1_seed_10 | Stage transition: BLENDING → CULLED
[11:17:44] env1_seed_10 | Culled (conv_enhance, Δacc +4.84%)
    [env1] Culled 'env1_seed_10' (conv_enhance, Δacc +4.84%)
[11:17:49] env1_seed_11 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_11' (norm, 0.1K params)
[11:17:49] env1_seed_11 | Stage transition: GERMINATED → TRAINING
[11:17:52] env0_seed_9 | Stage transition: BLENDING → SHADOWING
[11:17:52] env0_seed_9 | Stage transition: SHADOWING → CULLED
[11:17:52] env0_seed_9 | Culled (depthwise, Δacc -3.30%)
    [env0] Culled 'env0_seed_9' (depthwise, Δacc -3.30%)
[11:17:57] env0_seed_10 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_10' (norm, 0.1K params)
[11:17:57] env0_seed_10 | Stage transition: GERMINATED → TRAINING
[11:17:59] env1_seed_11 | Stage transition: TRAINING → BLENDING
Batch 4: Episodes 8/200
  Env accuracies: ['72.0%', '75.1%']
  Avg acc: 73.6% (rolling: 71.9%)
  Avg reward: 126.2
  Actions: {'WAIT': 21, 'GERMINATE_NORM': 24, 'GERMINATE_ATTENTION': 15, 'GERMINATE_DEPTHWISE': 21, 'GERMINATE_CONV_ENHANCE': 16, 'FOSSILIZE': 24, 'CULL': 29}
  Successful: {'WAIT': 21, 'GERMINATE_NORM': 11, 'GERMINATE_ATTENTION': 3, 'GERMINATE_DEPTHWISE': 5, 'GERMINATE_CONV_ENHANCE': 4, 'FOSSILIZE': 0, 'CULL': 21}
  Policy loss: -0.0242, Value loss: 309.2562, Entropy: 1.9230, Entropy coef: 0.1899
[11:18:04] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[11:18:04] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[11:18:05] env0_seed_0 | Stage transition: TRAINING → CULLED
[11:18:05] env0_seed_0 | Culled (norm, Δacc +0.00%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +0.00%)
[11:18:05] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[11:18:05] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[11:18:09] env0_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_1' (conv_enhance, 74.0K params)
[11:18:09] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[11:18:12] env1_seed_0 | Stage transition: TRAINING → BLENDING
[11:18:17] env0_seed_1 | Stage transition: TRAINING → BLENDING
[11:18:17] env1_seed_0 | Stage transition: BLENDING → CULLED
[11:18:17] env1_seed_0 | Culled (norm, Δacc +14.14%)
    [env1] Culled 'env1_seed_0' (norm, Δacc +14.14%)
[11:18:19] env1_seed_1 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_1' (norm, 0.1K params)
[11:18:19] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[11:18:25] env0_seed_1 | Stage transition: BLENDING → SHADOWING
[11:18:25] env1_seed_1 | Stage transition: TRAINING → BLENDING
[11:18:29] env0_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[11:18:34] env1_seed_1 | Stage transition: BLENDING → SHADOWING
[11:18:36] env0_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[11:18:36] env0_seed_1 | Fossilized (conv_enhance, Δacc +14.74%)
    [env0] Fossilized 'env0_seed_1' (conv_enhance, Δacc +14.74%)
[11:18:36] env1_seed_1 | Stage transition: SHADOWING → CULLED
[11:18:36] env1_seed_1 | Culled (norm, Δacc +14.94%)
    [env1] Culled 'env1_seed_1' (norm, Δacc +14.94%)
[11:18:37] env1_seed_2 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_2' (norm, 0.1K params)
[11:18:37] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[11:18:44] env1_seed_2 | Stage transition: TRAINING → BLENDING
[11:18:52] env1_seed_2 | Stage transition: BLENDING → SHADOWING
[11:18:56] env1_seed_2 | Stage transition: SHADOWING → PROBATIONARY
[11:18:57] env1_seed_2 | Stage transition: PROBATIONARY → CULLED
[11:18:57] env1_seed_2 | Culled (norm, Δacc +8.83%)
    [env1] Culled 'env1_seed_2' (norm, Δacc +8.83%)
[11:18:59] env1_seed_3 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_3' (norm, 0.1K params)
[11:18:59] env1_seed_3 | Stage transition: GERMINATED → TRAINING
[11:19:06] env1_seed_3 | Stage transition: TRAINING → BLENDING
[11:19:07] env1_seed_3 | Stage transition: BLENDING → CULLED
[11:19:07] env1_seed_3 | Culled (norm, Δacc +3.40%)
    [env1] Culled 'env1_seed_3' (norm, Δacc +3.40%)
[11:19:14] env1_seed_4 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_4' (depthwise, 4.8K params)
[11:19:14] env1_seed_4 | Stage transition: GERMINATED → TRAINING
[11:19:19] env1_seed_4 | Stage transition: TRAINING → CULLED
[11:19:19] env1_seed_4 | Culled (depthwise, Δacc -1.06%)
    [env1] Culled 'env1_seed_4' (depthwise, Δacc -1.06%)
[11:19:21] env1_seed_5 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_5' (attention, 2.0K params)
[11:19:21] env1_seed_5 | Stage transition: GERMINATED → TRAINING
[11:19:41] env1_seed_5 | Stage transition: TRAINING → CULLED
[11:19:41] env1_seed_5 | Culled (attention, Δacc -3.97%)
    [env1] Culled 'env1_seed_5' (attention, Δacc -3.97%)
[11:19:43] env1_seed_6 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_6' (norm, 0.1K params)
[11:19:43] env1_seed_6 | Stage transition: GERMINATED → TRAINING
[11:19:46] env1_seed_6 | Stage transition: TRAINING → CULLED
[11:19:46] env1_seed_6 | Culled (norm, Δacc -3.01%)
    [env1] Culled 'env1_seed_6' (norm, Δacc -3.01%)
[11:19:48] env1_seed_7 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_7' (conv_enhance, 74.0K params)
[11:19:48] env1_seed_7 | Stage transition: GERMINATED → TRAINING
[11:19:51] env1_seed_7 | Stage transition: TRAINING → CULLED
[11:19:51] env1_seed_7 | Culled (conv_enhance, Δacc +4.59%)
    [env1] Culled 'env1_seed_7' (conv_enhance, Δacc +4.59%)
[11:19:53] env1_seed_8 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_8' (norm, 0.1K params)
[11:19:53] env1_seed_8 | Stage transition: GERMINATED → TRAINING
[11:20:00] env1_seed_8 | Stage transition: TRAINING → CULLED
[11:20:00] env1_seed_8 | Culled (norm, Δacc -0.92%)
    [env1] Culled 'env1_seed_8' (norm, Δacc -0.92%)
[11:20:03] env1_seed_9 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_9' (depthwise, 4.8K params)
[11:20:03] env1_seed_9 | Stage transition: GERMINATED → TRAINING
[11:20:08] env1_seed_9 | Stage transition: TRAINING → CULLED
[11:20:08] env1_seed_9 | Culled (depthwise, Δacc +3.79%)
    [env1] Culled 'env1_seed_9' (depthwise, Δacc +3.79%)
Batch 5: Episodes 10/200
  Env accuracies: ['71.8%', '75.4%']
  Avg acc: 73.6% (rolling: 72.3%)
  Avg reward: 68.1
  Actions: {'WAIT': 21, 'GERMINATE_NORM': 29, 'GERMINATE_ATTENTION': 11, 'GERMINATE_DEPTHWISE': 16, 'GERMINATE_CONV_ENHANCE': 27, 'FOSSILIZE': 29, 'CULL': 17}
  Successful: {'WAIT': 21, 'GERMINATE_NORM': 7, 'GERMINATE_ATTENTION': 1, 'GERMINATE_DEPTHWISE': 2, 'GERMINATE_CONV_ENHANCE': 2, 'FOSSILIZE': 1, 'CULL': 14}
  Policy loss: -0.0081, Value loss: 422.1488, Entropy: 1.9280, Entropy coef: 0.1873

Blueprint Stats
  ---------------------------------------------------------------------------

Blueprint       Germ  Foss  Cull   Rate     ΔAcc    Churn
  ---------------------------------------------------------------------------

  attention         12     1    10   9.1%   +1.55%   -0.97%
  conv_enhance      19     1    16   5.9%  +14.74%   -0.34%
  depthwise         13     0    13   0.0%   +0.00%   +1.71%
  norm              32     2    28   6.7%   +8.43%   +3.86%
Seed Scoreboard (env 0):
  Fossilized: 2 (+76.0K params, +80.2% of host)
  Culled: 29
  Avg fossilize age: 16.0 epochs
  Avg cull age: 5.7 epochs
  Compute cost: 1.50x baseline
  Distribution: attention x1, conv_enhance x1
Seed Scoreboard (env 1):
  Fossilized: 2 (+0.3K params, +0.3% of host)
  Culled: 38
  Avg fossilize age: 13.0 epochs
  Avg cull age: 5.4 epochs
  Compute cost: 1.04x baseline
  Distribution: norm x2

[11:20:12] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[11:20:12] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[11:20:13] env0_seed_0 | Stage transition: TRAINING → CULLED
[11:20:13] env0_seed_0 | Culled (norm, Δacc +0.00%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +0.00%)
[11:20:13] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[11:20:13] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[11:20:18] env0_seed_1 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_1' (depthwise, 4.8K params)
[11:20:18] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[11:20:20] env1_seed_0 | Stage transition: TRAINING → BLENDING
[11:20:22] env1_seed_0 | Stage transition: BLENDING → CULLED
[11:20:22] env1_seed_0 | Culled (norm, Δacc +11.88%)
    [env1] Culled 'env1_seed_0' (norm, Δacc +11.88%)
[11:20:23] env1_seed_1 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_1' (norm, 0.1K params)
[11:20:23] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[11:20:30] env0_seed_1 | Stage transition: TRAINING → BLENDING
[11:20:30] env1_seed_1 | Stage transition: TRAINING → CULLED
[11:20:30] env1_seed_1 | Culled (norm, Δacc +2.25%)
    [env1] Culled 'env1_seed_1' (norm, Δacc +2.25%)
[11:20:32] env1_seed_2 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_2' (conv_enhance, 74.0K params)
[11:20:32] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[11:20:33] env1_seed_2 | Stage transition: TRAINING → CULLED
[11:20:33] env1_seed_2 | Culled (conv_enhance, Δacc +0.00%)
    [env1] Culled 'env1_seed_2' (conv_enhance, Δacc +0.00%)
[11:20:35] env1_seed_3 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_3' (norm, 0.1K params)
[11:20:35] env1_seed_3 | Stage transition: GERMINATED → TRAINING
[11:20:38] env0_seed_1 | Stage transition: BLENDING → SHADOWING
[11:20:42] env0_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[11:20:42] env1_seed_3 | Stage transition: TRAINING → BLENDING
[11:20:45] env0_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[11:20:45] env0_seed_1 | Fossilized (depthwise, Δacc +4.76%)
    [env0] Fossilized 'env0_seed_1' (depthwise, Δacc +4.76%)
[11:20:50] env1_seed_3 | Stage transition: BLENDING → SHADOWING
[11:20:50] env1_seed_3 | Stage transition: SHADOWING → CULLED
[11:20:50] env1_seed_3 | Culled (norm, Δacc +6.91%)
    [env1] Culled 'env1_seed_3' (norm, Δacc +6.91%)
[11:20:57] env1_seed_4 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_4' (depthwise, 4.8K params)
[11:20:57] env1_seed_4 | Stage transition: GERMINATED → TRAINING
[11:21:02] env1_seed_4 | Stage transition: TRAINING → CULLED
[11:21:02] env1_seed_4 | Culled (depthwise, Δacc +2.96%)
    [env1] Culled 'env1_seed_4' (depthwise, Δacc +2.96%)
[11:21:05] env1_seed_5 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_5' (norm, 0.1K params)
[11:21:05] env1_seed_5 | Stage transition: GERMINATED → TRAINING
[11:21:07] env1_seed_5 | Stage transition: TRAINING → CULLED
[11:21:07] env1_seed_5 | Culled (norm, Δacc +0.00%)
    [env1] Culled 'env1_seed_5' (norm, Δacc +0.00%)
[11:21:08] env1_seed_6 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_6' (norm, 0.1K params)
[11:21:08] env1_seed_6 | Stage transition: GERMINATED → TRAINING
[11:21:15] env1_seed_6 | Stage transition: TRAINING → BLENDING
[11:21:15] env1_seed_6 | Stage transition: BLENDING → CULLED
[11:21:15] env1_seed_6 | Culled (norm, Δacc +0.82%)
    [env1] Culled 'env1_seed_6' (norm, Δacc +0.82%)
[11:21:18] env1_seed_7 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_7' (norm, 0.1K params)
[11:21:18] env1_seed_7 | Stage transition: GERMINATED → TRAINING
[11:21:30] env1_seed_7 | Stage transition: TRAINING → BLENDING
[11:21:30] env1_seed_7 | Stage transition: BLENDING → CULLED
[11:21:30] env1_seed_7 | Culled (norm, Δacc -2.18%)
    [env1] Culled 'env1_seed_7' (norm, Δacc -2.18%)
[11:21:33] env1_seed_8 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_8' (attention, 2.0K params)
[11:21:33] env1_seed_8 | Stage transition: GERMINATED → TRAINING
[11:21:42] env1_seed_8 | Stage transition: TRAINING → CULLED
[11:21:42] env1_seed_8 | Culled (attention, Δacc -0.55%)
    [env1] Culled 'env1_seed_8' (attention, Δacc -0.55%)
[11:21:43] env1_seed_9 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_9' (attention, 2.0K params)
[11:21:43] env1_seed_9 | Stage transition: GERMINATED → TRAINING
[11:21:50] env1_seed_9 | Stage transition: TRAINING → CULLED
[11:21:50] env1_seed_9 | Culled (attention, Δacc +0.58%)
    [env1] Culled 'env1_seed_9' (attention, Δacc +0.58%)
[11:21:55] env1_seed_10 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_10' (attention, 2.0K params)
[11:21:55] env1_seed_10 | Stage transition: GERMINATED → TRAINING
[11:21:57] env1_seed_10 | Stage transition: TRAINING → CULLED
[11:21:57] env1_seed_10 | Culled (attention, Δacc +0.00%)
    [env1] Culled 'env1_seed_10' (attention, Δacc +0.00%)
[11:22:01] env1_seed_11 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_11' (attention, 2.0K params)
[11:22:01] env1_seed_11 | Stage transition: GERMINATED → TRAINING
[11:22:06] env1_seed_11 | Stage transition: TRAINING → CULLED
[11:22:06] env1_seed_11 | Culled (attention, Δacc +5.63%)
    [env1] Culled 'env1_seed_11' (attention, Δacc +5.63%)
[11:22:11] env1_seed_12 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_12' (conv_enhance, 74.0K params)
[11:22:11] env1_seed_12 | Stage transition: GERMINATED → TRAINING
[11:22:13] env1_seed_12 | Stage transition: TRAINING → CULLED
[11:22:13] env1_seed_12 | Culled (conv_enhance, Δacc +0.00%)
    [env1] Culled 'env1_seed_12' (conv_enhance, Δacc +0.00%)
Batch 6: Episodes 12/200
  Env accuracies: ['66.2%', '72.7%']
  Avg acc: 69.5% (rolling: 71.8%)
  Avg reward: 137.4
  Actions: {'WAIT': 20, 'GERMINATE_NORM': 25, 'GERMINATE_ATTENTION': 20, 'GERMINATE_DEPTHWISE': 13, 'GERMINATE_CONV_ENHANCE': 15, 'FOSSILIZE': 31, 'CULL': 26}
  Successful: {'WAIT': 20, 'GERMINATE_NORM': 7, 'GERMINATE_ATTENTION': 4, 'GERMINATE_DEPTHWISE': 2, 'GERMINATE_CONV_ENHANCE': 2, 'FOSSILIZE': 1, 'CULL': 22}
  Policy loss: -0.0212, Value loss: 316.9249, Entropy: 1.9248, Entropy coef: 0.1848
[11:22:16] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[11:22:16] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[11:22:18] env0_seed_0 | Stage transition: TRAINING → CULLED
[11:22:18] env0_seed_0 | Culled (norm, Δacc +0.00%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +0.00%)
[11:22:20] env0_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_1' (conv_enhance, 74.0K params)
[11:22:20] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[11:22:20] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[11:22:20] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[11:22:26] env0_seed_1 | Stage transition: TRAINING → BLENDING
[11:22:26] env1_seed_0 | Stage transition: TRAINING → BLENDING
[11:22:35] env0_seed_1 | Stage transition: BLENDING → SHADOWING
[11:22:35] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[11:22:35] env1_seed_0 | Stage transition: SHADOWING → CULLED
[11:22:35] env1_seed_0 | Culled (norm, Δacc +10.41%)
    [env1] Culled 'env1_seed_0' (norm, Δacc +10.41%)
[11:22:36] env1_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_1' (conv_enhance, 74.0K params)
[11:22:36] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[11:22:38] env0_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[11:22:49] env1_seed_1 | Stage transition: TRAINING → BLENDING
[11:22:49] env0_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[11:22:49] env0_seed_1 | Fossilized (conv_enhance, Δacc +17.22%)
    [env0] Fossilized 'env0_seed_1' (conv_enhance, Δacc +17.22%)
[11:22:54] env1_seed_1 | Stage transition: BLENDING → CULLED
[11:22:54] env1_seed_1 | Culled (conv_enhance, Δacc +4.29%)
    [env1] Culled 'env1_seed_1' (conv_enhance, Δacc +4.29%)
[11:22:56] env1_seed_2 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_2' (depthwise, 4.8K params)
[11:22:56] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[11:23:03] env1_seed_2 | Stage transition: TRAINING → CULLED
[11:23:03] env1_seed_2 | Culled (depthwise, Δacc +0.85%)
    [env1] Culled 'env1_seed_2' (depthwise, Δacc +0.85%)
[11:23:10] env1_seed_3 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_3' (attention, 2.0K params)
[11:23:10] env1_seed_3 | Stage transition: GERMINATED → TRAINING
[11:23:17] env1_seed_3 | Stage transition: TRAINING → CULLED
[11:23:17] env1_seed_3 | Culled (attention, Δacc +0.81%)
    [env1] Culled 'env1_seed_3' (attention, Δacc +0.81%)
[11:23:18] env1_seed_4 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_4' (norm, 0.1K params)
[11:23:18] env1_seed_4 | Stage transition: GERMINATED → TRAINING
[11:23:25] env1_seed_4 | Stage transition: TRAINING → BLENDING
[11:23:27] env1_seed_4 | Stage transition: BLENDING → CULLED
[11:23:27] env1_seed_4 | Culled (norm, Δacc +3.75%)
    [env1] Culled 'env1_seed_4' (norm, Δacc +3.75%)
[11:23:28] env1_seed_5 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_5' (attention, 2.0K params)
[11:23:28] env1_seed_5 | Stage transition: GERMINATED → TRAINING
[11:23:32] env1_seed_5 | Stage transition: TRAINING → CULLED
[11:23:32] env1_seed_5 | Culled (attention, Δacc -4.01%)
    [env1] Culled 'env1_seed_5' (attention, Δacc -4.01%)
[11:23:33] env1_seed_6 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_6' (attention, 2.0K params)
[11:23:33] env1_seed_6 | Stage transition: GERMINATED → TRAINING
[11:23:40] env1_seed_6 | Stage transition: TRAINING → BLENDING
[11:23:43] env1_seed_6 | Stage transition: BLENDING → CULLED
[11:23:43] env1_seed_6 | Culled (attention, Δacc +3.42%)
    [env1] Culled 'env1_seed_6' (attention, Δacc +3.42%)
[11:23:45] env1_seed_7 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_7' (attention, 2.0K params)
[11:23:45] env1_seed_7 | Stage transition: GERMINATED → TRAINING
[11:23:47] env1_seed_7 | Stage transition: TRAINING → CULLED
[11:23:47] env1_seed_7 | Culled (attention, Δacc +0.00%)
    [env1] Culled 'env1_seed_7' (attention, Δacc +0.00%)
[11:23:53] env1_seed_8 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_8' (depthwise, 4.8K params)
[11:23:53] env1_seed_8 | Stage transition: GERMINATED → TRAINING
[11:24:00] env1_seed_8 | Stage transition: TRAINING → CULLED
[11:24:00] env1_seed_8 | Culled (depthwise, Δacc +4.02%)
    [env1] Culled 'env1_seed_8' (depthwise, Δacc +4.02%)
[11:24:04] env1_seed_9 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_9' (attention, 2.0K params)
[11:24:04] env1_seed_9 | Stage transition: GERMINATED → TRAINING
[11:24:14] env1_seed_9 | Stage transition: TRAINING → CULLED
[11:24:14] env1_seed_9 | Culled (attention, Δacc -7.57%)
    [env1] Culled 'env1_seed_9' (attention, Δacc -7.57%)
Batch 7: Episodes 14/200
  Env accuracies: ['71.9%', '71.7%']
  Avg acc: 71.8% (rolling: 71.8%)
  Avg reward: 62.2
  Actions: {'WAIT': 32, 'GERMINATE_NORM': 18, 'GERMINATE_ATTENTION': 17, 'GERMINATE_DEPTHWISE': 23, 'GERMINATE_CONV_ENHANCE': 18, 'FOSSILIZE': 25, 'CULL': 17}
  Successful: {'WAIT': 32, 'GERMINATE_NORM': 3, 'GERMINATE_ATTENTION': 5, 'GERMINATE_DEPTHWISE': 2, 'GERMINATE_CONV_ENHANCE': 2, 'FOSSILIZE': 1, 'CULL': 14}
  Policy loss: -0.0065, Value loss: 425.3864, Entropy: 1.9094, Entropy coef: 0.1823
[11:24:22] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[11:24:22] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[11:24:26] env1_seed_0 | Stage transition: TRAINING → CULLED
[11:24:26] env1_seed_0 | Culled (norm, Δacc +1.96%)
    [env1] Culled 'env1_seed_0' (norm, Δacc +1.96%)
[11:24:27] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[11:24:27] env1_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_1' (conv_enhance, 74.0K params)
[11:24:27] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[11:24:27] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[11:24:33] env1_seed_1 | Stage transition: TRAINING → CULLED
[11:24:33] env1_seed_1 | Culled (conv_enhance, Δacc +3.20%)
    [env1] Culled 'env1_seed_1' (conv_enhance, Δacc +3.20%)
[11:24:34] env0_seed_0 | Stage transition: TRAINING → BLENDING
[11:24:34] env0_seed_0 | Stage transition: BLENDING → CULLED
[11:24:34] env0_seed_0 | Culled (norm, Δacc +1.16%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +1.16%)
[11:24:37] env0_seed_1 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_1' (depthwise, 4.8K params)
[11:24:37] env1_seed_2 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_2' (conv_enhance, 74.0K params)
[11:24:38] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[11:24:38] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[11:24:39] env1_seed_2 | Stage transition: TRAINING → CULLED
[11:24:39] env1_seed_2 | Culled (conv_enhance, Δacc +0.00%)
    [env1] Culled 'env1_seed_2' (conv_enhance, Δacc +0.00%)
[11:24:41] env1_seed_3 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_3' (conv_enhance, 74.0K params)
[11:24:41] env1_seed_3 | Stage transition: GERMINATED → TRAINING
[11:24:46] env0_seed_1 | Stage transition: TRAINING → BLENDING
[11:24:46] env1_seed_3 | Stage transition: TRAINING → CULLED
[11:24:46] env1_seed_3 | Culled (conv_enhance, Δacc +1.97%)
    [env1] Culled 'env1_seed_3' (conv_enhance, Δacc +1.97%)
[11:24:48] env1_seed_4 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_4' (depthwise, 4.8K params)
[11:24:48] env1_seed_4 | Stage transition: GERMINATED → TRAINING
[11:24:51] env1_seed_4 | Stage transition: TRAINING → CULLED
[11:24:51] env1_seed_4 | Culled (depthwise, Δacc +1.38%)
    [env1] Culled 'env1_seed_4' (depthwise, Δacc +1.38%)
[11:24:55] env0_seed_1 | Stage transition: BLENDING → SHADOWING
[11:24:55] env0_seed_1 | Stage transition: SHADOWING → CULLED
[11:24:55] env0_seed_1 | Culled (depthwise, Δacc +3.62%)
    [env0] Culled 'env0_seed_1' (depthwise, Δacc +3.62%)
[11:24:58] env1_seed_5 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_5' (depthwise, 4.8K params)
[11:24:58] env1_seed_5 | Stage transition: GERMINATED → TRAINING
[11:25:00] env0_seed_2 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_2' (norm, 0.1K params)
[11:25:00] env1_seed_5 | Stage transition: TRAINING → CULLED
[11:25:00] env1_seed_5 | Culled (depthwise, Δacc +0.00%)
    [env1] Culled 'env1_seed_5' (depthwise, Δacc +0.00%)
[11:25:00] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[11:25:01] env1_seed_6 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_6' (depthwise, 4.8K params)
[11:25:01] env1_seed_6 | Stage transition: GERMINATED → TRAINING
[11:25:08] env0_seed_2 | Stage transition: TRAINING → BLENDING
[11:25:08] env1_seed_6 | Stage transition: TRAINING → BLENDING
[11:25:10] env1_seed_6 | Stage transition: BLENDING → CULLED
[11:25:10] env1_seed_6 | Culled (depthwise, Δacc -1.97%)
    [env1] Culled 'env1_seed_6' (depthwise, Δacc -1.97%)
[11:25:12] env1_seed_7 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_7' (depthwise, 4.8K params)
[11:25:12] env1_seed_7 | Stage transition: GERMINATED → TRAINING
[11:25:17] env0_seed_2 | Stage transition: BLENDING → SHADOWING
[11:25:19] env1_seed_7 | Stage transition: TRAINING → BLENDING
[11:25:19] env1_seed_7 | Stage transition: BLENDING → CULLED
[11:25:19] env1_seed_7 | Culled (depthwise, Δacc -4.72%)
    [env1] Culled 'env1_seed_7' (depthwise, Δacc -4.72%)
[11:25:20] env0_seed_2 | Stage transition: SHADOWING → PROBATIONARY
[11:25:22] env0_seed_2 | Stage transition: PROBATIONARY → FOSSILIZED
[11:25:22] env0_seed_2 | Fossilized (norm, Δacc +4.73%)
    [env0] Fossilized 'env0_seed_2' (norm, Δacc +4.73%)
[11:25:25] env1_seed_8 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_8' (conv_enhance, 74.0K params)
[11:25:25] env1_seed_8 | Stage transition: GERMINATED → TRAINING
[11:25:36] env1_seed_8 | Stage transition: TRAINING → BLENDING
[11:25:45] env1_seed_8 | Stage transition: BLENDING → SHADOWING
[11:25:49] env1_seed_8 | Stage transition: SHADOWING → PROBATIONARY
[11:25:51] env1_seed_8 | Stage transition: PROBATIONARY → CULLED
[11:25:51] env1_seed_8 | Culled (conv_enhance, Δacc -14.86%)
    [env1] Culled 'env1_seed_8' (conv_enhance, Δacc -14.86%)
[11:25:54] env1_seed_9 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_9' (conv_enhance, 74.0K params)
[11:25:54] env1_seed_9 | Stage transition: GERMINATED → TRAINING
[11:25:57] env1_seed_9 | Stage transition: TRAINING → CULLED
[11:25:57] env1_seed_9 | Culled (conv_enhance, Δacc -2.15%)
    [env1] Culled 'env1_seed_9' (conv_enhance, Δacc -2.15%)
[11:25:59] env1_seed_10 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_10' (depthwise, 4.8K params)
[11:25:59] env1_seed_10 | Stage transition: GERMINATED → TRAINING
[11:26:03] env1_seed_10 | Stage transition: TRAINING → CULLED
[11:26:03] env1_seed_10 | Culled (depthwise, Δacc +1.41%)
    [env1] Culled 'env1_seed_10' (depthwise, Δacc +1.41%)
[11:26:09] env1_seed_11 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_11' (norm, 0.1K params)
[11:26:09] env1_seed_11 | Stage transition: GERMINATED → TRAINING
[11:26:14] env1_seed_11 | Stage transition: TRAINING → CULLED
[11:26:14] env1_seed_11 | Culled (norm, Δacc -3.33%)
    [env1] Culled 'env1_seed_11' (norm, Δacc -3.33%)
[11:26:17] env1_seed_12 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_12' (norm, 0.1K params)
[11:26:17] env1_seed_12 | Stage transition: GERMINATED → TRAINING
[11:26:26] env1_seed_12 | Stage transition: TRAINING → BLENDING
Batch 8: Episodes 16/200
  Env accuracies: ['76.5%', '74.6%']
  Avg acc: 75.5% (rolling: 72.3%)
  Avg reward: 141.2
  Actions: {'WAIT': 22, 'GERMINATE_NORM': 17, 'GERMINATE_ATTENTION': 12, 'GERMINATE_DEPTHWISE': 27, 'GERMINATE_CONV_ENHANCE': 21, 'FOSSILIZE': 24, 'CULL': 27}
  Successful: {'WAIT': 22, 'GERMINATE_NORM': 5, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 6, 'GERMINATE_CONV_ENHANCE': 5, 'FOSSILIZE': 1, 'CULL': 23}
  Policy loss: -0.0031, Value loss: 244.6480, Entropy: 1.9144, Entropy coef: 0.1797
[11:26:36] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[11:26:36] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[11:26:39] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[11:26:39] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[11:26:41] env0_seed_0 | Stage transition: TRAINING → CULLED
[11:26:41] env0_seed_0 | Culled (norm, Δacc +0.00%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +0.00%)
[11:26:42] env0_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_1' (conv_enhance, 74.0K params)
[11:26:42] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[11:26:44] env1_seed_0 | Stage transition: TRAINING → BLENDING
[11:26:46] env0_seed_1 | Stage transition: TRAINING → CULLED
[11:26:46] env0_seed_1 | Culled (conv_enhance, Δacc +2.93%)
    [env0] Culled 'env0_seed_1' (conv_enhance, Δacc +2.93%)
[11:26:49] env0_seed_2 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_2' (conv_enhance, 74.0K params)
[11:26:49] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[11:26:50] env1_seed_0 | Stage transition: BLENDING → CULLED
[11:26:50] env1_seed_0 | Culled (attention, Δacc +4.84%)
    [env1] Culled 'env1_seed_0' (attention, Δacc +4.84%)
[11:26:52] env0_seed_2 | Stage transition: TRAINING → CULLED
[11:26:52] env0_seed_2 | Culled (conv_enhance, Δacc +2.21%)
    [env0] Culled 'env0_seed_2' (conv_enhance, Δacc +2.21%)
[11:26:52] env1_seed_1 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_1' (depthwise, 4.8K params)
[11:26:52] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[11:26:54] env0_seed_3 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_3' (attention, 2.0K params)
[11:26:54] env0_seed_3 | Stage transition: GERMINATED → TRAINING
[11:26:57] env1_seed_1 | Stage transition: TRAINING → CULLED
[11:26:57] env1_seed_1 | Culled (depthwise, Δacc +4.19%)
    [env1] Culled 'env1_seed_1' (depthwise, Δacc +4.19%)
[11:26:59] env0_seed_3 | Stage transition: TRAINING → CULLED
[11:26:59] env0_seed_3 | Culled (attention, Δacc +7.92%)
    [env0] Culled 'env0_seed_3' (attention, Δacc +7.92%)
[11:27:01] env1_seed_2 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_2' (attention, 2.0K params)
[11:27:01] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[11:27:02] env0_seed_4 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_4' (depthwise, 4.8K params)
[11:27:02] env0_seed_4 | Stage transition: GERMINATED → TRAINING
[11:27:07] env1_seed_2 | Stage transition: TRAINING → BLENDING
[11:27:07] env1_seed_2 | Stage transition: BLENDING → CULLED
[11:27:07] env1_seed_2 | Culled (attention, Δacc -3.17%)
    [env1] Culled 'env1_seed_2' (attention, Δacc -3.17%)
[11:27:09] env0_seed_4 | Stage transition: TRAINING → BLENDING
[11:27:09] env1_seed_3 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_3' (conv_enhance, 74.0K params)
[11:27:09] env1_seed_3 | Stage transition: GERMINATED → TRAINING
[11:27:12] env1_seed_3 | Stage transition: TRAINING → CULLED
[11:27:12] env1_seed_3 | Culled (conv_enhance, Δacc +1.27%)
    [env1] Culled 'env1_seed_3' (conv_enhance, Δacc +1.27%)
[11:27:17] env0_seed_4 | Stage transition: BLENDING → SHADOWING
[11:27:17] env1_seed_4 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_4' (attention, 2.0K params)
[11:27:17] env1_seed_4 | Stage transition: GERMINATED → TRAINING
[11:27:21] env0_seed_4 | Stage transition: SHADOWING → PROBATIONARY
[11:27:21] env0_seed_4 | Stage transition: PROBATIONARY → CULLED
[11:27:21] env0_seed_4 | Culled (depthwise, Δacc +8.61%)
    [env0] Culled 'env0_seed_4' (depthwise, Δacc +8.61%)
[11:27:22] env0_seed_5 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_5' (norm, 0.1K params)
[11:27:22] env0_seed_5 | Stage transition: GERMINATED → TRAINING
[11:27:24] env1_seed_4 | Stage transition: TRAINING → BLENDING
[11:27:24] env0_seed_5 | Stage transition: TRAINING → CULLED
[11:27:24] env0_seed_5 | Culled (norm, Δacc +0.00%)
    [env0] Culled 'env0_seed_5' (norm, Δacc +0.00%)
[11:27:27] env0_seed_6 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_6' (conv_enhance, 74.0K params)
[11:27:27] env1_seed_4 | Stage transition: BLENDING → CULLED
[11:27:27] env1_seed_4 | Culled (attention, Δacc +4.13%)
    [env1] Culled 'env1_seed_4' (attention, Δacc +4.13%)
[11:27:27] env0_seed_6 | Stage transition: GERMINATED → TRAINING
[11:27:29] env1_seed_5 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_5' (attention, 2.0K params)
[11:27:29] env1_seed_5 | Stage transition: GERMINATED → TRAINING
[11:27:34] env0_seed_6 | Stage transition: TRAINING → BLENDING
[11:27:34] env1_seed_5 | Stage transition: TRAINING → CULLED
[11:27:34] env1_seed_5 | Culled (attention, Δacc -1.07%)
    [env1] Culled 'env1_seed_5' (attention, Δacc -1.07%)
[11:27:36] env1_seed_6 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_6' (attention, 2.0K params)
[11:27:36] env1_seed_6 | Stage transition: GERMINATED → TRAINING
[11:27:37] env1_seed_6 | Stage transition: TRAINING → CULLED
[11:27:37] env1_seed_6 | Culled (attention, Δacc +0.00%)
    [env1] Culled 'env1_seed_6' (attention, Δacc +0.00%)
[11:27:39] env1_seed_7 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_7' (norm, 0.1K params)
[11:27:39] env1_seed_7 | Stage transition: GERMINATED → TRAINING
[11:27:41] env0_seed_6 | Stage transition: BLENDING → CULLED
[11:27:41] env0_seed_6 | Culled (conv_enhance, Δacc +3.49%)
    [env0] Culled 'env0_seed_6' (conv_enhance, Δacc +3.49%)
[11:27:46] env1_seed_7 | Stage transition: TRAINING → BLENDING
[11:27:46] env1_seed_7 | Stage transition: BLENDING → CULLED
[11:27:46] env1_seed_7 | Culled (norm, Δacc +4.50%)
    [env1] Culled 'env1_seed_7' (norm, Δacc +4.50%)
[11:27:47] env0_seed_7 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_7' (conv_enhance, 74.0K params)
[11:27:47] env1_seed_8 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_8' (conv_enhance, 74.0K params)
[11:27:47] env0_seed_7 | Stage transition: GERMINATED → TRAINING
[11:27:47] env1_seed_8 | Stage transition: GERMINATED → TRAINING
[11:27:51] env0_seed_7 | Stage transition: TRAINING → CULLED
[11:27:51] env0_seed_7 | Culled (conv_enhance, Δacc -2.55%)
    [env0] Culled 'env0_seed_7' (conv_enhance, Δacc -2.55%)
[11:27:53] env0_seed_8 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_8' (norm, 0.1K params)
[11:27:53] env0_seed_8 | Stage transition: GERMINATED → TRAINING
[11:27:54] env1_seed_8 | Stage transition: TRAINING → BLENDING
[11:27:58] env1_seed_8 | Stage transition: BLENDING → CULLED
[11:27:58] env1_seed_8 | Culled (conv_enhance, Δacc +13.94%)
    [env1] Culled 'env1_seed_8' (conv_enhance, Δacc +13.94%)
[11:28:00] env0_seed_8 | Stage transition: TRAINING → BLENDING
[11:28:00] env1_seed_9 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_9' (conv_enhance, 74.0K params)
[11:28:00] env1_seed_9 | Stage transition: GERMINATED → TRAINING
[11:28:01] env1_seed_9 | Stage transition: TRAINING → CULLED
[11:28:01] env1_seed_9 | Culled (conv_enhance, Δacc +0.00%)
    [env1] Culled 'env1_seed_9' (conv_enhance, Δacc +0.00%)
[11:28:08] env0_seed_8 | Stage transition: BLENDING → SHADOWING
[11:28:08] env0_seed_8 | Stage transition: SHADOWING → CULLED
[11:28:08] env0_seed_8 | Culled (norm, Δacc +11.54%)
    [env0] Culled 'env0_seed_8' (norm, Δacc +11.54%)
[11:28:10] env0_seed_9 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_9' (attention, 2.0K params)
[11:28:10] env1_seed_10 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_10' (conv_enhance, 74.0K params)
[11:28:10] env0_seed_9 | Stage transition: GERMINATED → TRAINING
[11:28:10] env1_seed_10 | Stage transition: GERMINATED → TRAINING
[11:28:17] env1_seed_10 | Stage transition: TRAINING → BLENDING
[11:28:17] env0_seed_9 | Stage transition: TRAINING → CULLED
[11:28:17] env0_seed_9 | Culled (attention, Δacc +0.49%)
    [env0] Culled 'env0_seed_9' (attention, Δacc +0.49%)
[11:28:20] env0_seed_10 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_10' (norm, 0.1K params)
[11:28:20] env0_seed_10 | Stage transition: GERMINATED → TRAINING
[11:28:26] env1_seed_10 | Stage transition: BLENDING → SHADOWING
[11:28:26] env1_seed_10 | Stage transition: SHADOWING → CULLED
[11:28:26] env1_seed_10 | Culled (conv_enhance, Δacc -6.35%)
    [env1] Culled 'env1_seed_10' (conv_enhance, Δacc -6.35%)
[11:28:34] env0_seed_10 | Stage transition: TRAINING → CULLED
[11:28:34] env0_seed_10 | Culled (norm, Δacc -2.35%)
    [env0] Culled 'env0_seed_10' (norm, Δacc -2.35%)
[11:28:34] env1_seed_11 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_11' (norm, 0.1K params)
[11:28:34] env1_seed_11 | Stage transition: GERMINATED → TRAINING
[11:28:36] env0_seed_11 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_11' (depthwise, 4.8K params)
Batch 9: Episodes 18/200
  Env accuracies: ['76.4%', '73.8%']
  Avg acc: 75.1% (rolling: 72.6%)
  Avg reward: 128.5
  Actions: {'WAIT': 24, 'GERMINATE_NORM': 14, 'GERMINATE_ATTENTION': 20, 'GERMINATE_DEPTHWISE': 16, 'GERMINATE_CONV_ENHANCE': 22, 'FOSSILIZE': 26, 'CULL': 28}
  Successful: {'WAIT': 24, 'GERMINATE_NORM': 6, 'GERMINATE_ATTENTION': 7, 'GERMINATE_DEPTHWISE': 3, 'GERMINATE_CONV_ENHANCE': 8, 'FOSSILIZE': 0, 'CULL': 22}
  Policy loss: -0.0127, Value loss: 238.4540, Entropy: 1.8960, Entropy coef: 0.1772
[11:28:39] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[11:28:39] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[11:28:44] env0_seed_0 | Stage transition: TRAINING → CULLED
[11:28:44] env0_seed_0 | Culled (norm, Δacc -0.53%)
    [env0] Culled 'env0_seed_0' (norm, Δacc -0.53%)
[11:28:44] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[11:28:44] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[11:28:46] env0_seed_1 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_1' (attention, 2.0K params)
[11:28:46] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[11:28:52] env0_seed_1 | Stage transition: TRAINING → BLENDING
[11:28:54] env1_seed_0 | Stage transition: TRAINING → BLENDING
[11:28:54] env0_seed_1 | Stage transition: BLENDING → CULLED
[11:28:54] env0_seed_1 | Culled (attention, Δacc +15.31%)
    [env0] Culled 'env0_seed_1' (attention, Δacc +15.31%)
[11:28:54] env1_seed_0 | Stage transition: BLENDING → CULLED
[11:28:54] env1_seed_0 | Culled (norm, Δacc -3.79%)
    [env1] Culled 'env1_seed_0' (norm, Δacc -3.79%)
[11:28:56] env1_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_1' (conv_enhance, 74.0K params)
[11:28:56] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[11:28:57] env0_seed_2 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_2' (attention, 2.0K params)
[11:28:57] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[11:29:03] env1_seed_1 | Stage transition: TRAINING → BLENDING
[11:29:05] env0_seed_2 | Stage transition: TRAINING → BLENDING
[11:29:06] env1_seed_1 | Stage transition: BLENDING → CULLED
[11:29:06] env1_seed_1 | Culled (conv_enhance, Δacc +10.06%)
    [env1] Culled 'env1_seed_1' (conv_enhance, Δacc +10.06%)
[11:29:08] env1_seed_2 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_2' (attention, 2.0K params)
[11:29:08] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[11:29:13] env0_seed_2 | Stage transition: BLENDING → SHADOWING
[11:29:15] env1_seed_2 | Stage transition: TRAINING → BLENDING
[11:29:16] env0_seed_2 | Stage transition: SHADOWING → PROBATIONARY
[11:29:16] env0_seed_2 | Stage transition: PROBATIONARY → CULLED
[11:29:16] env0_seed_2 | Culled (attention, Δacc +9.83%)
    [env0] Culled 'env0_seed_2' (attention, Δacc +9.83%)
[11:29:18] env0_seed_3 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_3' (norm, 0.1K params)
[11:29:18] env0_seed_3 | Stage transition: GERMINATED → TRAINING
[11:29:23] env1_seed_2 | Stage transition: BLENDING → SHADOWING
[11:29:26] env1_seed_2 | Stage transition: SHADOWING → PROBATIONARY
[11:29:26] env1_seed_2 | Stage transition: PROBATIONARY → FOSSILIZED
[11:29:26] env1_seed_2 | Fossilized (attention, Δacc +7.68%)
    [env1] Fossilized 'env1_seed_2' (attention, Δacc +7.68%)
[11:29:33] env0_seed_3 | Stage transition: TRAINING → BLENDING
[11:29:35] env0_seed_3 | Stage transition: BLENDING → CULLED
[11:29:35] env0_seed_3 | Culled (norm, Δacc +1.22%)
    [env0] Culled 'env0_seed_3' (norm, Δacc +1.22%)
[11:29:40] env0_seed_4 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_4' (norm, 0.1K params)
[11:29:40] env0_seed_4 | Stage transition: GERMINATED → TRAINING
[11:29:46] env0_seed_4 | Stage transition: TRAINING → BLENDING
[11:29:51] env0_seed_4 | Stage transition: BLENDING → CULLED
[11:29:51] env0_seed_4 | Culled (norm, Δacc +0.35%)
    [env0] Culled 'env0_seed_4' (norm, Δacc +0.35%)
[11:29:53] env0_seed_5 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_5' (depthwise, 4.8K params)
[11:29:53] env0_seed_5 | Stage transition: GERMINATED → TRAINING
[11:29:55] env0_seed_5 | Stage transition: TRAINING → CULLED
[11:29:55] env0_seed_5 | Culled (depthwise, Δacc +0.00%)
    [env0] Culled 'env0_seed_5' (depthwise, Δacc +0.00%)
[11:30:00] env0_seed_6 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_6' (attention, 2.0K params)
[11:30:00] env0_seed_6 | Stage transition: GERMINATED → TRAINING
[11:30:08] env0_seed_6 | Stage transition: TRAINING → BLENDING
[11:30:08] env0_seed_6 | Stage transition: BLENDING → CULLED
[11:30:08] env0_seed_6 | Culled (attention, Δacc +0.66%)
    [env0] Culled 'env0_seed_6' (attention, Δacc +0.66%)
[11:30:15] env0_seed_7 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_7' (attention, 2.0K params)
[11:30:15] env0_seed_7 | Stage transition: GERMINATED → TRAINING
[11:30:26] env0_seed_7 | Stage transition: TRAINING → BLENDING
[11:30:35] env0_seed_7 | Stage transition: BLENDING → SHADOWING
[11:30:38] env0_seed_7 | Stage transition: SHADOWING → PROBATIONARY
[11:30:41] env0_seed_7 | Stage transition: PROBATIONARY → CULLED
[11:30:41] env0_seed_7 | Culled (attention, Δacc +1.44%)
    [env0] Culled 'env0_seed_7' (attention, Δacc +1.44%)
Batch 10: Episodes 20/200
  Env accuracies: ['73.5%', '69.0%']
  Avg acc: 71.3% (rolling: 72.5%)
  Avg reward: 144.2
  Actions: {'WAIT': 18, 'GERMINATE_NORM': 19, 'GERMINATE_ATTENTION': 26, 'GERMINATE_DEPTHWISE': 23, 'GERMINATE_CONV_ENHANCE': 15, 'FOSSILIZE': 30, 'CULL': 19}
  Successful: {'WAIT': 18, 'GERMINATE_NORM': 4, 'GERMINATE_ATTENTION': 5, 'GERMINATE_DEPTHWISE': 1, 'GERMINATE_CONV_ENHANCE': 1, 'FOSSILIZE': 1, 'CULL': 15}
  Policy loss: -0.0104, Value loss: 245.4015, Entropy: 1.8869, Entropy coef: 0.1747

Blueprint Stats
  ---------------------------------------------------------------------------

Blueprint       Germ  Foss  Cull   Rate     ΔAcc    Churn
  ---------------------------------------------------------------------------

  attention         33     2    30   6.2%   +4.61%   +0.97%
  conv_enhance      37     2    33   5.7%  +15.98%   +0.36%
  depthwise         27     1    25   3.8%   +4.76%   +1.70%
  norm              57     3    50   5.7%   +7.20%   +3.05%
Seed Scoreboard (env 0):
  Fossilized: 5 (+154.9K params, +163.5% of host)
  Culled: 52
  Avg fossilize age: 15.6 epochs
  Avg cull age: 5.6 epochs
  Compute cost: 1.75x baseline
  Distribution: attention x1, conv_enhance x2, depthwise x1, norm x1
Seed Scoreboard (env 1):
  Fossilized: 3 (+2.3K params, +2.4% of host)
  Culled: 86
  Avg fossilize age: 12.3 epochs
  Avg cull age: 4.7 epochs
  Compute cost: 1.39x baseline
  Distribution: norm x2, attention x1

[11:30:45] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[11:30:45] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[11:30:49] env1_seed_0 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_0' (depthwise, 4.8K params)
[11:30:50] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[11:30:51] env0_seed_0 | Stage transition: TRAINING → BLENDING
[11:30:55] env0_seed_0 | Stage transition: BLENDING → CULLED
[11:30:55] env0_seed_0 | Culled (norm, Δacc +10.90%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +10.90%)
[11:30:56] env1_seed_0 | Stage transition: TRAINING → BLENDING
[11:31:00] env0_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_1' (conv_enhance, 74.0K params)
[11:31:00] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[11:31:02] env0_seed_1 | Stage transition: TRAINING → CULLED
[11:31:02] env0_seed_1 | Culled (conv_enhance, Δacc +0.00%)
    [env0] Culled 'env0_seed_1' (conv_enhance, Δacc +0.00%)
[11:31:03] env0_seed_2 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_2' (norm, 0.1K params)
[11:31:04] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[11:31:05] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[11:31:07] env0_seed_2 | Stage transition: TRAINING → CULLED
[11:31:07] env0_seed_2 | Culled (norm, Δacc -0.16%)
    [env0] Culled 'env0_seed_2' (norm, Δacc -0.16%)
[11:31:09] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[11:31:09] env0_seed_3 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_3' (depthwise, 4.8K params)
[11:31:09] env0_seed_3 | Stage transition: GERMINATED → TRAINING
[11:31:10] env0_seed_3 | Stage transition: TRAINING → CULLED
[11:31:10] env0_seed_3 | Culled (depthwise, Δacc +0.00%)
    [env0] Culled 'env0_seed_3' (depthwise, Δacc +0.00%)
[11:31:12] env0_seed_4 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_4' (norm, 0.1K params)
[11:31:12] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[11:31:12] env1_seed_0 | Fossilized (depthwise, Δacc +18.95%)
    [env1] Fossilized 'env1_seed_0' (depthwise, Δacc +18.95%)
[11:31:12] env0_seed_4 | Stage transition: GERMINATED → TRAINING
[11:31:19] env0_seed_4 | Stage transition: TRAINING → BLENDING
[11:31:23] env0_seed_4 | Stage transition: BLENDING → CULLED
[11:31:23] env0_seed_4 | Culled (norm, Δacc +9.42%)
    [env0] Culled 'env0_seed_4' (norm, Δacc +9.42%)
[11:31:25] env0_seed_5 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_5' (depthwise, 4.8K params)
[11:31:25] env0_seed_5 | Stage transition: GERMINATED → TRAINING
[11:31:33] env0_seed_5 | Stage transition: TRAINING → BLENDING
[11:31:42] env0_seed_5 | Stage transition: BLENDING → SHADOWING
[11:31:44] env0_seed_5 | Stage transition: SHADOWING → CULLED
[11:31:44] env0_seed_5 | Culled (depthwise, Δacc -7.65%)
    [env0] Culled 'env0_seed_5' (depthwise, Δacc -7.65%)
[11:31:46] env0_seed_6 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_6' (norm, 0.1K params)
[11:31:46] env0_seed_6 | Stage transition: GERMINATED → TRAINING
[11:31:47] env0_seed_6 | Stage transition: TRAINING → CULLED
[11:31:47] env0_seed_6 | Culled (norm, Δacc +0.00%)
    [env0] Culled 'env0_seed_6' (norm, Δacc +0.00%)
[11:31:49] env0_seed_7 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_7' (norm, 0.1K params)
[11:31:49] env0_seed_7 | Stage transition: GERMINATED → TRAINING
[11:31:58] env0_seed_7 | Stage transition: TRAINING → BLENDING
[11:32:00] env0_seed_7 | Stage transition: BLENDING → CULLED
[11:32:00] env0_seed_7 | Culled (norm, Δacc +4.28%)
    [env0] Culled 'env0_seed_7' (norm, Δacc +4.28%)
[11:32:03] env0_seed_8 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_8' (norm, 0.1K params)
[11:32:03] env0_seed_8 | Stage transition: GERMINATED → TRAINING
[11:32:10] env0_seed_8 | Stage transition: TRAINING → BLENDING
[11:32:19] env0_seed_8 | Stage transition: BLENDING → SHADOWING
[11:32:23] env0_seed_8 | Stage transition: SHADOWING → PROBATIONARY
[11:32:23] env0_seed_8 | Stage transition: PROBATIONARY → CULLED
[11:32:23] env0_seed_8 | Culled (norm, Δacc +4.71%)
    [env0] Culled 'env0_seed_8' (norm, Δacc +4.71%)
[11:32:24] env0_seed_9 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_9' (attention, 2.0K params)
[11:32:24] env0_seed_9 | Stage transition: GERMINATED → TRAINING
[11:32:31] env0_seed_9 | Stage transition: TRAINING → BLENDING
[11:32:38] env0_seed_9 | Stage transition: BLENDING → CULLED
[11:32:38] env0_seed_9 | Culled (attention, Δacc +8.52%)
    [env0] Culled 'env0_seed_9' (attention, Δacc +8.52%)
[11:32:40] env0_seed_10 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_10' (attention, 2.0K params)
[11:32:40] env0_seed_10 | Stage transition: GERMINATED → TRAINING
[11:32:47] env0_seed_10 | Stage transition: TRAINING → BLENDING
Batch 11: Episodes 22/200
  Env accuracies: ['68.8%', '71.4%']
  Avg acc: 70.1% (rolling: 72.3%)
  Avg reward: 141.0
  Actions: {'WAIT': 12, 'GERMINATE_NORM': 23, 'GERMINATE_ATTENTION': 24, 'GERMINATE_DEPTHWISE': 21, 'GERMINATE_CONV_ENHANCE': 17, 'FOSSILIZE': 34, 'CULL': 19}
  Successful: {'WAIT': 12, 'GERMINATE_NORM': 6, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 3, 'GERMINATE_CONV_ENHANCE': 1, 'FOSSILIZE': 1, 'CULL': 18}
  Policy loss: -0.0255, Value loss: 266.3406, Entropy: 1.8657, Entropy coef: 0.1721
[11:32:56] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[11:32:56] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[11:32:59] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[11:32:59] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[11:33:02] env1_seed_0 | Stage transition: TRAINING → BLENDING
[11:33:06] env0_seed_0 | Stage transition: TRAINING → BLENDING
[11:33:06] env1_seed_0 | Stage transition: BLENDING → CULLED
[11:33:06] env1_seed_0 | Culled (norm, Δacc +12.27%)
    [env1] Culled 'env1_seed_0' (norm, Δacc +12.27%)
[11:33:14] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[11:33:17] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[11:33:17] env1_seed_1 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_1' (depthwise, 4.8K params)
[11:33:17] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[11:33:19] env0_seed_0 | Stage transition: PROBATIONARY → CULLED
[11:33:19] env0_seed_0 | Culled (norm, Δacc +12.47%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +12.47%)
[11:33:21] env1_seed_1 | Stage transition: TRAINING → CULLED
[11:33:21] env1_seed_1 | Culled (depthwise, Δacc -0.25%)
    [env1] Culled 'env1_seed_1' (depthwise, Δacc -0.25%)
[11:33:22] env1_seed_2 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_2' (attention, 2.0K params)
[11:33:22] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[11:33:24] env0_seed_1 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_1' (depthwise, 4.8K params)
[11:33:24] env0_seed_1 | Stage transition: GERMINATED → TRAINING
