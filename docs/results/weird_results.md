(.venv) john@nyx:~/esper-lite$ uv run python -m esper.scripts.train ppo     --vectorized     --n-envs 4    --devices cuda:0 cuda:1     --episodes 200     --entropy-coef-start 0.2     --entropy-coef-end 0.01     --entropy-anneal-episodes 150 --num-workers 4 --max-epochs 75
============================================================

PPO Vectorized Training (INVERTED CONTROL FLOW + CUDA STREAMS)
============================================================

Task: cifar10 (topology=cnn, type=classification)
Episodes: 200 (across 4 parallel envs)
Max epochs per episode: 75
Policy device: cuda:0
Env devices: ['cuda:0', 'cuda:1'] (2 envs per device)
Random seed: 42
Entropy annealing: 0.2 -> 0.01 over 150 episodes
Learning rate: 0.0003
Telemetry features: ENABLED

Loading cifar10 (4 independent DataLoaders)...
[00:15:43] env0_seed_0 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_0' (attention, 2.0K params)
[00:15:43] env3_seed_0 | Germinated (conv_enhance, 74.0K params)
    [env3] Germinated 'env3_seed_0' (conv_enhance, 74.0K params)
[00:15:43] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[00:15:43] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[00:15:47] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[00:15:47] env2_seed_0 | Germinated (conv_enhance, 74.0K params)
    [env2] Germinated 'env2_seed_0' (conv_enhance, 74.0K params)
[00:15:47] env3_seed_0 | Stage transition: TRAINING → CULLED
[00:15:47] env3_seed_0 | Culled (conv_enhance, Δacc +0.00%)
    [env3] Culled 'env3_seed_0' (conv_enhance, Δacc +0.00%)
[00:15:48] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[00:15:48] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[00:15:52] env0_seed_0 | Stage transition: TRAINING → CULLED
[00:15:52] env0_seed_0 | Culled (attention, Δacc +3.13%)
    [env0] Culled 'env0_seed_0' (attention, Δacc +3.13%)
[00:15:52] env3_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env3] Germinated 'env3_seed_1' (conv_enhance, 74.0K params)
[00:15:52] env3_seed_1 | Stage transition: GERMINATED → TRAINING
[00:15:57] env0_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_1' (conv_enhance, 74.0K params)
[00:15:57] env1_seed_0 | Stage transition: TRAINING → CULLED
[00:15:57] env1_seed_0 | Culled (attention, Δacc -2.62%)
    [env1] Culled 'env1_seed_0' (attention, Δacc -2.62%)
[00:15:57] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[00:16:02] env1_seed_1 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_1' (depthwise, 4.8K params)
[00:16:02] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[00:16:15] env2_seed_0 | Stage transition: TRAINING → BLENDING
[00:16:21] env0_seed_1 | Stage transition: TRAINING → BLENDING
[00:16:21] env0_seed_1 | Stage transition: BLENDING → CULLED
[00:16:21] env0_seed_1 | Culled (conv_enhance, Δacc +5.88%)
    [env0] Culled 'env0_seed_1' (conv_enhance, Δacc +5.88%)
[00:16:26] env1_seed_1 | Stage transition: TRAINING → BLENDING
[00:16:26] env3_seed_1 | Stage transition: TRAINING → BLENDING
[00:16:31] env0_seed_2 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_2' (attention, 2.0K params)
[00:16:31] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[00:16:43] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[00:16:48] env1_seed_1 | Stage transition: BLENDING → CULLED
[00:16:48] env1_seed_1 | Culled (depthwise, Δacc +5.60%)
    [env1] Culled 'env1_seed_1' (depthwise, Δacc +5.60%)
[00:16:53] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[00:16:53] env3_seed_1 | Stage transition: BLENDING → SHADOWING
[00:16:53] env1_seed_2 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_2' (norm, 0.1K params)
[00:16:53] env3_seed_1 | Stage transition: SHADOWING → CULLED
[00:16:53] env3_seed_1 | Culled (conv_enhance, Δacc +15.70%)
    [env3] Culled 'env3_seed_1' (conv_enhance, Δacc +15.70%)
[00:16:53] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[00:16:57] env0_seed_2 | Stage transition: TRAINING → BLENDING
[00:16:57] env0_seed_2 | Stage transition: BLENDING → CULLED
[00:16:57] env0_seed_2 | Culled (attention, Δacc +8.10%)
    [env0] Culled 'env0_seed_2' (attention, Δacc +8.10%)
[00:17:01] env0_seed_3 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_3' (attention, 2.0K params)
[00:17:01] env0_seed_3 | Stage transition: GERMINATED → TRAINING
[00:17:05] env2_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[00:17:05] env2_seed_0 | Fossilized (conv_enhance, Δacc +16.78%)
    [env2] Fossilized 'env2_seed_0' (conv_enhance, Δacc +16.78%)
[00:17:05] env3_seed_2 | Germinated (attention, 2.0K params)
    [env3] Germinated 'env3_seed_2' (attention, 2.0K params)
[00:17:05] env3_seed_2 | Stage transition: GERMINATED → TRAINING
[00:17:10] env1_seed_2 | Stage transition: TRAINING → BLENDING
[00:17:15] env2_seed_0 | Stage transition: FOSSILIZED → CULLED
[00:17:15] env2_seed_0 | Culled (conv_enhance, Δacc +18.86%)
    [env2] Culled 'env2_seed_0' (conv_enhance, Δacc +18.86%)
[00:17:19] env2_seed_1 | Germinated (norm, 0.1K params)
    [env2] Germinated 'env2_seed_1' (norm, 0.1K params)
[00:17:19] env2_seed_1 | Stage transition: GERMINATED → TRAINING
[00:17:25] env3_seed_2 | Stage transition: TRAINING → BLENDING
[00:17:25] env3_seed_2 | Stage transition: BLENDING → CULLED
[00:17:25] env3_seed_2 | Culled (attention, Δacc +13.03%)
    [env3] Culled 'env3_seed_2' (attention, Δacc +13.03%)
[00:17:30] env0_seed_3 | Stage transition: TRAINING → BLENDING
[00:17:30] env1_seed_2 | Stage transition: BLENDING → CULLED
[00:17:30] env1_seed_2 | Culled (norm, Δacc +6.63%)
    [env1] Culled 'env1_seed_2' (norm, Δacc +6.63%)
[00:17:30] env3_seed_3 | Germinated (conv_enhance, 74.0K params)
    [env3] Germinated 'env3_seed_3' (conv_enhance, 74.0K params)
[00:17:30] env3_seed_3 | Stage transition: GERMINATED → TRAINING
[00:17:34] env0_seed_3 | Stage transition: BLENDING → CULLED
[00:17:34] env0_seed_3 | Culled (attention, Δacc +2.35%)
    [env0] Culled 'env0_seed_3' (attention, Δacc +2.35%)
[00:17:39] env2_seed_1 | Stage transition: TRAINING → BLENDING
[00:17:39] env0_seed_4 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_4' (attention, 2.0K params)
[00:17:39] env1_seed_3 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_3' (conv_enhance, 74.0K params)
[00:17:39] env0_seed_4 | Stage transition: GERMINATED → TRAINING
[00:17:39] env1_seed_3 | Stage transition: GERMINATED → TRAINING
[00:17:51] env1_seed_3 | Stage transition: TRAINING → CULLED
[00:17:51] env1_seed_3 | Culled (conv_enhance, Δacc +0.33%)
    [env1] Culled 'env1_seed_3' (conv_enhance, Δacc +0.33%)
[00:17:51] env2_seed_1 | Stage transition: BLENDING → CULLED
[00:17:51] env2_seed_1 | Culled (norm, Δacc +7.44%)
    [env2] Culled 'env2_seed_1' (norm, Δacc +7.44%)
[00:17:56] env1_seed_4 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_4' (norm, 0.1K params)
[00:17:56] env2_seed_2 | Germinated (attention, 2.0K params)
    [env2] Germinated 'env2_seed_2' (attention, 2.0K params)
[00:17:56] env1_seed_4 | Stage transition: GERMINATED → TRAINING
[00:17:56] env2_seed_2 | Stage transition: GERMINATED → TRAINING
[00:18:01] env3_seed_3 | Stage transition: TRAINING → BLENDING
[00:18:25] env1_seed_4 | Stage transition: TRAINING → BLENDING
[00:18:31] env2_seed_2 | Stage transition: TRAINING → BLENDING
[00:18:31] env3_seed_3 | Stage transition: BLENDING → SHADOWING
[00:18:31] env0_seed_4 | Stage transition: TRAINING → CULLED
[00:18:31] env0_seed_4 | Culled (attention, Δacc +0.55%)
    [env0] Culled 'env0_seed_4' (attention, Δacc +0.55%)
[00:18:31] env2_seed_2 | Stage transition: BLENDING → CULLED
[00:18:31] env2_seed_2 | Culled (attention, Δacc -3.51%)
    [env2] Culled 'env2_seed_2' (attention, Δacc -3.51%)
[00:18:35] env2_seed_3 | Germinated (norm, 0.1K params)
    [env2] Germinated 'env2_seed_3' (norm, 0.1K params)
[00:18:35] env2_seed_3 | Stage transition: GERMINATED → TRAINING
[00:18:39] env3_seed_3 | Stage transition: SHADOWING → PROBATIONARY
[00:18:39] env0_seed_5 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_5' (norm, 0.1K params)
[00:18:39] env3_seed_3 | Stage transition: PROBATIONARY → CULLED
[00:18:39] env3_seed_3 | Culled (conv_enhance, Δacc +7.33%)
    [env3] Culled 'env3_seed_3' (conv_enhance, Δacc +7.33%)
[00:18:39] env0_seed_5 | Stage transition: GERMINATED → TRAINING
[00:18:44] env3_seed_4 | Germinated (attention, 2.0K params)
    [env3] Germinated 'env3_seed_4' (attention, 2.0K params)
[00:18:44] env3_seed_4 | Stage transition: GERMINATED → TRAINING
[00:18:50] env1_seed_4 | Stage transition: BLENDING → SHADOWING
[00:18:50] env3_seed_4 | Stage transition: TRAINING → CULLED
[00:18:50] env3_seed_4 | Culled (attention, Δacc +0.00%)
    [env3] Culled 'env3_seed_4' (attention, Δacc +0.00%)
[00:18:53] env2_seed_3 | Stage transition: TRAINING → BLENDING
[00:18:53] env3_seed_5 | Germinated (depthwise, 4.8K params)
    [env3] Germinated 'env3_seed_5' (depthwise, 4.8K params)
[00:18:53] env3_seed_5 | Stage transition: GERMINATED → TRAINING
[00:18:58] env1_seed_4 | Stage transition: SHADOWING → PROBATIONARY
[00:19:07] env0_seed_5 | Stage transition: TRAINING → BLENDING
[00:19:12] env1_seed_4 | Stage transition: PROBATIONARY → FOSSILIZED
[00:19:12] env1_seed_4 | Fossilized (norm, Δacc +7.60%)
    [env1] Fossilized 'env1_seed_4' (norm, Δacc +7.60%)
[00:19:16] env2_seed_3 | Stage transition: BLENDING → SHADOWING
[00:19:25] env2_seed_3 | Stage transition: SHADOWING → PROBATIONARY
[00:19:25] env3_seed_5 | Stage transition: TRAINING → BLENDING
[00:19:30] env0_seed_5 | Stage transition: BLENDING → SHADOWING
[00:19:30] env1_seed_4 | Stage transition: FOSSILIZED → CULLED
[00:19:30] env1_seed_4 | Culled (norm, Δacc +8.74%)
    [env1] Culled 'env1_seed_4' (norm, Δacc +8.74%)
[00:19:33] env0_seed_5 | Stage transition: SHADOWING → CULLED
[00:19:33] env0_seed_5 | Culled (norm, Δacc +1.14%)
    [env0] Culled 'env0_seed_5' (norm, Δacc +1.14%)
[00:19:33] env1_seed_5 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_5' (depthwise, 4.8K params)
[00:19:33] env1_seed_5 | Stage transition: GERMINATED → TRAINING
[00:19:37] env0_seed_6 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_6' (attention, 2.0K params)
[00:19:37] env0_seed_6 | Stage transition: GERMINATED → TRAINING
[00:19:47] env3_seed_5 | Stage transition: BLENDING → SHADOWING
[00:19:47] env1_seed_5 | Stage transition: TRAINING → CULLED
[00:19:47] env1_seed_5 | Culled (depthwise, Δacc -0.63%)
    [env1] Culled 'env1_seed_5' (depthwise, Δacc -0.63%)
[00:19:47] env2_seed_3 | Stage transition: PROBATIONARY → CULLED
[00:19:47] env2_seed_3 | Culled (norm, Δacc +6.21%)
    [env2] Culled 'env2_seed_3' (norm, Δacc +6.21%)
[00:19:50] env0_seed_6 | Stage transition: TRAINING → CULLED
[00:19:50] env0_seed_6 | Culled (attention, Δacc -1.72%)
    [env0] Culled 'env0_seed_6' (attention, Δacc -1.72%)
[00:19:50] env2_seed_4 | Germinated (attention, 2.0K params)
    [env2] Germinated 'env2_seed_4' (attention, 2.0K params)
[00:19:50] env3_seed_5 | Stage transition: SHADOWING → CULLED
[00:19:50] env3_seed_5 | Culled (depthwise, Δacc -2.86%)
    [env3] Culled 'env3_seed_5' (depthwise, Δacc -2.86%)
[00:19:50] env2_seed_4 | Stage transition: GERMINATED → TRAINING
[00:19:53] env3_seed_6 | Germinated (conv_enhance, 74.0K params)
    [env3] Germinated 'env3_seed_6' (conv_enhance, 74.0K params)
[00:19:53] env3_seed_6 | Stage transition: GERMINATED → TRAINING
[00:19:58] env1_seed_6 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_6' (depthwise, 4.8K params)
[00:19:58] env1_seed_6 | Stage transition: GERMINATED → TRAINING
[00:20:07] env0_seed_7 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_7' (attention, 2.0K params)
[00:20:07] env2_seed_4 | Stage transition: TRAINING → CULLED
[00:20:07] env2_seed_4 | Culled (attention, Δacc -5.30%)
    [env2] Culled 'env2_seed_4' (attention, Δacc -5.30%)
[00:20:07] env0_seed_7 | Stage transition: GERMINATED → TRAINING
[00:20:12] env3_seed_6 | Stage transition: TRAINING → BLENDING
[00:20:12] env1_seed_6 | Stage transition: TRAINING → CULLED
[00:20:12] env1_seed_6 | Culled (depthwise, Δacc +1.57%)
    [env1] Culled 'env1_seed_6' (depthwise, Δacc +1.57%)
[00:20:12] env2_seed_5 | Germinated (norm, 0.1K params)
    [env2] Germinated 'env2_seed_5' (norm, 0.1K params)
[00:20:12] env3_seed_6 | Stage transition: BLENDING → CULLED
[00:20:12] env3_seed_6 | Culled (conv_enhance, Δacc +2.37%)
    [env3] Culled 'env3_seed_6' (conv_enhance, Δacc +2.37%)
[00:20:12] env2_seed_5 | Stage transition: GERMINATED → TRAINING
[00:20:16] env1_seed_7 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_7' (attention, 2.0K params)
[00:20:16] env2_seed_5 | Stage transition: TRAINING → CULLED
[00:20:16] env2_seed_5 | Culled (norm, Δacc +0.00%)
    [env2] Culled 'env2_seed_5' (norm, Δacc +0.00%)
[00:20:16] env1_seed_7 | Stage transition: GERMINATED → TRAINING
[00:20:29] env1_seed_7 | Stage transition: TRAINING → CULLED
[00:20:29] env1_seed_7 | Culled (attention, Δacc -2.36%)
    [env1] Culled 'env1_seed_7' (attention, Δacc -2.36%)
[00:20:32] env1_seed_8 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_8' (depthwise, 4.8K params)
[00:20:32] env3_seed_7 | Germinated (depthwise, 4.8K params)
    [env3] Germinated 'env3_seed_7' (depthwise, 4.8K params)
[00:20:32] env1_seed_8 | Stage transition: GERMINATED → TRAINING
[00:20:32] env3_seed_7 | Stage transition: GERMINATED → TRAINING
[00:20:36] env2_seed_6 | Germinated (depthwise, 4.8K params)
    [env2] Germinated 'env2_seed_6' (depthwise, 4.8K params)
[00:20:37] env2_seed_6 | Stage transition: GERMINATED → TRAINING
[00:20:42] env2_seed_6 | Stage transition: TRAINING → CULLED
[00:20:42] env2_seed_6 | Culled (depthwise, Δacc +0.00%)
    [env2] Culled 'env2_seed_6' (depthwise, Δacc +0.00%)
[00:20:52] env1_seed_8 | Stage transition: TRAINING → BLENDING
[00:20:52] env2_seed_7 | Germinated (conv_enhance, 74.0K params)
    [env2] Germinated 'env2_seed_7' (conv_enhance, 74.0K params)
[00:20:52] env2_seed_7 | Stage transition: GERMINATED → TRAINING
[00:20:58] env3_seed_7 | Stage transition: TRAINING → BLENDING
[00:20:58] env1_seed_8 | Stage transition: BLENDING → CULLED
[00:20:58] env1_seed_8 | Culled (depthwise, Δacc -1.92%)
    [env1] Culled 'env1_seed_8' (depthwise, Δacc -1.92%)
[00:21:08] env1_seed_9 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_9' (norm, 0.1K params)
[00:21:08] env1_seed_9 | Stage transition: GERMINATED → TRAINING
[00:21:14] env1_seed_9 | Stage transition: TRAINING → CULLED
[00:21:14] env1_seed_9 | Culled (norm, Δacc +0.00%)
    [env1] Culled 'env1_seed_9' (norm, Δacc +0.00%)
[00:21:19] env2_seed_7 | Stage transition: TRAINING → BLENDING
[00:21:23] env3_seed_7 | Stage transition: BLENDING → SHADOWING
[00:21:27] env1_seed_10 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_10' (depthwise, 4.8K params)
[00:21:27] env1_seed_10 | Stage transition: GERMINATED → TRAINING
[00:21:32] env0_seed_7 | Stage transition: TRAINING → BLENDING
[00:21:32] env3_seed_7 | Stage transition: SHADOWING → PROBATIONARY
[00:21:32] env0_seed_7 | Stage transition: BLENDING → CULLED
[00:21:32] env0_seed_7 | Culled (attention, Δacc -0.74%)
    [env0] Culled 'env0_seed_7' (attention, Δacc -0.74%)
[00:21:36] env0_seed_8 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_8' (depthwise, 4.8K params)
[00:21:37] env0_seed_8 | Stage transition: GERMINATED → TRAINING
[00:21:41] env2_seed_7 | Stage transition: BLENDING → SHADOWING
Batch 1: Episodes 4/200
  Env accuracies: ['72.7%', '71.1%', '74.8%', '72.5%']
  Avg acc: 72.7% (rolling: 72.7%)
  Avg reward: -2.4
  Actions: {'WAIT': 42, 'GERMINATE_NORM': 46, 'GERMINATE_ATTENTION': 47, 'GERMINATE_DEPTHWISE': 39, 'GERMINATE_CONV_ENHANCE': 47, 'FOSSILIZE': 40, 'CULL': 39}
  Successful: {'WAIT': 42, 'GERMINATE_NORM': 7, 'GERMINATE_ATTENTION': 12, 'GERMINATE_DEPTHWISE': 9, 'GERMINATE_CONV_ENHANCE': 8, 'FOSSILIZE': 2, 'CULL': 32}
  Policy loss: -0.0169, Value loss: 79.2362, Entropy: 1.9405, Entropy coef: 0.1949
[00:21:44] env0_seed_0 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_0' (attention, 2.0K params)
[00:21:44] env2_seed_0 | Germinated (depthwise, 4.8K params)
    [env2] Germinated 'env2_seed_0' (depthwise, 4.8K params)
[00:21:44] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[00:21:44] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[00:21:48] env3_seed_0 | Germinated (conv_enhance, 74.0K params)
    [env3] Germinated 'env3_seed_0' (conv_enhance, 74.0K params)
[00:21:48] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[00:21:52] env2_seed_0 | Stage transition: TRAINING → CULLED
[00:21:52] env2_seed_0 | Culled (depthwise, Δacc +1.16%)
    [env2] Culled 'env2_seed_0' (depthwise, Δacc +1.16%)
[00:21:57] env2_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env2] Germinated 'env2_seed_1' (conv_enhance, 74.0K params)
[00:21:57] env2_seed_1 | Stage transition: GERMINATED → TRAINING
[00:22:02] env0_seed_0 | Stage transition: TRAINING → BLENDING
[00:22:02] env1_seed_0 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_0' (depthwise, 4.8K params)
[00:22:02] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[00:22:08] env3_seed_0 | Stage transition: TRAINING → BLENDING
[00:22:08] env3_seed_0 | Stage transition: BLENDING → CULLED
[00:22:08] env3_seed_0 | Culled (conv_enhance, Δacc -0.59%)
    [env3] Culled 'env3_seed_0' (conv_enhance, Δacc -0.59%)
[00:22:17] env2_seed_1 | Stage transition: TRAINING → CULLED
[00:22:17] env2_seed_1 | Culled (conv_enhance, Δacc +1.55%)
    [env2] Culled 'env2_seed_1' (conv_enhance, Δacc +1.55%)
[00:22:17] env3_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env3] Germinated 'env3_seed_1' (conv_enhance, 74.0K params)
[00:22:18] env3_seed_1 | Stage transition: GERMINATED → TRAINING
[00:22:22] env2_seed_2 | Germinated (norm, 0.1K params)
    [env2] Germinated 'env2_seed_2' (norm, 0.1K params)
[00:22:22] env2_seed_2 | Stage transition: GERMINATED → TRAINING
[00:22:28] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[00:22:33] env1_seed_0 | Stage transition: TRAINING → BLENDING
[00:22:33] env2_seed_2 | Stage transition: TRAINING → CULLED
[00:22:33] env2_seed_2 | Culled (norm, Δacc +1.59%)
    [env2] Culled 'env2_seed_2' (norm, Δacc +1.59%)
[00:22:37] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[00:22:37] env0_seed_0 | Stage transition: PROBATIONARY → CULLED
[00:22:37] env0_seed_0 | Culled (attention, Δacc +16.67%)
    [env0] Culled 'env0_seed_0' (attention, Δacc +16.67%)
[00:22:45] env0_seed_1 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_1' (norm, 0.1K params)
[00:22:45] env2_seed_3 | Germinated (attention, 2.0K params)
    [env2] Germinated 'env2_seed_3' (attention, 2.0K params)
[00:22:45] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[00:22:45] env2_seed_3 | Stage transition: GERMINATED → TRAINING
[00:22:57] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[00:22:57] env3_seed_1 | Stage transition: TRAINING → BLENDING
[00:22:57] env1_seed_0 | Stage transition: SHADOWING → CULLED
[00:22:57] env1_seed_0 | Culled (depthwise, Δacc +3.89%)
    [env1] Culled 'env1_seed_0' (depthwise, Δacc +3.89%)
[00:23:02] env0_seed_1 | Stage transition: TRAINING → CULLED
[00:23:02] env0_seed_1 | Culled (norm, Δacc -3.83%)
    [env0] Culled 'env0_seed_1' (norm, Δacc -3.83%)
[00:23:06] env2_seed_3 | Stage transition: TRAINING → BLENDING
[00:23:06] env0_seed_2 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_2' (conv_enhance, 74.0K params)
[00:23:06] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[00:23:11] env1_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_1' (conv_enhance, 74.0K params)
[00:23:11] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[00:23:17] env0_seed_2 | Stage transition: TRAINING → CULLED
[00:23:17] env0_seed_2 | Culled (conv_enhance, Δacc +3.20%)
    [env0] Culled 'env0_seed_2' (conv_enhance, Δacc +3.20%)
[00:23:22] env3_seed_1 | Stage transition: BLENDING → SHADOWING
[00:23:22] env2_seed_3 | Stage transition: BLENDING → CULLED
[00:23:22] env2_seed_3 | Culled (attention, Δacc +6.49%)
    [env2] Culled 'env2_seed_3' (attention, Δacc +6.49%)
[00:23:26] env0_seed_3 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_3' (conv_enhance, 74.0K params)
[00:23:26] env2_seed_4 | Germinated (attention, 2.0K params)
    [env2] Germinated 'env2_seed_4' (attention, 2.0K params)
[00:23:26] env0_seed_3 | Stage transition: GERMINATED → TRAINING
[00:23:26] env2_seed_4 | Stage transition: GERMINATED → TRAINING
[00:23:31] env3_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[00:23:36] env1_seed_1 | Stage transition: TRAINING → BLENDING
[00:23:36] env3_seed_1 | Stage transition: PROBATIONARY → CULLED
[00:23:36] env3_seed_1 | Culled (conv_enhance, Δacc +5.38%)
    [env3] Culled 'env3_seed_1' (conv_enhance, Δacc +5.38%)
[00:23:40] env3_seed_2 | Germinated (conv_enhance, 74.0K params)
    [env3] Germinated 'env3_seed_2' (conv_enhance, 74.0K params)
[00:23:41] env3_seed_2 | Stage transition: GERMINATED → TRAINING
[00:23:47] env0_seed_3 | Stage transition: TRAINING → BLENDING
[00:23:53] env2_seed_4 | Stage transition: TRAINING → BLENDING
[00:24:05] env1_seed_1 | Stage transition: BLENDING → SHADOWING
[00:24:05] env3_seed_2 | Stage transition: TRAINING → BLENDING
[00:24:05] env1_seed_1 | Stage transition: SHADOWING → CULLED
[00:24:05] env1_seed_1 | Culled (conv_enhance, Δacc +5.08%)
    [env1] Culled 'env1_seed_1' (conv_enhance, Δacc +5.08%)
[00:24:10] env0_seed_3 | Stage transition: BLENDING → CULLED
[00:24:10] env0_seed_3 | Culled (conv_enhance, Δacc +8.71%)
    [env0] Culled 'env0_seed_3' (conv_enhance, Δacc +8.71%)
[00:24:10] env1_seed_2 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_2' (conv_enhance, 74.0K params)
[00:24:10] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[00:24:20] env2_seed_4 | Stage transition: BLENDING → SHADOWING
[00:24:20] env0_seed_4 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_4' (conv_enhance, 74.0K params)
[00:24:20] env0_seed_4 | Stage transition: GERMINATED → TRAINING
[00:24:31] env1_seed_2 | Stage transition: TRAINING → BLENDING
[00:24:31] env2_seed_4 | Stage transition: SHADOWING → PROBATIONARY
[00:24:31] env3_seed_2 | Stage transition: BLENDING → SHADOWING
[00:24:31] env0_seed_4 | Stage transition: TRAINING → CULLED
[00:24:31] env0_seed_4 | Culled (conv_enhance, Δacc -1.14%)
    [env0] Culled 'env0_seed_4' (conv_enhance, Δacc -1.14%)
[00:24:34] env0_seed_5 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_5' (attention, 2.0K params)
[00:24:34] env0_seed_5 | Stage transition: GERMINATED → TRAINING
[00:24:39] env3_seed_2 | Stage transition: SHADOWING → PROBATIONARY
[00:24:39] env1_seed_2 | Stage transition: BLENDING → CULLED
[00:24:39] env1_seed_2 | Culled (conv_enhance, Δacc +6.89%)
    [env1] Culled 'env1_seed_2' (conv_enhance, Δacc +6.89%)
[00:24:42] env3_seed_2 | Stage transition: PROBATIONARY → FOSSILIZED
[00:24:42] env3_seed_2 | Fossilized (conv_enhance, Δacc +8.86%)
    [env3] Fossilized 'env3_seed_2' (conv_enhance, Δacc +8.86%)
[00:24:45] env1_seed_3 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_3' (depthwise, 4.8K params)
[00:24:45] env2_seed_4 | Stage transition: PROBATIONARY → FOSSILIZED
[00:24:45] env2_seed_4 | Fossilized (attention, Δacc +4.26%)
    [env2] Fossilized 'env2_seed_4' (attention, Δacc +4.26%)
[00:24:45] env1_seed_3 | Stage transition: GERMINATED → TRAINING
[00:24:49] env0_seed_5 | Stage transition: TRAINING → BLENDING
[00:24:54] env3_seed_2 | Stage transition: FOSSILIZED → CULLED
[00:24:54] env3_seed_2 | Culled (conv_enhance, Δacc +6.43%)
    [env3] Culled 'env3_seed_2' (conv_enhance, Δacc +6.43%)
[00:25:03] env3_seed_3 | Germinated (conv_enhance, 74.0K params)
    [env3] Germinated 'env3_seed_3' (conv_enhance, 74.0K params)
[00:25:03] env3_seed_3 | Stage transition: GERMINATED → TRAINING
[00:25:12] env0_seed_5 | Stage transition: BLENDING → SHADOWING
[00:25:20] env0_seed_5 | Stage transition: SHADOWING → PROBATIONARY
[00:25:20] env3_seed_3 | Stage transition: TRAINING → BLENDING
[00:25:24] env1_seed_3 | Stage transition: TRAINING → CULLED
[00:25:24] env1_seed_3 | Culled (depthwise, Δacc -1.55%)
    [env1] Culled 'env1_seed_3' (depthwise, Δacc -1.55%)
[00:25:28] env1_seed_4 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_4' (norm, 0.1K params)
[00:25:28] env1_seed_4 | Stage transition: GERMINATED → TRAINING
[00:25:31] env3_seed_3 | Stage transition: BLENDING → CULLED
[00:25:31] env3_seed_3 | Culled (conv_enhance, Δacc +6.15%)
    [env3] Culled 'env3_seed_3' (conv_enhance, Δacc +6.15%)
[00:25:35] env3_seed_4 | Germinated (norm, 0.1K params)
    [env3] Germinated 'env3_seed_4' (norm, 0.1K params)
[00:25:35] env3_seed_4 | Stage transition: GERMINATED → TRAINING
[00:25:42] env1_seed_4 | Stage transition: TRAINING → BLENDING
[00:25:50] env3_seed_4 | Stage transition: TRAINING → BLENDING
[00:25:50] env3_seed_4 | Stage transition: BLENDING → CULLED
[00:25:50] env3_seed_4 | Culled (norm, Δacc -8.14%)
    [env3] Culled 'env3_seed_4' (norm, Δacc -8.14%)
[00:25:53] env3_seed_5 | Germinated (conv_enhance, 74.0K params)
    [env3] Germinated 'env3_seed_5' (conv_enhance, 74.0K params)
[00:25:53] env3_seed_5 | Stage transition: GERMINATED → TRAINING
[00:25:57] env0_seed_5 | Stage transition: PROBATIONARY → CULLED
[00:25:57] env0_seed_5 | Culled (attention, Δacc -0.01%)
    [env0] Culled 'env0_seed_5' (attention, Δacc -0.01%)
[00:26:01] env1_seed_4 | Stage transition: BLENDING → SHADOWING
[00:26:01] env0_seed_6 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_6' (norm, 0.1K params)
[00:26:01] env0_seed_6 | Stage transition: GERMINATED → TRAINING
[00:26:10] env1_seed_4 | Stage transition: SHADOWING → PROBATIONARY
[00:26:14] env3_seed_5 | Stage transition: TRAINING → BLENDING
[00:26:19] env0_seed_6 | Stage transition: TRAINING → CULLED
[00:26:19] env0_seed_6 | Culled (norm, Δacc +0.97%)
    [env0] Culled 'env0_seed_6' (norm, Δacc +0.97%)
[00:26:22] env2_seed_4 | Stage transition: FOSSILIZED → CULLED
[00:26:22] env2_seed_4 | Culled (attention, Δacc +7.43%)
    [env2] Culled 'env2_seed_4' (attention, Δacc +7.43%)
[00:26:26] env0_seed_7 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_7' (conv_enhance, 74.0K params)
[00:26:26] env1_seed_4 | Stage transition: PROBATIONARY → CULLED
[00:26:26] env1_seed_4 | Culled (norm, Δacc +4.33%)
    [env1] Culled 'env1_seed_4' (norm, Δacc +4.33%)
[00:26:26] env0_seed_7 | Stage transition: GERMINATED → TRAINING
[00:26:31] env2_seed_5 | Germinated (depthwise, 4.8K params)
    [env2] Germinated 'env2_seed_5' (depthwise, 4.8K params)
[00:26:31] env2_seed_5 | Stage transition: GERMINATED → TRAINING
[00:26:36] env3_seed_5 | Stage transition: BLENDING → SHADOWING
[00:26:43] env3_seed_5 | Stage transition: SHADOWING → PROBATIONARY
[00:26:43] env1_seed_5 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_5' (conv_enhance, 74.0K params)
[00:26:43] env3_seed_5 | Stage transition: PROBATIONARY → CULLED
[00:26:43] env3_seed_5 | Culled (conv_enhance, Δacc +5.65%)
    [env3] Culled 'env3_seed_5' (conv_enhance, Δacc +5.65%)
[00:26:43] env1_seed_5 | Stage transition: GERMINATED → TRAINING
[00:26:48] env3_seed_6 | Germinated (depthwise, 4.8K params)
    [env3] Germinated 'env3_seed_6' (depthwise, 4.8K params)
[00:26:48] env3_seed_6 | Stage transition: GERMINATED → TRAINING
[00:26:54] env0_seed_7 | Stage transition: TRAINING → BLENDING
[00:27:00] env2_seed_5 | Stage transition: TRAINING → BLENDING
[00:27:00] env3_seed_6 | Stage transition: TRAINING → CULLED
[00:27:00] env3_seed_6 | Culled (depthwise, Δacc +2.47%)
    [env3] Culled 'env3_seed_6' (depthwise, Δacc +2.47%)
[00:27:05] env1_seed_5 | Stage transition: TRAINING → BLENDING
[00:27:05] env3_seed_7 | Germinated (depthwise, 4.8K params)
    [env3] Germinated 'env3_seed_7' (depthwise, 4.8K params)
[00:27:05] env3_seed_7 | Stage transition: GERMINATED → TRAINING
[00:27:18] env2_seed_5 | Stage transition: BLENDING → CULLED
[00:27:18] env2_seed_5 | Culled (depthwise, Δacc +0.52%)
    [env2] Culled 'env2_seed_5' (depthwise, Δacc +0.52%)
[00:27:23] env0_seed_7 | Stage transition: BLENDING → SHADOWING
[00:27:23] env2_seed_6 | Germinated (norm, 0.1K params)
    [env2] Germinated 'env2_seed_6' (norm, 0.1K params)
[00:27:23] env2_seed_6 | Stage transition: GERMINATED → TRAINING
[00:27:28] env0_seed_7 | Stage transition: SHADOWING → CULLED
[00:27:28] env0_seed_7 | Culled (conv_enhance, Δacc -0.73%)
    [env0] Culled 'env0_seed_7' (conv_enhance, Δacc -0.73%)
Batch 2: Episodes 8/200
  Env accuracies: ['71.1%', '73.3%', '70.6%', '75.5%']
  Avg acc: 72.6% (rolling: 72.7%)
  Avg reward: -0.2
  Actions: {'WAIT': 50, 'GERMINATE_NORM': 46, 'GERMINATE_ATTENTION': 33, 'GERMINATE_DEPTHWISE': 44, 'GERMINATE_CONV_ENHANCE': 51, 'FOSSILIZE': 46, 'CULL': 30}
  Successful: {'WAIT': 50, 'GERMINATE_NORM': 6, 'GERMINATE_ATTENTION': 4, 'GERMINATE_DEPTHWISE': 6, 'GERMINATE_CONV_ENHANCE': 13, 'FOSSILIZE': 2, 'CULL': 26}
  Policy loss: -0.0198, Value loss: 72.3110, Entropy: 1.9314, Entropy coef: 0.1897
[00:27:31] env0_seed_0 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_0' (conv_enhance, 74.0K params)
[00:27:31] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[00:27:31] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[00:27:31] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[00:27:35] env2_seed_0 | Germinated (depthwise, 4.8K params)
    [env2] Germinated 'env2_seed_0' (depthwise, 4.8K params)
[00:27:35] env3_seed_0 | Germinated (attention, 2.0K params)
    [env3] Germinated 'env3_seed_0' (attention, 2.0K params)
[00:27:35] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[00:27:35] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[00:27:53] env0_seed_0 | Stage transition: TRAINING → BLENDING
[00:27:53] env1_seed_0 | Stage transition: TRAINING → BLENDING
[00:27:53] env3_seed_0 | Stage transition: TRAINING → CULLED
[00:27:53] env3_seed_0 | Culled (attention, Δacc +7.22%)
    [env3] Culled 'env3_seed_0' (attention, Δacc +7.22%)
[00:27:58] env2_seed_0 | Stage transition: TRAINING → BLENDING
[00:27:58] env3_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env3] Germinated 'env3_seed_1' (conv_enhance, 74.0K params)
[00:27:58] env3_seed_1 | Stage transition: GERMINATED → TRAINING
[00:28:04] env1_seed_0 | Stage transition: BLENDING → CULLED
[00:28:04] env1_seed_0 | Culled (attention, Δacc +11.94%)
    [env1] Culled 'env1_seed_0' (attention, Δacc +11.94%)
[00:28:04] env2_seed_0 | Stage transition: BLENDING → CULLED
[00:28:04] env2_seed_0 | Culled (depthwise, Δacc +9.30%)
    [env2] Culled 'env2_seed_0' (depthwise, Δacc +9.30%)
[00:28:09] env1_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_1' (conv_enhance, 74.0K params)
[00:28:09] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[00:28:14] env1_seed_1 | Stage transition: TRAINING → CULLED
[00:28:14] env1_seed_1 | Culled (conv_enhance, Δacc +0.00%)
    [env1] Culled 'env1_seed_1' (conv_enhance, Δacc +0.00%)
[00:28:19] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[00:28:19] env3_seed_1 | Stage transition: TRAINING → BLENDING
[00:28:19] env3_seed_1 | Stage transition: BLENDING → CULLED
[00:28:19] env3_seed_1 | Culled (conv_enhance, Δacc +10.18%)
    [env3] Culled 'env3_seed_1' (conv_enhance, Δacc +10.18%)
[00:28:22] env1_seed_2 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_2' (attention, 2.0K params)
[00:28:22] env3_seed_2 | Germinated (depthwise, 4.8K params)
    [env3] Germinated 'env3_seed_2' (depthwise, 4.8K params)
[00:28:22] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[00:28:22] env3_seed_2 | Stage transition: GERMINATED → TRAINING
[00:28:25] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[00:28:25] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[00:28:25] env0_seed_0 | Fossilized (conv_enhance, Δacc +23.94%)
    [env0] Fossilized 'env0_seed_0' (conv_enhance, Δacc +23.94%)
[00:28:25] env1_seed_2 | Stage transition: TRAINING → CULLED
[00:28:25] env1_seed_2 | Culled (attention, Δacc +0.00%)
    [env1] Culled 'env1_seed_2' (attention, Δacc +0.00%)
[00:28:25] env2_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env2] Germinated 'env2_seed_1' (conv_enhance, 74.0K params)
[00:28:26] env2_seed_1 | Stage transition: GERMINATED → TRAINING
[00:28:30] env1_seed_3 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_3' (depthwise, 4.8K params)
[00:28:30] env3_seed_2 | Stage transition: TRAINING → CULLED
[00:28:30] env3_seed_2 | Culled (depthwise, Δacc +2.94%)
    [env3] Culled 'env3_seed_2' (depthwise, Δacc +2.94%)
[00:28:30] env1_seed_3 | Stage transition: GERMINATED → TRAINING
[00:28:35] env3_seed_3 | Germinated (depthwise, 4.8K params)
    [env3] Germinated 'env3_seed_3' (depthwise, 4.8K params)
[00:28:35] env3_seed_3 | Stage transition: GERMINATED → TRAINING
[00:28:50] env1_seed_3 | Stage transition: TRAINING → BLENDING
[00:28:50] env2_seed_1 | Stage transition: TRAINING → BLENDING
[00:28:50] env0_seed_0 | Stage transition: FOSSILIZED → CULLED
[00:28:50] env0_seed_0 | Culled (conv_enhance, Δacc +24.07%)
    [env0] Culled 'env0_seed_0' (conv_enhance, Δacc +24.07%)
[00:28:55] env0_seed_1 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_1' (attention, 2.0K params)
[00:28:55] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[00:29:01] env0_seed_1 | Stage transition: TRAINING → CULLED
[00:29:01] env0_seed_1 | Culled (attention, Δacc +0.00%)
    [env0] Culled 'env0_seed_1' (attention, Δacc +0.00%)
[00:29:06] env0_seed_2 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_2' (conv_enhance, 74.0K params)
[00:29:07] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[00:29:13] env0_seed_2 | Stage transition: TRAINING → CULLED
[00:29:13] env0_seed_2 | Culled (conv_enhance, Δacc +0.00%)
    [env0] Culled 'env0_seed_2' (conv_enhance, Δacc +0.00%)
[00:29:18] env1_seed_3 | Stage transition: BLENDING → SHADOWING
[00:29:18] env2_seed_1 | Stage transition: BLENDING → SHADOWING
[00:29:18] env0_seed_3 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_3' (depthwise, 4.8K params)
[00:29:18] env0_seed_3 | Stage transition: GERMINATED → TRAINING
[00:29:22] env3_seed_3 | Stage transition: TRAINING → CULLED
[00:29:22] env3_seed_3 | Culled (depthwise, Δacc -2.90%)
    [env3] Culled 'env3_seed_3' (depthwise, Δacc -2.90%)
[00:29:26] env1_seed_3 | Stage transition: SHADOWING → PROBATIONARY
[00:29:26] env2_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[00:29:29] env3_seed_4 | Germinated (attention, 2.0K params)
    [env3] Germinated 'env3_seed_4' (attention, 2.0K params)
[00:29:29] env3_seed_4 | Stage transition: GERMINATED → TRAINING
[00:29:34] env0_seed_3 | Stage transition: TRAINING → BLENDING
[00:29:34] env1_seed_3 | Stage transition: PROBATIONARY → CULLED
[00:29:34] env1_seed_3 | Culled (depthwise, Δacc +5.32%)
    [env1] Culled 'env1_seed_3' (depthwise, Δacc +5.32%)
[00:29:43] env2_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[00:29:43] env2_seed_1 | Fossilized (conv_enhance, Δacc +9.03%)
    [env2] Fossilized 'env2_seed_1' (conv_enhance, Δacc +9.03%)
[00:29:47] env1_seed_4 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_4' (attention, 2.0K params)
[00:29:47] env3_seed_4 | Stage transition: TRAINING → CULLED
[00:29:47] env3_seed_4 | Culled (attention, Δacc +0.91%)
    [env3] Culled 'env3_seed_4' (attention, Δacc +0.91%)
[00:29:47] env1_seed_4 | Stage transition: GERMINATED → TRAINING
[00:29:56] env0_seed_3 | Stage transition: BLENDING → SHADOWING
[00:30:00] env1_seed_4 | Stage transition: TRAINING → CULLED
[00:30:00] env1_seed_4 | Culled (attention, Δacc +0.17%)
    [env1] Culled 'env1_seed_4' (attention, Δacc +0.17%)
[00:30:03] env0_seed_3 | Stage transition: SHADOWING → PROBATIONARY
[00:30:03] env0_seed_3 | Stage transition: PROBATIONARY → FOSSILIZED
[00:30:03] env0_seed_3 | Fossilized (depthwise, Δacc +1.05%)
    [env0] Fossilized 'env0_seed_3' (depthwise, Δacc +1.05%)
[00:30:03] env1_seed_5 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_5' (depthwise, 4.8K params)
[00:30:03] env3_seed_5 | Germinated (attention, 2.0K params)
    [env3] Germinated 'env3_seed_5' (attention, 2.0K params)
[00:30:03] env1_seed_5 | Stage transition: GERMINATED → TRAINING
[00:30:03] env3_seed_5 | Stage transition: GERMINATED → TRAINING
[00:30:18] env1_seed_5 | Stage transition: TRAINING → BLENDING
[00:30:18] env3_seed_5 | Stage transition: TRAINING → CULLED
[00:30:18] env3_seed_5 | Culled (attention, Δacc +0.11%)
    [env3] Culled 'env3_seed_5' (attention, Δacc +0.11%)
[00:30:24] env3_seed_6 | Germinated (conv_enhance, 74.0K params)
    [env3] Germinated 'env3_seed_6' (conv_enhance, 74.0K params)
[00:30:25] env3_seed_6 | Stage transition: GERMINATED → TRAINING
[00:30:36] env1_seed_5 | Stage transition: BLENDING → SHADOWING
[00:30:40] env0_seed_3 | Stage transition: FOSSILIZED → CULLED
[00:30:40] env0_seed_3 | Culled (depthwise, Δacc +3.26%)
    [env0] Culled 'env0_seed_3' (depthwise, Δacc +3.26%)
[00:30:40] env2_seed_1 | Stage transition: FOSSILIZED → CULLED
[00:30:40] env2_seed_1 | Culled (conv_enhance, Δacc +10.40%)
    [env2] Culled 'env2_seed_1' (conv_enhance, Δacc +10.40%)
[00:30:43] env1_seed_5 | Stage transition: SHADOWING → PROBATIONARY
[00:30:47] env3_seed_6 | Stage transition: TRAINING → BLENDING
[00:30:47] env0_seed_4 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_4' (attention, 2.0K params)
[00:30:47] env0_seed_4 | Stage transition: GERMINATED → TRAINING
[00:30:51] env1_seed_5 | Stage transition: PROBATIONARY → FOSSILIZED
[00:30:51] env1_seed_5 | Fossilized (depthwise, Δacc +6.05%)
    [env1] Fossilized 'env1_seed_5' (depthwise, Δacc +6.05%)
[00:30:56] env2_seed_2 | Germinated (conv_enhance, 74.0K params)
    [env2] Germinated 'env2_seed_2' (conv_enhance, 74.0K params)
[00:30:56] env2_seed_2 | Stage transition: GERMINATED → TRAINING
[00:31:01] env0_seed_4 | Stage transition: TRAINING → CULLED
[00:31:01] env0_seed_4 | Culled (attention, Δacc -1.32%)
    [env0] Culled 'env0_seed_4' (attention, Δacc -1.32%)
[00:31:10] env3_seed_6 | Stage transition: BLENDING → SHADOWING
[00:31:10] env0_seed_5 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_5' (conv_enhance, 74.0K params)
[00:31:10] env2_seed_2 | Stage transition: TRAINING → CULLED
[00:31:10] env2_seed_2 | Culled (conv_enhance, Δacc -1.34%)
    [env2] Culled 'env2_seed_2' (conv_enhance, Δacc -1.34%)
[00:31:10] env3_seed_6 | Stage transition: SHADOWING → CULLED
[00:31:10] env3_seed_6 | Culled (conv_enhance, Δacc +4.32%)
    [env3] Culled 'env3_seed_6' (conv_enhance, Δacc +4.32%)
[00:31:10] env0_seed_5 | Stage transition: GERMINATED → TRAINING
[00:31:17] env3_seed_7 | Germinated (norm, 0.1K params)
    [env3] Germinated 'env3_seed_7' (norm, 0.1K params)
[00:31:17] env3_seed_7 | Stage transition: GERMINATED → TRAINING
[00:31:21] env2_seed_3 | Germinated (depthwise, 4.8K params)
    [env2] Germinated 'env2_seed_3' (depthwise, 4.8K params)
[00:31:22] env2_seed_3 | Stage transition: GERMINATED → TRAINING
[00:31:26] env0_seed_5 | Stage transition: TRAINING → BLENDING
[00:31:36] env2_seed_3 | Stage transition: TRAINING → CULLED
[00:31:36] env2_seed_3 | Culled (depthwise, Δacc +0.24%)
    [env2] Culled 'env2_seed_3' (depthwise, Δacc +0.24%)
[00:31:41] env2_seed_4 | Germinated (depthwise, 4.8K params)
    [env2] Germinated 'env2_seed_4' (depthwise, 4.8K params)
[00:31:41] env2_seed_4 | Stage transition: GERMINATED → TRAINING
[00:31:45] env3_seed_7 | Stage transition: TRAINING → BLENDING
[00:31:45] env0_seed_5 | Stage transition: BLENDING → CULLED
[00:31:45] env0_seed_5 | Culled (conv_enhance, Δacc +3.49%)
    [env0] Culled 'env0_seed_5' (conv_enhance, Δacc +3.49%)
[00:31:50] env0_seed_6 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_6' (attention, 2.0K params)
[00:31:50] env0_seed_6 | Stage transition: GERMINATED → TRAINING
[00:32:04] env2_seed_4 | Stage transition: TRAINING → BLENDING
[00:32:09] env3_seed_7 | Stage transition: BLENDING → SHADOWING
[00:32:12] env0_seed_6 | Stage transition: TRAINING → BLENDING
[00:32:16] env3_seed_7 | Stage transition: SHADOWING → PROBATIONARY
[00:32:16] env3_seed_7 | Stage transition: PROBATIONARY → FOSSILIZED
[00:32:16] env3_seed_7 | Fossilized (norm, Δacc +3.32%)
    [env3] Fossilized 'env3_seed_7' (norm, Δacc +3.32%)
[00:32:19] env3_seed_7 | Stage transition: FOSSILIZED → CULLED
[00:32:19] env3_seed_7 | Culled (norm, Δacc +2.61%)
    [env3] Culled 'env3_seed_7' (norm, Δacc +2.61%)
[00:32:23] env2_seed_4 | Stage transition: BLENDING → SHADOWING
[00:32:23] env0_seed_6 | Stage transition: BLENDING → CULLED
[00:32:23] env0_seed_6 | Culled (attention, Δacc +0.67%)
    [env0] Culled 'env0_seed_6' (attention, Δacc +0.67%)
[00:32:23] env3_seed_8 | Germinated (norm, 0.1K params)
    [env3] Germinated 'env3_seed_8' (norm, 0.1K params)
[00:32:23] env3_seed_8 | Stage transition: GERMINATED → TRAINING
[00:32:26] env0_seed_7 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_7' (depthwise, 4.8K params)
[00:32:26] env0_seed_7 | Stage transition: GERMINATED → TRAINING
[00:32:31] env2_seed_4 | Stage transition: SHADOWING → PROBATIONARY
[00:32:31] env1_seed_5 | Stage transition: FOSSILIZED → CULLED
[00:32:31] env1_seed_5 | Culled (depthwise, Δacc +4.34%)
    [env1] Culled 'env1_seed_5' (depthwise, Δacc +4.34%)
[00:32:40] env1_seed_6 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_6' (attention, 2.0K params)
[00:32:40] env1_seed_6 | Stage transition: GERMINATED → TRAINING
[00:32:49] env0_seed_7 | Stage transition: TRAINING → BLENDING
[00:32:59] env1_seed_6 | Stage transition: TRAINING → BLENDING
Batch 3: Episodes 12/200
  Env accuracies: ['74.9%', '71.8%', '70.6%', '73.0%']
  Avg acc: 72.6% (rolling: 72.7%)
  Avg reward: 2.9
  Actions: {'WAIT': 61, 'GERMINATE_NORM': 22, 'GERMINATE_ATTENTION': 39, 'GERMINATE_DEPTHWISE': 49, 'GERMINATE_CONV_ENHANCE': 46, 'FOSSILIZE': 52, 'CULL': 31}
  Successful: {'WAIT': 61, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 10, 'GERMINATE_DEPTHWISE': 9, 'GERMINATE_CONV_ENHANCE': 8, 'FOSSILIZE': 5, 'CULL': 25}
  Policy loss: -0.0286, Value loss: 81.2886, Entropy: 1.9223, Entropy coef: 0.1846
[00:33:02] env0_seed_0 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_0' (conv_enhance, 74.0K params)
[00:33:02] env2_seed_0 | Germinated (conv_enhance, 74.0K params)
    [env2] Germinated 'env2_seed_0' (conv_enhance, 74.0K params)
[00:33:02] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[00:33:02] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[00:33:06] env1_seed_0 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_0' (depthwise, 4.8K params)
[00:33:06] env3_seed_0 | Germinated (depthwise, 4.8K params)
    [env3] Germinated 'env3_seed_0' (depthwise, 4.8K params)
[00:33:06] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[00:33:06] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[00:33:12] env0_seed_0 | Stage transition: TRAINING → CULLED
[00:33:12] env0_seed_0 | Culled (conv_enhance, Δacc +2.05%)
    [env0] Culled 'env0_seed_0' (conv_enhance, Δacc +2.05%)
[00:33:17] env2_seed_0 | Stage transition: TRAINING → CULLED
[00:33:17] env2_seed_0 | Culled (conv_enhance, Δacc +8.97%)
    [env2] Culled 'env2_seed_0' (conv_enhance, Δacc +8.97%)
[00:33:17] env3_seed_0 | Stage transition: TRAINING → CULLED
[00:33:17] env3_seed_0 | Culled (depthwise, Δacc +3.99%)
    [env3] Culled 'env3_seed_0' (depthwise, Δacc +3.99%)
[00:33:20] env0_seed_1 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_1' (depthwise, 4.8K params)
[00:33:20] env3_seed_1 | Germinated (norm, 0.1K params)
    [env3] Germinated 'env3_seed_1' (norm, 0.1K params)
[00:33:20] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[00:33:20] env3_seed_1 | Stage transition: GERMINATED → TRAINING
[00:33:25] env1_seed_0 | Stage transition: TRAINING → BLENDING
[00:33:30] env2_seed_1 | Germinated (norm, 0.1K params)
    [env2] Germinated 'env2_seed_1' (norm, 0.1K params)
[00:33:30] env2_seed_1 | Stage transition: GERMINATED → TRAINING
[00:33:42] env0_seed_1 | Stage transition: TRAINING → BLENDING
[00:33:42] env3_seed_1 | Stage transition: TRAINING → BLENDING
[00:33:48] env1_seed_0 | Stage transition: BLENDING → CULLED
[00:33:48] env1_seed_0 | Culled (depthwise, Δacc +12.45%)
    [env1] Culled 'env1_seed_0' (depthwise, Δacc +12.45%)
[00:33:53] env2_seed_1 | Stage transition: TRAINING → BLENDING
[00:33:53] env1_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_1' (conv_enhance, 74.0K params)
[00:33:53] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[00:34:11] env0_seed_1 | Stage transition: BLENDING → SHADOWING
[00:34:11] env3_seed_1 | Stage transition: BLENDING → SHADOWING
[00:34:20] env0_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[00:34:20] env2_seed_1 | Stage transition: BLENDING → SHADOWING
[00:34:20] env3_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[00:34:24] env3_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[00:34:24] env3_seed_1 | Fossilized (norm, Δacc +16.37%)
    [env3] Fossilized 'env3_seed_1' (norm, Δacc +16.37%)
[00:34:27] env2_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[00:34:27] env0_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[00:34:27] env0_seed_1 | Fossilized (depthwise, Δacc +10.37%)
    [env0] Fossilized 'env0_seed_1' (depthwise, Δacc +10.37%)
[00:34:31] env1_seed_1 | Stage transition: TRAINING → BLENDING
[00:34:34] env3_seed_1 | Stage transition: FOSSILIZED → CULLED
[00:34:34] env3_seed_1 | Culled (norm, Δacc +17.01%)
    [env3] Culled 'env3_seed_1' (norm, Δacc +17.01%)
[00:34:41] env2_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[00:34:41] env2_seed_1 | Fossilized (norm, Δacc +7.70%)
    [env2] Fossilized 'env2_seed_1' (norm, Δacc +7.70%)
[00:34:44] env3_seed_2 | Germinated (depthwise, 4.8K params)
    [env3] Germinated 'env3_seed_2' (depthwise, 4.8K params)
[00:34:44] env3_seed_2 | Stage transition: GERMINATED → TRAINING
[00:34:48] env1_seed_1 | Stage transition: BLENDING → SHADOWING
[00:34:55] env1_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[00:34:55] env3_seed_2 | Stage transition: TRAINING → CULLED
[00:34:55] env3_seed_2 | Culled (depthwise, Δacc +1.88%)
    [env3] Culled 'env3_seed_2' (depthwise, Δacc +1.88%)
[00:35:01] env3_seed_3 | Germinated (depthwise, 4.8K params)
    [env3] Germinated 'env3_seed_3' (depthwise, 4.8K params)
[00:35:01] env3_seed_3 | Stage transition: GERMINATED → TRAINING
[00:35:05] env1_seed_1 | Stage transition: PROBATIONARY → CULLED
[00:35:05] env1_seed_1 | Culled (conv_enhance, Δacc +8.89%)
    [env1] Culled 'env1_seed_1' (conv_enhance, Δacc +8.89%)
[00:35:08] env1_seed_2 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_2' (norm, 0.1K params)
[00:35:08] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[00:35:16] env3_seed_3 | Stage transition: TRAINING → BLENDING
[00:35:20] env0_seed_1 | Stage transition: FOSSILIZED → CULLED
[00:35:20] env0_seed_1 | Culled (depthwise, Δacc +6.48%)
    [env0] Culled 'env0_seed_1' (depthwise, Δacc +6.48%)
[00:35:23] env2_seed_1 | Stage transition: FOSSILIZED → CULLED
[00:35:23] env2_seed_1 | Culled (norm, Δacc +12.60%)
    [env2] Culled 'env2_seed_1' (norm, Δacc +12.60%)
[00:35:27] env1_seed_2 | Stage transition: TRAINING → BLENDING
[00:35:27] env0_seed_2 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_2' (depthwise, 4.8K params)
[00:35:27] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[00:35:37] env3_seed_3 | Stage transition: BLENDING → SHADOWING
[00:35:41] env2_seed_2 | Germinated (norm, 0.1K params)
    [env2] Germinated 'env2_seed_2' (norm, 0.1K params)
[00:35:42] env2_seed_2 | Stage transition: GERMINATED → TRAINING
[00:35:46] env0_seed_2 | Stage transition: TRAINING → BLENDING
[00:35:46] env3_seed_3 | Stage transition: SHADOWING → PROBATIONARY
[00:35:46] env3_seed_3 | Stage transition: PROBATIONARY → FOSSILIZED
[00:35:46] env3_seed_3 | Fossilized (depthwise, Δacc +5.05%)
    [env3] Fossilized 'env3_seed_3' (depthwise, Δacc +5.05%)
[00:35:51] env1_seed_2 | Stage transition: BLENDING → SHADOWING
[00:35:51] env1_seed_2 | Stage transition: SHADOWING → CULLED
[00:35:51] env1_seed_2 | Culled (norm, Δacc -1.15%)
    [env1] Culled 'env1_seed_2' (norm, Δacc -1.15%)
[00:35:58] env1_seed_3 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_3' (norm, 0.1K params)
[00:35:58] env2_seed_2 | Stage transition: TRAINING → CULLED
[00:35:58] env2_seed_2 | Culled (norm, Δacc +1.04%)
    [env2] Culled 'env2_seed_2' (norm, Δacc +1.04%)
[00:35:58] env1_seed_3 | Stage transition: GERMINATED → TRAINING
[00:36:02] env0_seed_2 | Stage transition: BLENDING → CULLED
[00:36:02] env0_seed_2 | Culled (depthwise, Δacc +4.92%)
    [env0] Culled 'env0_seed_2' (depthwise, Δacc +4.92%)
[00:36:02] env2_seed_3 | Germinated (attention, 2.0K params)
    [env2] Germinated 'env2_seed_3' (attention, 2.0K params)
[00:36:03] env2_seed_3 | Stage transition: GERMINATED → TRAINING
[00:36:07] env0_seed_3 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_3' (norm, 0.1K params)
[00:36:07] env0_seed_3 | Stage transition: GERMINATED → TRAINING
[00:36:12] env2_seed_3 | Stage transition: TRAINING → CULLED
[00:36:12] env2_seed_3 | Culled (attention, Δacc -1.04%)
    [env2] Culled 'env2_seed_3' (attention, Δacc -1.04%)
[00:36:16] env2_seed_4 | Germinated (attention, 2.0K params)
    [env2] Germinated 'env2_seed_4' (attention, 2.0K params)
[00:36:16] env3_seed_3 | Stage transition: FOSSILIZED → CULLED
[00:36:16] env3_seed_3 | Culled (depthwise, Δacc +4.67%)
    [env3] Culled 'env3_seed_3' (depthwise, Δacc +4.67%)
[00:36:16] env2_seed_4 | Stage transition: GERMINATED → TRAINING
[00:36:25] env0_seed_3 | Stage transition: TRAINING → BLENDING
[00:36:25] env1_seed_3 | Stage transition: TRAINING → BLENDING
[00:36:25] env3_seed_4 | Germinated (conv_enhance, 74.0K params)
    [env3] Germinated 'env3_seed_4' (conv_enhance, 74.0K params)
[00:36:25] env3_seed_4 | Stage transition: GERMINATED → TRAINING
[00:36:37] env2_seed_4 | Stage transition: TRAINING → BLENDING
[00:36:55] env0_seed_3 | Stage transition: BLENDING → SHADOWING
[00:36:55] env1_seed_3 | Stage transition: BLENDING → SHADOWING
[00:37:00] env0_seed_3 | Stage transition: SHADOWING → CULLED
[00:37:00] env0_seed_3 | Culled (norm, Δacc +1.79%)
    [env0] Culled 'env0_seed_3' (norm, Δacc +1.79%)
[00:37:05] env1_seed_3 | Stage transition: SHADOWING → PROBATIONARY
[00:37:05] env2_seed_4 | Stage transition: BLENDING → SHADOWING
[00:37:05] env0_seed_4 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_4' (depthwise, 4.8K params)
[00:37:05] env0_seed_4 | Stage transition: GERMINATED → TRAINING
[00:37:09] env3_seed_4 | Stage transition: TRAINING → BLENDING
[00:37:14] env2_seed_4 | Stage transition: SHADOWING → PROBATIONARY
[00:37:14] env1_seed_3 | Stage transition: PROBATIONARY → FOSSILIZED
[00:37:14] env1_seed_3 | Fossilized (norm, Δacc +2.91%)
    [env1] Fossilized 'env1_seed_3' (norm, Δacc +2.91%)
[00:37:19] env2_seed_4 | Stage transition: PROBATIONARY → FOSSILIZED
[00:37:19] env2_seed_4 | Fossilized (attention, Δacc +3.01%)
    [env2] Fossilized 'env2_seed_4' (attention, Δacc +3.01%)
[00:37:28] env0_seed_4 | Stage transition: TRAINING → BLENDING
[00:37:28] env1_seed_3 | Stage transition: FOSSILIZED → CULLED
[00:37:28] env1_seed_3 | Culled (norm, Δacc +3.27%)
    [env1] Culled 'env1_seed_3' (norm, Δacc +3.27%)
[00:37:33] env3_seed_4 | Stage transition: BLENDING → SHADOWING
[00:37:33] env1_seed_4 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_4' (norm, 0.1K params)
[00:37:33] env1_seed_4 | Stage transition: GERMINATED → TRAINING
[00:37:42] env3_seed_4 | Stage transition: SHADOWING → PROBATIONARY
[00:37:42] env0_seed_4 | Stage transition: BLENDING → CULLED
[00:37:42] env0_seed_4 | Culled (depthwise, Δacc -4.49%)
    [env0] Culled 'env0_seed_4' (depthwise, Δacc -4.49%)
[00:37:45] env0_seed_5 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_5' (conv_enhance, 74.0K params)
[00:37:45] env0_seed_5 | Stage transition: GERMINATED → TRAINING
[00:37:58] env3_seed_4 | Stage transition: PROBATIONARY → FOSSILIZED
[00:37:58] env3_seed_4 | Fossilized (conv_enhance, Δacc +2.90%)
    [env3] Fossilized 'env3_seed_4' (conv_enhance, Δacc +2.90%)
[00:38:03] env0_seed_5 | Stage transition: TRAINING → BLENDING
[00:38:03] env1_seed_4 | Stage transition: TRAINING → BLENDING
[00:38:26] env0_seed_5 | Stage transition: BLENDING → SHADOWING
[00:38:26] env1_seed_4 | Stage transition: BLENDING → SHADOWING
[00:38:32] env0_seed_5 | Stage transition: SHADOWING → PROBATIONARY
[00:38:32] env1_seed_4 | Stage transition: SHADOWING → PROBATIONARY
[00:38:32] env0_seed_5 | Stage transition: PROBATIONARY → CULLED
[00:38:32] env0_seed_5 | Culled (conv_enhance, Δacc +2.59%)
    [env0] Culled 'env0_seed_5' (conv_enhance, Δacc +2.59%)
[00:38:32] env1_seed_4 | Stage transition: PROBATIONARY → FOSSILIZED
[00:38:32] env1_seed_4 | Fossilized (norm, Δacc +1.65%)
    [env1] Fossilized 'env1_seed_4' (norm, Δacc +1.65%)
Batch 4: Episodes 16/200
  Env accuracies: ['73.7%', '77.6%', '72.1%', '76.1%']
  Avg acc: 74.9% (rolling: 73.2%)
  Avg reward: 18.9
  Actions: {'WAIT': 49, 'GERMINATE_NORM': 45, 'GERMINATE_ATTENTION': 36, 'GERMINATE_DEPTHWISE': 46, 'GERMINATE_CONV_ENHANCE': 39, 'FOSSILIZE': 64, 'CULL': 21}
  Successful: {'WAIT': 49, 'GERMINATE_NORM': 7, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 7, 'GERMINATE_CONV_ENHANCE': 5, 'FOSSILIZE': 8, 'CULL': 18}
  Policy loss: -0.0216, Value loss: 54.2486, Entropy: 1.8971, Entropy coef: 0.1795
[00:38:35] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[00:38:35] env2_seed_0 | Germinated (depthwise, 4.8K params)
    [env2] Germinated 'env2_seed_0' (depthwise, 4.8K params)
[00:38:35] env3_seed_0 | Germinated (depthwise, 4.8K params)
    [env3] Germinated 'env3_seed_0' (depthwise, 4.8K params)
[00:38:35] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[00:38:35] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[00:38:35] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[00:38:40] env0_seed_0 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_0' (depthwise, 4.8K params)
[00:38:40] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[00:38:58] env1_seed_0 | Stage transition: TRAINING → BLENDING
[00:38:58] env3_seed_0 | Stage transition: TRAINING → BLENDING
[00:39:04] env0_seed_0 | Stage transition: TRAINING → BLENDING
[00:39:04] env2_seed_0 | Stage transition: TRAINING → BLENDING
[00:39:16] env1_seed_0 | Stage transition: BLENDING → CULLED
[00:39:16] env1_seed_0 | Culled (attention, Δacc +13.78%)
    [env1] Culled 'env1_seed_0' (attention, Δacc +13.78%)
[00:39:26] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[00:39:26] env1_seed_1 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_1' (attention, 2.0K params)
[00:39:26] env3_seed_0 | Stage transition: SHADOWING → CULLED
[00:39:26] env3_seed_0 | Culled (depthwise, Δacc +8.35%)
    [env3] Culled 'env3_seed_0' (depthwise, Δacc +8.35%)
[00:39:26] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[00:39:31] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[00:39:31] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[00:39:31] env2_seed_0 | Stage transition: SHADOWING → CULLED
[00:39:31] env2_seed_0 | Culled (depthwise, Δacc +4.51%)
    [env2] Culled 'env2_seed_0' (depthwise, Δacc +4.51%)
[00:39:37] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[00:39:40] env1_seed_1 | Stage transition: TRAINING → BLENDING
[00:39:40] env3_seed_1 | Germinated (norm, 0.1K params)
    [env3] Germinated 'env3_seed_1' (norm, 0.1K params)
[00:39:40] env3_seed_1 | Stage transition: GERMINATED → TRAINING
[00:39:44] env2_seed_1 | Germinated (norm, 0.1K params)
    [env2] Germinated 'env2_seed_1' (norm, 0.1K params)
[00:39:44] env2_seed_1 | Stage transition: GERMINATED → TRAINING
[00:39:58] env3_seed_1 | Stage transition: TRAINING → BLENDING
[00:40:03] env1_seed_1 | Stage transition: BLENDING → SHADOWING
[00:40:12] env1_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[00:40:12] env2_seed_1 | Stage transition: TRAINING → BLENDING
[00:40:17] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[00:40:17] env0_seed_0 | Fossilized (depthwise, Δacc +15.47%)
    [env0] Fossilized 'env0_seed_0' (depthwise, Δacc +15.47%)
[00:40:21] env3_seed_1 | Stage transition: BLENDING → SHADOWING
[00:40:25] env1_seed_1 | Stage transition: PROBATIONARY → CULLED
[00:40:25] env1_seed_1 | Culled (attention, Δacc +11.30%)
    [env1] Culled 'env1_seed_1' (attention, Δacc +11.30%)
[00:40:28] env3_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[00:40:28] env1_seed_2 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_2' (depthwise, 4.8K params)
[00:40:28] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[00:40:32] env2_seed_1 | Stage transition: BLENDING → SHADOWING
[00:40:39] env2_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[00:40:42] env1_seed_2 | Stage transition: TRAINING → BLENDING
[00:40:46] env3_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[00:40:46] env3_seed_1 | Fossilized (norm, Δacc +6.25%)
    [env3] Fossilized 'env3_seed_1' (norm, Δacc +6.25%)
[00:40:49] env2_seed_1 | Stage transition: PROBATIONARY → CULLED
[00:40:49] env2_seed_1 | Culled (norm, Δacc +9.42%)
    [env2] Culled 'env2_seed_1' (norm, Δacc +9.42%)
[00:40:53] env2_seed_2 | Germinated (attention, 2.0K params)
    [env2] Germinated 'env2_seed_2' (attention, 2.0K params)
[00:40:53] env2_seed_2 | Stage transition: GERMINATED → TRAINING
[00:41:02] env1_seed_2 | Stage transition: BLENDING → SHADOWING
[00:41:02] env1_seed_2 | Stage transition: SHADOWING → CULLED
[00:41:02] env1_seed_2 | Culled (depthwise, Δacc +0.90%)
    [env1] Culled 'env1_seed_2' (depthwise, Δacc +0.90%)
[00:41:05] env1_seed_3 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_3' (conv_enhance, 74.0K params)
[00:41:05] env1_seed_3 | Stage transition: GERMINATED → TRAINING
[00:41:14] env2_seed_2 | Stage transition: TRAINING → BLENDING
[00:41:14] env0_seed_0 | Stage transition: FOSSILIZED → CULLED
[00:41:14] env0_seed_0 | Culled (depthwise, Δacc +18.27%)
    [env0] Culled 'env0_seed_0' (depthwise, Δacc +18.27%)
[00:41:27] env1_seed_3 | Stage transition: TRAINING → BLENDING
[00:41:32] env0_seed_1 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_1' (norm, 0.1K params)
[00:41:32] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[00:41:36] env2_seed_2 | Stage transition: BLENDING → SHADOWING
[00:41:41] env3_seed_1 | Stage transition: FOSSILIZED → CULLED
[00:41:41] env3_seed_1 | Culled (norm, Δacc +6.19%)
    [env3] Culled 'env3_seed_1' (norm, Δacc +6.19%)
[00:41:46] env2_seed_2 | Stage transition: SHADOWING → PROBATIONARY
[00:41:50] env1_seed_3 | Stage transition: BLENDING → SHADOWING
[00:41:50] env3_seed_2 | Germinated (attention, 2.0K params)
    [env3] Germinated 'env3_seed_2' (attention, 2.0K params)
[00:41:50] env3_seed_2 | Stage transition: GERMINATED → TRAINING
[00:41:59] env1_seed_3 | Stage transition: SHADOWING → PROBATIONARY
[00:41:59] env1_seed_3 | Stage transition: PROBATIONARY → FOSSILIZED
[00:41:59] env1_seed_3 | Fossilized (conv_enhance, Δacc +4.95%)
    [env1] Fossilized 'env1_seed_3' (conv_enhance, Δacc +4.95%)
[00:42:08] env3_seed_2 | Stage transition: TRAINING → BLENDING
[00:42:12] env0_seed_1 | Stage transition: TRAINING → BLENDING
[00:42:30] env3_seed_2 | Stage transition: BLENDING → SHADOWING
[00:42:34] env0_seed_1 | Stage transition: BLENDING → SHADOWING
[00:42:37] env3_seed_2 | Stage transition: SHADOWING → PROBATIONARY
[00:42:37] env2_seed_2 | Stage transition: PROBATIONARY → FOSSILIZED
[00:42:37] env2_seed_2 | Fossilized (attention, Δacc +1.45%)
    [env2] Fossilized 'env2_seed_2' (attention, Δacc +1.45%)
[00:42:40] env0_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[00:42:46] env2_seed_2 | Stage transition: FOSSILIZED → CULLED
[00:42:46] env2_seed_2 | Culled (attention, Δacc +2.43%)
    [env2] Culled 'env2_seed_2' (attention, Δacc +2.43%)
[00:42:52] env0_seed_1 | Stage transition: PROBATIONARY → CULLED
[00:42:52] env0_seed_1 | Culled (norm, Δacc +4.98%)
    [env0] Culled 'env0_seed_1' (norm, Δacc +4.98%)
[00:42:52] env2_seed_3 | Germinated (attention, 2.0K params)
    [env2] Germinated 'env2_seed_3' (attention, 2.0K params)
[00:42:52] env2_seed_3 | Stage transition: GERMINATED → TRAINING
[00:42:58] env0_seed_2 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_2' (norm, 0.1K params)
[00:42:58] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[00:43:08] env2_seed_3 | Stage transition: TRAINING → BLENDING
[00:43:08] env0_seed_2 | Stage transition: TRAINING → CULLED
[00:43:08] env0_seed_2 | Culled (norm, Δacc -7.17%)
    [env0] Culled 'env0_seed_2' (norm, Δacc -7.17%)
[00:43:21] env0_seed_3 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_3' (norm, 0.1K params)
[00:43:21] env0_seed_3 | Stage transition: GERMINATED → TRAINING
[00:43:25] env2_seed_3 | Stage transition: BLENDING → SHADOWING
[00:43:25] env0_seed_3 | Stage transition: TRAINING → CULLED
[00:43:25] env0_seed_3 | Culled (norm, Δacc +0.00%)
    [env0] Culled 'env0_seed_3' (norm, Δacc +0.00%)
[00:43:28] env0_seed_4 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_4' (depthwise, 4.8K params)
[00:43:28] env3_seed_2 | Stage transition: PROBATIONARY → FOSSILIZED
[00:43:28] env3_seed_2 | Fossilized (attention, Δacc +3.01%)
    [env3] Fossilized 'env3_seed_2' (attention, Δacc +3.01%)
[00:43:28] env0_seed_4 | Stage transition: GERMINATED → TRAINING
[00:43:31] env2_seed_3 | Stage transition: SHADOWING → PROBATIONARY
[00:43:34] env2_seed_3 | Stage transition: PROBATIONARY → CULLED
[00:43:34] env2_seed_3 | Culled (attention, Δacc -0.88%)
    [env2] Culled 'env2_seed_3' (attention, Δacc -0.88%)
[00:43:37] env2_seed_4 | Germinated (conv_enhance, 74.0K params)
    [env2] Germinated 'env2_seed_4' (conv_enhance, 74.0K params)
Batch 5: Episodes 20/200
  Env accuracies: ['75.5%', '78.5%', '73.6%', '75.5%']
  Avg acc: 75.8% (rolling: 73.7%)
  Avg reward: 18.1
  Actions: {'WAIT': 43, 'GERMINATE_NORM': 52, 'GERMINATE_ATTENTION': 52, 'GERMINATE_DEPTHWISE': 40, 'GERMINATE_CONV_ENHANCE': 42, 'FOSSILIZE': 53, 'CULL': 18}
  Successful: {'WAIT': 43, 'GERMINATE_NORM': 5, 'GERMINATE_ATTENTION': 5, 'GERMINATE_DEPTHWISE': 5, 'GERMINATE_CONV_ENHANCE': 2, 'FOSSILIZE': 5, 'CULL': 13}
  Policy loss: -0.0225, Value loss: 59.3976, Entropy: 1.8783, Entropy coef: 0.1743

Blueprint Stats
  ---------------------------------------------------------------------------

Blueprint       Germ  Foss  Cull   Rate     ΔAcc    Churn
  ---------------------------------------------------------------------------

  attention         33     4    30  11.8%   +2.93%   +2.89%
  conv_enhance      36     6    31  16.2%  +11.08%   +5.51%
  depthwise         36     5    29  14.7%   +7.60%   +3.20%
  norm              27     7    24  22.6%   +6.54%   +3.15%
Seed Scoreboard (env 0):
  Fossilized: 4 (+88.4K params, +93.3% of host)
  Culled: 33
  Avg fossilize age: 13.8 epochs
  Avg cull age: 9.3 epochs
  Compute cost: 1.39x baseline
  Distribution: conv_enhance x1, depthwise x3
Seed Scoreboard (env 1):
  Fossilized: 5 (+79.2K params, +83.5% of host)
  Culled: 28
  Avg fossilize age: 13.8 epochs
  Avg cull age: 8.9 epochs
  Compute cost: 1.29x baseline
  Distribution: norm x3, depthwise x1, conv_enhance x1
Seed Scoreboard (env 2):
  Fossilized: 6 (+154.2K params, +162.8% of host)
  Culled: 25
  Avg fossilize age: 16.3 epochs
  Avg cull age: 10.4 epochs
  Compute cost: 2.37x baseline
  Distribution: conv_enhance x2, attention x3, norm x1
Seed Scoreboard (env 3):
  Fossilized: 7 (+155.2K params, +163.8% of host)
  Culled: 28
  Avg fossilize age: 15.7 epochs
  Avg cull age: 8.4 epochs
  Compute cost: 1.79x baseline
  Distribution: conv_enhance x2, norm x3, depthwise x1, attention x1

[00:43:40] env2_seed_0 | Germinated (conv_enhance, 74.0K params)
    [env2] Germinated 'env2_seed_0' (conv_enhance, 74.0K params)
[00:43:40] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[00:43:44] env0_seed_0 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_0' (attention, 2.0K params)
[00:43:44] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[00:43:47] env3_seed_0 | Germinated (depthwise, 4.8K params)
    [env3] Germinated 'env3_seed_0' (depthwise, 4.8K params)
[00:43:47] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[00:43:52] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[00:43:52] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[00:43:58] env2_seed_0 | Stage transition: TRAINING → BLENDING
[00:44:04] env0_seed_0 | Stage transition: TRAINING → BLENDING
[00:44:10] env3_seed_0 | Stage transition: TRAINING → BLENDING
[00:44:16] env1_seed_0 | Stage transition: TRAINING → BLENDING
[00:44:16] env0_seed_0 | Stage transition: BLENDING → CULLED
[00:44:16] env0_seed_0 | Culled (attention, Δacc +10.10%)
    [env0] Culled 'env0_seed_0' (attention, Δacc +10.10%)
[00:44:26] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[00:44:34] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[00:44:34] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[00:44:37] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[00:44:37] env0_seed_1 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_1' (attention, 2.0K params)
[00:44:37] env2_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[00:44:37] env2_seed_0 | Fossilized (conv_enhance, Δacc +27.28%)
    [env2] Fossilized 'env2_seed_0' (conv_enhance, Δacc +27.28%)
[00:44:37] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[00:44:40] env3_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[00:44:40] env1_seed_0 | Stage transition: SHADOWING → CULLED
[00:44:40] env1_seed_0 | Culled (attention, Δacc +7.51%)
    [env1] Culled 'env1_seed_0' (attention, Δacc +7.51%)
[00:44:44] env1_seed_1 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_1' (attention, 2.0K params)
[00:44:44] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[00:44:57] env0_seed_1 | Stage transition: TRAINING → BLENDING
[00:45:02] env1_seed_1 | Stage transition: TRAINING → BLENDING
[00:45:02] env3_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[00:45:02] env3_seed_0 | Fossilized (depthwise, Δacc +14.46%)
    [env3] Fossilized 'env3_seed_0' (depthwise, Δacc +14.46%)
[00:45:20] env0_seed_1 | Stage transition: BLENDING → SHADOWING
[00:45:23] env1_seed_1 | Stage transition: BLENDING → SHADOWING
[00:45:26] env0_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[00:45:29] env1_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[00:45:29] env3_seed_0 | Stage transition: FOSSILIZED → CULLED
[00:45:29] env3_seed_0 | Culled (depthwise, Δacc +11.69%)
    [env3] Culled 'env3_seed_0' (depthwise, Δacc +11.69%)
[00:45:32] env3_seed_1 | Germinated (attention, 2.0K params)
    [env3] Germinated 'env3_seed_1' (attention, 2.0K params)
[00:45:32] env3_seed_1 | Stage transition: GERMINATED → TRAINING
[00:45:35] env1_seed_1 | Stage transition: PROBATIONARY → CULLED
[00:45:35] env1_seed_1 | Culled (attention, Δacc +4.56%)
    [env1] Culled 'env1_seed_1' (attention, Δacc +4.56%)
[00:45:39] env0_seed_1 | Stage transition: PROBATIONARY → CULLED
[00:45:39] env0_seed_1 | Culled (attention, Δacc +1.90%)
    [env0] Culled 'env0_seed_1' (attention, Δacc +1.90%)
[00:45:39] env1_seed_2 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_2' (attention, 2.0K params)
[00:45:39] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[00:45:46] env3_seed_1 | Stage transition: TRAINING → BLENDING
[00:45:46] env0_seed_2 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_2' (depthwise, 4.8K params)
[00:45:46] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[00:45:56] env1_seed_2 | Stage transition: TRAINING → BLENDING
[00:46:10] env3_seed_1 | Stage transition: BLENDING → SHADOWING
[00:46:10] env1_seed_2 | Stage transition: BLENDING → CULLED
[00:46:10] env1_seed_2 | Culled (attention, Δacc +8.79%)
    [env1] Culled 'env1_seed_2' (attention, Δacc +8.79%)
[00:46:17] env3_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[00:46:17] env1_seed_3 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_3' (attention, 2.0K params)
[00:46:17] env3_seed_1 | Stage transition: PROBATIONARY → CULLED
[00:46:17] env3_seed_1 | Culled (attention, Δacc +1.07%)
    [env3] Culled 'env3_seed_1' (attention, Δacc +1.07%)
[00:46:17] env1_seed_3 | Stage transition: GERMINATED → TRAINING
[00:46:21] env1_seed_3 | Stage transition: TRAINING → CULLED
[00:46:21] env1_seed_3 | Culled (attention, Δacc +0.00%)
    [env1] Culled 'env1_seed_3' (attention, Δacc +0.00%)
[00:46:21] env2_seed_0 | Stage transition: FOSSILIZED → CULLED
[00:46:21] env2_seed_0 | Culled (conv_enhance, Δacc +20.04%)
    [env2] Culled 'env2_seed_0' (conv_enhance, Δacc +20.04%)
[00:46:21] env3_seed_2 | Germinated (norm, 0.1K params)
    [env3] Germinated 'env3_seed_2' (norm, 0.1K params)
[00:46:22] env3_seed_2 | Stage transition: GERMINATED → TRAINING
[00:46:31] env1_seed_4 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_4' (norm, 0.1K params)
[00:46:31] env2_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env2] Germinated 'env2_seed_1' (conv_enhance, 74.0K params)
[00:46:31] env1_seed_4 | Stage transition: GERMINATED → TRAINING
[00:46:31] env2_seed_1 | Stage transition: GERMINATED → TRAINING
[00:46:43] env3_seed_2 | Stage transition: TRAINING → BLENDING
[00:46:49] env0_seed_2 | Stage transition: TRAINING → BLENDING
[00:47:01] env1_seed_4 | Stage transition: TRAINING → BLENDING
[00:47:07] env2_seed_1 | Stage transition: TRAINING → BLENDING
[00:47:07] env2_seed_1 | Stage transition: BLENDING → CULLED
[00:47:07] env2_seed_1 | Culled (conv_enhance, Δacc +5.28%)
    [env2] Culled 'env2_seed_1' (conv_enhance, Δacc +5.28%)
[00:47:12] env3_seed_2 | Stage transition: BLENDING → SHADOWING
[00:47:16] env0_seed_2 | Stage transition: BLENDING → SHADOWING
[00:47:20] env3_seed_2 | Stage transition: SHADOWING → PROBATIONARY
[00:47:20] env1_seed_4 | Stage transition: BLENDING → CULLED
[00:47:20] env1_seed_4 | Culled (norm, Δacc +6.07%)
    [env1] Culled 'env1_seed_4' (norm, Δacc +6.07%)
[00:47:20] env3_seed_2 | Stage transition: PROBATIONARY → FOSSILIZED
[00:47:20] env3_seed_2 | Fossilized (norm, Δacc +4.13%)
    [env3] Fossilized 'env3_seed_2' (norm, Δacc +4.13%)
[00:47:23] env0_seed_2 | Stage transition: SHADOWING → PROBATIONARY
[00:47:23] env2_seed_2 | Germinated (depthwise, 4.8K params)
    [env2] Germinated 'env2_seed_2' (depthwise, 4.8K params)
[00:47:23] env2_seed_2 | Stage transition: GERMINATED → TRAINING
[00:47:32] env1_seed_5 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_5' (attention, 2.0K params)
[00:47:33] env1_seed_5 | Stage transition: GERMINATED → TRAINING
[00:47:37] env2_seed_2 | Stage transition: TRAINING → BLENDING
[00:47:46] env0_seed_2 | Stage transition: PROBATIONARY → CULLED
[00:47:46] env0_seed_2 | Culled (depthwise, Δacc -1.78%)
    [env0] Culled 'env0_seed_2' (depthwise, Δacc -1.78%)
[00:47:46] env2_seed_2 | Stage transition: BLENDING → CULLED
[00:47:46] env2_seed_2 | Culled (depthwise, Δacc +4.50%)
    [env2] Culled 'env2_seed_2' (depthwise, Δacc +4.50%)
[00:47:49] env1_seed_5 | Stage transition: TRAINING → BLENDING
[00:47:49] env2_seed_3 | Germinated (attention, 2.0K params)
    [env2] Germinated 'env2_seed_3' (attention, 2.0K params)
[00:47:49] env2_seed_3 | Stage transition: GERMINATED → TRAINING
[00:47:58] env2_seed_3 | Stage transition: TRAINING → CULLED
[00:47:58] env2_seed_3 | Culled (attention, Δacc +3.15%)
    [env2] Culled 'env2_seed_3' (attention, Δacc +3.15%)
[00:48:01] env0_seed_3 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_3' (depthwise, 4.8K params)
[00:48:01] env2_seed_4 | Germinated (norm, 0.1K params)
    [env2] Germinated 'env2_seed_4' (norm, 0.1K params)
[00:48:01] env0_seed_3 | Stage transition: GERMINATED → TRAINING
[00:48:01] env2_seed_4 | Stage transition: GERMINATED → TRAINING
[00:48:11] env1_seed_5 | Stage transition: BLENDING → SHADOWING
[00:48:11] env1_seed_5 | Stage transition: SHADOWING → CULLED
[00:48:11] env1_seed_5 | Culled (attention, Δacc +4.87%)
    [env1] Culled 'env1_seed_5' (attention, Δacc +4.87%)
[00:48:14] env1_seed_6 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_6' (norm, 0.1K params)
[00:48:14] env1_seed_6 | Stage transition: GERMINATED → TRAINING
[00:48:24] env0_seed_3 | Stage transition: TRAINING → BLENDING
[00:48:28] env2_seed_4 | Stage transition: TRAINING → BLENDING
[00:48:33] env1_seed_6 | Stage transition: TRAINING → BLENDING
[00:48:38] env0_seed_3 | Stage transition: BLENDING → CULLED
[00:48:38] env0_seed_3 | Culled (depthwise, Δacc -5.43%)
    [env0] Culled 'env0_seed_3' (depthwise, Δacc -5.43%)
[00:48:42] env0_seed_4 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_4' (depthwise, 4.8K params)
[00:48:43] env0_seed_4 | Stage transition: GERMINATED → TRAINING
[00:48:52] env2_seed_4 | Stage transition: BLENDING → SHADOWING
[00:48:57] env1_seed_6 | Stage transition: BLENDING → SHADOWING
[00:49:00] env2_seed_4 | Stage transition: SHADOWING → PROBATIONARY
[00:49:03] env1_seed_6 | Stage transition: SHADOWING → PROBATIONARY
[00:49:03] env1_seed_6 | Stage transition: PROBATIONARY → FOSSILIZED
[00:49:03] env1_seed_6 | Fossilized (norm, Δacc +7.52%)
    [env1] Fossilized 'env1_seed_6' (norm, Δacc +7.52%)
Batch 6: Episodes 24/200
  Env accuracies: ['72.0%', '75.8%', '75.2%', '77.4%']
  Avg acc: 75.1% (rolling: 74.0%)
  Avg reward: 15.6
  Actions: {'WAIT': 58, 'GERMINATE_NORM': 46, 'GERMINATE_ATTENTION': 46, 'GERMINATE_DEPTHWISE': 36, 'GERMINATE_CONV_ENHANCE': 39, 'FOSSILIZE': 55, 'CULL': 20}
  Successful: {'WAIT': 58, 'GERMINATE_NORM': 4, 'GERMINATE_ATTENTION': 9, 'GERMINATE_DEPTHWISE': 5, 'GERMINATE_CONV_ENHANCE': 2, 'FOSSILIZE': 4, 'CULL': 16}
  Policy loss: -0.0253, Value loss: 67.6903, Entropy: 1.8731, Entropy coef: 0.1692
[00:49:06] env2_seed_0 | Germinated (depthwise, 4.8K params)
    [env2] Germinated 'env2_seed_0' (depthwise, 4.8K params)
[00:49:06] env3_seed_0 | Germinated (conv_enhance, 74.0K params)
    [env3] Germinated 'env3_seed_0' (conv_enhance, 74.0K params)
[00:49:06] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[00:49:06] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[00:49:11] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[00:49:11] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[00:49:21] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[00:49:21] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[00:49:27] env2_seed_0 | Stage transition: TRAINING → BLENDING
[00:49:27] env3_seed_0 | Stage transition: TRAINING → BLENDING
[00:49:39] env1_seed_0 | Stage transition: TRAINING → BLENDING
[00:49:39] env2_seed_0 | Stage transition: BLENDING → CULLED
[00:49:39] env2_seed_0 | Culled (depthwise, Δacc +14.02%)
    [env2] Culled 'env2_seed_0' (depthwise, Δacc +14.02%)
[00:49:44] env0_seed_0 | Stage transition: TRAINING → BLENDING
[00:49:49] env2_seed_1 | Germinated (norm, 0.1K params)
    [env2] Germinated 'env2_seed_1' (norm, 0.1K params)
[00:49:49] env2_seed_1 | Stage transition: GERMINATED → TRAINING
[00:49:55] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[00:49:55] env0_seed_0 | Stage transition: BLENDING → CULLED
[00:49:55] env0_seed_0 | Culled (norm, Δacc +10.57%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +10.57%)
[00:50:04] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[00:50:04] env3_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[00:50:04] env1_seed_0 | Stage transition: SHADOWING → CULLED
[00:50:04] env1_seed_0 | Culled (attention, Δacc +7.99%)
    [env1] Culled 'env1_seed_0' (attention, Δacc +7.99%)
[00:50:04] env3_seed_0 | Stage transition: PROBATIONARY → CULLED
[00:50:04] env3_seed_0 | Culled (conv_enhance, Δacc +23.41%)
    [env3] Culled 'env3_seed_0' (conv_enhance, Δacc +23.41%)
[00:50:07] env2_seed_1 | Stage transition: TRAINING → BLENDING
[00:50:07] env0_seed_1 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_1' (attention, 2.0K params)
[00:50:07] env1_seed_1 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_1' (norm, 0.1K params)
[00:50:07] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[00:50:07] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[00:50:12] env3_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env3] Germinated 'env3_seed_1' (conv_enhance, 74.0K params)
[00:50:12] env3_seed_1 | Stage transition: GERMINATED → TRAINING
[00:50:30] env0_seed_1 | Stage transition: TRAINING → BLENDING
[00:50:30] env1_seed_1 | Stage transition: TRAINING → BLENDING
[00:50:30] env1_seed_1 | Stage transition: BLENDING → CULLED
[00:50:30] env1_seed_1 | Culled (norm, Δacc -1.49%)
    [env1] Culled 'env1_seed_1' (norm, Δacc -1.49%)
[00:50:34] env2_seed_1 | Stage transition: BLENDING → SHADOWING
[00:50:34] env3_seed_1 | Stage transition: TRAINING → BLENDING
[00:50:34] env1_seed_2 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_2' (norm, 0.1K params)
[00:50:35] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[00:50:44] env2_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[00:50:54] env0_seed_1 | Stage transition: BLENDING → SHADOWING
[00:50:54] env1_seed_2 | Stage transition: TRAINING → BLENDING
[00:50:54] env2_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[00:50:54] env2_seed_1 | Fossilized (norm, Δacc +14.50%)
    [env2] Fossilized 'env2_seed_1' (norm, Δacc +14.50%)
[00:50:58] env3_seed_1 | Stage transition: BLENDING → SHADOWING
[00:50:58] env3_seed_1 | Stage transition: SHADOWING → CULLED
[00:50:58] env3_seed_1 | Culled (conv_enhance, Δacc +13.62%)
    [env3] Culled 'env3_seed_1' (conv_enhance, Δacc +13.62%)
[00:51:01] env0_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[00:51:05] env0_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[00:51:05] env0_seed_1 | Fossilized (attention, Δacc +6.15%)
    [env0] Fossilized 'env0_seed_1' (attention, Δacc +6.15%)
[00:51:08] env3_seed_2 | Germinated (norm, 0.1K params)
    [env3] Germinated 'env3_seed_2' (norm, 0.1K params)
[00:51:08] env3_seed_2 | Stage transition: GERMINATED → TRAINING
[00:51:12] env1_seed_2 | Stage transition: BLENDING → SHADOWING
[00:51:18] env1_seed_2 | Stage transition: SHADOWING → PROBATIONARY
[00:51:18] env1_seed_2 | Stage transition: PROBATIONARY → FOSSILIZED
[00:51:18] env1_seed_2 | Fossilized (norm, Δacc +7.39%)
    [env1] Fossilized 'env1_seed_2' (norm, Δacc +7.39%)
[00:51:18] env3_seed_2 | Stage transition: TRAINING → CULLED
[00:51:18] env3_seed_2 | Culled (norm, Δacc +2.41%)
    [env3] Culled 'env3_seed_2' (norm, Δacc +2.41%)
[00:51:21] env2_seed_1 | Stage transition: FOSSILIZED → CULLED
[00:51:21] env2_seed_1 | Culled (norm, Δacc +16.54%)
    [env2] Culled 'env2_seed_1' (norm, Δacc +16.54%)
[00:51:21] env3_seed_3 | Germinated (conv_enhance, 74.0K params)
    [env3] Germinated 'env3_seed_3' (conv_enhance, 74.0K params)
[00:51:21] env3_seed_3 | Stage transition: GERMINATED → TRAINING
[00:51:25] env1_seed_2 | Stage transition: FOSSILIZED → CULLED
[00:51:25] env1_seed_2 | Culled (norm, Δacc +8.19%)
    [env1] Culled 'env1_seed_2' (norm, Δacc +8.19%)
[00:51:25] env2_seed_2 | Germinated (conv_enhance, 74.0K params)
    [env2] Germinated 'env2_seed_2' (conv_enhance, 74.0K params)
[00:51:25] env2_seed_2 | Stage transition: GERMINATED → TRAINING
[00:51:39] env1_seed_3 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_3' (norm, 0.1K params)
[00:51:39] env1_seed_3 | Stage transition: GERMINATED → TRAINING
[00:51:44] env2_seed_2 | Stage transition: TRAINING → BLENDING
[00:51:44] env2_seed_2 | Stage transition: BLENDING → CULLED
[00:51:44] env2_seed_2 | Culled (conv_enhance, Δacc +1.58%)
    [env2] Culled 'env2_seed_2' (conv_enhance, Δacc +1.58%)
[00:51:48] env2_seed_3 | Germinated (conv_enhance, 74.0K params)
    [env2] Germinated 'env2_seed_3' (conv_enhance, 74.0K params)
[00:51:48] env2_seed_3 | Stage transition: GERMINATED → TRAINING
[00:52:03] env1_seed_3 | Stage transition: TRAINING → BLENDING
[00:52:13] env0_seed_1 | Stage transition: FOSSILIZED → CULLED
[00:52:13] env0_seed_1 | Culled (attention, Δacc -3.52%)
    [env0] Culled 'env0_seed_1' (attention, Δacc -3.52%)
[00:52:13] env2_seed_3 | Stage transition: TRAINING → CULLED
[00:52:13] env2_seed_3 | Culled (conv_enhance, Δacc +1.51%)
    [env2] Culled 'env2_seed_3' (conv_enhance, Δacc +1.51%)
[00:52:17] env3_seed_3 | Stage transition: TRAINING → BLENDING
[00:52:17] env0_seed_2 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_2' (conv_enhance, 74.0K params)
[00:52:17] env2_seed_4 | Germinated (conv_enhance, 74.0K params)
    [env2] Germinated 'env2_seed_4' (conv_enhance, 74.0K params)
[00:52:17] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[00:52:17] env2_seed_4 | Stage transition: GERMINATED → TRAINING
[00:52:30] env1_seed_3 | Stage transition: BLENDING → SHADOWING
[00:52:35] env3_seed_3 | Stage transition: BLENDING → CULLED
[00:52:35] env3_seed_3 | Culled (conv_enhance, Δacc +2.71%)
    [env3] Culled 'env3_seed_3' (conv_enhance, Δacc +2.71%)
[00:52:38] env1_seed_3 | Stage transition: SHADOWING → PROBATIONARY
[00:52:38] env2_seed_4 | Stage transition: TRAINING → BLENDING
[00:52:38] env3_seed_4 | Germinated (conv_enhance, 74.0K params)
    [env3] Germinated 'env3_seed_4' (conv_enhance, 74.0K params)
[00:52:38] env3_seed_4 | Stage transition: GERMINATED → TRAINING
[00:52:48] env2_seed_4 | Stage transition: BLENDING → CULLED
[00:52:48] env2_seed_4 | Culled (conv_enhance, Δacc +5.28%)
    [env2] Culled 'env2_seed_4' (conv_enhance, Δacc +5.28%)
[00:52:53] env2_seed_5 | Germinated (depthwise, 4.8K params)
    [env2] Germinated 'env2_seed_5' (depthwise, 4.8K params)
[00:52:53] env2_seed_5 | Stage transition: GERMINATED → TRAINING
[00:52:58] env0_seed_2 | Stage transition: TRAINING → BLENDING
[00:52:58] env3_seed_4 | Stage transition: TRAINING → BLENDING
[00:53:03] env3_seed_4 | Stage transition: BLENDING → CULLED
[00:53:03] env3_seed_4 | Culled (conv_enhance, Δacc +1.33%)
    [env3] Culled 'env3_seed_4' (conv_enhance, Δacc +1.33%)
[00:53:06] env3_seed_5 | Germinated (norm, 0.1K params)
    [env3] Germinated 'env3_seed_5' (norm, 0.1K params)
[00:53:06] env3_seed_5 | Stage transition: GERMINATED → TRAINING
[00:53:16] env2_seed_5 | Stage transition: TRAINING → BLENDING
[00:53:21] env0_seed_2 | Stage transition: BLENDING → SHADOWING
[00:53:26] env3_seed_5 | Stage transition: TRAINING → BLENDING
[00:53:26] env1_seed_3 | Stage transition: PROBATIONARY → FOSSILIZED
[00:53:26] env1_seed_3 | Fossilized (norm, Δacc +4.54%)
    [env1] Fossilized 'env1_seed_3' (norm, Δacc +4.54%)
[00:53:30] env0_seed_2 | Stage transition: SHADOWING → PROBATIONARY
[00:53:30] env0_seed_2 | Stage transition: PROBATIONARY → CULLED
[00:53:30] env0_seed_2 | Culled (conv_enhance, Δacc +4.82%)
    [env0] Culled 'env0_seed_2' (conv_enhance, Δacc +4.82%)
[00:53:39] env2_seed_5 | Stage transition: BLENDING → SHADOWING
[00:53:39] env0_seed_3 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_3' (depthwise, 4.8K params)
[00:53:39] env0_seed_3 | Stage transition: GERMINATED → TRAINING
[00:53:49] env2_seed_5 | Stage transition: SHADOWING → PROBATIONARY
[00:53:49] env3_seed_5 | Stage transition: BLENDING → SHADOWING
[00:53:52] env0_seed_3 | Stage transition: TRAINING → CULLED
[00:53:52] env0_seed_3 | Culled (depthwise, Δacc +5.46%)
    [env0] Culled 'env0_seed_3' (depthwise, Δacc +5.46%)
[00:53:55] env3_seed_5 | Stage transition: SHADOWING → PROBATIONARY
[00:53:58] env0_seed_4 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_4' (attention, 2.0K params)
[00:53:58] env1_seed_3 | Stage transition: FOSSILIZED → CULLED
[00:53:58] env1_seed_3 | Culled (norm, Δacc +4.72%)
    [env1] Culled 'env1_seed_3' (norm, Δacc +4.72%)
[00:53:58] env0_seed_4 | Stage transition: GERMINATED → TRAINING
[00:54:04] env1_seed_4 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_4' (depthwise, 4.8K params)
[00:54:04] env1_seed_4 | Stage transition: GERMINATED → TRAINING
[00:54:13] env3_seed_5 | Stage transition: PROBATIONARY → FOSSILIZED
[00:54:13] env3_seed_5 | Fossilized (norm, Δacc +3.75%)
    [env3] Fossilized 'env3_seed_5' (norm, Δacc +3.75%)
[00:54:17] env0_seed_4 | Stage transition: TRAINING → BLENDING
[00:54:22] env1_seed_4 | Stage transition: TRAINING → BLENDING
[00:54:35] env0_seed_4 | Stage transition: BLENDING → CULLED
[00:54:35] env0_seed_4 | Culled (attention, Δacc +0.48%)
    [env0] Culled 'env0_seed_4' (attention, Δacc +0.48%)
[00:54:39] env0_seed_5 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_5' (conv_enhance, 74.0K params)
[00:54:39] env2_seed_5 | Stage transition: PROBATIONARY → FOSSILIZED
[00:54:39] env2_seed_5 | Fossilized (depthwise, Δacc +0.42%)
    [env2] Fossilized 'env2_seed_5' (depthwise, Δacc +0.42%)
[00:54:39] env0_seed_5 | Stage transition: GERMINATED → TRAINING
[00:54:43] env1_seed_4 | Stage transition: BLENDING → SHADOWING
Batch 7: Episodes 28/200
  Env accuracies: ['73.6%', '67.7%', '75.2%', '75.3%']
  Avg acc: 73.0% (rolling: 73.8%)
  Avg reward: 15.2
  Actions: {'WAIT': 49, 'GERMINATE_NORM': 60, 'GERMINATE_ATTENTION': 36, 'GERMINATE_DEPTHWISE': 47, 'GERMINATE_CONV_ENHANCE': 34, 'FOSSILIZE': 52, 'CULL': 22}
  Successful: {'WAIT': 49, 'GERMINATE_NORM': 7, 'GERMINATE_ATTENTION': 3, 'GERMINATE_DEPTHWISE': 4, 'GERMINATE_CONV_ENHANCE': 9, 'FOSSILIZE': 6, 'CULL': 19}
  Policy loss: -0.0190, Value loss: 55.9255, Entropy: 1.8480, Entropy coef: 0.1641
[00:54:46] env0_seed_0 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_0' (conv_enhance, 74.0K params)
[00:54:46] env3_seed_0 | Germinated (conv_enhance, 74.0K params)
    [env3] Germinated 'env3_seed_0' (conv_enhance, 74.0K params)
[00:54:46] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[00:54:46] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[00:54:51] env2_seed_0 | Germinated (conv_enhance, 74.0K params)
    [env2] Germinated 'env2_seed_0' (conv_enhance, 74.0K params)
[00:54:51] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[00:54:56] env0_seed_0 | Stage transition: TRAINING → CULLED
[00:54:56] env0_seed_0 | Culled (conv_enhance, Δacc +6.92%)
    [env0] Culled 'env0_seed_0' (conv_enhance, Δacc +6.92%)
[00:54:56] env1_seed_0 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_0' (depthwise, 4.8K params)
[00:54:56] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[00:55:01] env0_seed_1 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_1' (norm, 0.1K params)
[00:55:01] env3_seed_0 | Stage transition: TRAINING → CULLED
[00:55:01] env3_seed_0 | Culled (conv_enhance, Δacc +10.75%)
    [env3] Culled 'env3_seed_0' (conv_enhance, Δacc +10.75%)
[00:55:01] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[00:55:06] env3_seed_1 | Germinated (attention, 2.0K params)
    [env3] Germinated 'env3_seed_1' (attention, 2.0K params)
[00:55:06] env3_seed_1 | Stage transition: GERMINATED → TRAINING
[00:55:12] env2_seed_0 | Stage transition: TRAINING → BLENDING
[00:55:12] env3_seed_1 | Stage transition: TRAINING → CULLED
[00:55:12] env3_seed_1 | Culled (attention, Δacc +0.00%)
    [env3] Culled 'env3_seed_1' (attention, Δacc +0.00%)
[00:55:17] env1_seed_0 | Stage transition: TRAINING → BLENDING
[00:55:21] env0_seed_1 | Stage transition: TRAINING → BLENDING
[00:55:21] env3_seed_2 | Germinated (conv_enhance, 74.0K params)
    [env3] Germinated 'env3_seed_2' (conv_enhance, 74.0K params)
[00:55:22] env3_seed_2 | Stage transition: GERMINATED → TRAINING
[00:55:40] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[00:55:45] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[00:55:45] env0_seed_1 | Stage transition: BLENDING → CULLED
[00:55:45] env0_seed_1 | Culled (norm, Δacc +12.22%)
    [env0] Culled 'env0_seed_1' (norm, Δacc +12.22%)
[00:55:49] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[00:55:49] env0_seed_2 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_2' (depthwise, 4.8K params)
[00:55:49] env2_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[00:55:49] env2_seed_0 | Fossilized (conv_enhance, Δacc +19.57%)
    [env2] Fossilized 'env2_seed_0' (conv_enhance, Δacc +19.57%)
[00:55:49] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[00:55:53] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[00:55:53] env3_seed_2 | Stage transition: TRAINING → BLENDING
[00:55:53] env0_seed_2 | Stage transition: TRAINING → CULLED
[00:55:53] env0_seed_2 | Culled (depthwise, Δacc +0.00%)
    [env0] Culled 'env0_seed_2' (depthwise, Δacc +0.00%)
[00:55:57] env0_seed_3 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_3' (conv_enhance, 74.0K params)
[00:55:57] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[00:55:57] env1_seed_0 | Fossilized (depthwise, Δacc +9.21%)
    [env1] Fossilized 'env1_seed_0' (depthwise, Δacc +9.21%)
[00:55:57] env0_seed_3 | Stage transition: GERMINATED → TRAINING
[00:56:16] env3_seed_2 | Stage transition: BLENDING → SHADOWING
[00:56:19] env0_seed_3 | Stage transition: TRAINING → BLENDING
[00:56:23] env3_seed_2 | Stage transition: SHADOWING → PROBATIONARY
[00:56:33] env1_seed_0 | Stage transition: FOSSILIZED → CULLED
[00:56:33] env1_seed_0 | Culled (depthwise, Δacc +14.26%)
    [env1] Culled 'env1_seed_0' (depthwise, Δacc +14.26%)
[00:56:37] env0_seed_3 | Stage transition: BLENDING → SHADOWING
[00:56:40] env1_seed_1 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_1' (norm, 0.1K params)
[00:56:40] env3_seed_2 | Stage transition: PROBATIONARY → FOSSILIZED
[00:56:40] env3_seed_2 | Fossilized (conv_enhance, Δacc +11.63%)
    [env3] Fossilized 'env3_seed_2' (conv_enhance, Δacc +11.63%)
[00:56:40] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[00:56:43] env0_seed_3 | Stage transition: SHADOWING → PROBATIONARY
[00:56:57] env1_seed_1 | Stage transition: TRAINING → BLENDING
[00:57:14] env1_seed_1 | Stage transition: BLENDING → SHADOWING
[00:57:14] env0_seed_3 | Stage transition: PROBATIONARY → FOSSILIZED
[00:57:14] env0_seed_3 | Fossilized (conv_enhance, Δacc +5.95%)
    [env0] Fossilized 'env0_seed_3' (conv_enhance, Δacc +5.95%)
[00:57:21] env1_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[00:57:21] env1_seed_1 | Stage transition: PROBATIONARY → CULLED
[00:57:21] env1_seed_1 | Culled (norm, Δacc +1.61%)
    [env1] Culled 'env1_seed_1' (norm, Δacc +1.61%)
[00:57:24] env1_seed_2 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_2' (depthwise, 4.8K params)
[00:57:24] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[00:57:42] env2_seed_0 | Stage transition: FOSSILIZED → CULLED
[00:57:42] env2_seed_0 | Culled (conv_enhance, Δacc +20.40%)
    [env2] Culled 'env2_seed_0' (conv_enhance, Δacc +20.40%)
[00:57:55] env2_seed_1 | Germinated (norm, 0.1K params)
    [env2] Germinated 'env2_seed_1' (norm, 0.1K params)
[00:57:56] env2_seed_1 | Stage transition: GERMINATED → TRAINING
[00:58:05] env1_seed_2 | Stage transition: TRAINING → BLENDING
[00:58:14] env2_seed_1 | Stage transition: TRAINING → BLENDING
[00:58:23] env0_seed_3 | Stage transition: FOSSILIZED → CULLED
[00:58:23] env0_seed_3 | Culled (conv_enhance, Δacc +6.56%)
    [env0] Culled 'env0_seed_3' (conv_enhance, Δacc +6.56%)
[00:58:28] env1_seed_2 | Stage transition: BLENDING → SHADOWING
[00:58:34] env1_seed_2 | Stage transition: SHADOWING → PROBATIONARY
[00:58:34] env2_seed_1 | Stage transition: BLENDING → SHADOWING
[00:58:41] env2_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[00:58:47] env0_seed_4 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_4' (conv_enhance, 74.0K params)
[00:58:47] env1_seed_2 | Stage transition: PROBATIONARY → CULLED
[00:58:47] env1_seed_2 | Culled (depthwise, Δacc +0.86%)
    [env1] Culled 'env1_seed_2' (depthwise, Δacc +0.86%)
[00:58:47] env0_seed_4 | Stage transition: GERMINATED → TRAINING
[00:58:50] env1_seed_3 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_3' (conv_enhance, 74.0K params)
[00:58:50] env1_seed_3 | Stage transition: GERMINATED → TRAINING
[00:58:59] env2_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[00:58:59] env2_seed_1 | Fossilized (norm, Δacc +9.23%)
    [env2] Fossilized 'env2_seed_1' (norm, Δacc +9.23%)
[00:59:04] env0_seed_4 | Stage transition: TRAINING → BLENDING
[00:59:27] env0_seed_4 | Stage transition: BLENDING → SHADOWING
[00:59:27] env1_seed_3 | Stage transition: TRAINING → CULLED
[00:59:27] env1_seed_3 | Culled (conv_enhance, Δacc -4.82%)
    [env1] Culled 'env1_seed_3' (conv_enhance, Δacc -4.82%)
[00:59:30] env1_seed_4 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_4' (norm, 0.1K params)
[00:59:30] env1_seed_4 | Stage transition: GERMINATED → TRAINING
[00:59:34] env0_seed_4 | Stage transition: SHADOWING → PROBATIONARY
[00:59:34] env0_seed_4 | Stage transition: PROBATIONARY → FOSSILIZED
[00:59:34] env0_seed_4 | Fossilized (conv_enhance, Δacc +8.53%)
    [env0] Fossilized 'env0_seed_4' (conv_enhance, Δacc +8.53%)
[00:59:37] env2_seed_1 | Stage transition: FOSSILIZED → CULLED
[00:59:37] env2_seed_1 | Culled (norm, Δacc +9.66%)
    [env2] Culled 'env2_seed_1' (norm, Δacc +9.66%)
[00:59:40] env2_seed_2 | Germinated (depthwise, 4.8K params)
    [env2] Germinated 'env2_seed_2' (depthwise, 4.8K params)
[00:59:41] env2_seed_2 | Stage transition: GERMINATED → TRAINING
Batch 8: Episodes 32/200
  Env accuracies: ['76.8%', '75.3%', '73.9%', '75.2%']
  Avg acc: 75.3% (rolling: 74.0%)
  Avg reward: 6.5
  Actions: {'WAIT': 42, 'GERMINATE_NORM': 80, 'GERMINATE_ATTENTION': 43, 'GERMINATE_DEPTHWISE': 43, 'GERMINATE_CONV_ENHANCE': 23, 'FOSSILIZE': 56, 'CULL': 13}
  Successful: {'WAIT': 42, 'GERMINATE_NORM': 4, 'GERMINATE_ATTENTION': 1, 'GERMINATE_DEPTHWISE': 4, 'GERMINATE_CONV_ENHANCE': 7, 'FOSSILIZE': 6, 'CULL': 12}
  Policy loss: -0.0234, Value loss: 57.2831, Entropy: 1.8132, Entropy coef: 0.1589
[00:59:48] env1_seed_0 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_0' (conv_enhance, 74.0K params)
[00:59:48] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[00:59:51] env2_seed_0 | Germinated (depthwise, 4.8K params)
    [env2] Germinated 'env2_seed_0' (depthwise, 4.8K params)
[00:59:51] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[00:59:56] env0_seed_0 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_0' (conv_enhance, 74.0K params)
[00:59:56] env3_seed_0 | Germinated (conv_enhance, 74.0K params)
    [env3] Germinated 'env3_seed_0' (conv_enhance, 74.0K params)
[00:59:56] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[00:59:56] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[01:00:09] env1_seed_0 | Stage transition: TRAINING → BLENDING
[01:00:15] env2_seed_0 | Stage transition: TRAINING → BLENDING
[01:00:15] env2_seed_0 | Stage transition: BLENDING → CULLED
[01:00:15] env2_seed_0 | Culled (depthwise, Δacc +7.49%)
    [env2] Culled 'env2_seed_0' (depthwise, Δacc +7.49%)
[01:00:20] env0_seed_0 | Stage transition: TRAINING → BLENDING
[01:00:20] env3_seed_0 | Stage transition: TRAINING → BLENDING
[01:00:36] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[01:00:36] env2_seed_1 | Germinated (depthwise, 4.8K params)
    [env2] Germinated 'env2_seed_1' (depthwise, 4.8K params)
[01:00:36] env2_seed_1 | Stage transition: GERMINATED → TRAINING
[01:00:46] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[01:00:46] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:00:46] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[01:00:53] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:00:53] env2_seed_1 | Stage transition: TRAINING → BLENDING
[01:00:53] env3_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:00:56] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:00:56] env0_seed_0 | Fossilized (conv_enhance, Δacc +12.07%)
    [env0] Fossilized 'env0_seed_0' (conv_enhance, Δacc +12.07%)
[01:01:03] env3_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:01:03] env3_seed_0 | Fossilized (conv_enhance, Δacc +17.51%)
    [env3] Fossilized 'env3_seed_0' (conv_enhance, Δacc +17.51%)
[01:01:10] env2_seed_1 | Stage transition: BLENDING → SHADOWING
[01:01:10] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:01:10] env1_seed_0 | Fossilized (conv_enhance, Δacc +19.76%)
    [env1] Fossilized 'env1_seed_0' (conv_enhance, Δacc +19.76%)
[01:01:16] env2_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[01:01:29] env0_seed_0 | Stage transition: FOSSILIZED → CULLED
[01:01:29] env0_seed_0 | Culled (conv_enhance, Δacc +16.94%)
    [env0] Culled 'env0_seed_0' (conv_enhance, Δacc +16.94%)
[01:01:32] env1_seed_0 | Stage transition: FOSSILIZED → CULLED
[01:01:32] env1_seed_0 | Culled (conv_enhance, Δacc +20.93%)
    [env1] Culled 'env1_seed_0' (conv_enhance, Δacc +20.93%)
[01:01:35] env0_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_1' (conv_enhance, 74.0K params)
[01:01:35] env1_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_1' (conv_enhance, 74.0K params)
[01:01:35] env2_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[01:01:35] env2_seed_1 | Fossilized (depthwise, Δacc +8.56%)
    [env2] Fossilized 'env2_seed_1' (depthwise, Δacc +8.56%)
[01:01:35] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[01:01:35] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[01:01:54] env0_seed_1 | Stage transition: TRAINING → BLENDING
[01:02:12] env1_seed_1 | Stage transition: TRAINING → BLENDING
[01:02:12] env0_seed_1 | Stage transition: BLENDING → CULLED
[01:02:12] env0_seed_1 | Culled (conv_enhance, Δacc +10.64%)
    [env0] Culled 'env0_seed_1' (conv_enhance, Δacc +10.64%)
[01:02:27] env0_seed_2 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_2' (attention, 2.0K params)
[01:02:27] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[01:02:31] env1_seed_1 | Stage transition: BLENDING → SHADOWING
[01:02:31] env0_seed_2 | Stage transition: TRAINING → CULLED
[01:02:31] env0_seed_2 | Culled (attention, Δacc +0.00%)
    [env0] Culled 'env0_seed_2' (attention, Δacc +0.00%)
[01:02:35] env0_seed_3 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_3' (conv_enhance, 74.0K params)
[01:02:35] env0_seed_3 | Stage transition: GERMINATED → TRAINING
[01:02:38] env1_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[01:02:38] env1_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[01:02:38] env1_seed_1 | Fossilized (conv_enhance, Δacc +7.94%)
    [env1] Fossilized 'env1_seed_1' (conv_enhance, Δacc +7.94%)
[01:03:05] env0_seed_3 | Stage transition: TRAINING → BLENDING
[01:03:09] env1_seed_1 | Stage transition: FOSSILIZED → CULLED
[01:03:09] env1_seed_1 | Culled (conv_enhance, Δacc +8.85%)
    [env1] Culled 'env1_seed_1' (conv_enhance, Δacc +8.85%)
[01:03:12] env1_seed_2 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_2' (norm, 0.1K params)
[01:03:12] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[01:03:16] env0_seed_3 | Stage transition: BLENDING → CULLED
[01:03:16] env0_seed_3 | Culled (conv_enhance, Δacc +1.35%)
    [env0] Culled 'env0_seed_3' (conv_enhance, Δacc +1.35%)
[01:03:30] env1_seed_2 | Stage transition: TRAINING → BLENDING
[01:03:33] env0_seed_4 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_4' (norm, 0.1K params)
[01:03:33] env0_seed_4 | Stage transition: GERMINATED → TRAINING
[01:03:51] env0_seed_4 | Stage transition: TRAINING → BLENDING
[01:03:51] env1_seed_2 | Stage transition: BLENDING → SHADOWING
[01:03:58] env1_seed_2 | Stage transition: SHADOWING → PROBATIONARY
[01:03:58] env1_seed_2 | Stage transition: PROBATIONARY → FOSSILIZED
[01:03:58] env1_seed_2 | Fossilized (norm, Δacc +5.95%)
    [env1] Fossilized 'env1_seed_2' (norm, Δacc +5.95%)
[01:04:07] env0_seed_4 | Stage transition: BLENDING → SHADOWING
[01:04:13] env0_seed_4 | Stage transition: SHADOWING → PROBATIONARY
[01:04:13] env0_seed_4 | Stage transition: PROBATIONARY → FOSSILIZED
[01:04:13] env0_seed_4 | Fossilized (norm, Δacc +2.08%)
    [env0] Fossilized 'env0_seed_4' (norm, Δacc +2.08%)
Batch 9: Episodes 36/200
  Env accuracies: ['76.6%', '77.5%', '71.6%', '68.2%']
  Avg acc: 73.5% (rolling: 73.9%)
  Avg reward: 13.0
  Actions: {'WAIT': 52, 'GERMINATE_NORM': 52, 'GERMINATE_ATTENTION': 32, 'GERMINATE_DEPTHWISE': 42, 'GERMINATE_CONV_ENHANCE': 33, 'FOSSILIZE': 78, 'CULL': 11}
  Successful: {'WAIT': 52, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 1, 'GERMINATE_DEPTHWISE': 2, 'GERMINATE_CONV_ENHANCE': 6, 'FOSSILIZE': 7, 'CULL': 7}
  Policy loss: -0.0144, Value loss: 58.9292, Entropy: 1.8272, Entropy coef: 0.1538
[01:04:35] env3_seed_0 | Germinated (conv_enhance, 74.0K params)
    [env3] Germinated 'env3_seed_0' (conv_enhance, 74.0K params)
[01:04:35] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[01:04:38] env0_seed_0 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_0' (depthwise, 4.8K params)
[01:04:38] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[01:04:38] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[01:04:38] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[01:04:43] env2_seed_0 | Germinated (attention, 2.0K params)
    [env2] Germinated 'env2_seed_0' (attention, 2.0K params)
[01:04:43] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[01:04:55] env3_seed_0 | Stage transition: TRAINING → BLENDING
[01:05:01] env1_seed_0 | Stage transition: TRAINING → BLENDING
[01:05:07] env0_seed_0 | Stage transition: TRAINING → BLENDING
[01:05:14] env2_seed_0 | Stage transition: TRAINING → BLENDING
[01:05:26] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[01:05:31] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[01:05:34] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[01:05:34] env3_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:05:37] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:05:37] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[01:05:41] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:05:41] env3_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:05:41] env3_seed_0 | Fossilized (conv_enhance, Δacc +16.69%)
    [env3] Fossilized 'env3_seed_0' (conv_enhance, Δacc +16.69%)
[01:05:44] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:05:44] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:05:44] env0_seed_0 | Fossilized (depthwise, Δacc +20.21%)
    [env0] Fossilized 'env0_seed_0' (depthwise, Δacc +20.21%)
[01:05:44] env2_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:05:44] env2_seed_0 | Fossilized (attention, Δacc +3.16%)
    [env2] Fossilized 'env2_seed_0' (attention, Δacc +3.16%)
[01:05:53] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:05:53] env1_seed_0 | Fossilized (attention, Δacc +14.48%)
    [env1] Fossilized 'env1_seed_0' (attention, Δacc +14.48%)
[01:05:59] env3_seed_0 | Stage transition: FOSSILIZED → CULLED
[01:05:59] env3_seed_0 | Culled (conv_enhance, Δacc +17.93%)
    [env3] Culled 'env3_seed_0' (conv_enhance, Δacc +17.93%)
[01:06:02] env3_seed_1 | Germinated (depthwise, 4.8K params)
    [env3] Germinated 'env3_seed_1' (depthwise, 4.8K params)
[01:06:02] env3_seed_1 | Stage transition: GERMINATED → TRAINING
[01:06:15] env1_seed_0 | Stage transition: FOSSILIZED → CULLED
[01:06:15] env1_seed_0 | Culled (attention, Δacc +16.85%)
    [env1] Culled 'env1_seed_0' (attention, Δacc +16.85%)
[01:06:19] env3_seed_1 | Stage transition: TRAINING → CULLED
[01:06:19] env3_seed_1 | Culled (depthwise, Δacc +5.10%)
    [env3] Culled 'env3_seed_1' (depthwise, Δacc +5.10%)
[01:06:22] env0_seed_0 | Stage transition: FOSSILIZED → CULLED
[01:06:22] env0_seed_0 | Culled (depthwise, Δacc +19.18%)
    [env0] Culled 'env0_seed_0' (depthwise, Δacc +19.18%)
[01:06:22] env1_seed_1 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_1' (norm, 0.1K params)
[01:06:22] env3_seed_2 | Germinated (conv_enhance, 74.0K params)
    [env3] Germinated 'env3_seed_2' (conv_enhance, 74.0K params)
[01:06:22] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[01:06:22] env3_seed_2 | Stage transition: GERMINATED → TRAINING
[01:06:26] env0_seed_1 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_1' (norm, 0.1K params)
[01:06:26] env1_seed_1 | Stage transition: TRAINING → CULLED
[01:06:26] env1_seed_1 | Culled (norm, Δacc +0.00%)
    [env1] Culled 'env1_seed_1' (norm, Δacc +0.00%)
[01:06:26] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[01:06:30] env1_seed_2 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_2' (norm, 0.1K params)
[01:06:30] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[01:06:45] env0_seed_1 | Stage transition: TRAINING → BLENDING
[01:06:45] env3_seed_2 | Stage transition: TRAINING → BLENDING
[01:06:55] env1_seed_2 | Stage transition: TRAINING → BLENDING
[01:07:10] env0_seed_1 | Stage transition: BLENDING → SHADOWING
[01:07:10] env3_seed_2 | Stage transition: BLENDING → SHADOWING
[01:07:16] env0_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[01:07:16] env1_seed_2 | Stage transition: BLENDING → SHADOWING
[01:07:16] env3_seed_2 | Stage transition: SHADOWING → PROBATIONARY
[01:07:23] env1_seed_2 | Stage transition: SHADOWING → PROBATIONARY
[01:07:35] env0_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[01:07:35] env0_seed_1 | Fossilized (norm, Δacc +7.43%)
    [env0] Fossilized 'env0_seed_1' (norm, Δacc +7.43%)
[01:07:35] env3_seed_2 | Stage transition: PROBATIONARY → FOSSILIZED
[01:07:35] env3_seed_2 | Fossilized (conv_enhance, Δacc +10.01%)
    [env3] Fossilized 'env3_seed_2' (conv_enhance, Δacc +10.01%)
[01:07:41] env1_seed_2 | Stage transition: PROBATIONARY → FOSSILIZED
[01:07:41] env1_seed_2 | Fossilized (norm, Δacc +7.31%)
    [env1] Fossilized 'env1_seed_2' (norm, Δacc +7.31%)
[01:08:57] env1_seed_2 | Stage transition: FOSSILIZED → CULLED
[01:08:57] env1_seed_2 | Culled (norm, Δacc +6.97%)
    [env1] Culled 'env1_seed_2' (norm, Δacc +6.97%)
[01:09:00] env1_seed_3 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_3' (depthwise, 4.8K params)
[01:09:00] env1_seed_3 | Stage transition: GERMINATED → TRAINING
[01:09:04] env2_seed_0 | Stage transition: FOSSILIZED → CULLED
[01:09:04] env2_seed_0 | Culled (attention, Δacc +16.91%)
    [env2] Culled 'env2_seed_0' (attention, Δacc +16.91%)
Batch 10: Episodes 40/200
  Env accuracies: ['75.4%', '58.1%', '73.5%', '77.0%']
  Avg acc: 71.0% (rolling: 73.6%)
  Avg reward: 22.9
  Actions: {'WAIT': 49, 'GERMINATE_NORM': 68, 'GERMINATE_ATTENTION': 50, 'GERMINATE_DEPTHWISE': 42, 'GERMINATE_CONV_ENHANCE': 29, 'FOSSILIZE': 55, 'CULL': 7}
  Successful: {'WAIT': 49, 'GERMINATE_NORM': 3, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 3, 'GERMINATE_CONV_ENHANCE': 2, 'FOSSILIZE': 7, 'CULL': 7}
  Policy loss: -0.0231, Value loss: 47.3296, Entropy: 1.8068, Entropy coef: 0.1486

Blueprint Stats
  ---------------------------------------------------------------------------

Blueprint       Germ  Foss  Cull   Rate     ΔAcc    Churn
  ---------------------------------------------------------------------------

  attention         49     7    46  13.2%   +5.07%   +3.64%
  conv_enhance      62    17    52  24.6%  +13.14%   +7.05%
  depthwise         54    10    41  19.6%   +9.08%   +4.10%
  norm              47    18    36  33.3%   +6.65%   +4.25%
Seed Scoreboard (env 0):
  Fossilized: 11 (+317.4K params, +335.0% of host)
  Culled: 51
  Avg fossilize age: 13.8 epochs
  Avg cull age: 10.6 epochs
  Compute cost: 2.31x baseline
  Distribution: conv_enhance x4, depthwise x4, attention x1, norm x2
Seed Scoreboard (env 1):
  Fossilized: 14 (+234.6K params, +247.6% of host)
  Culled: 47
  Avg fossilize age: 14.6 epochs
  Avg cull age: 11.4 epochs
  Compute cost: 2.12x baseline
  Distribution: norm x8, depthwise x2, conv_enhance x3, attention x1
Seed Scoreboard (env 2):
  Fossilized: 13 (+314.1K params, +331.5% of host)
  Culled: 38
  Avg fossilize age: 15.7 epochs
  Avg cull age: 13.0 epochs
  Compute cost: 3.22x baseline
  Distribution: conv_enhance x4, attention x4, norm x3, depthwise x2
Seed Scoreboard (env 3):
  Fossilized: 14 (+456.2K params, +481.4% of host)
  Culled: 39
  Avg fossilize age: 15.4 epochs
  Avg cull age: 8.7 epochs
  Compute cost: 2.51x baseline
  Distribution: conv_enhance x6, norm x5, depthwise x2, attention x1

[01:09:13] env0_seed_0 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_0' (depthwise, 4.8K params)
[01:09:13] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[01:09:16] env3_seed_0 | Germinated (depthwise, 4.8K params)
    [env3] Germinated 'env3_seed_0' (depthwise, 4.8K params)
[01:09:16] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[01:09:20] env1_seed_0 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_0' (depthwise, 4.8K params)
[01:09:20] env2_seed_0 | Germinated (norm, 0.1K params)
    [env2] Germinated 'env2_seed_0' (norm, 0.1K params)
[01:09:20] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[01:09:20] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[01:09:33] env0_seed_0 | Stage transition: TRAINING → BLENDING
[01:09:39] env3_seed_0 | Stage transition: TRAINING → BLENDING
[01:09:45] env2_seed_0 | Stage transition: TRAINING → BLENDING
[01:10:03] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[01:10:03] env1_seed_0 | Stage transition: TRAINING → BLENDING
[01:10:08] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[01:10:13] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:10:13] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[01:10:16] env3_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:10:16] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:10:16] env0_seed_0 | Fossilized (depthwise, Δacc +11.23%)
    [env0] Fossilized 'env0_seed_0' (depthwise, Δacc +11.23%)
[01:10:20] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:10:23] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[01:10:29] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:10:29] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:10:29] env1_seed_0 | Fossilized (depthwise, Δacc +7.21%)
    [env1] Fossilized 'env1_seed_0' (depthwise, Δacc +7.21%)
[01:10:32] env2_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:10:32] env2_seed_0 | Fossilized (norm, Δacc +12.31%)
    [env2] Fossilized 'env2_seed_0' (norm, Δacc +12.31%)
[01:10:41] env3_seed_0 | Stage transition: PROBATIONARY → CULLED
[01:10:41] env3_seed_0 | Culled (depthwise, Δacc +9.87%)
    [env3] Culled 'env3_seed_0' (depthwise, Δacc +9.87%)
[01:10:50] env3_seed_1 | Germinated (attention, 2.0K params)
    [env3] Germinated 'env3_seed_1' (attention, 2.0K params)
[01:10:50] env3_seed_1 | Stage transition: GERMINATED → TRAINING
[01:11:03] env3_seed_1 | Stage transition: TRAINING → BLENDING
[01:11:20] env3_seed_1 | Stage transition: BLENDING → SHADOWING
[01:11:23] env2_seed_0 | Stage transition: FOSSILIZED → CULLED
[01:11:23] env2_seed_0 | Culled (norm, Δacc +16.00%)
    [env2] Culled 'env2_seed_0' (norm, Δacc +16.00%)
[01:11:26] env3_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[01:11:26] env2_seed_1 | Germinated (depthwise, 4.8K params)
    [env2] Germinated 'env2_seed_1' (depthwise, 4.8K params)
[01:11:26] env2_seed_1 | Stage transition: GERMINATED → TRAINING
[01:11:33] env3_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[01:11:33] env3_seed_1 | Fossilized (attention, Δacc +11.88%)
    [env3] Fossilized 'env3_seed_1' (attention, Δacc +11.88%)
[01:11:40] env2_seed_1 | Stage transition: TRAINING → BLENDING
[01:11:49] env1_seed_0 | Stage transition: FOSSILIZED → CULLED
[01:11:49] env1_seed_0 | Culled (depthwise, Δacc +8.78%)
    [env1] Culled 'env1_seed_0' (depthwise, Δacc +8.78%)
[01:11:53] env1_seed_1 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_1' (norm, 0.1K params)
[01:11:53] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[01:11:57] env2_seed_1 | Stage transition: BLENDING → SHADOWING
[01:12:04] env2_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[01:12:07] env1_seed_1 | Stage transition: TRAINING → BLENDING
[01:12:07] env3_seed_1 | Stage transition: FOSSILIZED → CULLED
[01:12:07] env3_seed_1 | Culled (attention, Δacc +9.44%)
    [env3] Culled 'env3_seed_1' (attention, Δacc +9.44%)
[01:12:14] env3_seed_2 | Germinated (norm, 0.1K params)
    [env3] Germinated 'env3_seed_2' (norm, 0.1K params)
[01:12:14] env3_seed_2 | Stage transition: GERMINATED → TRAINING
[01:12:18] env2_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[01:12:18] env2_seed_1 | Fossilized (depthwise, Δacc +4.56%)
    [env2] Fossilized 'env2_seed_1' (depthwise, Δacc +4.56%)
[01:12:25] env1_seed_1 | Stage transition: BLENDING → SHADOWING
[01:12:32] env1_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[01:12:35] env3_seed_2 | Stage transition: TRAINING → BLENDING
[01:12:39] env1_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[01:12:39] env1_seed_1 | Fossilized (norm, Δacc +10.48%)
    [env1] Fossilized 'env1_seed_1' (norm, Δacc +10.48%)
[01:12:52] env3_seed_2 | Stage transition: BLENDING → SHADOWING
[01:12:58] env3_seed_2 | Stage transition: SHADOWING → PROBATIONARY
[01:12:58] env3_seed_2 | Stage transition: PROBATIONARY → FOSSILIZED
[01:12:58] env3_seed_2 | Fossilized (norm, Δacc +1.65%)
    [env3] Fossilized 'env3_seed_2' (norm, Δacc +1.65%)
[01:13:07] env3_seed_2 | Stage transition: FOSSILIZED → CULLED
[01:13:07] env3_seed_2 | Culled (norm, Δacc +2.91%)
    [env3] Culled 'env3_seed_2' (norm, Δacc +2.91%)
[01:13:17] env3_seed_3 | Germinated (norm, 0.1K params)
    [env3] Germinated 'env3_seed_3' (norm, 0.1K params)
[01:13:17] env3_seed_3 | Stage transition: GERMINATED → TRAINING
[01:13:34] env3_seed_3 | Stage transition: TRAINING → BLENDING
Batch 11: Episodes 44/200
  Env accuracies: ['73.8%', '77.1%', '70.9%', '72.0%']
  Avg acc: 73.4% (rolling: 73.7%)
  Avg reward: 39.8
  Actions: {'WAIT': 62, 'GERMINATE_NORM': 63, 'GERMINATE_ATTENTION': 41, 'GERMINATE_DEPTHWISE': 34, 'GERMINATE_CONV_ENHANCE': 34, 'FOSSILIZE': 59, 'CULL': 7}
  Successful: {'WAIT': 62, 'GERMINATE_NORM': 4, 'GERMINATE_ATTENTION': 1, 'GERMINATE_DEPTHWISE': 4, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 7, 'CULL': 5}
  Policy loss: -0.0336, Value loss: 37.5370, Entropy: 1.7978, Entropy coef: 0.1435
[01:13:43] env2_seed_0 | Germinated (depthwise, 4.8K params)
    [env2] Germinated 'env2_seed_0' (depthwise, 4.8K params)
[01:13:43] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[01:13:49] env3_seed_0 | Germinated (norm, 0.1K params)
    [env3] Germinated 'env3_seed_0' (norm, 0.1K params)
[01:13:49] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[01:13:54] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[01:13:54] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[01:13:54] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[01:13:54] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[01:14:00] env2_seed_0 | Stage transition: TRAINING → BLENDING
[01:14:12] env3_seed_0 | Stage transition: TRAINING → BLENDING
[01:14:18] env0_seed_0 | Stage transition: TRAINING → BLENDING
[01:14:18] env1_seed_0 | Stage transition: TRAINING → BLENDING
[01:14:30] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[01:14:39] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:14:39] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[01:14:44] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[01:14:44] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[01:14:44] env2_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:14:44] env2_seed_0 | Fossilized (depthwise, Δacc +6.91%)
    [env2] Fossilized 'env2_seed_0' (depthwise, Δacc +6.91%)
[01:14:47] env3_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:14:50] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:14:50] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:14:50] env1_seed_0 | Stage transition: PROBATIONARY → CULLED
[01:14:50] env1_seed_0 | Culled (attention, Δacc +3.72%)
    [env1] Culled 'env1_seed_0' (attention, Δacc +3.72%)
[01:14:50] env3_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:14:50] env3_seed_0 | Fossilized (norm, Δacc +9.26%)
    [env3] Fossilized 'env3_seed_0' (norm, Δacc +9.26%)
[01:14:53] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:14:53] env0_seed_0 | Fossilized (norm, Δacc +16.54%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +16.54%)
[01:14:53] env1_seed_1 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_1' (attention, 2.0K params)
[01:14:53] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[01:14:56] env1_seed_1 | Stage transition: TRAINING → CULLED
[01:14:56] env1_seed_1 | Culled (attention, Δacc +0.00%)
    [env1] Culled 'env1_seed_1' (attention, Δacc +0.00%)
[01:14:59] env1_seed_2 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_2' (attention, 2.0K params)
[01:14:59] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[01:15:12] env1_seed_2 | Stage transition: TRAINING → BLENDING
[01:15:29] env1_seed_2 | Stage transition: BLENDING → SHADOWING
[01:15:35] env1_seed_2 | Stage transition: SHADOWING → PROBATIONARY
[01:15:41] env1_seed_2 | Stage transition: PROBATIONARY → FOSSILIZED
[01:15:41] env1_seed_2 | Fossilized (attention, Δacc +4.80%)
    [env1] Fossilized 'env1_seed_2' (attention, Δacc +4.80%)
[01:15:59] env2_seed_0 | Stage transition: FOSSILIZED → CULLED
[01:15:59] env2_seed_0 | Culled (depthwise, Δacc +14.81%)
    [env2] Culled 'env2_seed_0' (depthwise, Δacc +14.81%)
[01:16:02] env2_seed_1 | Germinated (depthwise, 4.8K params)
    [env2] Germinated 'env2_seed_1' (depthwise, 4.8K params)
[01:16:02] env2_seed_1 | Stage transition: GERMINATED → TRAINING
[01:16:35] env2_seed_1 | Stage transition: TRAINING → BLENDING
[01:16:51] env2_seed_1 | Stage transition: BLENDING → SHADOWING
[01:16:57] env2_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[01:17:09] env2_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[01:17:09] env2_seed_1 | Fossilized (depthwise, Δacc +2.37%)
    [env2] Fossilized 'env2_seed_1' (depthwise, Δacc +2.37%)
[01:17:24] env1_seed_2 | Stage transition: FOSSILIZED → CULLED
[01:17:24] env1_seed_2 | Culled (attention, Δacc +3.05%)
    [env1] Culled 'env1_seed_2' (attention, Δacc +3.05%)
[01:17:30] env1_seed_3 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_3' (norm, 0.1K params)
[01:17:30] env1_seed_3 | Stage transition: GERMINATED → TRAINING
Batch 12: Episodes 48/200
  Env accuracies: ['77.0%', '73.5%', '66.2%', '75.7%']
  Avg acc: 73.1% (rolling: 73.8%)
  Avg reward: 44.9
  Actions: {'WAIT': 63, 'GERMINATE_NORM': 69, 'GERMINATE_ATTENTION': 32, 'GERMINATE_DEPTHWISE': 50, 'GERMINATE_CONV_ENHANCE': 31, 'FOSSILIZE': 51, 'CULL': 4}
  Successful: {'WAIT': 63, 'GERMINATE_NORM': 3, 'GERMINATE_ATTENTION': 3, 'GERMINATE_DEPTHWISE': 2, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 5, 'CULL': 4}
  Policy loss: -0.0167, Value loss: 33.0311, Entropy: 1.7708, Entropy coef: 0.1384
[01:18:03] env2_seed_0 | Germinated (depthwise, 4.8K params)
    [env2] Germinated 'env2_seed_0' (depthwise, 4.8K params)
[01:18:03] env3_seed_0 | Germinated (attention, 2.0K params)
    [env3] Germinated 'env3_seed_0' (attention, 2.0K params)
[01:18:03] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[01:18:03] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[01:18:07] env0_seed_0 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_0' (attention, 2.0K params)
[01:18:07] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[01:18:21] env3_seed_0 | Stage transition: TRAINING → BLENDING
[01:18:21] env1_seed_0 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_0' (conv_enhance, 74.0K params)
[01:18:21] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[01:18:27] env0_seed_0 | Stage transition: TRAINING → BLENDING
[01:18:27] env2_seed_0 | Stage transition: TRAINING → BLENDING
[01:18:39] env1_seed_0 | Stage transition: TRAINING → CULLED
[01:18:39] env1_seed_0 | Culled (conv_enhance, Δacc +5.22%)
    [env1] Culled 'env1_seed_0' (conv_enhance, Δacc +5.22%)
[01:18:49] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[01:18:49] env1_seed_1 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_1' (attention, 2.0K params)
[01:18:49] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[01:18:54] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[01:18:54] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[01:18:57] env3_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:18:57] env3_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:18:57] env3_seed_0 | Fossilized (attention, Δacc +11.56%)
    [env3] Fossilized 'env3_seed_0' (attention, Δacc +11.56%)
[01:19:00] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:19:00] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:19:00] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:19:00] env0_seed_0 | Fossilized (attention, Δacc +9.16%)
    [env0] Fossilized 'env0_seed_0' (attention, Δacc +9.16%)
[01:19:03] env1_seed_1 | Stage transition: TRAINING → BLENDING
[01:19:13] env2_seed_0 | Stage transition: PROBATIONARY → CULLED
[01:19:13] env2_seed_0 | Culled (depthwise, Δacc +14.01%)
    [env2] Culled 'env2_seed_0' (depthwise, Δacc +14.01%)
[01:19:17] env2_seed_1 | Germinated (norm, 0.1K params)
    [env2] Germinated 'env2_seed_1' (norm, 0.1K params)
[01:19:17] env2_seed_1 | Stage transition: GERMINATED → TRAINING
[01:19:21] env1_seed_1 | Stage transition: BLENDING → SHADOWING
[01:19:27] env1_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[01:19:27] env0_seed_0 | Stage transition: FOSSILIZED → CULLED
[01:19:27] env0_seed_0 | Culled (attention, Δacc +16.35%)
    [env0] Culled 'env0_seed_0' (attention, Δacc +16.35%)
[01:19:31] env3_seed_0 | Stage transition: FOSSILIZED → CULLED
[01:19:31] env3_seed_0 | Culled (attention, Δacc +14.23%)
    [env3] Culled 'env3_seed_0' (attention, Δacc +14.23%)
[01:19:34] env2_seed_1 | Stage transition: TRAINING → BLENDING
[01:19:34] env1_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[01:19:34] env1_seed_1 | Fossilized (attention, Δacc +10.17%)
    [env1] Fossilized 'env1_seed_1' (attention, Δacc +10.17%)
[01:19:37] env0_seed_1 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_1' (norm, 0.1K params)
[01:19:37] env3_seed_1 | Germinated (attention, 2.0K params)
    [env3] Germinated 'env3_seed_1' (attention, 2.0K params)
[01:19:37] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[01:19:37] env3_seed_1 | Stage transition: GERMINATED → TRAINING
[01:19:46] env3_seed_1 | Stage transition: TRAINING → CULLED
[01:19:46] env3_seed_1 | Culled (attention, Δacc -1.53%)
    [env3] Culled 'env3_seed_1' (attention, Δacc -1.53%)
[01:19:50] env2_seed_1 | Stage transition: BLENDING → CULLED
[01:19:50] env2_seed_1 | Culled (norm, Δacc +2.73%)
    [env2] Culled 'env2_seed_1' (norm, Δacc +2.73%)
[01:19:50] env3_seed_2 | Germinated (norm, 0.1K params)
    [env3] Germinated 'env3_seed_2' (norm, 0.1K params)
[01:19:50] env3_seed_2 | Stage transition: GERMINATED → TRAINING
[01:19:54] env0_seed_1 | Stage transition: TRAINING → BLENDING
[01:20:03] env2_seed_2 | Germinated (depthwise, 4.8K params)
    [env2] Germinated 'env2_seed_2' (depthwise, 4.8K params)
[01:20:03] env2_seed_2 | Stage transition: GERMINATED → TRAINING
[01:20:13] env3_seed_2 | Stage transition: TRAINING → BLENDING
[01:20:18] env0_seed_1 | Stage transition: BLENDING → SHADOWING
[01:20:27] env0_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[01:20:27] env2_seed_2 | Stage transition: TRAINING → BLENDING
[01:20:27] env3_seed_2 | Stage transition: BLENDING → CULLED
[01:20:27] env3_seed_2 | Culled (norm, Δacc +3.82%)
    [env3] Culled 'env3_seed_2' (norm, Δacc +3.82%)
[01:20:37] env1_seed_1 | Stage transition: FOSSILIZED → CULLED
[01:20:37] env1_seed_1 | Culled (attention, Δacc +11.43%)
    [env1] Culled 'env1_seed_1' (attention, Δacc +11.43%)
[01:20:37] env3_seed_3 | Germinated (attention, 2.0K params)
    [env3] Germinated 'env3_seed_3' (attention, 2.0K params)
[01:20:37] env3_seed_3 | Stage transition: GERMINATED → TRAINING
[01:20:46] env2_seed_2 | Stage transition: BLENDING → SHADOWING
[01:20:46] env1_seed_2 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_2' (conv_enhance, 74.0K params)
[01:20:46] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[01:20:53] env2_seed_2 | Stage transition: SHADOWING → PROBATIONARY
[01:21:05] env3_seed_3 | Stage transition: TRAINING → CULLED
[01:21:05] env3_seed_3 | Culled (attention, Δacc -0.96%)
    [env3] Culled 'env3_seed_3' (attention, Δacc -0.96%)
[01:21:08] env1_seed_2 | Stage transition: TRAINING → BLENDING
[01:21:08] env0_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[01:21:08] env0_seed_1 | Fossilized (norm, Δacc +8.68%)
    [env0] Fossilized 'env0_seed_1' (norm, Δacc +8.68%)
[01:21:08] env3_seed_4 | Germinated (conv_enhance, 74.0K params)
    [env3] Germinated 'env3_seed_4' (conv_enhance, 74.0K params)
[01:21:08] env3_seed_4 | Stage transition: GERMINATED → TRAINING
[01:21:20] env1_seed_2 | Stage transition: BLENDING → CULLED
[01:21:20] env1_seed_2 | Culled (conv_enhance, Δacc +2.88%)
    [env1] Culled 'env1_seed_2' (conv_enhance, Δacc +2.88%)
[01:21:24] env1_seed_3 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_3' (attention, 2.0K params)
[01:21:24] env1_seed_3 | Stage transition: GERMINATED → TRAINING
[01:21:27] env2_seed_2 | Stage transition: PROBATIONARY → FOSSILIZED
[01:21:27] env2_seed_2 | Fossilized (depthwise, Δacc +0.61%)
    [env2] Fossilized 'env2_seed_2' (depthwise, Δacc +0.61%)
[01:21:39] env1_seed_3 | Stage transition: TRAINING → BLENDING
[01:21:43] env2_seed_2 | Stage transition: FOSSILIZED → CULLED
[01:21:43] env2_seed_2 | Culled (depthwise, Δacc -2.12%)
    [env2] Culled 'env2_seed_2' (depthwise, Δacc -2.12%)
[01:21:58] env1_seed_3 | Stage transition: BLENDING → SHADOWING
[01:21:58] env2_seed_3 | Germinated (attention, 2.0K params)
    [env2] Germinated 'env2_seed_3' (attention, 2.0K params)
[01:21:58] env2_seed_3 | Stage transition: GERMINATED → TRAINING
[01:22:03] env3_seed_4 | Stage transition: TRAINING → BLENDING
[01:22:07] env1_seed_3 | Stage transition: SHADOWING → PROBATIONARY
[01:22:12] env2_seed_3 | Stage transition: TRAINING → CULLED
[01:22:12] env2_seed_3 | Culled (attention, Δacc -3.48%)
    [env2] Culled 'env2_seed_3' (attention, Δacc -3.48%)
[01:22:15] env2_seed_4 | Germinated (depthwise, 4.8K params)
    [env2] Germinated 'env2_seed_4' (depthwise, 4.8K params)
[01:22:15] env2_seed_4 | Stage transition: GERMINATED → TRAINING
[01:22:25] env3_seed_4 | Stage transition: BLENDING → SHADOWING
[01:22:31] env3_seed_4 | Stage transition: SHADOWING → PROBATIONARY
[01:22:45] env1_seed_3 | Stage transition: PROBATIONARY → FOSSILIZED
[01:22:45] env1_seed_3 | Fossilized (attention, Δacc +3.91%)
    [env1] Fossilized 'env1_seed_3' (attention, Δacc +3.91%)
[01:22:45] env3_seed_4 | Stage transition: PROBATIONARY → CULLED
[01:22:45] env3_seed_4 | Culled (conv_enhance, Δacc +2.26%)
    [env3] Culled 'env3_seed_4' (conv_enhance, Δacc +2.26%)
[01:22:48] env3_seed_5 | Germinated (norm, 0.1K params)
    [env3] Germinated 'env3_seed_5' (norm, 0.1K params)
[01:22:48] env3_seed_5 | Stage transition: GERMINATED → TRAINING
Batch 13: Episodes 52/200
  Env accuracies: ['77.1%', '72.5%', '74.8%', '72.3%']
  Avg acc: 74.2% (rolling: 73.9%)
  Avg reward: 22.7
  Actions: {'WAIT': 54, 'GERMINATE_NORM': 49, 'GERMINATE_ATTENTION': 51, 'GERMINATE_DEPTHWISE': 47, 'GERMINATE_CONV_ENHANCE': 33, 'FOSSILIZE': 50, 'CULL': 16}
  Successful: {'WAIT': 54, 'GERMINATE_NORM': 4, 'GERMINATE_ATTENTION': 7, 'GERMINATE_DEPTHWISE': 3, 'GERMINATE_CONV_ENHANCE': 3, 'FOSSILIZE': 6, 'CULL': 13}
  Policy loss: -0.0126, Value loss: 38.1297, Entropy: 1.7841, Entropy coef: 0.1332
[01:23:03] env1_seed_0 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_0' (depthwise, 4.8K params)
[01:23:03] env2_seed_0 | Germinated (depthwise, 4.8K params)
    [env2] Germinated 'env2_seed_0' (depthwise, 4.8K params)
[01:23:03] env3_seed_0 | Germinated (depthwise, 4.8K params)
    [env3] Germinated 'env3_seed_0' (depthwise, 4.8K params)
[01:23:03] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[01:23:03] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[01:23:03] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[01:23:08] env0_seed_0 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_0' (depthwise, 4.8K params)
[01:23:08] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[01:23:26] env1_seed_0 | Stage transition: TRAINING → BLENDING
[01:23:26] env2_seed_0 | Stage transition: TRAINING → BLENDING
[01:23:26] env3_seed_0 | Stage transition: TRAINING → BLENDING
[01:23:33] env0_seed_0 | Stage transition: TRAINING → BLENDING
[01:23:58] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[01:23:58] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[01:23:58] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[01:24:01] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[01:24:04] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:24:04] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:24:04] env3_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:24:04] env2_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:24:04] env2_seed_0 | Fossilized (depthwise, Δacc +13.33%)
    [env2] Fossilized 'env2_seed_0' (depthwise, Δacc +13.33%)
[01:24:07] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:24:07] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:24:07] env0_seed_0 | Fossilized (depthwise, Δacc +2.29%)
    [env0] Fossilized 'env0_seed_0' (depthwise, Δacc +2.29%)
[01:24:16] env3_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:24:16] env3_seed_0 | Fossilized (depthwise, Δacc +12.38%)
    [env3] Fossilized 'env3_seed_0' (depthwise, Δacc +12.38%)
[01:24:22] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:24:22] env1_seed_0 | Fossilized (depthwise, Δacc +17.55%)
    [env1] Fossilized 'env1_seed_0' (depthwise, Δacc +17.55%)
[01:24:31] env2_seed_0 | Stage transition: FOSSILIZED → CULLED
[01:24:31] env2_seed_0 | Culled (depthwise, Δacc +9.85%)
    [env2] Culled 'env2_seed_0' (depthwise, Δacc +9.85%)
[01:24:37] env2_seed_1 | Germinated (depthwise, 4.8K params)
    [env2] Germinated 'env2_seed_1' (depthwise, 4.8K params)
[01:24:37] env2_seed_1 | Stage transition: GERMINATED → TRAINING
[01:24:50] env2_seed_1 | Stage transition: TRAINING → BLENDING
[01:25:07] env2_seed_1 | Stage transition: BLENDING → SHADOWING
[01:25:13] env2_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[01:25:13] env3_seed_0 | Stage transition: FOSSILIZED → CULLED
[01:25:13] env3_seed_0 | Culled (depthwise, Δacc +11.50%)
    [env3] Culled 'env3_seed_0' (depthwise, Δacc +11.50%)
[01:25:16] env2_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[01:25:16] env2_seed_1 | Fossilized (depthwise, Δacc +3.99%)
    [env2] Fossilized 'env2_seed_1' (depthwise, Δacc +3.99%)
[01:25:16] env3_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env3] Germinated 'env3_seed_1' (conv_enhance, 74.0K params)
[01:25:16] env3_seed_1 | Stage transition: GERMINATED → TRAINING
[01:25:23] env3_seed_1 | Stage transition: TRAINING → CULLED
[01:25:23] env3_seed_1 | Culled (conv_enhance, Δacc +6.48%)
    [env3] Culled 'env3_seed_1' (conv_enhance, Δacc +6.48%)
[01:25:26] env3_seed_2 | Germinated (norm, 0.1K params)
    [env3] Germinated 'env3_seed_2' (norm, 0.1K params)
[01:25:26] env3_seed_2 | Stage transition: GERMINATED → TRAINING
[01:25:53] env3_seed_2 | Stage transition: TRAINING → BLENDING
[01:25:56] env2_seed_1 | Stage transition: FOSSILIZED → CULLED
[01:25:56] env2_seed_1 | Culled (depthwise, Δacc +3.47%)
    [env2] Culled 'env2_seed_1' (depthwise, Δacc +3.47%)
[01:26:00] env2_seed_2 | Germinated (depthwise, 4.8K params)
    [env2] Germinated 'env2_seed_2' (depthwise, 4.8K params)
[01:26:00] env2_seed_2 | Stage transition: GERMINATED → TRAINING
[01:26:14] env3_seed_2 | Stage transition: BLENDING → SHADOWING
[01:26:17] env2_seed_2 | Stage transition: TRAINING → BLENDING
[01:26:20] env3_seed_2 | Stage transition: SHADOWING → PROBATIONARY
[01:26:33] env2_seed_2 | Stage transition: BLENDING → SHADOWING
[01:26:36] env3_seed_2 | Stage transition: PROBATIONARY → FOSSILIZED
[01:26:36] env3_seed_2 | Fossilized (norm, Δacc +3.19%)
    [env3] Fossilized 'env3_seed_2' (norm, Δacc +3.19%)
[01:26:39] env2_seed_2 | Stage transition: SHADOWING → PROBATIONARY
[01:26:45] env2_seed_2 | Stage transition: PROBATIONARY → FOSSILIZED
[01:26:45] env2_seed_2 | Fossilized (depthwise, Δacc +4.64%)
    [env2] Fossilized 'env2_seed_2' (depthwise, Δacc +4.64%)
Batch 14: Episodes 56/200
  Env accuracies: ['71.0%', '73.6%', '73.9%', '76.3%']
  Avg acc: 73.7% (rolling: 73.8%)
  Avg reward: 45.6
  Actions: {'WAIT': 55, 'GERMINATE_NORM': 57, 'GERMINATE_ATTENTION': 35, 'GERMINATE_DEPTHWISE': 69, 'GERMINATE_CONV_ENHANCE': 24, 'FOSSILIZE': 56, 'CULL': 4}
  Successful: {'WAIT': 55, 'GERMINATE_NORM': 1, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 6, 'GERMINATE_CONV_ENHANCE': 1, 'FOSSILIZE': 7, 'CULL': 4}
  Policy loss: -0.0071, Value loss: 28.8428, Entropy: 1.7729, Entropy coef: 0.1281
[01:27:30] env2_seed_0 | Germinated (attention, 2.0K params)
    [env2] Germinated 'env2_seed_0' (attention, 2.0K params)
[01:27:30] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[01:27:37] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[01:27:37] env3_seed_0 | Germinated (depthwise, 4.8K params)
    [env3] Germinated 'env3_seed_0' (depthwise, 4.8K params)
[01:27:37] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[01:27:37] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[01:27:46] env2_seed_0 | Stage transition: TRAINING → BLENDING
[01:27:51] env0_seed_0 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_0' (depthwise, 4.8K params)
[01:27:51] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[01:27:57] env1_seed_0 | Stage transition: TRAINING → CULLED
[01:27:57] env1_seed_0 | Culled (norm, Δacc -1.35%)
    [env1] Culled 'env1_seed_0' (norm, Δacc -1.35%)
[01:27:57] env2_seed_0 | Stage transition: BLENDING → CULLED
[01:27:57] env2_seed_0 | Culled (attention, Δacc +11.67%)
    [env2] Culled 'env2_seed_0' (attention, Δacc +11.67%)
[01:28:02] env3_seed_0 | Stage transition: TRAINING → BLENDING
[01:28:02] env1_seed_1 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_1' (attention, 2.0K params)
[01:28:02] env2_seed_1 | Germinated (depthwise, 4.8K params)
    [env2] Germinated 'env2_seed_1' (depthwise, 4.8K params)
[01:28:02] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[01:28:02] env2_seed_1 | Stage transition: GERMINATED → TRAINING
[01:28:26] env0_seed_0 | Stage transition: TRAINING → BLENDING
[01:28:26] env2_seed_1 | Stage transition: TRAINING → BLENDING
[01:28:32] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[01:28:37] env1_seed_1 | Stage transition: TRAINING → BLENDING
[01:28:42] env3_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:28:42] env3_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:28:42] env3_seed_0 | Fossilized (depthwise, Δacc +8.15%)
    [env3] Fossilized 'env3_seed_0' (depthwise, Δacc +8.15%)
[01:28:52] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[01:28:52] env2_seed_1 | Stage transition: BLENDING → SHADOWING
[01:28:58] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:28:58] env1_seed_1 | Stage transition: BLENDING → SHADOWING
[01:28:58] env2_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[01:29:01] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:29:01] env0_seed_0 | Fossilized (depthwise, Δacc +6.70%)
    [env0] Fossilized 'env0_seed_0' (depthwise, Δacc +6.70%)
[01:29:04] env1_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[01:29:10] env2_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[01:29:10] env2_seed_1 | Fossilized (depthwise, Δacc +10.46%)
    [env2] Fossilized 'env2_seed_1' (depthwise, Δacc +10.46%)
[01:29:22] env1_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[01:29:22] env1_seed_1 | Fossilized (attention, Δacc +1.73%)
    [env1] Fossilized 'env1_seed_1' (attention, Δacc +1.73%)
[01:29:31] env2_seed_1 | Stage transition: FOSSILIZED → CULLED
[01:29:31] env2_seed_1 | Culled (depthwise, Δacc +15.49%)
    [env2] Culled 'env2_seed_1' (depthwise, Δacc +15.49%)
[01:29:37] env2_seed_2 | Germinated (norm, 0.1K params)
    [env2] Germinated 'env2_seed_2' (norm, 0.1K params)
[01:29:37] env2_seed_2 | Stage transition: GERMINATED → TRAINING
[01:29:53] env2_seed_2 | Stage transition: TRAINING → CULLED
[01:29:53] env2_seed_2 | Culled (norm, Δacc -2.97%)
    [env2] Culled 'env2_seed_2' (norm, Δacc -2.97%)
[01:29:56] env2_seed_3 | Germinated (attention, 2.0K params)
    [env2] Germinated 'env2_seed_3' (attention, 2.0K params)
[01:29:56] env2_seed_3 | Stage transition: GERMINATED → TRAINING
[01:30:09] env2_seed_3 | Stage transition: TRAINING → BLENDING
[01:30:09] env3_seed_0 | Stage transition: FOSSILIZED → CULLED
[01:30:09] env3_seed_0 | Culled (depthwise, Δacc +10.71%)
    [env3] Culled 'env3_seed_0' (depthwise, Δacc +10.71%)
[01:30:15] env3_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env3] Germinated 'env3_seed_1' (conv_enhance, 74.0K params)
[01:30:16] env3_seed_1 | Stage transition: GERMINATED → TRAINING
[01:30:29] env2_seed_3 | Stage transition: BLENDING → SHADOWING
[01:30:36] env2_seed_3 | Stage transition: SHADOWING → PROBATIONARY
[01:30:36] env3_seed_1 | Stage transition: TRAINING → BLENDING
[01:30:54] env3_seed_1 | Stage transition: BLENDING → SHADOWING
[01:31:00] env3_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[01:31:00] env2_seed_3 | Stage transition: PROBATIONARY → FOSSILIZED
[01:31:00] env2_seed_3 | Fossilized (attention, Δacc +5.14%)
    [env2] Fossilized 'env2_seed_3' (attention, Δacc +5.14%)
[01:31:24] env0_seed_0 | Stage transition: FOSSILIZED → CULLED
[01:31:24] env0_seed_0 | Culled (depthwise, Δacc +7.82%)
    [env0] Culled 'env0_seed_0' (depthwise, Δacc +7.82%)
[01:31:27] env0_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_1' (conv_enhance, 74.0K params)
[01:31:27] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[01:31:40] env0_seed_1 | Stage transition: TRAINING → BLENDING
[01:31:57] env0_seed_1 | Stage transition: BLENDING → SHADOWING
[01:31:57] env2_seed_3 | Stage transition: FOSSILIZED → CULLED
[01:31:57] env2_seed_3 | Culled (attention, Δacc +6.81%)
    [env2] Culled 'env2_seed_3' (attention, Δacc +6.81%)
Batch 15: Episodes 60/200
  Env accuracies: ['73.8%', '73.3%', '74.2%', '77.5%']
  Avg acc: 74.7% (rolling: 73.7%)
  Avg reward: 31.9
  Actions: {'WAIT': 67, 'GERMINATE_NORM': 54, 'GERMINATE_ATTENTION': 65, 'GERMINATE_DEPTHWISE': 47, 'GERMINATE_CONV_ENHANCE': 17, 'FOSSILIZE': 43, 'CULL': 7}
  Successful: {'WAIT': 67, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 3, 'GERMINATE_DEPTHWISE': 3, 'GERMINATE_CONV_ENHANCE': 2, 'FOSSILIZE': 5, 'CULL': 7}
  Policy loss: -0.0148, Value loss: 24.4705, Entropy: 1.7696, Entropy coef: 0.1230

Blueprint Stats
  ---------------------------------------------------------------------------

Blueprint       Germ  Foss  Cull   Rate     ΔAcc    Churn
  ---------------------------------------------------------------------------

  attention         63    15    58  20.5%   +6.26%   +4.11%
  conv_enhance      68    17    56  23.3%  +13.14%   +6.85%
  depthwise         72    25    52  32.5%   +8.13%   +5.23%
  norm              61    25    42  37.3%   +7.27%   +4.15%
Seed Scoreboard (env 0):
  Fossilized: 17 (+334.1K params, +352.6% of host)
  Culled: 53
  Avg fossilize age: 13.8 epochs
  Avg cull age: 11.7 epochs
  Compute cost: 2.94x baseline
  Distribution: conv_enhance x4, depthwise x7, attention x2, norm x4
Seed Scoreboard (env 1):
  Fossilized: 21 (+252.6K params, +266.5% of host)
  Culled: 55
  Avg fossilize age: 15.0 epochs
  Avg cull age: 12.3 epochs
  Compute cost: 3.70x baseline
  Distribution: norm x9, depthwise x4, conv_enhance x3, attention x5
Seed Scoreboard (env 2):
  Fossilized: 23 (+354.7K params, +374.3% of host)
  Culled: 50
  Avg fossilize age: 15.5 epochs
  Avg cull age: 14.6 epochs
  Compute cost: 4.23x baseline
  Distribution: conv_enhance x4, attention x5, norm x4, depthwise x10
Seed Scoreboard (env 3):
  Fossilized: 21 (+470.3K params, +496.3% of host)
  Culled: 50
  Avg fossilize age: 14.9 epochs
  Avg cull age: 10.7 epochs
  Compute cost: 3.43x baseline
  Distribution: conv_enhance x6, norm x8, depthwise x4, attention x3

[01:32:03] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[01:32:03] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[01:32:06] env2_seed_0 | Germinated (depthwise, 4.8K params)
    [env2] Germinated 'env2_seed_0' (depthwise, 4.8K params)
[01:32:06] env3_seed_0 | Germinated (depthwise, 4.8K params)
    [env3] Germinated 'env3_seed_0' (depthwise, 4.8K params)
[01:32:06] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[01:32:06] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[01:32:11] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[01:32:11] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[01:32:23] env0_seed_0 | Stage transition: TRAINING → BLENDING
[01:32:29] env2_seed_0 | Stage transition: TRAINING → BLENDING
[01:32:29] env3_seed_0 | Stage transition: TRAINING → BLENDING
[01:32:35] env1_seed_0 | Stage transition: TRAINING → BLENDING
[01:32:53] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[01:32:58] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[01:32:58] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[01:33:02] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:33:02] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[01:33:05] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:33:05] env3_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:33:08] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:33:08] env3_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:33:08] env3_seed_0 | Fossilized (depthwise, Δacc +12.45%)
    [env3] Fossilized 'env3_seed_0' (depthwise, Δacc +12.45%)
[01:33:11] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:33:11] env0_seed_0 | Fossilized (norm, Δacc +15.70%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +15.70%)
[01:33:17] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:33:17] env1_seed_0 | Fossilized (norm, Δacc +13.14%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +13.14%)
[01:33:20] env2_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:33:20] env2_seed_0 | Fossilized (depthwise, Δacc +12.47%)
    [env2] Fossilized 'env2_seed_0' (depthwise, Δacc +12.47%)
[01:33:37] env1_seed_0 | Stage transition: FOSSILIZED → CULLED
[01:33:37] env1_seed_0 | Culled (norm, Δacc +14.06%)
    [env1] Culled 'env1_seed_0' (norm, Δacc +14.06%)
[01:33:40] env1_seed_1 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_1' (norm, 0.1K params)
[01:33:41] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[01:33:44] env1_seed_1 | Stage transition: TRAINING → CULLED
[01:33:44] env1_seed_1 | Culled (norm, Δacc +0.00%)
    [env1] Culled 'env1_seed_1' (norm, Δacc +0.00%)
[01:33:53] env1_seed_2 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_2' (attention, 2.0K params)
[01:33:53] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[01:34:09] env1_seed_2 | Stage transition: TRAINING → BLENDING
[01:34:09] env3_seed_0 | Stage transition: FOSSILIZED → CULLED
[01:34:09] env3_seed_0 | Culled (depthwise, Δacc +16.73%)
    [env3] Culled 'env3_seed_0' (depthwise, Δacc +16.73%)
[01:34:12] env3_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env3] Germinated 'env3_seed_1' (conv_enhance, 74.0K params)
[01:34:13] env3_seed_1 | Stage transition: GERMINATED → TRAINING
[01:34:28] env1_seed_2 | Stage transition: BLENDING → SHADOWING
[01:34:31] env3_seed_1 | Stage transition: TRAINING → BLENDING
[01:34:35] env1_seed_2 | Stage transition: SHADOWING → PROBATIONARY
[01:34:49] env3_seed_1 | Stage transition: BLENDING → SHADOWING
[01:34:55] env3_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[01:34:55] env2_seed_0 | Stage transition: FOSSILIZED → CULLED
[01:34:55] env2_seed_0 | Culled (depthwise, Δacc +16.58%)
    [env2] Culled 'env2_seed_0' (depthwise, Δacc +16.58%)
[01:34:58] env2_seed_1 | Germinated (depthwise, 4.8K params)
    [env2] Germinated 'env2_seed_1' (depthwise, 4.8K params)
[01:34:58] env2_seed_1 | Stage transition: GERMINATED → TRAINING
[01:35:05] env3_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[01:35:05] env3_seed_1 | Fossilized (conv_enhance, Δacc +9.17%)
    [env3] Fossilized 'env3_seed_1' (conv_enhance, Δacc +9.17%)
[01:35:11] env2_seed_1 | Stage transition: TRAINING → BLENDING
[01:35:11] env2_seed_1 | Stage transition: BLENDING → CULLED
[01:35:11] env2_seed_1 | Culled (depthwise, Δacc -2.03%)
    [env2] Culled 'env2_seed_1' (depthwise, Δacc -2.03%)
[01:35:14] env1_seed_2 | Stage transition: PROBATIONARY → FOSSILIZED
[01:35:14] env1_seed_2 | Fossilized (attention, Δacc +4.09%)
    [env1] Fossilized 'env1_seed_2' (attention, Δacc +4.09%)
[01:35:14] env2_seed_2 | Germinated (attention, 2.0K params)
    [env2] Germinated 'env2_seed_2' (attention, 2.0K params)
[01:35:14] env2_seed_2 | Stage transition: GERMINATED → TRAINING
[01:35:21] env1_seed_2 | Stage transition: FOSSILIZED → CULLED
[01:35:21] env1_seed_2 | Culled (attention, Δacc -3.09%)
    [env1] Culled 'env1_seed_2' (attention, Δacc -3.09%)
[01:35:24] env1_seed_3 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_3' (conv_enhance, 74.0K params)
[01:35:24] env1_seed_3 | Stage transition: GERMINATED → TRAINING
[01:35:43] env2_seed_2 | Stage transition: TRAINING → BLENDING
[01:36:06] env2_seed_2 | Stage transition: BLENDING → SHADOWING
[01:36:14] env1_seed_3 | Stage transition: TRAINING → BLENDING
[01:36:14] env2_seed_2 | Stage transition: SHADOWING → PROBATIONARY
[01:36:21] env2_seed_2 | Stage transition: PROBATIONARY → FOSSILIZED
[01:36:21] env2_seed_2 | Fossilized (attention, Δacc +3.94%)
    [env2] Fossilized 'env2_seed_2' (attention, Δacc +3.94%)
[01:36:32] env1_seed_3 | Stage transition: BLENDING → SHADOWING
[01:36:38] env1_seed_3 | Stage transition: SHADOWING → PROBATIONARY
[01:36:38] env1_seed_3 | Stage transition: PROBATIONARY → FOSSILIZED
[01:36:38] env1_seed_3 | Fossilized (conv_enhance, Δacc +0.76%)
    [env1] Fossilized 'env1_seed_3' (conv_enhance, Δacc +0.76%)
Batch 16: Episodes 64/200
  Env accuracies: ['76.9%', '74.0%', '69.1%', '76.1%']
  Avg acc: 74.0% (rolling: 73.6%)
  Avg reward: 34.7
  Actions: {'WAIT': 41, 'GERMINATE_NORM': 52, 'GERMINATE_ATTENTION': 43, 'GERMINATE_DEPTHWISE': 56, 'GERMINATE_CONV_ENHANCE': 39, 'FOSSILIZE': 63, 'CULL': 6}
  Successful: {'WAIT': 41, 'GERMINATE_NORM': 3, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 3, 'GERMINATE_CONV_ENHANCE': 2, 'FOSSILIZE': 8, 'CULL': 6}
  Policy loss: -0.0175, Value loss: 27.7340, Entropy: 1.7650, Entropy coef: 0.1178
[01:36:44] env2_seed_0 | Germinated (depthwise, 4.8K params)
    [env2] Germinated 'env2_seed_0' (depthwise, 4.8K params)
[01:36:44] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[01:36:47] env1_seed_0 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_0' (depthwise, 4.8K params)
[01:36:47] env3_seed_0 | Germinated (attention, 2.0K params)
    [env3] Germinated 'env3_seed_0' (attention, 2.0K params)
[01:36:47] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[01:36:47] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[01:36:52] env0_seed_0 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_0' (conv_enhance, 74.0K params)
[01:36:52] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[01:37:10] env1_seed_0 | Stage transition: TRAINING → BLENDING
[01:37:10] env3_seed_0 | Stage transition: TRAINING → BLENDING
[01:37:17] env0_seed_0 | Stage transition: TRAINING → BLENDING
[01:37:17] env2_seed_0 | Stage transition: TRAINING → BLENDING
[01:37:29] env1_seed_0 | Stage transition: BLENDING → CULLED
[01:37:29] env1_seed_0 | Culled (depthwise, Δacc +9.16%)
    [env1] Culled 'env1_seed_0' (depthwise, Δacc +9.16%)
[01:37:39] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[01:37:39] env1_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_1' (conv_enhance, 74.0K params)
[01:37:39] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[01:37:44] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[01:37:44] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[01:37:47] env3_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:37:51] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:37:51] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:37:51] env3_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:37:51] env3_seed_0 | Fossilized (attention, Δacc +12.85%)
    [env3] Fossilized 'env3_seed_0' (attention, Δacc +12.85%)
[01:37:54] env1_seed_1 | Stage transition: TRAINING → BLENDING
[01:37:57] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:37:57] env0_seed_0 | Fossilized (conv_enhance, Δacc +16.45%)
    [env0] Fossilized 'env0_seed_0' (conv_enhance, Δacc +16.45%)
[01:37:57] env2_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:37:57] env2_seed_0 | Fossilized (depthwise, Δacc +14.73%)
    [env2] Fossilized 'env2_seed_0' (depthwise, Δacc +14.73%)
[01:38:01] env3_seed_0 | Stage transition: FOSSILIZED → CULLED
[01:38:01] env3_seed_0 | Culled (attention, Δacc +13.96%)
    [env3] Culled 'env3_seed_0' (attention, Δacc +13.96%)
[01:38:11] env1_seed_1 | Stage transition: BLENDING → SHADOWING
[01:38:14] env3_seed_1 | Germinated (norm, 0.1K params)
    [env3] Germinated 'env3_seed_1' (norm, 0.1K params)
[01:38:14] env3_seed_1 | Stage transition: GERMINATED → TRAINING
[01:38:18] env1_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[01:38:21] env1_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[01:38:21] env1_seed_1 | Fossilized (conv_enhance, Δacc +13.07%)
    [env1] Fossilized 'env1_seed_1' (conv_enhance, Δacc +13.07%)
[01:38:28] env3_seed_1 | Stage transition: TRAINING → CULLED
[01:38:28] env3_seed_1 | Culled (norm, Δacc -0.91%)
    [env3] Culled 'env3_seed_1' (norm, Δacc -0.91%)
[01:38:31] env3_seed_2 | Germinated (norm, 0.1K params)
    [env3] Germinated 'env3_seed_2' (norm, 0.1K params)
[01:38:31] env3_seed_2 | Stage transition: GERMINATED → TRAINING
[01:38:59] env3_seed_2 | Stage transition: TRAINING → BLENDING
[01:39:06] env1_seed_1 | Stage transition: FOSSILIZED → CULLED
[01:39:06] env1_seed_1 | Culled (conv_enhance, Δacc +8.25%)
    [env1] Culled 'env1_seed_1' (conv_enhance, Δacc +8.25%)
[01:39:09] env1_seed_2 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_2' (norm, 0.1K params)
[01:39:09] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[01:39:17] env3_seed_2 | Stage transition: BLENDING → SHADOWING
[01:39:23] env1_seed_2 | Stage transition: TRAINING → BLENDING
[01:39:23] env3_seed_2 | Stage transition: SHADOWING → PROBATIONARY
[01:39:27] env3_seed_2 | Stage transition: PROBATIONARY → FOSSILIZED
[01:39:27] env3_seed_2 | Fossilized (norm, Δacc +3.68%)
    [env3] Fossilized 'env3_seed_2' (norm, Δacc +3.68%)
[01:39:40] env1_seed_2 | Stage transition: BLENDING → SHADOWING
[01:39:46] env1_seed_2 | Stage transition: SHADOWING → PROBATIONARY
[01:39:46] env1_seed_2 | Stage transition: PROBATIONARY → FOSSILIZED
[01:39:46] env1_seed_2 | Fossilized (norm, Δacc +4.18%)
    [env1] Fossilized 'env1_seed_2' (norm, Δacc +4.18%)
Batch 17: Episodes 68/200
  Env accuracies: ['75.2%', '77.4%', '66.7%', '75.7%']
  Avg acc: 73.7% (rolling: 73.7%)
  Avg reward: 34.7
  Actions: {'WAIT': 61, 'GERMINATE_NORM': 42, 'GERMINATE_ATTENTION': 33, 'GERMINATE_DEPTHWISE': 48, 'GERMINATE_CONV_ENHANCE': 34, 'FOSSILIZE': 78, 'CULL': 4}
  Successful: {'WAIT': 61, 'GERMINATE_NORM': 3, 'GERMINATE_ATTENTION': 1, 'GERMINATE_DEPTHWISE': 2, 'GERMINATE_CONV_ENHANCE': 2, 'FOSSILIZE': 6, 'CULL': 4}
  Policy loss: -0.0105, Value loss: 19.8492, Entropy: 1.7674, Entropy coef: 0.1127
[01:41:13] env1_seed_0 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_0' (depthwise, 4.8K params)
[01:41:13] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[01:41:19] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[01:41:19] env2_seed_0 | Germinated (norm, 0.1K params)
    [env2] Germinated 'env2_seed_0' (norm, 0.1K params)
[01:41:19] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[01:41:19] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[01:41:29] env1_seed_0 | Stage transition: TRAINING → BLENDING
[01:41:29] env3_seed_0 | Germinated (norm, 0.1K params)
    [env3] Germinated 'env3_seed_0' (norm, 0.1K params)
[01:41:29] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[01:41:41] env0_seed_0 | Stage transition: TRAINING → BLENDING
[01:41:41] env2_seed_0 | Stage transition: TRAINING → BLENDING
[01:41:53] env3_seed_0 | Stage transition: TRAINING → BLENDING
[01:41:59] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[01:42:08] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[01:42:08] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:42:08] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[01:42:15] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:42:15] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:42:15] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[01:42:15] env2_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:42:15] env2_seed_0 | Fossilized (norm, Δacc +10.12%)
    [env2] Fossilized 'env2_seed_0' (norm, Δacc +10.12%)
[01:42:21] env3_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:42:21] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:42:21] env1_seed_0 | Fossilized (depthwise, Δacc +20.32%)
    [env1] Fossilized 'env1_seed_0' (depthwise, Δacc +20.32%)
[01:42:27] env3_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:42:27] env3_seed_0 | Fossilized (norm, Δacc +9.03%)
    [env3] Fossilized 'env3_seed_0' (norm, Δacc +9.03%)
[01:42:39] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:42:39] env0_seed_0 | Fossilized (norm, Δacc +12.35%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +12.35%)
[01:42:50] env1_seed_0 | Stage transition: FOSSILIZED → CULLED
[01:42:50] env1_seed_0 | Culled (depthwise, Δacc +21.17%)
    [env1] Culled 'env1_seed_0' (depthwise, Δacc +21.17%)
[01:42:53] env1_seed_1 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_1' (depthwise, 4.8K params)
[01:42:53] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[01:43:17] env1_seed_1 | Stage transition: TRAINING → BLENDING
[01:43:34] env1_seed_1 | Stage transition: BLENDING → SHADOWING
[01:43:40] env1_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[01:44:10] env1_seed_1 | Stage transition: PROBATIONARY → CULLED
[01:44:10] env1_seed_1 | Culled (depthwise, Δacc +7.65%)
    [env1] Culled 'env1_seed_1' (depthwise, Δacc +7.65%)
[01:44:18] env1_seed_2 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_2' (depthwise, 4.8K params)
[01:44:18] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[01:44:35] env1_seed_2 | Stage transition: TRAINING → BLENDING
[01:44:35] env1_seed_2 | Stage transition: BLENDING → CULLED
[01:44:35] env1_seed_2 | Culled (depthwise, Δacc -0.60%)
    [env1] Culled 'env1_seed_2' (depthwise, Δacc -0.60%)
[01:44:38] env1_seed_3 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_3' (conv_enhance, 74.0K params)
[01:44:38] env1_seed_3 | Stage transition: GERMINATED → TRAINING
[01:44:49] env1_seed_3 | Stage transition: TRAINING → CULLED
[01:44:49] env1_seed_3 | Culled (conv_enhance, Δacc +1.02%)
    [env1] Culled 'env1_seed_3' (conv_enhance, Δacc +1.02%)
[01:44:52] env1_seed_4 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_4' (depthwise, 4.8K params)
[01:44:52] env1_seed_4 | Stage transition: GERMINATED → TRAINING
[01:44:55] env1_seed_4 | Stage transition: TRAINING → CULLED
[01:44:55] env1_seed_4 | Culled (depthwise, Δacc +0.00%)
    [env1] Culled 'env1_seed_4' (depthwise, Δacc +0.00%)
[01:44:58] env1_seed_5 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_5' (depthwise, 4.8K params)
[01:44:58] env1_seed_5 | Stage transition: GERMINATED → TRAINING
[01:45:11] env1_seed_5 | Stage transition: TRAINING → BLENDING
Batch 18: Episodes 72/200
  Env accuracies: ['74.6%', '71.5%', '76.5%', '75.7%']
  Avg acc: 74.5% (rolling: 73.6%)
  Avg reward: 44.2
  Actions: {'WAIT': 41, 'GERMINATE_NORM': 50, 'GERMINATE_ATTENTION': 49, 'GERMINATE_DEPTHWISE': 59, 'GERMINATE_CONV_ENHANCE': 41, 'FOSSILIZE': 55, 'CULL': 5}
  Successful: {'WAIT': 41, 'GERMINATE_NORM': 3, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 5, 'GERMINATE_CONV_ENHANCE': 1, 'FOSSILIZE': 4, 'CULL': 5}
  Policy loss: -0.0196, Value loss: 21.5226, Entropy: 1.7584, Entropy coef: 0.1076
[01:45:31] env1_seed_0 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_0' (depthwise, 4.8K params)
[01:45:31] env2_seed_0 | Germinated (depthwise, 4.8K params)
    [env2] Germinated 'env2_seed_0' (depthwise, 4.8K params)
[01:45:31] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[01:45:31] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[01:45:35] env0_seed_0 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_0' (depthwise, 4.8K params)
[01:45:35] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[01:45:40] env3_seed_0 | Germinated (depthwise, 4.8K params)
    [env3] Germinated 'env3_seed_0' (depthwise, 4.8K params)
[01:45:40] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[01:45:52] env1_seed_0 | Stage transition: TRAINING → BLENDING
[01:45:52] env2_seed_0 | Stage transition: TRAINING → BLENDING
[01:45:58] env0_seed_0 | Stage transition: TRAINING → BLENDING
[01:46:05] env3_seed_0 | Stage transition: TRAINING → BLENDING
[01:46:17] env1_seed_0 | Stage transition: BLENDING → CULLED
[01:46:17] env1_seed_0 | Culled (depthwise, Δacc +5.83%)
    [env1] Culled 'env1_seed_0' (depthwise, Δacc +5.83%)
[01:46:22] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[01:46:22] env1_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_1' (conv_enhance, 74.0K params)
[01:46:22] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[01:46:27] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[01:46:31] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:46:31] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[01:46:34] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:46:34] env3_seed_0 | Stage transition: SHADOWING → CULLED
[01:46:34] env3_seed_0 | Culled (depthwise, Δacc +7.72%)
    [env3] Culled 'env3_seed_0' (depthwise, Δacc +7.72%)
[01:46:38] env1_seed_1 | Stage transition: TRAINING → BLENDING
[01:46:38] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:46:38] env0_seed_0 | Fossilized (depthwise, Δacc +11.21%)
    [env0] Fossilized 'env0_seed_0' (depthwise, Δacc +11.21%)
[01:46:41] env0_seed_0 | Stage transition: FOSSILIZED → CULLED
[01:46:41] env0_seed_0 | Culled (depthwise, Δacc +12.48%)
    [env0] Culled 'env0_seed_0' (depthwise, Δacc +12.48%)
[01:46:45] env3_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env3] Germinated 'env3_seed_1' (conv_enhance, 74.0K params)
[01:46:45] env3_seed_1 | Stage transition: GERMINATED → TRAINING
[01:46:49] env0_seed_1 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_1' (norm, 0.1K params)
[01:46:49] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[01:46:59] env1_seed_1 | Stage transition: BLENDING → SHADOWING
[01:47:03] env3_seed_1 | Stage transition: TRAINING → BLENDING
[01:47:08] env1_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[01:47:08] env1_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[01:47:08] env1_seed_1 | Fossilized (conv_enhance, Δacc +9.69%)
    [env1] Fossilized 'env1_seed_1' (conv_enhance, Δacc +9.69%)
[01:47:27] env3_seed_1 | Stage transition: BLENDING → SHADOWING
[01:47:30] env0_seed_1 | Stage transition: TRAINING → BLENDING
[01:47:30] env2_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:47:30] env2_seed_0 | Fossilized (depthwise, Δacc +18.90%)
    [env2] Fossilized 'env2_seed_0' (depthwise, Δacc +18.90%)
[01:47:34] env3_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[01:47:34] env3_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[01:47:34] env3_seed_1 | Fossilized (conv_enhance, Δacc +7.86%)
    [env3] Fossilized 'env3_seed_1' (conv_enhance, Δacc +7.86%)
[01:47:47] env0_seed_1 | Stage transition: BLENDING → SHADOWING
[01:47:53] env0_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[01:48:18] env0_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[01:48:18] env0_seed_1 | Fossilized (norm, Δacc +1.74%)
    [env0] Fossilized 'env0_seed_1' (norm, Δacc +1.74%)
[01:49:56] env1_seed_1 | Stage transition: FOSSILIZED → CULLED
[01:49:56] env1_seed_1 | Culled (conv_enhance, Δacc +13.41%)
    [env1] Culled 'env1_seed_1' (conv_enhance, Δacc +13.41%)
[01:50:02] env1_seed_2 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_2' (norm, 0.1K params)
Batch 19: Episodes 76/200
  Env accuracies: ['76.9%', '68.7%', '67.0%', '75.5%']
  Avg acc: 72.0% (rolling: 73.4%)
  Avg reward: 26.0
  Actions: {'WAIT': 44, 'GERMINATE_NORM': 63, 'GERMINATE_ATTENTION': 32, 'GERMINATE_DEPTHWISE': 46, 'GERMINATE_CONV_ENHANCE': 38, 'FOSSILIZE': 73, 'CULL': 4}
  Successful: {'WAIT': 44, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 4, 'GERMINATE_CONV_ENHANCE': 2, 'FOSSILIZE': 5, 'CULL': 4}
  Policy loss: -0.0128, Value loss: 22.5147, Entropy: 1.7514, Entropy coef: 0.1024
[01:50:08] env2_seed_0 | Germinated (conv_enhance, 74.0K params)
    [env2] Germinated 'env2_seed_0' (conv_enhance, 74.0K params)
[01:50:08] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[01:50:11] env0_seed_0 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_0' (attention, 2.0K params)
[01:50:11] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[01:50:11] env3_seed_0 | Germinated (depthwise, 4.8K params)
    [env3] Germinated 'env3_seed_0' (depthwise, 4.8K params)
[01:50:11] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[01:50:11] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[01:50:11] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[01:50:30] env2_seed_0 | Stage transition: TRAINING → BLENDING
[01:50:36] env0_seed_0 | Stage transition: TRAINING → BLENDING
[01:50:36] env1_seed_0 | Stage transition: TRAINING → BLENDING
[01:50:36] env3_seed_0 | Stage transition: TRAINING → BLENDING
[01:50:36] env3_seed_0 | Stage transition: BLENDING → CULLED
[01:50:36] env3_seed_0 | Culled (depthwise, Δacc -0.58%)
    [env3] Culled 'env3_seed_0' (depthwise, Δacc -0.58%)
[01:50:41] env3_seed_1 | Germinated (attention, 2.0K params)
    [env3] Germinated 'env3_seed_1' (attention, 2.0K params)
[01:50:41] env3_seed_1 | Stage transition: GERMINATED → TRAINING
[01:50:59] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[01:51:03] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[01:51:03] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[01:51:03] env3_seed_1 | Stage transition: TRAINING → BLENDING
[01:51:07] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:51:10] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:51:10] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:51:10] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:51:10] env0_seed_0 | Fossilized (attention, Δacc +12.15%)
    [env0] Fossilized 'env0_seed_0' (attention, Δacc +12.15%)
[01:51:14] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:51:14] env1_seed_0 | Fossilized (norm, Δacc +12.69%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +12.69%)
[01:51:14] env2_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:51:14] env2_seed_0 | Fossilized (conv_enhance, Δacc +18.59%)
    [env2] Fossilized 'env2_seed_0' (conv_enhance, Δacc +18.59%)
[01:51:20] env3_seed_1 | Stage transition: BLENDING → SHADOWING
[01:51:26] env3_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[01:51:48] env3_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[01:51:48] env3_seed_1 | Fossilized (attention, Δacc +8.79%)
    [env3] Fossilized 'env3_seed_1' (attention, Δacc +8.79%)
[01:52:39] env1_seed_0 | Stage transition: FOSSILIZED → CULLED
[01:52:39] env1_seed_0 | Culled (norm, Δacc +18.32%)
    [env1] Culled 'env1_seed_0' (norm, Δacc +18.32%)
[01:52:45] env1_seed_1 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_1' (depthwise, 4.8K params)
[01:52:45] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[01:53:29] env1_seed_1 | Stage transition: TRAINING → BLENDING
[01:53:46] env1_seed_1 | Stage transition: BLENDING → SHADOWING
[01:53:52] env1_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[01:53:55] env1_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[01:53:55] env1_seed_1 | Fossilized (depthwise, Δacc +0.62%)
    [env1] Fossilized 'env1_seed_1' (depthwise, Δacc +0.62%)
Batch 20: Episodes 80/200
  Env accuracies: ['73.8%', '74.0%', '70.8%', '74.3%']
  Avg acc: 73.2% (rolling: 73.7%)
  Avg reward: 40.4
  Actions: {'WAIT': 41, 'GERMINATE_NORM': 64, 'GERMINATE_ATTENTION': 40, 'GERMINATE_DEPTHWISE': 41, 'GERMINATE_CONV_ENHANCE': 42, 'FOSSILIZE': 70, 'CULL': 2}
  Successful: {'WAIT': 41, 'GERMINATE_NORM': 1, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 2, 'GERMINATE_CONV_ENHANCE': 1, 'FOSSILIZE': 5, 'CULL': 2}
  Policy loss: -0.0132, Value loss: 18.9065, Entropy: 1.7559, Entropy coef: 0.0973

Blueprint Stats
  ---------------------------------------------------------------------------

Blueprint       Germ  Foss  Cull   Rate     ΔAcc    Churn
  ---------------------------------------------------------------------------

  attention         68    20    60  25.0%   +6.78%   +4.15%
  conv_enhance      76    24    59  28.9%  +12.46%   +6.89%
  depthwise         88    32    64  33.3%   +9.19%   +5.72%
  norm              73    34    46  42.5%   +7.78%   +4.47%
Seed Scoreboard (env 0):
  Fossilized: 23 (+415.4K params, +438.3% of host)
  Culled: 54
  Avg fossilize age: 14.2 epochs
  Avg cull age: 11.7 epochs
  Compute cost: 3.58x baseline
  Distribution: conv_enhance x5, depthwise x8, attention x3, norm x7
Seed Scoreboard (env 1):
  Fossilized: 30 (+486.5K params, +513.4% of host)
  Culled: 68
  Avg fossilize age: 15.1 epochs
  Avg cull age: 13.6 epochs
  Compute cost: 4.72x baseline
  Distribution: norm x12, depthwise x6, conv_enhance x6, attention x6
Seed Scoreboard (env 2):
  Fossilized: 29 (+445.3K params, +469.9% of host)
  Culled: 52
  Avg fossilize age: 15.6 epochs
  Avg cull age: 14.9 epochs
  Compute cost: 4.99x baseline
  Distribution: conv_enhance x5, attention x6, norm x5, depthwise x13
Seed Scoreboard (env 3):
  Fossilized: 28 (+627.4K params, +662.1% of host)
  Culled: 55
  Avg fossilize age: 14.6 epochs
  Avg cull age: 10.9 epochs
  Compute cost: 4.55x baseline
  Distribution: conv_enhance x8, norm x10, depthwise x5, attention x5

[01:54:31] env1_seed_0 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_0' (conv_enhance, 74.0K params)
[01:54:31] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[01:54:35] env0_seed_0 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_0' (attention, 2.0K params)
[01:54:35] env2_seed_0 | Germinated (depthwise, 4.8K params)
    [env2] Germinated 'env2_seed_0' (depthwise, 4.8K params)
[01:54:35] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[01:54:35] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[01:54:44] env3_seed_0 | Germinated (norm, 0.1K params)
    [env3] Germinated 'env3_seed_0' (norm, 0.1K params)
[01:54:44] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[01:54:50] env1_seed_0 | Stage transition: TRAINING → BLENDING
[01:55:02] env0_seed_0 | Stage transition: TRAINING → BLENDING
[01:55:02] env2_seed_0 | Stage transition: TRAINING → BLENDING
[01:55:09] env3_seed_0 | Stage transition: TRAINING → BLENDING
[01:55:21] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[01:55:30] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[01:55:30] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:55:30] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[01:55:34] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[01:55:37] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:55:37] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:55:37] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:55:37] env0_seed_0 | Fossilized (attention, Δacc +12.04%)
    [env0] Fossilized 'env0_seed_0' (attention, Δacc +12.04%)
[01:55:37] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:55:37] env1_seed_0 | Fossilized (conv_enhance, Δacc +9.98%)
    [env1] Fossilized 'env1_seed_0' (conv_enhance, Δacc +9.98%)
[01:55:40] env3_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:55:49] env3_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:55:49] env3_seed_0 | Fossilized (norm, Δacc +12.57%)
    [env3] Fossilized 'env3_seed_0' (norm, Δacc +12.57%)
[01:56:07] env2_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:56:07] env2_seed_0 | Fossilized (depthwise, Δacc +8.87%)
    [env2] Fossilized 'env2_seed_0' (depthwise, Δacc +8.87%)
Batch 21: Episodes 84/200
  Env accuracies: ['64.7%', '71.2%', '75.4%', '77.7%']
  Avg acc: 72.3% (rolling: 73.5%)
  Avg reward: 43.4
  Actions: {'WAIT': 58, 'GERMINATE_NORM': 40, 'GERMINATE_ATTENTION': 62, 'GERMINATE_DEPTHWISE': 51, 'GERMINATE_CONV_ENHANCE': 35, 'FOSSILIZE': 54, 'CULL': 0}
  Successful: {'WAIT': 58, 'GERMINATE_NORM': 1, 'GERMINATE_ATTENTION': 1, 'GERMINATE_DEPTHWISE': 1, 'GERMINATE_CONV_ENHANCE': 1, 'FOSSILIZE': 4, 'CULL': 0}
  Policy loss: -0.0117, Value loss: 14.3689, Entropy: 1.7561, Entropy coef: 0.0922
[01:58:43] env0_seed_0 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_0' (conv_enhance, 74.0K params)
[01:58:43] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[01:58:43] env2_seed_0 | Germinated (norm, 0.1K params)
    [env2] Germinated 'env2_seed_0' (norm, 0.1K params)
[01:58:43] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[01:58:43] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[01:58:43] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[01:58:52] env3_seed_0 | Germinated (depthwise, 4.8K params)
    [env3] Germinated 'env3_seed_0' (depthwise, 4.8K params)
[01:58:52] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[01:59:04] env0_seed_0 | Stage transition: TRAINING → BLENDING
[01:59:04] env1_seed_0 | Stage transition: TRAINING → BLENDING
[01:59:04] env2_seed_0 | Stage transition: TRAINING → BLENDING
[01:59:17] env3_seed_0 | Stage transition: TRAINING → BLENDING
[01:59:35] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[01:59:35] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[01:59:35] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[01:59:42] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:59:42] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:59:42] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:59:42] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[01:59:45] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:59:45] env0_seed_0 | Fossilized (conv_enhance, Δacc +14.23%)
    [env0] Fossilized 'env0_seed_0' (conv_enhance, Δacc +14.23%)
[01:59:48] env3_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[01:59:54] env2_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:59:54] env2_seed_0 | Fossilized (norm, Δacc +17.43%)
    [env2] Fossilized 'env2_seed_0' (norm, Δacc +17.43%)
[01:59:57] env3_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[01:59:57] env3_seed_0 | Fossilized (depthwise, Δacc +13.98%)
    [env3] Fossilized 'env3_seed_0' (depthwise, Δacc +13.98%)
[02:00:10] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:00:10] env1_seed_0 | Fossilized (norm, Δacc +18.83%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +18.83%)
[02:01:39] env0_seed_0 | Stage transition: FOSSILIZED → CULLED
[02:01:39] env0_seed_0 | Culled (conv_enhance, Δacc +18.70%)
    [env0] Culled 'env0_seed_0' (conv_enhance, Δacc +18.70%)
[02:01:45] env0_seed_1 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_1' (depthwise, 4.8K params)
[02:01:45] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[02:02:01] env0_seed_1 | Stage transition: TRAINING → BLENDING
[02:02:17] env0_seed_1 | Stage transition: BLENDING → SHADOWING
[02:02:23] env0_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[02:02:23] env0_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[02:02:23] env0_seed_1 | Fossilized (depthwise, Δacc +0.86%)
    [env0] Fossilized 'env0_seed_1' (depthwise, Δacc +0.86%)
[02:02:29] env0_seed_1 | Stage transition: FOSSILIZED → CULLED
[02:02:29] env0_seed_1 | Culled (depthwise, Δacc +5.69%)
    [env0] Culled 'env0_seed_1' (depthwise, Δacc +5.69%)
[02:02:32] env0_seed_2 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_2' (conv_enhance, 74.0K params)
[02:02:32] env0_seed_2 | Stage transition: GERMINATED → TRAINING
Batch 22: Episodes 88/200
  Env accuracies: ['70.3%', '75.1%', '75.8%', '70.4%']
  Avg acc: 72.9% (rolling: 73.5%)
  Avg reward: 40.1
  Actions: {'WAIT': 71, 'GERMINATE_NORM': 50, 'GERMINATE_ATTENTION': 37, 'GERMINATE_DEPTHWISE': 29, 'GERMINATE_CONV_ENHANCE': 50, 'FOSSILIZE': 61, 'CULL': 2}
  Successful: {'WAIT': 71, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 2, 'GERMINATE_CONV_ENHANCE': 2, 'FOSSILIZE': 5, 'CULL': 2}
  Policy loss: -0.0155, Value loss: 18.9290, Entropy: 1.7264, Entropy coef: 0.0870
[02:03:04] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[02:03:04] env1_seed_0 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_0' (depthwise, 4.8K params)
[02:03:04] env2_seed_0 | Germinated (norm, 0.1K params)
    [env2] Germinated 'env2_seed_0' (norm, 0.1K params)
[02:03:04] env3_seed_0 | Germinated (conv_enhance, 74.0K params)
    [env3] Germinated 'env3_seed_0' (conv_enhance, 74.0K params)
[02:03:04] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[02:03:04] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[02:03:04] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[02:03:04] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[02:03:28] env0_seed_0 | Stage transition: TRAINING → BLENDING
[02:03:28] env2_seed_0 | Stage transition: TRAINING → BLENDING
[02:03:34] env1_seed_0 | Stage transition: TRAINING → BLENDING
[02:03:34] env3_seed_0 | Stage transition: TRAINING → BLENDING
[02:03:47] env1_seed_0 | Stage transition: BLENDING → CULLED
[02:03:47] env1_seed_0 | Culled (depthwise, Δacc +6.45%)
    [env1] Culled 'env1_seed_0' (depthwise, Δacc +6.45%)
[02:03:52] env1_seed_1 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_1' (depthwise, 4.8K params)
[02:03:52] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[02:03:58] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[02:03:58] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[02:04:02] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[02:04:05] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:04:05] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:04:05] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:04:05] env0_seed_0 | Fossilized (norm, Δacc +10.88%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +10.88%)
[02:04:09] env1_seed_1 | Stage transition: TRAINING → BLENDING
[02:04:09] env3_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:04:09] env2_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:04:09] env2_seed_0 | Fossilized (norm, Δacc +14.45%)
    [env2] Fossilized 'env2_seed_0' (norm, Δacc +14.45%)
[02:04:16] env3_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:04:16] env3_seed_0 | Fossilized (conv_enhance, Δacc +18.00%)
    [env3] Fossilized 'env3_seed_0' (conv_enhance, Δacc +18.00%)
[02:04:26] env1_seed_1 | Stage transition: BLENDING → SHADOWING
[02:04:32] env1_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[02:04:32] env1_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[02:04:32] env1_seed_1 | Fossilized (depthwise, Δacc +8.10%)
    [env1] Fossilized 'env1_seed_1' (depthwise, Δacc +8.10%)
Batch 23: Episodes 92/200
  Env accuracies: ['76.5%', '73.8%', '75.5%', '74.3%']
  Avg acc: 75.0% (rolling: 73.6%)
  Avg reward: 48.9
  Actions: {'WAIT': 79, 'GERMINATE_NORM': 66, 'GERMINATE_ATTENTION': 34, 'GERMINATE_DEPTHWISE': 20, 'GERMINATE_CONV_ENHANCE': 29, 'FOSSILIZE': 71, 'CULL': 1}
  Successful: {'WAIT': 79, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 2, 'GERMINATE_CONV_ENHANCE': 1, 'FOSSILIZE': 4, 'CULL': 1}
  Policy loss: -0.0106, Value loss: 17.1978, Entropy: 1.7192, Entropy coef: 0.0819
[02:07:21] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[02:07:21] env2_seed_0 | Germinated (depthwise, 4.8K params)
    [env2] Germinated 'env2_seed_0' (depthwise, 4.8K params)
[02:07:21] env3_seed_0 | Germinated (depthwise, 4.8K params)
    [env3] Germinated 'env3_seed_0' (depthwise, 4.8K params)
[02:07:21] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[02:07:22] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[02:07:22] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[02:07:41] env1_seed_0 | Stage transition: TRAINING → BLENDING
[02:07:41] env2_seed_0 | Stage transition: TRAINING → BLENDING
[02:07:41] env3_seed_0 | Stage transition: TRAINING → BLENDING
[02:07:41] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[02:07:41] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[02:08:05] env0_seed_0 | Stage transition: TRAINING → BLENDING
[02:08:05] env1_seed_0 | Stage transition: BLENDING → CULLED
[02:08:05] env1_seed_0 | Culled (norm, Δacc +14.93%)
    [env1] Culled 'env1_seed_0' (norm, Δacc +14.93%)
[02:08:10] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[02:08:10] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[02:08:10] env1_seed_1 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_1' (norm, 0.1K params)
[02:08:10] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[02:08:19] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:08:19] env3_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:08:28] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[02:08:35] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:08:38] env1_seed_1 | Stage transition: TRAINING → BLENDING
[02:08:38] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:08:38] env0_seed_0 | Fossilized (norm, Δacc +7.20%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +7.20%)
[02:08:38] env3_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:08:38] env3_seed_0 | Fossilized (depthwise, Δacc +11.66%)
    [env3] Fossilized 'env3_seed_0' (depthwise, Δacc +11.66%)
[02:08:52] env2_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:08:52] env2_seed_0 | Fossilized (depthwise, Δacc +19.30%)
    [env2] Fossilized 'env2_seed_0' (depthwise, Δacc +19.30%)
[02:08:55] env1_seed_1 | Stage transition: BLENDING → SHADOWING
[02:09:01] env1_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[02:09:51] env1_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[02:09:51] env1_seed_1 | Fossilized (norm, Δacc +5.19%)
    [env1] Fossilized 'env1_seed_1' (norm, Δacc +5.19%)
[02:09:54] env0_seed_0 | Stage transition: FOSSILIZED → CULLED
[02:09:54] env0_seed_0 | Culled (norm, Δacc +14.18%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +14.18%)
[02:09:57] env0_seed_1 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_1' (attention, 2.0K params)
[02:09:57] env2_seed_0 | Stage transition: FOSSILIZED → CULLED
[02:09:57] env2_seed_0 | Culled (depthwise, Δacc +18.91%)
    [env2] Culled 'env2_seed_0' (depthwise, Δacc +18.91%)
[02:09:57] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[02:10:01] env2_seed_1 | Germinated (norm, 0.1K params)
    [env2] Germinated 'env2_seed_1' (norm, 0.1K params)
[02:10:01] env2_seed_1 | Stage transition: GERMINATED → TRAINING
[02:10:14] env2_seed_1 | Stage transition: TRAINING → CULLED
[02:10:14] env2_seed_1 | Culled (norm, Δacc +1.89%)
    [env2] Culled 'env2_seed_1' (norm, Δacc +1.89%)
[02:10:17] env2_seed_2 | Germinated (conv_enhance, 74.0K params)
    [env2] Germinated 'env2_seed_2' (conv_enhance, 74.0K params)
[02:10:17] env2_seed_2 | Stage transition: GERMINATED → TRAINING
[02:10:24] env0_seed_1 | Stage transition: TRAINING → BLENDING
[02:10:39] env2_seed_2 | Stage transition: TRAINING → BLENDING
[02:10:42] env0_seed_1 | Stage transition: BLENDING → SHADOWING
[02:10:49] env0_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[02:10:56] env2_seed_2 | Stage transition: BLENDING → SHADOWING
[02:11:02] env2_seed_2 | Stage transition: SHADOWING → PROBATIONARY
[02:11:02] env0_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[02:11:02] env0_seed_1 | Fossilized (attention, Δacc +2.59%)
    [env0] Fossilized 'env0_seed_1' (attention, Δacc +2.59%)
Batch 24: Episodes 96/200
  Env accuracies: ['72.0%', '77.2%', '77.2%', '72.6%']
  Avg acc: 74.7% (rolling: 73.7%)
  Avg reward: 43.7
  Actions: {'WAIT': 79, 'GERMINATE_NORM': 69, 'GERMINATE_ATTENTION': 37, 'GERMINATE_DEPTHWISE': 34, 'GERMINATE_CONV_ENHANCE': 29, 'FOSSILIZE': 48, 'CULL': 4}
  Successful: {'WAIT': 79, 'GERMINATE_NORM': 4, 'GERMINATE_ATTENTION': 1, 'GERMINATE_DEPTHWISE': 2, 'GERMINATE_CONV_ENHANCE': 1, 'FOSSILIZE': 5, 'CULL': 4}
  Policy loss: -0.0036, Value loss: 19.5375, Entropy: 1.7345, Entropy coef: 0.0768
[02:11:44] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[02:11:44] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[02:11:44] env3_seed_0 | Germinated (norm, 0.1K params)
    [env3] Germinated 'env3_seed_0' (norm, 0.1K params)
[02:11:44] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[02:11:44] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[02:11:44] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[02:11:54] env2_seed_0 | Germinated (norm, 0.1K params)
    [env2] Germinated 'env2_seed_0' (norm, 0.1K params)
[02:11:54] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[02:12:06] env1_seed_0 | Stage transition: TRAINING → BLENDING
[02:12:06] env3_seed_0 | Stage transition: TRAINING → BLENDING
[02:12:11] env0_seed_0 | Stage transition: TRAINING → BLENDING
[02:12:17] env2_seed_0 | Stage transition: TRAINING → BLENDING
[02:12:35] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[02:12:35] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[02:12:39] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[02:12:42] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:12:42] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[02:12:42] env3_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:12:45] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:12:45] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:12:45] env0_seed_0 | Fossilized (norm, Δacc +8.76%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +8.76%)
[02:12:45] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:12:45] env1_seed_0 | Fossilized (norm, Δacc +17.57%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +17.57%)
[02:12:48] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:13:09] env3_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:13:09] env3_seed_0 | Fossilized (norm, Δacc +17.20%)
    [env3] Fossilized 'env3_seed_0' (norm, Δacc +17.20%)
[02:13:35] env3_seed_0 | Stage transition: FOSSILIZED → CULLED
[02:13:35] env3_seed_0 | Culled (norm, Δacc +18.96%)
    [env3] Culled 'env3_seed_0' (norm, Δacc +18.96%)
[02:13:41] env2_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:13:41] env2_seed_0 | Fossilized (norm, Δacc +13.89%)
    [env2] Fossilized 'env2_seed_0' (norm, Δacc +13.89%)
[02:13:41] env3_seed_1 | Germinated (depthwise, 4.8K params)
    [env3] Germinated 'env3_seed_1' (depthwise, 4.8K params)
[02:13:41] env3_seed_1 | Stage transition: GERMINATED → TRAINING
[02:13:58] env3_seed_1 | Stage transition: TRAINING → BLENDING
[02:14:16] env3_seed_1 | Stage transition: BLENDING → SHADOWING
[02:14:21] env3_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[02:14:42] env3_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[02:14:42] env3_seed_1 | Fossilized (depthwise, Δacc +3.95%)
    [env3] Fossilized 'env3_seed_1' (depthwise, Δacc +3.95%)
Batch 25: Episodes 100/200
  Env accuracies: ['75.9%', '73.4%', '75.3%', '66.1%']
  Avg acc: 72.7% (rolling: 73.5%)
  Avg reward: 52.8
  Actions: {'WAIT': 64, 'GERMINATE_NORM': 60, 'GERMINATE_ATTENTION': 62, 'GERMINATE_DEPTHWISE': 24, 'GERMINATE_CONV_ENHANCE': 35, 'FOSSILIZE': 54, 'CULL': 1}
  Successful: {'WAIT': 64, 'GERMINATE_NORM': 4, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 1, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 5, 'CULL': 1}
  Policy loss: -0.0123, Value loss: 23.8143, Entropy: 1.7191, Entropy coef: 0.0716

Blueprint Stats
  ---------------------------------------------------------------------------

Blueprint       Germ  Foss  Cull   Rate     ΔAcc    Churn
  ---------------------------------------------------------------------------

  attention         70    22    60  26.8%   +6.83%   +4.15%
  conv_enhance      81    27    60  31.0%  +12.64%   +7.08%
  depthwise         96    39    67  36.8%   +9.25%   +5.93%
  norm              86    45    50  47.4%   +9.07%   +5.11%
Seed Scoreboard (env 0):
  Fossilized: 30 (+498.6K params, +526.2% of host)
  Culled: 57
  Avg fossilize age: 13.9 epochs
  Avg cull age: 12.8 epochs
  Compute cost: 4.57x baseline
  Distribution: conv_enhance x6, depthwise x9, attention x5, norm x10
Seed Scoreboard (env 1):
  Fossilized: 35 (+565.7K params, +597.0% of host)
  Culled: 70
  Avg fossilize age: 15.4 epochs
  Avg cull age: 13.4 epochs
  Compute cost: 5.01x baseline
  Distribution: norm x15, depthwise x7, conv_enhance x7, attention x6
Seed Scoreboard (env 2):
  Fossilized: 34 (+455.2K params, +480.4% of host)
  Culled: 54
  Avg fossilize age: 16.2 epochs
  Avg cull age: 15.2 epochs
  Compute cost: 5.21x baseline
  Distribution: conv_enhance x5, attention x6, norm x8, depthwise x15
Seed Scoreboard (env 3):
  Fossilized: 34 (+716.0K params, +755.6% of host)
  Culled: 56
  Avg fossilize age: 14.9 epochs
  Avg cull age: 11.2 epochs
  Compute cost: 4.98x baseline
  Distribution: conv_enhance x9, norm x12, depthwise x8, attention x5

[02:15:56] env2_seed_0 | Germinated (norm, 0.1K params)
    [env2] Germinated 'env2_seed_0' (norm, 0.1K params)
[02:15:56] env3_seed_0 | Germinated (depthwise, 4.8K params)
    [env3] Germinated 'env3_seed_0' (depthwise, 4.8K params)
[02:15:56] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[02:15:56] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[02:16:00] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[02:16:00] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[02:16:00] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[02:16:01] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[02:16:18] env2_seed_0 | Stage transition: TRAINING → BLENDING
[02:16:18] env3_seed_0 | Stage transition: TRAINING → BLENDING
[02:16:24] env0_seed_0 | Stage transition: TRAINING → BLENDING
[02:16:24] env1_seed_0 | Stage transition: TRAINING → BLENDING
[02:16:49] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[02:16:49] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[02:16:53] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[02:16:53] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[02:16:56] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:16:56] env3_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:16:56] env2_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:16:56] env2_seed_0 | Fossilized (norm, Δacc +14.00%)
    [env2] Fossilized 'env2_seed_0' (norm, Δacc +14.00%)
[02:16:59] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:16:59] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:16:59] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:16:59] env0_seed_0 | Fossilized (norm, Δacc +10.03%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +10.03%)
[02:16:59] env3_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:16:59] env3_seed_0 | Fossilized (depthwise, Δacc +11.66%)
    [env3] Fossilized 'env3_seed_0' (depthwise, Δacc +11.66%)
[02:17:05] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:17:05] env1_seed_0 | Fossilized (norm, Δacc +14.83%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +14.83%)
[02:17:49] env1_seed_0 | Stage transition: FOSSILIZED → CULLED
[02:17:49] env1_seed_0 | Culled (norm, Δacc +17.51%)
    [env1] Culled 'env1_seed_0' (norm, Δacc +17.51%)
[02:17:52] env1_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_1' (conv_enhance, 74.0K params)
[02:17:52] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[02:18:10] env1_seed_1 | Stage transition: TRAINING → BLENDING
[02:18:27] env1_seed_1 | Stage transition: BLENDING → SHADOWING
[02:18:33] env1_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[02:18:33] env1_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[02:18:33] env1_seed_1 | Fossilized (conv_enhance, Δacc +8.36%)
    [env1] Fossilized 'env1_seed_1' (conv_enhance, Δacc +8.36%)
[02:20:03] env2_seed_0 | Stage transition: FOSSILIZED → CULLED
[02:20:03] env2_seed_0 | Culled (norm, Δacc +22.62%)
    [env2] Culled 'env2_seed_0' (norm, Δacc +22.62%)
[02:20:09] env2_seed_1 | Germinated (norm, 0.1K params)
    [env2] Germinated 'env2_seed_1' (norm, 0.1K params)
Batch 26: Episodes 104/200
  Env accuracies: ['75.7%', '75.5%', '64.4%', '65.2%']
  Avg acc: 70.2% (rolling: 73.1%)
  Avg reward: 43.7
  Actions: {'WAIT': 74, 'GERMINATE_NORM': 58, 'GERMINATE_ATTENTION': 44, 'GERMINATE_DEPTHWISE': 29, 'GERMINATE_CONV_ENHANCE': 31, 'FOSSILIZE': 62, 'CULL': 2}
  Successful: {'WAIT': 74, 'GERMINATE_NORM': 4, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 1, 'GERMINATE_CONV_ENHANCE': 1, 'FOSSILIZE': 5, 'CULL': 2}
  Policy loss: -0.0159, Value loss: 22.8117, Entropy: 1.7372, Entropy coef: 0.0665
[02:20:15] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[02:20:15] env2_seed_0 | Germinated (depthwise, 4.8K params)
    [env2] Germinated 'env2_seed_0' (depthwise, 4.8K params)
[02:20:15] env3_seed_0 | Germinated (norm, 0.1K params)
    [env3] Germinated 'env3_seed_0' (norm, 0.1K params)
[02:20:15] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[02:20:15] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[02:20:15] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[02:20:20] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[02:20:20] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[02:20:38] env1_seed_0 | Stage transition: TRAINING → BLENDING
[02:20:38] env3_seed_0 | Stage transition: TRAINING → BLENDING
[02:20:44] env0_seed_0 | Stage transition: TRAINING → BLENDING
[02:20:44] env2_seed_0 | Stage transition: TRAINING → BLENDING
[02:21:08] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[02:21:08] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[02:21:11] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[02:21:11] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[02:21:14] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:21:14] env3_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:21:14] env3_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:21:14] env3_seed_0 | Fossilized (norm, Δacc +11.83%)
    [env3] Fossilized 'env3_seed_0' (norm, Δacc +11.83%)
[02:21:17] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:21:17] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:21:17] env2_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:21:17] env2_seed_0 | Fossilized (depthwise, Δacc +14.20%)
    [env2] Fossilized 'env2_seed_0' (depthwise, Δacc +14.20%)
[02:21:20] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:21:20] env0_seed_0 | Fossilized (norm, Δacc +8.88%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +8.88%)
[02:21:20] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:21:20] env1_seed_0 | Fossilized (norm, Δacc +12.94%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +12.94%)
Batch 27: Episodes 108/200
  Env accuracies: ['74.3%', '75.6%', '71.5%', '76.8%']
  Avg acc: 74.6% (rolling: 73.2%)
  Avg reward: 57.0
  Actions: {'WAIT': 75, 'GERMINATE_NORM': 64, 'GERMINATE_ATTENTION': 47, 'GERMINATE_DEPTHWISE': 29, 'GERMINATE_CONV_ENHANCE': 36, 'FOSSILIZE': 49, 'CULL': 0}
  Successful: {'WAIT': 75, 'GERMINATE_NORM': 3, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 1, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 4, 'CULL': 0}
  Policy loss: -0.0153, Value loss: 20.1028, Entropy: 1.7057, Entropy coef: 0.0614
[02:24:25] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[02:24:25] env2_seed_0 | Germinated (attention, 2.0K params)
    [env2] Germinated 'env2_seed_0' (attention, 2.0K params)
[02:24:25] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[02:24:25] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[02:24:28] env3_seed_0 | Germinated (norm, 0.1K params)
    [env3] Germinated 'env3_seed_0' (norm, 0.1K params)
[02:24:28] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[02:24:33] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[02:24:33] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[02:24:45] env0_seed_0 | Stage transition: TRAINING → BLENDING
[02:24:45] env2_seed_0 | Stage transition: TRAINING → BLENDING
[02:24:51] env3_seed_0 | Stage transition: TRAINING → BLENDING
[02:25:02] env1_seed_0 | Stage transition: TRAINING → BLENDING
[02:25:14] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[02:25:14] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[02:25:18] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[02:25:21] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:25:21] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:25:25] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[02:25:25] env3_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:25:25] env2_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:25:25] env2_seed_0 | Fossilized (attention, Δacc +9.84%)
    [env2] Fossilized 'env2_seed_0' (attention, Δacc +9.84%)
[02:25:25] env3_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:25:25] env3_seed_0 | Fossilized (norm, Δacc +11.61%)
    [env3] Fossilized 'env3_seed_0' (norm, Δacc +11.61%)
[02:25:30] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:25:30] env1_seed_0 | Stage transition: PROBATIONARY → CULLED
[02:25:30] env1_seed_0 | Culled (norm, Δacc +6.96%)
    [env1] Culled 'env1_seed_0' (norm, Δacc +6.96%)
[02:25:33] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:25:33] env0_seed_0 | Fossilized (norm, Δacc +16.13%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +16.13%)
[02:25:36] env1_seed_1 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_1' (norm, 0.1K params)
[02:25:36] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[02:25:50] env1_seed_1 | Stage transition: TRAINING → BLENDING
[02:25:50] env0_seed_0 | Stage transition: FOSSILIZED → CULLED
[02:25:50] env0_seed_0 | Culled (norm, Δacc +18.09%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +18.09%)
[02:25:53] env0_seed_1 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_1' (depthwise, 4.8K params)
[02:25:53] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[02:26:11] env1_seed_1 | Stage transition: BLENDING → SHADOWING
[02:26:14] env3_seed_0 | Stage transition: FOSSILIZED → CULLED
[02:26:14] env3_seed_0 | Culled (norm, Δacc +19.92%)
    [env3] Culled 'env3_seed_0' (norm, Δacc +19.92%)
[02:26:17] env0_seed_1 | Stage transition: TRAINING → BLENDING
[02:26:17] env1_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[02:26:17] env3_seed_1 | Germinated (attention, 2.0K params)
    [env3] Germinated 'env3_seed_1' (attention, 2.0K params)
[02:26:17] env3_seed_1 | Stage transition: GERMINATED → TRAINING
[02:26:40] env0_seed_1 | Stage transition: BLENDING → SHADOWING
[02:26:44] env1_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[02:26:44] env1_seed_1 | Fossilized (norm, Δacc +3.39%)
    [env1] Fossilized 'env1_seed_1' (norm, Δacc +3.39%)
[02:26:47] env0_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[02:26:54] env3_seed_1 | Stage transition: TRAINING → BLENDING
[02:27:11] env3_seed_1 | Stage transition: BLENDING → SHADOWING
[02:27:17] env3_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[02:27:28] env0_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[02:27:28] env0_seed_1 | Fossilized (depthwise, Δacc +4.99%)
    [env0] Fossilized 'env0_seed_1' (depthwise, Δacc +4.99%)
[02:27:37] env3_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[02:27:37] env3_seed_1 | Fossilized (attention, Δacc +0.96%)
    [env3] Fossilized 'env3_seed_1' (attention, Δacc +0.96%)
[02:28:39] env3_seed_1 | Stage transition: FOSSILIZED → CULLED
[02:28:39] env3_seed_1 | Culled (attention, Δacc +0.31%)
    [env3] Culled 'env3_seed_1' (attention, Δacc +0.31%)
[02:28:42] env3_seed_2 | Germinated (norm, 0.1K params)
    [env3] Germinated 'env3_seed_2' (norm, 0.1K params)
[02:28:42] env3_seed_2 | Stage transition: GERMINATED → TRAINING
Batch 28: Episodes 112/200
  Env accuracies: ['75.0%', '76.4%', '71.8%', '74.3%']
  Avg acc: 74.4% (rolling: 73.2%)
  Avg reward: 44.6
  Actions: {'WAIT': 71, 'GERMINATE_NORM': 56, 'GERMINATE_ATTENTION': 36, 'GERMINATE_DEPTHWISE': 35, 'GERMINATE_CONV_ENHANCE': 28, 'FOSSILIZE': 70, 'CULL': 4}
  Successful: {'WAIT': 71, 'GERMINATE_NORM': 5, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 1, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 6, 'CULL': 4}
  Policy loss: -0.0080, Value loss: 19.2598, Entropy: 1.7370, Entropy coef: 0.0562
[02:28:51] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[02:28:51] env3_seed_0 | Germinated (depthwise, 4.8K params)
    [env3] Germinated 'env3_seed_0' (depthwise, 4.8K params)
[02:28:51] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[02:28:51] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[02:28:55] env2_seed_0 | Germinated (norm, 0.1K params)
    [env2] Germinated 'env2_seed_0' (norm, 0.1K params)
[02:28:55] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[02:29:00] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[02:29:00] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[02:29:12] env0_seed_0 | Stage transition: TRAINING → BLENDING
[02:29:12] env3_seed_0 | Stage transition: TRAINING → BLENDING
[02:29:18] env2_seed_0 | Stage transition: TRAINING → BLENDING
[02:29:24] env1_seed_0 | Stage transition: TRAINING → BLENDING
[02:29:42] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[02:29:42] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[02:29:46] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[02:29:50] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:29:50] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[02:29:50] env3_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:29:53] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:29:53] env2_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:29:53] env2_seed_0 | Fossilized (norm, Δacc +11.15%)
    [env2] Fossilized 'env2_seed_0' (norm, Δacc +11.15%)
[02:29:56] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:30:01] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:30:01] env0_seed_0 | Fossilized (norm, Δacc +14.14%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +14.14%)
[02:30:04] env3_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:30:04] env3_seed_0 | Fossilized (depthwise, Δacc +16.27%)
    [env3] Fossilized 'env3_seed_0' (depthwise, Δacc +16.27%)
[02:30:07] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:30:07] env1_seed_0 | Fossilized (attention, Δacc +9.87%)
    [env1] Fossilized 'env1_seed_0' (attention, Δacc +9.87%)
[02:30:40] env2_seed_0 | Stage transition: FOSSILIZED → CULLED
[02:30:40] env2_seed_0 | Culled (norm, Δacc +19.72%)
    [env2] Culled 'env2_seed_0' (norm, Δacc +19.72%)
[02:30:51] env2_seed_1 | Germinated (attention, 2.0K params)
    [env2] Germinated 'env2_seed_1' (attention, 2.0K params)
[02:30:51] env2_seed_1 | Stage transition: GERMINATED → TRAINING
[02:31:04] env3_seed_0 | Stage transition: FOSSILIZED → CULLED
[02:31:04] env3_seed_0 | Culled (depthwise, Δacc +17.54%)
    [env3] Culled 'env3_seed_0' (depthwise, Δacc +17.54%)
[02:31:07] env3_seed_1 | Germinated (attention, 2.0K params)
    [env3] Germinated 'env3_seed_1' (attention, 2.0K params)
[02:31:07] env3_seed_1 | Stage transition: GERMINATED → TRAINING
[02:31:21] env2_seed_1 | Stage transition: TRAINING → BLENDING
[02:31:25] env3_seed_1 | Stage transition: TRAINING → BLENDING
[02:31:43] env2_seed_1 | Stage transition: BLENDING → SHADOWING
[02:31:47] env3_seed_1 | Stage transition: BLENDING → SHADOWING
[02:31:50] env2_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[02:31:53] env3_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[02:31:58] env2_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[02:31:58] env2_seed_1 | Fossilized (attention, Δacc +2.20%)
    [env2] Fossilized 'env2_seed_1' (attention, Δacc +2.20%)
[02:32:19] env3_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[02:32:19] env3_seed_1 | Fossilized (attention, Δacc +1.91%)
    [env3] Fossilized 'env3_seed_1' (attention, Δacc +1.91%)
Batch 29: Episodes 116/200
  Env accuracies: ['74.4%', '71.8%', '73.2%', '75.2%']
  Avg acc: 73.7% (rolling: 73.4%)
  Avg reward: 50.3
  Actions: {'WAIT': 62, 'GERMINATE_NORM': 60, 'GERMINATE_ATTENTION': 40, 'GERMINATE_DEPTHWISE': 29, 'GERMINATE_CONV_ENHANCE': 34, 'FOSSILIZE': 73, 'CULL': 2}
  Successful: {'WAIT': 62, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 3, 'GERMINATE_DEPTHWISE': 1, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 6, 'CULL': 2}
  Policy loss: -0.0223, Value loss: 20.3449, Entropy: 1.7273, Entropy coef: 0.0511
[02:33:11] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[02:33:11] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[02:33:11] env2_seed_0 | Germinated (norm, 0.1K params)
    [env2] Germinated 'env2_seed_0' (norm, 0.1K params)
[02:33:11] env3_seed_0 | Germinated (norm, 0.1K params)
    [env3] Germinated 'env3_seed_0' (norm, 0.1K params)
[02:33:11] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[02:33:11] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[02:33:11] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[02:33:11] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[02:33:35] env0_seed_0 | Stage transition: TRAINING → BLENDING
[02:33:35] env1_seed_0 | Stage transition: TRAINING → BLENDING
[02:33:35] env2_seed_0 | Stage transition: TRAINING → BLENDING
[02:33:35] env3_seed_0 | Stage transition: TRAINING → BLENDING
[02:34:04] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[02:34:04] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[02:34:04] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[02:34:04] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[02:34:10] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:34:10] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:34:10] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:34:10] env3_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:34:10] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:34:10] env0_seed_0 | Fossilized (norm, Δacc +8.82%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +8.82%)
[02:34:10] env2_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:34:10] env2_seed_0 | Fossilized (norm, Δacc +9.99%)
    [env2] Fossilized 'env2_seed_0' (norm, Δacc +9.99%)
[02:34:13] env3_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:34:13] env3_seed_0 | Fossilized (norm, Δacc +8.08%)
    [env3] Fossilized 'env3_seed_0' (norm, Δacc +8.08%)
[02:34:25] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:34:25] env1_seed_0 | Fossilized (norm, Δacc +13.79%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +13.79%)
Batch 30: Episodes 120/200
  Env accuracies: ['75.7%', '75.6%', '76.5%', '76.1%']
  Avg acc: 76.0% (rolling: 73.6%)
  Avg reward: 59.9
  Actions: {'WAIT': 68, 'GERMINATE_NORM': 36, 'GERMINATE_ATTENTION': 49, 'GERMINATE_DEPTHWISE': 39, 'GERMINATE_CONV_ENHANCE': 35, 'FOSSILIZE': 73, 'CULL': 0}
  Successful: {'WAIT': 68, 'GERMINATE_NORM': 4, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 4, 'CULL': 0}
  Policy loss: -0.0146, Value loss: 21.3275, Entropy: 1.6808, Entropy coef: 0.0459

Blueprint Stats
  ---------------------------------------------------------------------------

Blueprint       Germ  Foss  Cull   Rate     ΔAcc    Churn
  ---------------------------------------------------------------------------

  attention         75    27    61  30.7%   +6.49%   +4.09%
  conv_enhance      82    28    60  31.8%  +12.48%   +7.08%
  depthwise        100    43    68  38.7%   +9.48%   +6.10%
  norm             104    60    56  51.7%   +9.63%   +6.44%
Seed Scoreboard (env 0):
  Fossilized: 36 (+504.1K params, +531.9% of host)
  Culled: 58
  Avg fossilize age: 14.1 epochs
  Avg cull age: 13.0 epochs
  Compute cost: 4.75x baseline
  Distribution: conv_enhance x6, depthwise x10, attention x5, norm x15
Seed Scoreboard (env 1):
  Fossilized: 41 (+642.3K params, +677.8% of host)
  Culled: 72
  Avg fossilize age: 15.2 epochs
  Avg cull age: 13.6 epochs
  Compute cost: 5.59x baseline
  Distribution: norm x19, depthwise x7, conv_enhance x8, attention x7
Seed Scoreboard (env 2):
  Fossilized: 40 (+464.5K params, +490.2% of host)
  Culled: 56
  Avg fossilize age: 15.7 epochs
  Avg cull age: 16.4 epochs
  Compute cost: 6.05x baseline
  Distribution: conv_enhance x5, attention x8, norm x11, depthwise x16
Seed Scoreboard (env 3):
  Fossilized: 41 (+730.1K params, +770.5% of host)
  Culled: 59
  Avg fossilize age: 14.9 epochs
  Avg cull age: 12.4 epochs
  Compute cost: 5.90x baseline
  Distribution: conv_enhance x9, norm x15, depthwise x10, attention x7

[02:37:18] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[02:37:18] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[02:37:18] env2_seed_0 | Germinated (norm, 0.1K params)
    [env2] Germinated 'env2_seed_0' (norm, 0.1K params)
[02:37:18] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[02:37:18] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[02:37:18] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[02:37:22] env3_seed_0 | Germinated (depthwise, 4.8K params)
    [env3] Germinated 'env3_seed_0' (depthwise, 4.8K params)
[02:37:23] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[02:37:40] env0_seed_0 | Stage transition: TRAINING → BLENDING
[02:37:40] env1_seed_0 | Stage transition: TRAINING → BLENDING
[02:37:40] env2_seed_0 | Stage transition: TRAINING → BLENDING
[02:37:46] env3_seed_0 | Stage transition: TRAINING → BLENDING
[02:38:11] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[02:38:11] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[02:38:11] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[02:38:14] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[02:38:17] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:38:17] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:38:17] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:38:20] env3_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:38:20] env2_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:38:20] env2_seed_0 | Fossilized (norm, Δacc +15.67%)
    [env2] Fossilized 'env2_seed_0' (norm, Δacc +15.67%)
[02:38:23] env3_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:38:23] env3_seed_0 | Fossilized (depthwise, Δacc +8.37%)
    [env3] Fossilized 'env3_seed_0' (depthwise, Δacc +8.37%)
[02:38:29] env2_seed_0 | Stage transition: FOSSILIZED → CULLED
[02:38:29] env2_seed_0 | Culled (norm, Δacc +10.28%)
    [env2] Culled 'env2_seed_0' (norm, Δacc +10.28%)
[02:38:32] env2_seed_1 | Germinated (norm, 0.1K params)
    [env2] Germinated 'env2_seed_1' (norm, 0.1K params)
[02:38:32] env2_seed_1 | Stage transition: GERMINATED → TRAINING
[02:38:35] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:38:35] env1_seed_0 | Fossilized (attention, Δacc +5.66%)
    [env1] Fossilized 'env1_seed_0' (attention, Δacc +5.66%)
[02:38:45] env2_seed_1 | Stage transition: TRAINING → BLENDING
[02:38:58] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:38:58] env0_seed_0 | Fossilized (norm, Δacc +21.85%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +21.85%)
[02:39:01] env2_seed_1 | Stage transition: BLENDING → SHADOWING
[02:39:07] env2_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[02:39:07] env2_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[02:39:07] env2_seed_1 | Fossilized (norm, Δacc +7.29%)
    [env2] Fossilized 'env2_seed_1' (norm, Δacc +7.29%)
[02:41:22] env2_seed_1 | Stage transition: FOSSILIZED → CULLED
[02:41:22] env2_seed_1 | Culled (norm, Δacc +8.92%)
    [env2] Culled 'env2_seed_1' (norm, Δacc +8.92%)
Batch 31: Episodes 124/200
  Env accuracies: ['73.7%', '70.6%', '74.9%', '73.7%']
  Avg acc: 73.2% (rolling: 73.7%)
  Avg reward: 51.7
  Actions: {'WAIT': 84, 'GERMINATE_NORM': 42, 'GERMINATE_ATTENTION': 34, 'GERMINATE_DEPTHWISE': 37, 'GERMINATE_CONV_ENHANCE': 32, 'FOSSILIZE': 69, 'CULL': 2}
  Successful: {'WAIT': 84, 'GERMINATE_NORM': 3, 'GERMINATE_ATTENTION': 1, 'GERMINATE_DEPTHWISE': 1, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 5, 'CULL': 2}
  Policy loss: -0.0152, Value loss: 20.0636, Entropy: 1.6666, Entropy coef: 0.0408
[02:41:27] env1_seed_0 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_0' (depthwise, 4.8K params)
[02:41:27] env2_seed_0 | Germinated (attention, 2.0K params)
    [env2] Germinated 'env2_seed_0' (attention, 2.0K params)
[02:41:27] env3_seed_0 | Germinated (depthwise, 4.8K params)
    [env3] Germinated 'env3_seed_0' (depthwise, 4.8K params)
[02:41:28] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[02:41:28] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[02:41:28] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[02:41:32] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[02:41:32] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[02:41:50] env1_seed_0 | Stage transition: TRAINING → BLENDING
[02:41:50] env2_seed_0 | Stage transition: TRAINING → BLENDING
[02:41:50] env3_seed_0 | Stage transition: TRAINING → BLENDING
[02:41:57] env0_seed_0 | Stage transition: TRAINING → BLENDING
[02:42:21] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[02:42:21] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[02:42:21] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[02:42:24] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[02:42:27] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:42:27] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:42:27] env3_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:42:27] env3_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:42:27] env3_seed_0 | Fossilized (depthwise, Δacc +8.56%)
    [env3] Fossilized 'env3_seed_0' (depthwise, Δacc +8.56%)
[02:42:30] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:42:30] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:42:30] env0_seed_0 | Fossilized (norm, Δacc +11.79%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +11.79%)
[02:42:33] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:42:33] env1_seed_0 | Fossilized (depthwise, Δacc +12.91%)
    [env1] Fossilized 'env1_seed_0' (depthwise, Δacc +12.91%)
[02:42:42] env2_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:42:42] env2_seed_0 | Fossilized (attention, Δacc +8.27%)
    [env2] Fossilized 'env2_seed_0' (attention, Δacc +8.27%)
[02:43:21] env2_seed_0 | Stage transition: FOSSILIZED → CULLED
[02:43:21] env2_seed_0 | Culled (attention, Δacc +15.47%)
    [env2] Culled 'env2_seed_0' (attention, Δacc +15.47%)
[02:43:33] env2_seed_1 | Germinated (norm, 0.1K params)
    [env2] Germinated 'env2_seed_1' (norm, 0.1K params)
[02:43:34] env2_seed_1 | Stage transition: GERMINATED → TRAINING
[02:43:50] env2_seed_1 | Stage transition: TRAINING → BLENDING
[02:44:06] env2_seed_1 | Stage transition: BLENDING → SHADOWING
[02:44:12] env2_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[02:44:12] env2_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[02:44:12] env2_seed_1 | Fossilized (norm, Δacc +1.69%)
    [env2] Fossilized 'env2_seed_1' (norm, Δacc +1.69%)
[02:44:42] env3_seed_0 | Stage transition: FOSSILIZED → CULLED
[02:44:42] env3_seed_0 | Culled (depthwise, Δacc +12.29%)
    [env3] Culled 'env3_seed_0' (depthwise, Δacc +12.29%)
[02:44:45] env3_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env3] Germinated 'env3_seed_1' (conv_enhance, 74.0K params)
[02:44:45] env3_seed_1 | Stage transition: GERMINATED → TRAINING
[02:44:59] env3_seed_1 | Stage transition: TRAINING → BLENDING
[02:45:17] env3_seed_1 | Stage transition: BLENDING → SHADOWING
[02:45:24] env3_seed_1 | Stage transition: SHADOWING → PROBATIONARY
Batch 32: Episodes 128/200
  Env accuracies: ['76.8%', '66.7%', '76.5%', '77.9%']
  Avg acc: 74.5% (rolling: 73.9%)
  Avg reward: 51.6
  Actions: {'WAIT': 98, 'GERMINATE_NORM': 26, 'GERMINATE_ATTENTION': 33, 'GERMINATE_DEPTHWISE': 37, 'GERMINATE_CONV_ENHANCE': 29, 'FOSSILIZE': 74, 'CULL': 3}
  Successful: {'WAIT': 98, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 1, 'GERMINATE_DEPTHWISE': 2, 'GERMINATE_CONV_ENHANCE': 1, 'FOSSILIZE': 5, 'CULL': 2}
  Policy loss: -0.0181, Value loss: 21.0538, Entropy: 1.6130, Entropy coef: 0.0357
[02:45:51] env0_seed_0 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_0' (conv_enhance, 74.0K params)
[02:45:51] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[02:45:51] env2_seed_0 | Germinated (conv_enhance, 74.0K params)
    [env2] Germinated 'env2_seed_0' (conv_enhance, 74.0K params)
[02:45:51] env3_seed_0 | Germinated (norm, 0.1K params)
    [env3] Germinated 'env3_seed_0' (norm, 0.1K params)
[02:45:51] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[02:45:51] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[02:45:51] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[02:45:51] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[02:46:15] env0_seed_0 | Stage transition: TRAINING → BLENDING
[02:46:15] env1_seed_0 | Stage transition: TRAINING → BLENDING
[02:46:15] env2_seed_0 | Stage transition: TRAINING → BLENDING
[02:46:21] env3_seed_0 | Stage transition: TRAINING → BLENDING
[02:46:46] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[02:46:46] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[02:46:46] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[02:46:50] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[02:46:53] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:46:53] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:46:53] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:46:53] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:46:53] env0_seed_0 | Fossilized (conv_enhance, Δacc +15.68%)
    [env0] Fossilized 'env0_seed_0' (conv_enhance, Δacc +15.68%)
[02:46:56] env3_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:46:59] env2_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:46:59] env2_seed_0 | Fossilized (conv_enhance, Δacc +16.78%)
    [env2] Fossilized 'env2_seed_0' (conv_enhance, Δacc +16.78%)
[02:47:02] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:47:02] env1_seed_0 | Fossilized (norm, Δacc +13.71%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +13.71%)
[02:47:09] env3_seed_0 | Stage transition: PROBATIONARY → CULLED
[02:47:09] env3_seed_0 | Culled (norm, Δacc +15.65%)
    [env3] Culled 'env3_seed_0' (norm, Δacc +15.65%)
[02:47:12] env3_seed_1 | Germinated (norm, 0.1K params)
    [env3] Germinated 'env3_seed_1' (norm, 0.1K params)
[02:47:12] env3_seed_1 | Stage transition: GERMINATED → TRAINING
[02:47:26] env3_seed_1 | Stage transition: TRAINING → BLENDING
[02:47:43] env3_seed_1 | Stage transition: BLENDING → SHADOWING
[02:47:50] env3_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[02:47:50] env3_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[02:47:50] env3_seed_1 | Fossilized (norm, Δacc +5.95%)
    [env3] Fossilized 'env3_seed_1' (norm, Δacc +5.95%)
Batch 33: Episodes 132/200
  Env accuracies: ['74.3%', '75.8%', '71.9%', '76.1%']
  Avg acc: 74.5% (rolling: 73.8%)
  Avg reward: 38.9
  Actions: {'WAIT': 110, 'GERMINATE_NORM': 47, 'GERMINATE_ATTENTION': 21, 'GERMINATE_DEPTHWISE': 24, 'GERMINATE_CONV_ENHANCE': 22, 'FOSSILIZE': 75, 'CULL': 1}
  Successful: {'WAIT': 110, 'GERMINATE_NORM': 3, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 2, 'FOSSILIZE': 4, 'CULL': 1}
  Policy loss: -0.0087, Value loss: 11.7827, Entropy: 1.6362, Entropy coef: 0.0305
[02:50:17] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[02:50:18] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[02:50:21] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[02:50:21] env2_seed_0 | Germinated (norm, 0.1K params)
    [env2] Germinated 'env2_seed_0' (norm, 0.1K params)
[02:50:21] env3_seed_0 | Germinated (attention, 2.0K params)
    [env3] Germinated 'env3_seed_0' (attention, 2.0K params)
[02:50:21] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[02:50:21] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[02:50:21] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[02:50:38] env1_seed_0 | Stage transition: TRAINING → BLENDING
[02:50:44] env0_seed_0 | Stage transition: TRAINING → BLENDING
[02:50:44] env3_seed_0 | Stage transition: TRAINING → BLENDING
[02:50:50] env2_seed_0 | Stage transition: TRAINING → BLENDING
[02:51:08] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[02:51:13] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[02:51:13] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[02:51:16] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:51:16] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[02:51:19] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:51:19] env3_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:51:19] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:51:19] env0_seed_0 | Fossilized (norm, Δacc +14.43%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +14.43%)
[02:51:22] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:51:25] env2_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:51:25] env2_seed_0 | Fossilized (norm, Δacc +12.14%)
    [env2] Fossilized 'env2_seed_0' (norm, Δacc +12.14%)
[02:51:28] env3_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:51:28] env3_seed_0 | Fossilized (attention, Δacc +2.63%)
    [env3] Fossilized 'env3_seed_0' (attention, Δacc +2.63%)
[02:51:31] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:51:31] env1_seed_0 | Fossilized (norm, Δacc +21.25%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +21.25%)
[02:51:45] env0_seed_0 | Stage transition: FOSSILIZED → CULLED
[02:51:45] env0_seed_0 | Culled (norm, Δacc +20.10%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +20.10%)
[02:51:51] env0_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_1' (conv_enhance, 74.0K params)
[02:51:51] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[02:52:05] env0_seed_1 | Stage transition: TRAINING → BLENDING
[02:52:21] env0_seed_1 | Stage transition: BLENDING → SHADOWING
[02:52:27] env0_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[02:52:43] env0_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[02:52:43] env0_seed_1 | Fossilized (conv_enhance, Δacc +6.66%)
    [env0] Fossilized 'env0_seed_1' (conv_enhance, Δacc +6.66%)
[02:54:27] env3_seed_0 | Stage transition: FOSSILIZED → CULLED
[02:54:27] env3_seed_0 | Culled (attention, Δacc +19.22%)
    [env3] Culled 'env3_seed_0' (attention, Δacc +19.22%)
Batch 34: Episodes 136/200
  Env accuracies: ['77.0%', '74.1%', '75.8%', '75.0%']
  Avg acc: 75.5% (rolling: 73.9%)
  Avg reward: 46.5
  Actions: {'WAIT': 70, 'GERMINATE_NORM': 41, 'GERMINATE_ATTENTION': 42, 'GERMINATE_DEPTHWISE': 28, 'GERMINATE_CONV_ENHANCE': 39, 'FOSSILIZE': 78, 'CULL': 2}
  Successful: {'WAIT': 70, 'GERMINATE_NORM': 3, 'GERMINATE_ATTENTION': 1, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 1, 'FOSSILIZE': 5, 'CULL': 2}
  Policy loss: -0.0106, Value loss: 15.1196, Entropy: 1.6245, Entropy coef: 0.0254
[02:54:32] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[02:54:32] env2_seed_0 | Germinated (norm, 0.1K params)
    [env2] Germinated 'env2_seed_0' (norm, 0.1K params)
[02:54:32] env3_seed_0 | Germinated (norm, 0.1K params)
    [env3] Germinated 'env3_seed_0' (norm, 0.1K params)
[02:54:32] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[02:54:32] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[02:54:32] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[02:54:37] env1_seed_0 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_0' (conv_enhance, 74.0K params)
[02:54:37] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[02:54:55] env2_seed_0 | Stage transition: TRAINING → BLENDING
[02:54:55] env3_seed_0 | Stage transition: TRAINING → BLENDING
[02:55:01] env0_seed_0 | Stage transition: TRAINING → BLENDING
[02:55:01] env1_seed_0 | Stage transition: TRAINING → BLENDING
[02:55:25] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[02:55:25] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[02:55:30] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[02:55:30] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[02:55:33] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:55:33] env3_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:55:36] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:55:36] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:55:36] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:55:36] env0_seed_0 | Fossilized (norm, Δacc +11.08%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +11.08%)
[02:55:39] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:55:39] env1_seed_0 | Fossilized (conv_enhance, Δacc +16.98%)
    [env1] Fossilized 'env1_seed_0' (conv_enhance, Δacc +16.98%)
[02:55:42] env2_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:55:42] env2_seed_0 | Fossilized (norm, Δacc +13.45%)
    [env2] Fossilized 'env2_seed_0' (norm, Δacc +13.45%)
[02:55:51] env3_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:55:51] env3_seed_0 | Fossilized (norm, Δacc +17.39%)
    [env3] Fossilized 'env3_seed_0' (norm, Δacc +17.39%)
Batch 35: Episodes 140/200
  Env accuracies: ['75.3%', '72.2%', '76.3%', '73.8%']
  Avg acc: 74.4% (rolling: 74.1%)
  Avg reward: 49.9
  Actions: {'WAIT': 96, 'GERMINATE_NORM': 45, 'GERMINATE_ATTENTION': 37, 'GERMINATE_DEPTHWISE': 25, 'GERMINATE_CONV_ENHANCE': 30, 'FOSSILIZE': 67, 'CULL': 0}
  Successful: {'WAIT': 96, 'GERMINATE_NORM': 3, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 1, 'FOSSILIZE': 4, 'CULL': 0}
  Policy loss: -0.0095, Value loss: 12.2838, Entropy: 1.5858, Entropy coef: 0.0203

Blueprint Stats
  ---------------------------------------------------------------------------

Blueprint       Germ  Foss  Cull   Rate     ΔAcc    Churn
  ---------------------------------------------------------------------------

  attention         78    30    63  32.3%   +6.39%   +4.51%
  conv_enhance      87    32    60  34.8%  +12.68%   +7.08%
  depthwise        103    46    69  40.0%   +9.51%   +6.19%
  norm             118    73    60  54.9%  +10.21%   +6.92%
Seed Scoreboard (env 0):
  Fossilized: 42 (+652.6K params, +688.6% of host)
  Culled: 59
  Avg fossilize age: 14.1 epochs
  Avg cull age: 13.1 epochs
  Compute cost: 5.13x baseline
  Distribution: conv_enhance x8, depthwise x10, attention x5, norm x19
Seed Scoreboard (env 1):
  Fossilized: 46 (+723.3K params, +763.3% of host)
  Culled: 72
  Avg fossilize age: 15.2 epochs
  Avg cull age: 13.6 epochs
  Compute cost: 6.21x baseline
  Distribution: norm x21, depthwise x8, conv_enhance x9, attention x8
Seed Scoreboard (env 2):
  Fossilized: 47 (+541.2K params, +571.1% of host)
  Culled: 59
  Avg fossilize age: 15.3 epochs
  Avg cull age: 17.3 epochs
  Compute cost: 6.65x baseline
  Distribution: conv_enhance x6, attention x9, norm x16, depthwise x16
Seed Scoreboard (env 3):
  Fossilized: 46 (+742.0K params, +783.0% of host)
  Culled: 62
  Avg fossilize age: 14.7 epochs
  Avg cull age: 14.1 epochs
  Compute cost: 6.45x baseline
  Distribution: conv_enhance x9, norm x17, depthwise x12, attention x8

[02:58:48] env2_seed_0 | Germinated (norm, 0.1K params)
    [env2] Germinated 'env2_seed_0' (norm, 0.1K params)
[02:58:48] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[02:58:51] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[02:58:51] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[02:58:55] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[02:58:55] env3_seed_0 | Germinated (attention, 2.0K params)
    [env3] Germinated 'env3_seed_0' (attention, 2.0K params)
[02:58:56] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[02:58:56] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[02:59:13] env1_seed_0 | Stage transition: TRAINING → BLENDING
[02:59:19] env0_seed_0 | Stage transition: TRAINING → BLENDING
[02:59:19] env2_seed_0 | Stage transition: TRAINING → BLENDING
[02:59:19] env3_seed_0 | Stage transition: TRAINING → BLENDING
[02:59:43] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[02:59:48] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[02:59:48] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[02:59:48] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[02:59:51] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:59:51] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:59:51] env1_seed_0 | Fossilized (norm, Δacc +8.38%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +8.38%)
[02:59:54] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:59:54] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:59:54] env3_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[02:59:54] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:59:54] env0_seed_0 | Fossilized (norm, Δacc +9.85%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +9.85%)
[02:59:56] env2_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:59:56] env2_seed_0 | Fossilized (norm, Δacc +11.10%)
    [env2] Fossilized 'env2_seed_0' (norm, Δacc +11.10%)
[02:59:59] env3_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[02:59:59] env3_seed_0 | Fossilized (attention, Δacc +0.49%)
    [env3] Fossilized 'env3_seed_0' (attention, Δacc +0.49%)
Batch 36: Episodes 144/200
  Env accuracies: ['76.2%', '76.4%', '77.0%', '71.2%']
  Avg acc: 75.2% (rolling: 74.6%)
  Avg reward: 60.8
  Actions: {'WAIT': 93, 'GERMINATE_NORM': 47, 'GERMINATE_ATTENTION': 30, 'GERMINATE_DEPTHWISE': 23, 'GERMINATE_CONV_ENHANCE': 20, 'FOSSILIZE': 87, 'CULL': 0}
  Successful: {'WAIT': 93, 'GERMINATE_NORM': 3, 'GERMINATE_ATTENTION': 1, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 4, 'CULL': 0}
  Policy loss: -0.0060, Value loss: 18.4132, Entropy: 1.5534, Entropy coef: 0.0151
[03:02:55] env2_seed_0 | Germinated (norm, 0.1K params)
    [env2] Germinated 'env2_seed_0' (norm, 0.1K params)
[03:02:55] env3_seed_0 | Germinated (norm, 0.1K params)
    [env3] Germinated 'env3_seed_0' (norm, 0.1K params)
[03:02:55] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[03:02:56] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[03:03:13] env2_seed_0 | Stage transition: TRAINING → BLENDING
[03:03:13] env3_seed_0 | Stage transition: TRAINING → BLENDING
[03:03:13] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[03:03:13] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[03:03:23] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[03:03:23] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[03:03:35] env1_seed_0 | Stage transition: TRAINING → BLENDING
[03:03:41] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[03:03:41] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[03:03:45] env0_seed_0 | Stage transition: TRAINING → BLENDING
[03:03:50] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:03:50] env3_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:03:50] env3_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:03:50] env3_seed_0 | Fossilized (norm, Δacc +15.12%)
    [env3] Fossilized 'env3_seed_0' (norm, Δacc +15.12%)
[03:03:59] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[03:04:05] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[03:04:05] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:04:05] env2_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:04:05] env2_seed_0 | Fossilized (norm, Δacc +17.13%)
    [env2] Fossilized 'env2_seed_0' (norm, Δacc +17.13%)
[03:04:11] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:04:11] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:04:11] env1_seed_0 | Fossilized (norm, Δacc +11.59%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +11.59%)
[03:04:20] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:04:20] env0_seed_0 | Fossilized (norm, Δacc +16.04%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +16.04%)
[03:06:29] env1_seed_0 | Stage transition: FOSSILIZED → CULLED
[03:06:29] env1_seed_0 | Culled (norm, Δacc +16.58%)
    [env1] Culled 'env1_seed_0' (norm, Δacc +16.58%)
[03:06:32] env1_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_1' (conv_enhance, 74.0K params)
[03:06:32] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[03:06:45] env1_seed_1 | Stage transition: TRAINING → BLENDING
Batch 37: Episodes 148/200
  Env accuracies: ['76.3%', '74.7%', '76.6%', '74.6%']
  Avg acc: 75.6% (rolling: 74.7%)
  Avg reward: 58.9
  Actions: {'WAIT': 121, 'GERMINATE_NORM': 39, 'GERMINATE_ATTENTION': 33, 'GERMINATE_DEPTHWISE': 23, 'GERMINATE_CONV_ENHANCE': 21, 'FOSSILIZE': 62, 'CULL': 1}
  Successful: {'WAIT': 121, 'GERMINATE_NORM': 4, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 1, 'FOSSILIZE': 4, 'CULL': 1}
  Policy loss: -0.0146, Value loss: 20.8429, Entropy: 1.5531, Entropy coef: 0.0100
[03:07:05] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[03:07:05] env2_seed_0 | Germinated (norm, 0.1K params)
    [env2] Germinated 'env2_seed_0' (norm, 0.1K params)
[03:07:05] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[03:07:05] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[03:07:10] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[03:07:10] env3_seed_0 | Germinated (norm, 0.1K params)
    [env3] Germinated 'env3_seed_0' (norm, 0.1K params)
[03:07:10] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[03:07:10] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[03:07:27] env1_seed_0 | Stage transition: TRAINING → BLENDING
[03:07:27] env2_seed_0 | Stage transition: TRAINING → BLENDING
[03:07:33] env0_seed_0 | Stage transition: TRAINING → BLENDING
[03:07:33] env3_seed_0 | Stage transition: TRAINING → BLENDING
[03:07:57] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[03:07:57] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[03:08:02] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[03:08:02] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[03:08:05] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:08:05] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:08:05] env2_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:08:05] env2_seed_0 | Fossilized (norm, Δacc +14.04%)
    [env2] Fossilized 'env2_seed_0' (norm, Δacc +14.04%)
[03:08:08] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:08:08] env3_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:08:08] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:08:08] env1_seed_0 | Fossilized (norm, Δacc +19.21%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +19.21%)
[03:08:11] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:08:11] env0_seed_0 | Fossilized (norm, Δacc +13.05%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +13.05%)
[03:08:25] env3_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:08:25] env3_seed_0 | Fossilized (norm, Δacc +14.72%)
    [env3] Fossilized 'env3_seed_0' (norm, Δacc +14.72%)
Batch 38: Episodes 152/200
  Env accuracies: ['75.0%', '76.2%', '76.4%', '75.5%']
  Avg acc: 75.8% (rolling: 74.8%)
  Avg reward: 63.0
  Actions: {'WAIT': 105, 'GERMINATE_NORM': 43, 'GERMINATE_ATTENTION': 29, 'GERMINATE_DEPTHWISE': 34, 'GERMINATE_CONV_ENHANCE': 24, 'FOSSILIZE': 65, 'CULL': 0}
  Successful: {'WAIT': 105, 'GERMINATE_NORM': 4, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 4, 'CULL': 0}
  Policy loss: -0.0127, Value loss: 18.5399, Entropy: 1.4738, Entropy coef: 0.0100
[03:11:12] env3_seed_0 | Germinated (conv_enhance, 74.0K params)
    [env3] Germinated 'env3_seed_0' (conv_enhance, 74.0K params)
[03:11:13] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[03:11:16] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[03:11:16] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[03:11:30] env3_seed_0 | Stage transition: TRAINING → BLENDING
[03:11:30] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[03:11:30] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[03:11:34] env0_seed_0 | Stage transition: TRAINING → BLENDING
[03:11:49] env1_seed_0 | Stage transition: TRAINING → BLENDING
[03:11:49] env2_seed_0 | Germinated (norm, 0.1K params)
    [env2] Germinated 'env2_seed_0' (norm, 0.1K params)
[03:11:49] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[03:11:55] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[03:12:00] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[03:12:05] env3_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:12:09] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:12:09] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:12:09] env0_seed_0 | Fossilized (norm, Δacc +8.37%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +8.37%)
[03:12:09] env3_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:12:09] env3_seed_0 | Fossilized (conv_enhance, Δacc +18.60%)
    [env3] Fossilized 'env3_seed_0' (conv_enhance, Δacc +18.60%)
[03:12:14] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[03:12:14] env2_seed_0 | Stage transition: TRAINING → BLENDING
[03:12:20] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:12:20] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:12:20] env1_seed_0 | Fossilized (norm, Δacc +12.62%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +12.62%)
[03:12:30] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[03:12:36] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:12:36] env2_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:12:36] env2_seed_0 | Fossilized (norm, Δacc +6.90%)
    [env2] Fossilized 'env2_seed_0' (norm, Δacc +6.90%)
Batch 39: Episodes 156/200
  Env accuracies: ['76.6%', '75.2%', '77.0%', '76.3%']
  Avg acc: 76.3% (rolling: 75.1%)
  Avg reward: 53.0
  Actions: {'WAIT': 109, 'GERMINATE_NORM': 32, 'GERMINATE_ATTENTION': 29, 'GERMINATE_DEPTHWISE': 31, 'GERMINATE_CONV_ENHANCE': 15, 'FOSSILIZE': 84, 'CULL': 0}
  Successful: {'WAIT': 109, 'GERMINATE_NORM': 3, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 1, 'FOSSILIZE': 4, 'CULL': 0}
  Policy loss: 0.0019, Value loss: 11.1325, Entropy: 1.5025, Entropy coef: 0.0100
[03:15:24] env3_seed_0 | Germinated (norm, 0.1K params)
    [env3] Germinated 'env3_seed_0' (norm, 0.1K params)
[03:15:24] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[03:15:30] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[03:15:30] env2_seed_0 | Germinated (norm, 0.1K params)
    [env2] Germinated 'env2_seed_0' (norm, 0.1K params)
[03:15:30] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[03:15:30] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[03:15:35] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[03:15:35] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[03:15:47] env3_seed_0 | Stage transition: TRAINING → BLENDING
[03:15:53] env0_seed_0 | Stage transition: TRAINING → BLENDING
[03:15:59] env1_seed_0 | Stage transition: TRAINING → BLENDING
[03:15:59] env2_seed_0 | Stage transition: TRAINING → BLENDING
[03:16:17] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[03:16:21] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[03:16:26] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[03:16:26] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[03:16:26] env3_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:16:29] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:16:29] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:16:29] env0_seed_0 | Fossilized (norm, Δacc +14.28%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +14.28%)
[03:16:29] env3_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:16:29] env3_seed_0 | Fossilized (norm, Δacc +10.82%)
    [env3] Fossilized 'env3_seed_0' (norm, Δacc +10.82%)
[03:16:32] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:16:32] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:16:32] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:16:32] env1_seed_0 | Fossilized (norm, Δacc +10.88%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +10.88%)
[03:16:43] env2_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:16:43] env2_seed_0 | Fossilized (norm, Δacc +11.80%)
    [env2] Fossilized 'env2_seed_0' (norm, Δacc +11.80%)
[03:17:33] env2_seed_0 | Stage transition: FOSSILIZED → CULLED
[03:17:33] env2_seed_0 | Culled (norm, Δacc +14.96%)
    [env2] Culled 'env2_seed_0' (norm, Δacc +14.96%)
[03:17:48] env2_seed_1 | Germinated (norm, 0.1K params)
    [env2] Germinated 'env2_seed_1' (norm, 0.1K params)
[03:17:48] env2_seed_1 | Stage transition: GERMINATED → TRAINING
[03:18:01] env2_seed_1 | Stage transition: TRAINING → BLENDING
[03:18:17] env2_seed_1 | Stage transition: BLENDING → SHADOWING
[03:18:23] env2_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[03:18:26] env2_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[03:18:26] env2_seed_1 | Fossilized (norm, Δacc +10.39%)
    [env2] Fossilized 'env2_seed_1' (norm, Δacc +10.39%)
Batch 40: Episodes 160/200
  Env accuracies: ['74.8%', '75.7%', '76.2%', '76.3%']
  Avg acc: 75.7% (rolling: 75.1%)
  Avg reward: 58.5
  Actions: {'WAIT': 108, 'GERMINATE_NORM': 35, 'GERMINATE_ATTENTION': 38, 'GERMINATE_DEPTHWISE': 40, 'GERMINATE_CONV_ENHANCE': 7, 'FOSSILIZE': 71, 'CULL': 1}
  Successful: {'WAIT': 108, 'GERMINATE_NORM': 5, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 5, 'CULL': 1}
  Policy loss: -0.0144, Value loss: 16.2405, Entropy: 1.4183, Entropy coef: 0.0100

Blueprint Stats
  ---------------------------------------------------------------------------

Blueprint       Germ  Foss  Cull   Rate     ΔAcc    Churn
  ---------------------------------------------------------------------------

  attention         79    31    63  33.0%   +6.20%   +4.51%
  conv_enhance      89    33    60  35.5%  +12.86%   +7.08%
  depthwise        103    46    69  40.0%   +9.51%   +6.19%
  norm             137    92    62  59.7%  +10.67%   +7.21%
Seed Scoreboard (env 0):
  Fossilized: 47 (+653.2K params, +689.3% of host)
  Culled: 59
  Avg fossilize age: 13.9 epochs
  Avg cull age: 13.1 epochs
  Compute cost: 5.23x baseline
  Distribution: conv_enhance x8, depthwise x10, attention x5, norm x24
Seed Scoreboard (env 1):
  Fossilized: 51 (+724.0K params, +764.0% of host)
  Culled: 73
  Avg fossilize age: 14.8 epochs
  Avg cull age: 14.2 epochs
  Compute cost: 6.31x baseline
  Distribution: norm x26, depthwise x8, conv_enhance x9, attention x8
Seed Scoreboard (env 2):
  Fossilized: 53 (+542.0K params, +571.9% of host)
  Culled: 60
  Avg fossilize age: 15.0 epochs
  Avg cull age: 17.6 epochs
  Compute cost: 6.77x baseline
  Distribution: conv_enhance x6, attention x9, norm x22, depthwise x16
Seed Scoreboard (env 3):
  Fossilized: 51 (+818.5K params, +863.7% of host)
  Culled: 62
  Avg fossilize age: 14.5 epochs
  Avg cull age: 14.1 epochs
  Compute cost: 7.01x baseline
  Distribution: conv_enhance x10, norm x20, depthwise x12, attention x9

[03:19:36] env0_seed_0 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_0' (attention, 2.0K params)
[03:19:36] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[03:19:39] env2_seed_0 | Germinated (norm, 0.1K params)
    [env2] Germinated 'env2_seed_0' (norm, 0.1K params)
[03:19:39] env3_seed_0 | Germinated (norm, 0.1K params)
    [env3] Germinated 'env3_seed_0' (norm, 0.1K params)
[03:19:39] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[03:19:39] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[03:19:49] env1_seed_0 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_0' (depthwise, 4.8K params)
[03:19:49] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[03:19:55] env0_seed_0 | Stage transition: TRAINING → BLENDING
[03:20:00] env2_seed_0 | Stage transition: TRAINING → BLENDING
[03:20:00] env3_seed_0 | Stage transition: TRAINING → BLENDING
[03:20:12] env1_seed_0 | Stage transition: TRAINING → BLENDING
[03:20:24] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[03:20:29] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[03:20:29] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[03:20:33] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:20:36] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[03:20:36] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:20:36] env3_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:20:36] env2_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:20:36] env2_seed_0 | Fossilized (norm, Δacc +12.49%)
    [env2] Fossilized 'env2_seed_0' (norm, Δacc +12.49%)
[03:20:36] env3_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:20:36] env3_seed_0 | Fossilized (norm, Δacc +14.81%)
    [env3] Fossilized 'env3_seed_0' (norm, Δacc +14.81%)
[03:20:42] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:20:45] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:20:45] env0_seed_0 | Fossilized (attention, Δacc +9.70%)
    [env0] Fossilized 'env0_seed_0' (attention, Δacc +9.70%)
[03:20:57] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:20:57] env1_seed_0 | Fossilized (depthwise, Δacc +5.69%)
    [env1] Fossilized 'env1_seed_0' (depthwise, Δacc +5.69%)
Batch 41: Episodes 164/200
  Env accuracies: ['75.0%', '66.6%', '73.9%', '76.6%']
  Avg acc: 73.0% (rolling: 75.0%)
  Avg reward: 60.5
  Actions: {'WAIT': 102, 'GERMINATE_NORM': 25, 'GERMINATE_ATTENTION': 30, 'GERMINATE_DEPTHWISE': 39, 'GERMINATE_CONV_ENHANCE': 8, 'FOSSILIZE': 96, 'CULL': 0}
  Successful: {'WAIT': 102, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 1, 'GERMINATE_DEPTHWISE': 1, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 4, 'CULL': 0}
  Policy loss: -0.0165, Value loss: 16.5219, Entropy: 1.4075, Entropy coef: 0.0100
[03:23:42] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[03:23:42] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[03:23:42] env3_seed_0 | Germinated (depthwise, 4.8K params)
    [env3] Germinated 'env3_seed_0' (depthwise, 4.8K params)
[03:23:42] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[03:23:42] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[03:23:42] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[03:24:01] env0_seed_0 | Stage transition: TRAINING → BLENDING
[03:24:01] env1_seed_0 | Stage transition: TRAINING → BLENDING
[03:24:01] env3_seed_0 | Stage transition: TRAINING → BLENDING
[03:24:01] env2_seed_0 | Germinated (norm, 0.1K params)
    [env2] Germinated 'env2_seed_0' (norm, 0.1K params)
[03:24:01] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[03:24:31] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[03:24:31] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[03:24:31] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[03:24:38] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:24:38] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:24:38] env2_seed_0 | Stage transition: TRAINING → BLENDING
[03:24:38] env3_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:24:38] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:24:38] env1_seed_0 | Fossilized (norm, Δacc +8.34%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +8.34%)
[03:24:41] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:24:41] env0_seed_0 | Fossilized (norm, Δacc +6.02%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +6.02%)
[03:24:44] env3_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:24:44] env3_seed_0 | Fossilized (depthwise, Δacc +11.00%)
    [env3] Fossilized 'env3_seed_0' (depthwise, Δacc +11.00%)
[03:24:54] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[03:25:00] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:25:00] env2_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:25:00] env2_seed_0 | Fossilized (norm, Δacc +12.25%)
    [env2] Fossilized 'env2_seed_0' (norm, Δacc +12.25%)
Batch 42: Episodes 168/200
  Env accuracies: ['75.0%', '76.2%', '75.1%', '71.2%']
  Avg acc: 74.4% (rolling: 75.0%)
  Avg reward: 62.7
  Actions: {'WAIT': 117, 'GERMINATE_NORM': 27, 'GERMINATE_ATTENTION': 25, 'GERMINATE_DEPTHWISE': 28, 'GERMINATE_CONV_ENHANCE': 10, 'FOSSILIZE': 93, 'CULL': 0}
  Successful: {'WAIT': 117, 'GERMINATE_NORM': 3, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 1, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 4, 'CULL': 0}
  Policy loss: -0.0160, Value loss: 18.7541, Entropy: 1.3506, Entropy coef: 0.0100
[03:27:48] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[03:27:48] env2_seed_0 | Germinated (depthwise, 4.8K params)
    [env2] Germinated 'env2_seed_0' (depthwise, 4.8K params)
[03:27:48] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[03:27:48] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[03:27:53] env3_seed_0 | Germinated (depthwise, 4.8K params)
    [env3] Germinated 'env3_seed_0' (depthwise, 4.8K params)
[03:27:53] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[03:28:02] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[03:28:03] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[03:28:08] env1_seed_0 | Stage transition: TRAINING → BLENDING
[03:28:15] env2_seed_0 | Stage transition: TRAINING → BLENDING
[03:28:21] env3_seed_0 | Stage transition: TRAINING → BLENDING
[03:28:27] env0_seed_0 | Stage transition: TRAINING → BLENDING
[03:28:39] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[03:28:44] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[03:28:48] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:28:48] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[03:28:52] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[03:28:52] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:28:52] env2_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:28:52] env2_seed_0 | Fossilized (depthwise, Δacc +9.16%)
    [env2] Fossilized 'env2_seed_0' (depthwise, Δacc +9.16%)
[03:28:55] env3_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:28:55] env3_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:28:55] env3_seed_0 | Fossilized (depthwise, Δacc +5.91%)
    [env3] Fossilized 'env3_seed_0' (depthwise, Δacc +5.91%)
[03:28:58] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:28:58] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:28:58] env0_seed_0 | Fossilized (norm, Δacc +9.59%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +9.59%)
[03:29:01] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:29:01] env1_seed_0 | Fossilized (norm, Δacc +18.76%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +18.76%)
Batch 43: Episodes 172/200
  Env accuracies: ['76.2%', '75.5%', '75.0%', '73.8%']
  Avg acc: 75.1% (rolling: 75.1%)
  Avg reward: 65.4
  Actions: {'WAIT': 119, 'GERMINATE_NORM': 25, 'GERMINATE_ATTENTION': 18, 'GERMINATE_DEPTHWISE': 12, 'GERMINATE_CONV_ENHANCE': 11, 'FOSSILIZE': 115, 'CULL': 0}
  Successful: {'WAIT': 119, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 2, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 4, 'CULL': 0}
  Policy loss: -0.0102, Value loss: 20.7990, Entropy: 1.3425, Entropy coef: 0.0100
[03:31:59] env3_seed_0 | Germinated (depthwise, 4.8K params)
    [env3] Germinated 'env3_seed_0' (depthwise, 4.8K params)
[03:31:59] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[03:32:02] env0_seed_0 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_0' (depthwise, 4.8K params)
[03:32:02] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[03:32:02] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[03:32:02] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[03:32:12] env2_seed_0 | Germinated (conv_enhance, 74.0K params)
    [env2] Germinated 'env2_seed_0' (conv_enhance, 74.0K params)
[03:32:12] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[03:32:18] env3_seed_0 | Stage transition: TRAINING → BLENDING
[03:32:24] env0_seed_0 | Stage transition: TRAINING → BLENDING
[03:32:24] env1_seed_0 | Stage transition: TRAINING → BLENDING
[03:32:36] env2_seed_0 | Stage transition: TRAINING → BLENDING
[03:32:49] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[03:32:54] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[03:32:54] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[03:32:57] env3_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:33:01] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:33:01] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:33:01] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[03:33:01] env3_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:33:01] env3_seed_0 | Fossilized (depthwise, Δacc +9.42%)
    [env3] Fossilized 'env3_seed_0' (depthwise, Δacc +9.42%)
[03:33:07] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:33:07] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:33:07] env0_seed_0 | Fossilized (depthwise, Δacc +13.67%)
    [env0] Fossilized 'env0_seed_0' (depthwise, Δacc +13.67%)
[03:33:07] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:33:07] env1_seed_0 | Fossilized (norm, Δacc +10.38%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +10.38%)
[03:33:16] env2_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:33:16] env2_seed_0 | Fossilized (conv_enhance, Δacc +14.40%)
    [env2] Fossilized 'env2_seed_0' (conv_enhance, Δacc +14.40%)
[03:35:48] env2_seed_0 | Stage transition: FOSSILIZED → CULLED
[03:35:48] env2_seed_0 | Culled (conv_enhance, Δacc +14.37%)
    [env2] Culled 'env2_seed_0' (conv_enhance, Δacc +14.37%)
[03:35:51] env2_seed_1 | Germinated (attention, 2.0K params)
    [env2] Germinated 'env2_seed_1' (attention, 2.0K params)
[03:35:52] env2_seed_1 | Stage transition: GERMINATED → TRAINING
[03:36:04] env2_seed_1 | Stage transition: TRAINING → BLENDING
Batch 44: Episodes 176/200
  Env accuracies: ['74.0%', '74.5%', '71.5%', '68.9%']
  Avg acc: 72.2% (rolling: 74.8%)
  Avg reward: 50.5
  Actions: {'WAIT': 108, 'GERMINATE_NORM': 38, 'GERMINATE_ATTENTION': 14, 'GERMINATE_DEPTHWISE': 13, 'GERMINATE_CONV_ENHANCE': 9, 'FOSSILIZE': 117, 'CULL': 1}
  Successful: {'WAIT': 108, 'GERMINATE_NORM': 1, 'GERMINATE_ATTENTION': 1, 'GERMINATE_DEPTHWISE': 2, 'GERMINATE_CONV_ENHANCE': 1, 'FOSSILIZE': 4, 'CULL': 1}
  Policy loss: -0.0190, Value loss: 16.0144, Entropy: 1.2780, Entropy coef: 0.0100
[03:36:16] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[03:36:16] env2_seed_0 | Germinated (depthwise, 4.8K params)
    [env2] Germinated 'env2_seed_0' (depthwise, 4.8K params)
[03:36:16] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[03:36:16] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[03:36:20] env0_seed_0 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_0' (conv_enhance, 74.0K params)
[03:36:20] env3_seed_0 | Germinated (conv_enhance, 74.0K params)
    [env3] Germinated 'env3_seed_0' (conv_enhance, 74.0K params)
[03:36:20] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[03:36:20] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[03:36:39] env1_seed_0 | Stage transition: TRAINING → BLENDING
[03:36:45] env2_seed_0 | Stage transition: TRAINING → BLENDING
[03:36:45] env3_seed_0 | Stage transition: TRAINING → BLENDING
[03:36:58] env0_seed_0 | Stage transition: TRAINING → BLENDING
[03:37:10] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[03:37:15] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[03:37:15] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[03:37:19] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:37:19] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:37:19] env1_seed_0 | Fossilized (attention, Δacc +6.86%)
    [env1] Fossilized 'env1_seed_0' (attention, Δacc +6.86%)
[03:37:22] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[03:37:22] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:37:22] env3_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:37:22] env3_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:37:22] env3_seed_0 | Fossilized (conv_enhance, Δacc +18.20%)
    [env3] Fossilized 'env3_seed_0' (conv_enhance, Δacc +18.20%)
[03:37:28] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:37:28] env2_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:37:28] env2_seed_0 | Fossilized (depthwise, Δacc +13.61%)
    [env2] Fossilized 'env2_seed_0' (depthwise, Δacc +13.61%)
[03:37:37] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:37:37] env0_seed_0 | Fossilized (conv_enhance, Δacc +11.60%)
    [env0] Fossilized 'env0_seed_0' (conv_enhance, Δacc +11.60%)
[03:38:55] env1_seed_0 | Stage transition: FOSSILIZED → CULLED
[03:38:55] env1_seed_0 | Culled (attention, Δacc +14.35%)
    [env1] Culled 'env1_seed_0' (attention, Δacc +14.35%)
[03:38:58] env1_seed_1 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_1' (norm, 0.1K params)
[03:38:58] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[03:39:22] env1_seed_1 | Stage transition: TRAINING → BLENDING
[03:39:39] env1_seed_1 | Stage transition: BLENDING → SHADOWING
[03:39:45] env1_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[03:39:45] env1_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[03:39:45] env1_seed_1 | Fossilized (norm, Δacc +3.92%)
    [env1] Fossilized 'env1_seed_1' (norm, Δacc +3.92%)
Batch 45: Episodes 180/200
  Env accuracies: ['75.9%', '76.9%', '72.1%', '72.3%']
  Avg acc: 74.3% (rolling: 74.8%)
  Avg reward: 41.4
  Actions: {'WAIT': 117, 'GERMINATE_NORM': 29, 'GERMINATE_ATTENTION': 15, 'GERMINATE_DEPTHWISE': 16, 'GERMINATE_CONV_ENHANCE': 9, 'FOSSILIZE': 113, 'CULL': 1}
  Successful: {'WAIT': 117, 'GERMINATE_NORM': 1, 'GERMINATE_ATTENTION': 1, 'GERMINATE_DEPTHWISE': 1, 'GERMINATE_CONV_ENHANCE': 2, 'FOSSILIZE': 5, 'CULL': 1}
  Policy loss: -0.0146, Value loss: 15.9990, Entropy: 1.2684, Entropy coef: 0.0100

Blueprint Stats
  ---------------------------------------------------------------------------

Blueprint       Germ  Foss  Cull   Rate     ΔAcc    Churn
  ---------------------------------------------------------------------------

  attention         82    33    64  34.0%   +6.32%   +4.66%
  conv_enhance      92    36    61  37.1%  +13.01%   +7.20%
  depthwise        110    53    69  43.4%   +9.55%   +6.19%
  norm             146   101    62  62.0%  +10.68%   +7.21%
Seed Scoreboard (env 0):
  Fossilized: 52 (+734.3K params, +774.9% of host)
  Culled: 59
  Avg fossilize age: 13.8 epochs
  Avg cull age: 13.1 epochs
  Compute cost: 5.85x baseline
  Distribution: conv_enhance x9, depthwise x11, attention x6, norm x26
Seed Scoreboard (env 1):
  Fossilized: 57 (+731.4K params, +771.8% of host)
  Culled: 74
  Avg fossilize age: 14.6 epochs
  Avg cull age: 14.6 epochs
  Compute cost: 6.82x baseline
  Distribution: norm x30, depthwise x9, conv_enhance x9, attention x9
Seed Scoreboard (env 2):
  Fossilized: 58 (+625.8K params, +660.4% of host)
  Culled: 61
  Avg fossilize age: 14.9 epochs
  Avg cull age: 18.3 epochs
  Compute cost: 7.12x baseline
  Distribution: conv_enhance x7, attention x9, norm x24, depthwise x18
Seed Scoreboard (env 3):
  Fossilized: 56 (+907.0K params, +957.1% of host)
  Culled: 62
  Avg fossilize age: 14.3 epochs
  Avg cull age: 14.1 epochs
  Compute cost: 7.42x baseline
  Distribution: conv_enhance x11, norm x21, depthwise x15, attention x9

[03:40:40] env2_seed_0 | Germinated (conv_enhance, 74.0K params)
    [env2] Germinated 'env2_seed_0' (conv_enhance, 74.0K params)
[03:40:40] env3_seed_0 | Germinated (depthwise, 4.8K params)
    [env3] Germinated 'env3_seed_0' (depthwise, 4.8K params)
[03:40:40] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[03:40:40] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[03:40:45] env1_seed_0 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_0' (conv_enhance, 74.0K params)
[03:40:45] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[03:41:00] env2_seed_0 | Stage transition: TRAINING → BLENDING
[03:41:00] env0_seed_0 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_0' (attention, 2.0K params)
[03:41:00] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[03:41:06] env1_seed_0 | Stage transition: TRAINING → BLENDING
[03:41:06] env3_seed_0 | Stage transition: TRAINING → BLENDING
[03:41:32] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[03:41:37] env0_seed_0 | Stage transition: TRAINING → BLENDING
[03:41:37] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[03:41:37] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[03:41:40] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:41:40] env2_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:41:40] env2_seed_0 | Fossilized (conv_enhance, Δacc +12.70%)
    [env2] Fossilized 'env2_seed_0' (conv_enhance, Δacc +12.70%)
[03:41:43] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:41:43] env3_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:41:46] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:41:46] env1_seed_0 | Fossilized (conv_enhance, Δacc +17.47%)
    [env1] Fossilized 'env1_seed_0' (conv_enhance, Δacc +17.47%)
[03:41:53] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[03:41:56] env3_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:41:56] env3_seed_0 | Fossilized (depthwise, Δacc +11.05%)
    [env3] Fossilized 'env3_seed_0' (depthwise, Δacc +11.05%)
[03:41:59] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:42:05] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:42:05] env0_seed_0 | Fossilized (attention, Δacc +4.72%)
    [env0] Fossilized 'env0_seed_0' (attention, Δacc +4.72%)
Batch 46: Episodes 184/200
  Env accuracies: ['73.3%', '72.5%', '73.8%', '70.2%']
  Avg acc: 72.5% (rolling: 74.5%)
  Avg reward: 42.2
  Actions: {'WAIT': 112, 'GERMINATE_NORM': 25, 'GERMINATE_ATTENTION': 18, 'GERMINATE_DEPTHWISE': 24, 'GERMINATE_CONV_ENHANCE': 9, 'FOSSILIZE': 112, 'CULL': 0}
  Successful: {'WAIT': 112, 'GERMINATE_NORM': 0, 'GERMINATE_ATTENTION': 1, 'GERMINATE_DEPTHWISE': 1, 'GERMINATE_CONV_ENHANCE': 2, 'FOSSILIZE': 4, 'CULL': 0}
  Policy loss: -0.0157, Value loss: 12.2393, Entropy: 1.2675, Entropy coef: 0.0100
[03:44:54] env3_seed_0 | Germinated (depthwise, 4.8K params)
    [env3] Germinated 'env3_seed_0' (depthwise, 4.8K params)
[03:44:54] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[03:44:57] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[03:44:57] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[03:45:08] env3_seed_0 | Stage transition: TRAINING → BLENDING
[03:45:12] env1_seed_0 | Stage transition: TRAINING → BLENDING
[03:45:12] env0_seed_0 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_0' (attention, 2.0K params)
[03:45:12] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[03:45:32] env0_seed_0 | Stage transition: TRAINING → BLENDING
[03:45:32] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[03:45:32] env2_seed_0 | Germinated (norm, 0.1K params)
    [env2] Germinated 'env2_seed_0' (norm, 0.1K params)
[03:45:32] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[03:45:36] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[03:45:40] env3_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:45:43] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:45:43] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:45:43] env1_seed_0 | Fossilized (attention, Δacc +9.52%)
    [env1] Fossilized 'env1_seed_0' (attention, Δacc +9.52%)
[03:45:47] env2_seed_0 | Stage transition: TRAINING → BLENDING
[03:45:50] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[03:45:50] env3_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:45:50] env3_seed_0 | Fossilized (depthwise, Δacc +5.11%)
    [env3] Fossilized 'env3_seed_0' (depthwise, Δacc +5.11%)
[03:45:57] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:46:03] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[03:46:09] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:46:09] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:46:09] env0_seed_0 | Fossilized (attention, Δacc +9.07%)
    [env0] Fossilized 'env0_seed_0' (attention, Δacc +9.07%)
[03:46:15] env2_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:46:15] env2_seed_0 | Fossilized (norm, Δacc +7.29%)
    [env2] Fossilized 'env2_seed_0' (norm, Δacc +7.29%)
Batch 47: Episodes 188/200
  Env accuracies: ['75.9%', '66.3%', '77.1%', '72.8%']
  Avg acc: 73.0% (rolling: 74.2%)
  Avg reward: 61.9
  Actions: {'WAIT': 123, 'GERMINATE_NORM': 25, 'GERMINATE_ATTENTION': 22, 'GERMINATE_DEPTHWISE': 20, 'GERMINATE_CONV_ENHANCE': 9, 'FOSSILIZE': 101, 'CULL': 0}
  Successful: {'WAIT': 123, 'GERMINATE_NORM': 1, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 1, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 4, 'CULL': 0}
  Policy loss: -0.0236, Value loss: 17.4231, Entropy: 1.2900, Entropy coef: 0.0100
[03:48:52] env0_seed_0 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_0' (depthwise, 4.8K params)
[03:48:52] env1_seed_0 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_0' (depthwise, 4.8K params)
[03:48:52] env3_seed_0 | Germinated (conv_enhance, 74.0K params)
    [env3] Germinated 'env3_seed_0' (conv_enhance, 74.0K params)
[03:48:52] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[03:48:52] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[03:48:52] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[03:49:12] env0_seed_0 | Stage transition: TRAINING → BLENDING
[03:49:12] env1_seed_0 | Stage transition: TRAINING → BLENDING
[03:49:12] env3_seed_0 | Stage transition: TRAINING → BLENDING
[03:49:12] env2_seed_0 | Germinated (norm, 0.1K params)
    [env2] Germinated 'env2_seed_0' (norm, 0.1K params)
[03:49:12] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[03:49:37] env2_seed_0 | Stage transition: TRAINING → BLENDING
[03:49:43] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[03:49:43] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[03:49:43] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[03:49:50] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:49:50] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:49:50] env3_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:49:50] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:49:50] env0_seed_0 | Fossilized (depthwise, Δacc +12.62%)
    [env0] Fossilized 'env0_seed_0' (depthwise, Δacc +12.62%)
[03:49:56] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[03:49:56] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:49:56] env1_seed_0 | Fossilized (depthwise, Δacc +6.66%)
    [env1] Fossilized 'env1_seed_0' (depthwise, Δacc +6.66%)
[03:50:02] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:50:12] env3_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:50:12] env3_seed_0 | Fossilized (conv_enhance, Δacc +15.88%)
    [env3] Fossilized 'env3_seed_0' (conv_enhance, Δacc +15.88%)
[03:50:24] env2_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:50:24] env2_seed_0 | Fossilized (norm, Δacc +9.37%)
    [env2] Fossilized 'env2_seed_0' (norm, Δacc +9.37%)
[03:52:15] env1_seed_0 | Stage transition: FOSSILIZED → CULLED
[03:52:15] env1_seed_0 | Culled (depthwise, Δacc +14.27%)
    [env1] Culled 'env1_seed_0' (depthwise, Δacc +14.27%)
[03:52:18] env1_seed_1 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_1' (depthwise, 4.8K params)
[03:52:18] env2_seed_0 | Stage transition: FOSSILIZED → CULLED
[03:52:18] env2_seed_0 | Culled (norm, Δacc +13.18%)
    [env2] Culled 'env2_seed_0' (norm, Δacc +13.18%)
[03:52:18] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[03:52:21] env2_seed_1 | Germinated (attention, 2.0K params)
    [env2] Germinated 'env2_seed_1' (attention, 2.0K params)
[03:52:21] env2_seed_1 | Stage transition: GERMINATED → TRAINING
[03:52:26] env2_seed_1 | Stage transition: TRAINING → CULLED
[03:52:26] env2_seed_1 | Culled (attention, Δacc +0.00%)
    [env2] Culled 'env2_seed_1' (attention, Δacc +0.00%)
[03:52:29] env2_seed_2 | Germinated (depthwise, 4.8K params)
    [env2] Germinated 'env2_seed_2' (depthwise, 4.8K params)
[03:52:29] env2_seed_2 | Stage transition: GERMINATED → TRAINING
[03:52:53] env1_seed_1 | Stage transition: TRAINING → BLENDING
[03:53:17] env1_seed_1 | Stage transition: BLENDING → SHADOWING
Batch 48: Episodes 192/200
  Env accuracies: ['71.8%', '68.3%', '67.7%', '76.5%']
  Avg acc: 71.1% (rolling: 73.8%)
  Avg reward: 43.7
  Actions: {'WAIT': 136, 'GERMINATE_NORM': 17, 'GERMINATE_ATTENTION': 21, 'GERMINATE_DEPTHWISE': 10, 'GERMINATE_CONV_ENHANCE': 11, 'FOSSILIZE': 102, 'CULL': 3}
  Successful: {'WAIT': 136, 'GERMINATE_NORM': 1, 'GERMINATE_ATTENTION': 1, 'GERMINATE_DEPTHWISE': 4, 'GERMINATE_CONV_ENHANCE': 1, 'FOSSILIZE': 4, 'CULL': 3}
  Policy loss: -0.0105, Value loss: 23.7175, Entropy: 1.1917, Entropy coef: 0.0100
[03:53:22] env1_seed_0 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_0' (conv_enhance, 74.0K params)
[03:53:22] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[03:53:29] env0_seed_0 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_0' (depthwise, 4.8K params)
[03:53:29] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[03:53:38] env1_seed_0 | Stage transition: TRAINING → BLENDING
[03:53:38] env3_seed_0 | Germinated (conv_enhance, 74.0K params)
    [env3] Germinated 'env3_seed_0' (conv_enhance, 74.0K params)
[03:53:38] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[03:53:48] env0_seed_0 | Stage transition: TRAINING → BLENDING
[03:53:53] env2_seed_0 | Germinated (conv_enhance, 74.0K params)
    [env2] Germinated 'env2_seed_0' (conv_enhance, 74.0K params)
[03:53:54] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[03:54:00] env3_seed_0 | Stage transition: TRAINING → BLENDING
[03:54:06] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[03:54:16] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[03:54:16] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:54:16] env2_seed_0 | Stage transition: TRAINING → BLENDING
[03:54:26] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:54:26] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[03:54:26] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:54:26] env0_seed_0 | Fossilized (depthwise, Δacc +11.97%)
    [env0] Fossilized 'env0_seed_0' (depthwise, Δacc +11.97%)
[03:54:26] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:54:26] env1_seed_0 | Fossilized (conv_enhance, Δacc +15.89%)
    [env1] Fossilized 'env1_seed_0' (conv_enhance, Δacc +15.89%)
[03:54:33] env3_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:54:36] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[03:54:43] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:54:43] env2_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:54:43] env2_seed_0 | Fossilized (conv_enhance, Δacc +13.39%)
    [env2] Fossilized 'env2_seed_0' (conv_enhance, Δacc +13.39%)
[03:54:43] env3_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:54:43] env3_seed_0 | Fossilized (conv_enhance, Δacc +12.81%)
    [env3] Fossilized 'env3_seed_0' (conv_enhance, Δacc +12.81%)
Batch 49: Episodes 196/200
  Env accuracies: ['73.8%', '76.1%', '76.1%', '74.2%']
  Avg acc: 75.0% (rolling: 73.6%)
  Avg reward: 35.1
  Actions: {'WAIT': 131, 'GERMINATE_NORM': 23, 'GERMINATE_ATTENTION': 16, 'GERMINATE_DEPTHWISE': 12, 'GERMINATE_CONV_ENHANCE': 13, 'FOSSILIZE': 105, 'CULL': 0}
  Successful: {'WAIT': 131, 'GERMINATE_NORM': 0, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 1, 'GERMINATE_CONV_ENHANCE': 3, 'FOSSILIZE': 4, 'CULL': 0}
  Policy loss: -0.0057, Value loss: 15.1011, Entropy: 1.1970, Entropy coef: 0.0100
[03:57:43] env0_seed_0 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_0' (conv_enhance, 74.0K params)
[03:57:43] env1_seed_0 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_0' (conv_enhance, 74.0K params)
[03:57:43] env2_seed_0 | Germinated (conv_enhance, 74.0K params)
    [env2] Germinated 'env2_seed_0' (conv_enhance, 74.0K params)
[03:57:43] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[03:57:43] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[03:57:43] env2_seed_0 | Stage transition: GERMINATED → TRAINING
[03:57:48] env3_seed_0 | Germinated (conv_enhance, 74.0K params)
    [env3] Germinated 'env3_seed_0' (conv_enhance, 74.0K params)
[03:57:48] env3_seed_0 | Stage transition: GERMINATED → TRAINING
[03:58:07] env0_seed_0 | Stage transition: TRAINING → BLENDING
[03:58:07] env1_seed_0 | Stage transition: TRAINING → BLENDING
[03:58:07] env2_seed_0 | Stage transition: TRAINING → BLENDING
[03:58:13] env3_seed_0 | Stage transition: TRAINING → BLENDING
[03:58:39] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[03:58:39] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[03:58:39] env2_seed_0 | Stage transition: BLENDING → SHADOWING
[03:58:43] env3_seed_0 | Stage transition: BLENDING → SHADOWING
[03:58:46] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:58:46] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:58:46] env2_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:58:49] env3_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[03:58:49] env2_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:58:49] env2_seed_0 | Fossilized (conv_enhance, Δacc +14.01%)
    [env2] Fossilized 'env2_seed_0' (conv_enhance, Δacc +14.01%)
[03:58:56] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:58:56] env0_seed_0 | Fossilized (conv_enhance, Δacc +16.51%)
    [env0] Fossilized 'env0_seed_0' (conv_enhance, Δacc +16.51%)
[03:58:56] env3_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:58:56] env3_seed_0 | Fossilized (conv_enhance, Δacc +22.65%)
    [env3] Fossilized 'env3_seed_0' (conv_enhance, Δacc +22.65%)
[03:58:59] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[03:58:59] env1_seed_0 | Fossilized (conv_enhance, Δacc +21.38%)
    [env1] Fossilized 'env1_seed_0' (conv_enhance, Δacc +21.38%)
Batch 50: Episodes 200/200
  Env accuracies: ['76.0%', '73.8%', '75.6%', '75.5%']
  Avg acc: 75.2% (rolling: 73.6%)
  Avg reward: 26.3
  Actions: {'WAIT': 117, 'GERMINATE_NORM': 24, 'GERMINATE_ATTENTION': 14, 'GERMINATE_DEPTHWISE': 13, 'GERMINATE_CONV_ENHANCE': 10, 'FOSSILIZE': 122, 'CULL': 0}
  Successful: {'WAIT': 117, 'GERMINATE_NORM': 0, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 4, 'FOSSILIZE': 4, 'CULL': 0}
  Policy loss: -0.0066, Value loss: 25.3186, Entropy: 1.1171, Entropy coef: 0.0100

Blueprint Stats
  ---------------------------------------------------------------------------

Blueprint       Germ  Foss  Cull   Rate     ΔAcc    Churn
  ---------------------------------------------------------------------------

  attention         86    36    65  35.6%   +6.45%   +4.59%
  conv_enhance     102    46    61  43.0%  +13.72%   +7.20%
  depthwise        117    58    70  45.3%   +9.54%   +6.31%
  norm             148   103    63  62.0%  +10.63%   +7.30%
Seed Scoreboard (env 0):
  Fossilized: 57 (+822.0K params, +867.4% of host)
  Culled: 59
  Avg fossilize age: 13.8 epochs
  Avg cull age: 13.1 epochs
  Compute cost: 6.86x baseline
  Distribution: conv_enhance x10, depthwise x13, attention x8, norm x26
Seed Scoreboard (env 1):
  Fossilized: 62 (+960.2K params, +1013.2% of host)
  Culled: 75
  Avg fossilize age: 14.5 epochs
  Avg cull age: 15.2 epochs
  Compute cost: 7.70x baseline
  Distribution: norm x30, depthwise x10, conv_enhance x12, attention x10
Seed Scoreboard (env 2):
  Fossilized: 63 (+848.0K params, +894.9% of host)
  Culled: 63
  Avg fossilize age: 14.7 epochs
  Avg cull age: 18.6 epochs
  Compute cost: 7.61x baseline
  Distribution: conv_enhance x10, attention x9, norm x26, depthwise x18
Seed Scoreboard (env 3):
  Fossilized: 61 (+1138.5K params, +1201.4% of host)
  Culled: 62
  Avg fossilize age: 14.4 epochs
  Avg cull age: 14.1 epochs
  Compute cost: 8.03x baseline
  Distribution: conv_enhance x14, norm x21, depthwise x17, attention x9

Loaded best weights (avg_acc=75.1%)
(.venv) john@nyx:~/esper-lite$
