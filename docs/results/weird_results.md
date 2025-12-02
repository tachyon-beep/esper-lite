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
[11:54:18] env1_seed_0 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_0' (conv_enhance, 74.0K params)
[11:54:18] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[11:54:20] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[11:54:20] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[11:54:25] env1_seed_0 | Stage transition: TRAINING → BLENDING
[11:54:25] env1_seed_0 | Stage transition: BLENDING → CULLED
[11:54:25] env1_seed_0 | Culled (conv_enhance, Δacc +8.85%)
    [env1] Culled 'env1_seed_0' (conv_enhance, Δacc +8.85%)
[11:54:27] env0_seed_0 | Stage transition: TRAINING → BLENDING
[11:54:27] env0_seed_0 | Stage transition: BLENDING → CULLED
[11:54:27] env0_seed_0 | Culled (norm, Δacc +3.94%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +3.94%)
[11:54:29] env1_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_1' (conv_enhance, 74.0K params)
[11:54:29] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[11:54:30] env0_seed_1 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_1' (depthwise, 4.8K params)
[11:54:30] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[11:54:34] env1_seed_1 | Stage transition: TRAINING → CULLED
[11:54:34] env1_seed_1 | Culled (conv_enhance, Δacc -1.93%)
    [env1] Culled 'env1_seed_1' (conv_enhance, Δacc -1.93%)
[11:54:36] env1_seed_2 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_2' (attention, 2.0K params)
[11:54:36] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[11:54:42] env0_seed_1 | Stage transition: TRAINING → BLENDING
[11:54:46] env1_seed_2 | Stage transition: TRAINING → BLENDING
[11:54:47] env0_seed_1 | Stage transition: BLENDING → CULLED
[11:54:47] env0_seed_1 | Culled (depthwise, Δacc +7.44%)
    [env0] Culled 'env0_seed_1' (depthwise, Δacc +7.44%)
[11:54:52] env0_seed_2 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_2' (conv_enhance, 74.0K params)
[11:54:52] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[11:54:54] env1_seed_2 | Stage transition: BLENDING → SHADOWING
[11:54:54] env0_seed_2 | Stage transition: TRAINING → CULLED
[11:54:54] env0_seed_2 | Culled (conv_enhance, Δacc +0.00%)
    [env0] Culled 'env0_seed_2' (conv_enhance, Δacc +0.00%)
[11:54:56] env0_seed_3 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_3' (depthwise, 4.8K params)
[11:54:56] env0_seed_3 | Stage transition: GERMINATED → TRAINING
[11:54:57] env1_seed_2 | Stage transition: SHADOWING → PROBATIONARY
[11:54:57] env1_seed_2 | Stage transition: PROBATIONARY → FOSSILIZED
[11:54:57] env1_seed_2 | Fossilized (attention, Δacc +0.03%)
    [env1] Fossilized 'env1_seed_2' (attention, Δacc +0.03%)
[11:55:12] env0_seed_3 | Stage transition: TRAINING → BLENDING
[11:55:14] env0_seed_3 | Stage transition: BLENDING → CULLED
[11:55:14] env0_seed_3 | Culled (depthwise, Δacc +4.38%)
    [env0] Culled 'env0_seed_3' (depthwise, Δacc +4.38%)
[11:55:17] env0_seed_4 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_4' (norm, 0.1K params)
[11:55:17] env0_seed_4 | Stage transition: GERMINATED → TRAINING
[11:55:26] env0_seed_4 | Stage transition: TRAINING → BLENDING
[11:55:31] env0_seed_4 | Stage transition: BLENDING → CULLED
[11:55:31] env0_seed_4 | Culled (norm, Δacc +3.13%)
    [env0] Culled 'env0_seed_4' (norm, Δacc +3.13%)
[11:55:36] env0_seed_5 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_5' (norm, 0.1K params)
[11:55:36] env0_seed_5 | Stage transition: GERMINATED → TRAINING
[11:55:41] env0_seed_5 | Stage transition: TRAINING → CULLED
[11:55:41] env0_seed_5 | Culled (norm, Δacc -3.83%)
    [env0] Culled 'env0_seed_5' (norm, Δacc -3.83%)
[11:55:42] env0_seed_6 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_6' (depthwise, 4.8K params)
[11:55:43] env0_seed_6 | Stage transition: GERMINATED → TRAINING
[11:55:49] env0_seed_6 | Stage transition: TRAINING → BLENDING
[11:55:52] env0_seed_6 | Stage transition: BLENDING → CULLED
[11:55:52] env0_seed_6 | Culled (depthwise, Δacc +1.97%)
    [env0] Culled 'env0_seed_6' (depthwise, Δacc +1.97%)
[11:55:54] env0_seed_7 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_7' (attention, 2.0K params)
[11:55:54] env0_seed_7 | Stage transition: GERMINATED → TRAINING
[11:56:11] env0_seed_7 | Stage transition: TRAINING → CULLED
[11:56:11] env0_seed_7 | Culled (attention, Δacc -0.77%)
    [env0] Culled 'env0_seed_7' (attention, Δacc -0.77%)
[11:56:14] env0_seed_8 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_8' (depthwise, 4.8K params)
[11:56:14] env0_seed_8 | Stage transition: GERMINATED → TRAINING
[11:56:18] env0_seed_8 | Stage transition: TRAINING → CULLED
[11:56:18] env0_seed_8 | Culled (depthwise, Δacc +0.74%)
    [env0] Culled 'env0_seed_8' (depthwise, Δacc +0.74%)
[11:56:21] env0_seed_9 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_9' (conv_enhance, 74.0K params)
[11:56:21] env0_seed_9 | Stage transition: GERMINATED → TRAINING
Batch 1: Episodes 2/200
  Env accuracies: ['72.3%', '74.1%']
  Avg acc: 73.2% (rolling: 73.2%)
  Avg reward: 139.7
  Actions: {'WAIT': 16, 'GERMINATE_NORM': 23, 'GERMINATE_ATTENTION': 23, 'GERMINATE_DEPTHWISE': 20, 'GERMINATE_CONV_ENHANCE': 27, 'FOSSILIZE': 16, 'CULL': 25}
  Successful: {'WAIT': 16, 'GERMINATE_NORM': 3, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 4, 'GERMINATE_CONV_ENHANCE': 4, 'FOSSILIZE': 1, 'CULL': 20}
  Policy loss: -0.0112, Value loss: 306.9798, Entropy: 1.9437, Entropy coef: 0.1975
[11:56:24] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[11:56:24] env1_seed_0 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_0' (depthwise, 4.8K params)
[11:56:24] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[11:56:24] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[11:56:31] env0_seed_0 | Stage transition: TRAINING → BLENDING
[11:56:31] env1_seed_0 | Stage transition: TRAINING → BLENDING
[11:56:37] env1_seed_0 | Stage transition: BLENDING → CULLED
[11:56:37] env1_seed_0 | Culled (depthwise, Δacc +18.25%)
    [env1] Culled 'env1_seed_0' (depthwise, Δacc +18.25%)
[11:56:38] env1_seed_1 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_1' (norm, 0.1K params)
[11:56:38] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[11:56:40] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[11:56:43] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[11:56:43] env0_seed_0 | Stage transition: PROBATIONARY → CULLED
[11:56:43] env0_seed_0 | Culled (norm, Δacc +17.85%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +17.85%)
[11:56:45] env1_seed_1 | Stage transition: TRAINING → BLENDING
[11:56:45] env0_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_1' (conv_enhance, 74.0K params)
[11:56:45] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[11:56:50] env0_seed_1 | Stage transition: TRAINING → CULLED
[11:56:50] env0_seed_1 | Culled (conv_enhance, Δacc -8.53%)
    [env0] Culled 'env0_seed_1' (conv_enhance, Δacc -8.53%)
[11:56:53] env1_seed_1 | Stage transition: BLENDING → SHADOWING
[11:56:53] env0_seed_2 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_2' (norm, 0.1K params)
[11:56:53] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[11:56:57] env1_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[11:56:58] env1_seed_1 | Stage transition: PROBATIONARY → CULLED
[11:56:58] env1_seed_1 | Culled (norm, Δacc +10.33%)
    [env1] Culled 'env1_seed_1' (norm, Δacc +10.33%)
[11:57:02] env0_seed_2 | Stage transition: TRAINING → BLENDING
[11:57:02] env1_seed_2 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_2' (norm, 0.1K params)
[11:57:02] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[11:57:08] env1_seed_2 | Stage transition: TRAINING → BLENDING
[11:57:10] env0_seed_2 | Stage transition: BLENDING → SHADOWING
[11:57:13] env0_seed_2 | Stage transition: SHADOWING → PROBATIONARY
[11:57:15] env0_seed_2 | Stage transition: PROBATIONARY → CULLED
[11:57:15] env0_seed_2 | Culled (norm, Δacc +6.50%)
    [env0] Culled 'env0_seed_2' (norm, Δacc +6.50%)
[11:57:17] env1_seed_2 | Stage transition: BLENDING → SHADOWING
[11:57:17] env0_seed_3 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_3' (conv_enhance, 74.0K params)
[11:57:17] env0_seed_3 | Stage transition: GERMINATED → TRAINING
[11:57:20] env1_seed_2 | Stage transition: SHADOWING → PROBATIONARY
[11:57:22] env1_seed_2 | Stage transition: PROBATIONARY → FOSSILIZED
[11:57:22] env1_seed_2 | Fossilized (norm, Δacc +8.14%)
    [env1] Fossilized 'env1_seed_2' (norm, Δacc +8.14%)
[11:57:39] env0_seed_3 | Stage transition: TRAINING → BLENDING
[11:57:47] env0_seed_3 | Stage transition: BLENDING → SHADOWING
[11:57:50] env0_seed_3 | Stage transition: SHADOWING → PROBATIONARY
[11:57:50] env0_seed_3 | Stage transition: PROBATIONARY → CULLED
[11:57:50] env0_seed_3 | Culled (conv_enhance, Δacc -53.09%)
    [env0] Culled 'env0_seed_3' (conv_enhance, Δacc -53.09%)
[11:57:52] env0_seed_4 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_4' (norm, 0.1K params)
[11:57:52] env0_seed_4 | Stage transition: GERMINATED → TRAINING
[11:57:54] env0_seed_4 | Stage transition: TRAINING → CULLED
[11:57:54] env0_seed_4 | Culled (norm, Δacc +0.00%)
    [env0] Culled 'env0_seed_4' (norm, Δacc +0.00%)
[11:57:56] env0_seed_5 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_5' (conv_enhance, 74.0K params)
[11:57:56] env0_seed_5 | Stage transition: GERMINATED → TRAINING
[11:58:02] env0_seed_5 | Stage transition: TRAINING → BLENDING
[11:58:11] env0_seed_5 | Stage transition: BLENDING → SHADOWING
[11:58:14] env0_seed_5 | Stage transition: SHADOWING → PROBATIONARY
Batch 2: Episodes 4/200
  Env accuracies: ['65.3%', '76.0%']
  Avg acc: 70.6% (rolling: 71.9%)
  Avg reward: 120.1
  Actions: {'WAIT': 24, 'GERMINATE_NORM': 28, 'GERMINATE_ATTENTION': 17, 'GERMINATE_DEPTHWISE': 30, 'GERMINATE_CONV_ENHANCE': 22, 'FOSSILIZE': 18, 'CULL': 11}
  Successful: {'WAIT': 24, 'GERMINATE_NORM': 5, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 1, 'GERMINATE_CONV_ENHANCE': 3, 'FOSSILIZE': 1, 'CULL': 11}
  Policy loss: -0.0387, Value loss: 457.1027, Entropy: 1.9263, Entropy coef: 0.1949
[11:58:31] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[11:58:31] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[11:58:34] env0_seed_0 | Stage transition: TRAINING → CULLED
[11:58:34] env0_seed_0 | Culled (norm, Δacc +4.78%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +4.78%)
[11:58:34] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[11:58:34] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[11:58:38] env1_seed_0 | Stage transition: TRAINING → CULLED
[11:58:38] env1_seed_0 | Culled (norm, Δacc +3.56%)
    [env1] Culled 'env1_seed_0' (norm, Δacc +3.56%)
[11:58:39] env0_seed_1 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_1' (norm, 0.1K params)
[11:58:39] env1_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_1' (conv_enhance, 74.0K params)
[11:58:39] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[11:58:39] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[11:58:43] env0_seed_1 | Stage transition: TRAINING → CULLED
[11:58:43] env0_seed_1 | Culled (norm, Δacc +5.00%)
    [env0] Culled 'env0_seed_1' (norm, Δacc +5.00%)
[11:58:48] env0_seed_2 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_2' (norm, 0.1K params)
[11:58:48] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[11:58:50] env1_seed_1 | Stage transition: TRAINING → BLENDING
[11:58:50] env0_seed_2 | Stage transition: TRAINING → CULLED
[11:58:50] env0_seed_2 | Culled (norm, Δacc +0.00%)
    [env0] Culled 'env0_seed_2' (norm, Δacc +0.00%)
[11:58:57] env0_seed_3 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_3' (norm, 0.1K params)
[11:58:57] env1_seed_1 | Stage transition: BLENDING → CULLED
[11:58:57] env1_seed_1 | Culled (conv_enhance, Δacc -0.19%)
    [env1] Culled 'env1_seed_1' (conv_enhance, Δacc -0.19%)
[11:58:57] env0_seed_3 | Stage transition: GERMINATED → TRAINING
[11:58:59] env1_seed_2 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_2' (attention, 2.0K params)
[11:58:59] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[11:59:04] env0_seed_3 | Stage transition: TRAINING → BLENDING
[11:59:04] env0_seed_3 | Stage transition: BLENDING → CULLED
[11:59:04] env0_seed_3 | Culled (norm, Δacc +4.12%)
    [env0] Culled 'env0_seed_3' (norm, Δacc +4.12%)
[11:59:06] env1_seed_2 | Stage transition: TRAINING → BLENDING
[11:59:09] env0_seed_4 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_4' (attention, 2.0K params)
[11:59:09] env1_seed_2 | Stage transition: BLENDING → CULLED
[11:59:09] env1_seed_2 | Culled (attention, Δacc +6.75%)
    [env1] Culled 'env1_seed_2' (attention, Δacc +6.75%)
[11:59:09] env0_seed_4 | Stage transition: GERMINATED → TRAINING
[11:59:11] env1_seed_3 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_3' (attention, 2.0K params)
[11:59:11] env1_seed_3 | Stage transition: GERMINATED → TRAINING
[11:59:16] env0_seed_4 | Stage transition: TRAINING → BLENDING
[11:59:16] env1_seed_3 | Stage transition: TRAINING → CULLED
[11:59:16] env1_seed_3 | Culled (attention, Δacc -1.86%)
    [env1] Culled 'env1_seed_3' (attention, Δacc -1.86%)
[11:59:19] env1_seed_4 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_4' (conv_enhance, 74.0K params)
[11:59:19] env1_seed_4 | Stage transition: GERMINATED → TRAINING
[11:59:21] env1_seed_4 | Stage transition: TRAINING → CULLED
[11:59:21] env1_seed_4 | Culled (conv_enhance, Δacc +0.00%)
    [env1] Culled 'env1_seed_4' (conv_enhance, Δacc +0.00%)
[11:59:22] env1_seed_5 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_5' (conv_enhance, 74.0K params)
[11:59:22] env1_seed_5 | Stage transition: GERMINATED → TRAINING
[11:59:24] env0_seed_4 | Stage transition: BLENDING → SHADOWING
[11:59:28] env0_seed_4 | Stage transition: SHADOWING → PROBATIONARY
[11:59:31] env1_seed_5 | Stage transition: TRAINING → BLENDING
[11:59:37] env0_seed_4 | Stage transition: PROBATIONARY → FOSSILIZED
[11:59:37] env0_seed_4 | Fossilized (attention, Δacc +1.74%)
    [env0] Fossilized 'env0_seed_4' (attention, Δacc +1.74%)
[11:59:41] env1_seed_5 | Stage transition: BLENDING → SHADOWING
[11:59:44] env1_seed_5 | Stage transition: SHADOWING → PROBATIONARY
[11:59:57] env1_seed_5 | Stage transition: PROBATIONARY → CULLED
[11:59:57] env1_seed_5 | Culled (conv_enhance, Δacc +4.12%)
    [env1] Culled 'env1_seed_5' (conv_enhance, Δacc +4.12%)
[11:59:59] env1_seed_6 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_6' (norm, 0.1K params)
[11:59:59] env1_seed_6 | Stage transition: GERMINATED → TRAINING
[12:00:05] env1_seed_6 | Stage transition: TRAINING → BLENDING
[12:00:07] env1_seed_6 | Stage transition: BLENDING → CULLED
[12:00:07] env1_seed_6 | Culled (norm, Δacc +4.75%)
    [env1] Culled 'env1_seed_6' (norm, Δacc +4.75%)
[12:00:09] env1_seed_7 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_7' (depthwise, 4.8K params)
[12:00:09] env1_seed_7 | Stage transition: GERMINATED → TRAINING
[12:00:16] env1_seed_7 | Stage transition: TRAINING → BLENDING
[12:00:24] env1_seed_7 | Stage transition: BLENDING → SHADOWING
[12:00:26] env1_seed_7 | Stage transition: SHADOWING → CULLED
[12:00:26] env1_seed_7 | Culled (depthwise, Δacc +0.68%)
    [env1] Culled 'env1_seed_7' (depthwise, Δacc +0.68%)
[12:00:30] env1_seed_8 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_8' (conv_enhance, 74.0K params)
[12:00:30] env1_seed_8 | Stage transition: GERMINATED → TRAINING
[12:00:33] env1_seed_8 | Stage transition: TRAINING → CULLED
[12:00:33] env1_seed_8 | Culled (conv_enhance, Δacc -9.86%)
    [env1] Culled 'env1_seed_8' (conv_enhance, Δacc -9.86%)
[12:00:36] env1_seed_9 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_9' (attention, 2.0K params)
[12:00:36] env1_seed_9 | Stage transition: GERMINATED → TRAINING
[12:00:38] env1_seed_9 | Stage transition: TRAINING → CULLED
[12:00:38] env1_seed_9 | Culled (attention, Δacc +0.00%)
    [env1] Culled 'env1_seed_9' (attention, Δacc +0.00%)
Batch 3: Episodes 6/200
  Env accuracies: ['71.3%', '71.9%']
  Avg acc: 71.6% (rolling: 71.8%)
  Avg reward: 124.7
  Actions: {'WAIT': 20, 'GERMINATE_NORM': 21, 'GERMINATE_ATTENTION': 21, 'GERMINATE_DEPTHWISE': 22, 'GERMINATE_CONV_ENHANCE': 15, 'FOSSILIZE': 24, 'CULL': 27}
  Successful: {'WAIT': 20, 'GERMINATE_NORM': 6, 'GERMINATE_ATTENTION': 4, 'GERMINATE_DEPTHWISE': 1, 'GERMINATE_CONV_ENHANCE': 4, 'FOSSILIZE': 1, 'CULL': 19}
  Policy loss: -0.0105, Value loss: 310.4442, Entropy: 1.9306, Entropy coef: 0.1924
[12:00:43] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[12:00:43] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[12:00:46] env0_seed_0 | Stage transition: TRAINING → CULLED
[12:00:46] env0_seed_0 | Culled (norm, Δacc +4.96%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +4.96%)
[12:00:48] env0_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_1' (conv_enhance, 74.0K params)
[12:00:48] env1_seed_0 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_0' (depthwise, 4.8K params)
[12:00:48] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[12:00:48] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[12:00:51] env1_seed_0 | Stage transition: TRAINING → CULLED
[12:00:51] env1_seed_0 | Culled (depthwise, Δacc -1.28%)
    [env1] Culled 'env1_seed_0' (depthwise, Δacc -1.28%)
[12:00:53] env0_seed_1 | Stage transition: TRAINING → CULLED
[12:00:53] env0_seed_1 | Culled (conv_enhance, Δacc +4.12%)
    [env0] Culled 'env0_seed_1' (conv_enhance, Δacc +4.12%)
[12:00:53] env1_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_1' (conv_enhance, 74.0K params)
[12:00:53] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[12:00:55] env0_seed_2 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_2' (depthwise, 4.8K params)
[12:00:55] env1_seed_1 | Stage transition: TRAINING → CULLED
[12:00:55] env1_seed_1 | Culled (conv_enhance, Δacc +0.00%)
    [env1] Culled 'env1_seed_1' (conv_enhance, Δacc +0.00%)
[12:00:55] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[12:00:57] env1_seed_2 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_2' (depthwise, 4.8K params)
[12:00:57] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[12:01:02] env0_seed_2 | Stage transition: TRAINING → CULLED
[12:01:02] env0_seed_2 | Culled (depthwise, Δacc -2.02%)
    [env0] Culled 'env0_seed_2' (depthwise, Δacc -2.02%)
[12:01:07] env0_seed_3 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_3' (depthwise, 4.8K params)
[12:01:07] env1_seed_2 | Stage transition: TRAINING → CULLED
[12:01:07] env1_seed_2 | Culled (depthwise, Δacc -3.90%)
    [env1] Culled 'env1_seed_2' (depthwise, Δacc -3.90%)
[12:01:07] env0_seed_3 | Stage transition: GERMINATED → TRAINING
[12:01:09] env1_seed_3 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_3' (norm, 0.1K params)
[12:01:09] env1_seed_3 | Stage transition: GERMINATED → TRAINING
[12:01:15] env0_seed_3 | Stage transition: TRAINING → BLENDING
[12:01:17] env0_seed_3 | Stage transition: BLENDING → CULLED
[12:01:17] env0_seed_3 | Culled (depthwise, Δacc +3.12%)
    [env0] Culled 'env0_seed_3' (depthwise, Δacc +3.12%)
[12:01:20] env1_seed_3 | Stage transition: TRAINING → BLENDING
[12:01:20] env1_seed_3 | Stage transition: BLENDING → CULLED
[12:01:20] env1_seed_3 | Culled (norm, Δacc -1.20%)
    [env1] Culled 'env1_seed_3' (norm, Δacc -1.20%)
[12:01:22] env1_seed_4 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_4' (norm, 0.1K params)
[12:01:22] env1_seed_4 | Stage transition: GERMINATED → TRAINING
[12:01:24] env0_seed_4 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_4' (norm, 0.1K params)
[12:01:24] env1_seed_4 | Stage transition: TRAINING → CULLED
[12:01:24] env1_seed_4 | Culled (norm, Δacc +0.00%)
    [env1] Culled 'env1_seed_4' (norm, Δacc +0.00%)
[12:01:24] env0_seed_4 | Stage transition: GERMINATED → TRAINING
[12:01:25] env0_seed_4 | Stage transition: TRAINING → CULLED
[12:01:25] env0_seed_4 | Culled (norm, Δacc +0.00%)
    [env0] Culled 'env0_seed_4' (norm, Δacc +0.00%)
[12:01:29] env0_seed_5 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_5' (norm, 0.1K params)
[12:01:29] env1_seed_5 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_5' (norm, 0.1K params)
[12:01:29] env0_seed_5 | Stage transition: GERMINATED → TRAINING
[12:01:29] env1_seed_5 | Stage transition: GERMINATED → TRAINING
[12:01:34] env1_seed_5 | Stage transition: TRAINING → CULLED
[12:01:34] env1_seed_5 | Culled (norm, Δacc +2.01%)
    [env1] Culled 'env1_seed_5' (norm, Δacc +2.01%)
[12:01:35] env0_seed_5 | Stage transition: TRAINING → BLENDING
[12:01:37] env0_seed_5 | Stage transition: BLENDING → CULLED
[12:01:37] env0_seed_5 | Culled (norm, Δacc +9.48%)
    [env0] Culled 'env0_seed_5' (norm, Δacc +9.48%)
[12:01:37] env1_seed_6 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_6' (norm, 0.1K params)
[12:01:37] env1_seed_6 | Stage transition: GERMINATED → TRAINING
[12:01:39] env0_seed_6 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_6' (attention, 2.0K params)
[12:01:39] env0_seed_6 | Stage transition: GERMINATED → TRAINING
[12:01:40] env0_seed_6 | Stage transition: TRAINING → CULLED
[12:01:40] env0_seed_6 | Culled (attention, Δacc +0.00%)
    [env0] Culled 'env0_seed_6' (attention, Δacc +0.00%)
[12:01:40] env1_seed_6 | Stage transition: TRAINING → CULLED
[12:01:40] env1_seed_6 | Culled (norm, Δacc +0.73%)
    [env1] Culled 'env1_seed_6' (norm, Δacc +0.73%)
[12:01:42] env1_seed_7 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_7' (norm, 0.1K params)
[12:01:42] env1_seed_7 | Stage transition: GERMINATED → TRAINING
[12:01:44] env0_seed_7 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_7' (attention, 2.0K params)
[12:01:44] env0_seed_7 | Stage transition: GERMINATED → TRAINING
[12:01:45] env0_seed_7 | Stage transition: TRAINING → CULLED
[12:01:45] env0_seed_7 | Culled (attention, Δacc +0.00%)
    [env0] Culled 'env0_seed_7' (attention, Δacc +0.00%)
[12:01:45] env1_seed_7 | Stage transition: TRAINING → CULLED
[12:01:45] env1_seed_7 | Culled (norm, Δacc -2.00%)
    [env1] Culled 'env1_seed_7' (norm, Δacc -2.00%)
[12:01:47] env0_seed_8 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_8' (conv_enhance, 74.0K params)
[12:01:47] env1_seed_8 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_8' (norm, 0.1K params)
[12:01:47] env0_seed_8 | Stage transition: GERMINATED → TRAINING
[12:01:47] env1_seed_8 | Stage transition: GERMINATED → TRAINING
[12:01:52] env1_seed_8 | Stage transition: TRAINING → CULLED
[12:01:52] env1_seed_8 | Culled (norm, Δacc -2.59%)
    [env1] Culled 'env1_seed_8' (norm, Δacc -2.59%)
[12:01:59] env1_seed_9 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_9' (attention, 2.0K params)
[12:01:59] env1_seed_9 | Stage transition: GERMINATED → TRAINING
[12:02:07] env1_seed_9 | Stage transition: TRAINING → CULLED
[12:02:07] env1_seed_9 | Culled (attention, Δacc -3.67%)
    [env1] Culled 'env1_seed_9' (attention, Δacc -3.67%)
[12:02:09] env1_seed_10 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_10' (attention, 2.0K params)
[12:02:09] env1_seed_10 | Stage transition: GERMINATED → TRAINING
[12:02:14] env1_seed_10 | Stage transition: TRAINING → CULLED
[12:02:14] env1_seed_10 | Culled (attention, Δacc -4.00%)
    [env1] Culled 'env1_seed_10' (attention, Δacc -4.00%)
[12:02:15] env0_seed_8 | Stage transition: TRAINING → CULLED
[12:02:15] env0_seed_8 | Culled (conv_enhance, Δacc -1.75%)
    [env0] Culled 'env0_seed_8' (conv_enhance, Δacc -1.75%)
[12:02:17] env0_seed_9 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_9' (depthwise, 4.8K params)
[12:02:17] env1_seed_11 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_11' (conv_enhance, 74.0K params)
[12:02:17] env0_seed_9 | Stage transition: GERMINATED → TRAINING
[12:02:17] env1_seed_11 | Stage transition: GERMINATED → TRAINING
[12:02:24] env0_seed_9 | Stage transition: TRAINING → BLENDING
[12:02:24] env1_seed_11 | Stage transition: TRAINING → BLENDING
[12:02:28] env1_seed_11 | Stage transition: BLENDING → CULLED
[12:02:28] env1_seed_11 | Culled (conv_enhance, Δacc +3.41%)
    [env1] Culled 'env1_seed_11' (conv_enhance, Δacc +3.41%)
[12:02:33] env0_seed_9 | Stage transition: BLENDING → SHADOWING
[12:02:33] env1_seed_12 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_12' (norm, 0.1K params)
[12:02:33] env1_seed_12 | Stage transition: GERMINATED → TRAINING
[12:02:36] env0_seed_9 | Stage transition: SHADOWING → PROBATIONARY
[12:02:36] env0_seed_9 | Stage transition: PROBATIONARY → CULLED
[12:02:36] env0_seed_9 | Culled (depthwise, Δacc -0.80%)
    [env0] Culled 'env0_seed_9' (depthwise, Δacc -0.80%)
[12:02:41] env0_seed_10 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_10' (norm, 0.1K params)
[12:02:41] env0_seed_10 | Stage transition: GERMINATED → TRAINING
[12:02:43] env1_seed_12 | Stage transition: TRAINING → BLENDING
Batch 4: Episodes 8/200
  Env accuracies: ['71.2%', '75.3%']
  Avg acc: 73.2% (rolling: 72.2%)
  Avg reward: 124.7
  Actions: {'WAIT': 18, 'GERMINATE_NORM': 26, 'GERMINATE_ATTENTION': 15, 'GERMINATE_DEPTHWISE': 22, 'GERMINATE_CONV_ENHANCE': 19, 'FOSSILIZE': 16, 'CULL': 34}
  Successful: {'WAIT': 18, 'GERMINATE_NORM': 11, 'GERMINATE_ATTENTION': 4, 'GERMINATE_DEPTHWISE': 5, 'GERMINATE_CONV_ENHANCE': 4, 'FOSSILIZE': 0, 'CULL': 22}
  Policy loss: -0.0242, Value loss: 286.7065, Entropy: 1.9308, Entropy coef: 0.1899
[12:02:48] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[12:02:48] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[12:02:49] env0_seed_0 | Stage transition: TRAINING → CULLED
[12:02:49] env0_seed_0 | Culled (norm, Δacc +0.00%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +0.00%)
[12:02:49] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[12:02:49] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[12:02:53] env0_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_1' (conv_enhance, 74.0K params)
[12:02:53] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[12:02:56] env1_seed_0 | Stage transition: TRAINING → BLENDING
[12:03:01] env0_seed_1 | Stage transition: TRAINING → BLENDING
[12:03:01] env1_seed_0 | Stage transition: BLENDING → CULLED
[12:03:01] env1_seed_0 | Culled (norm, Δacc +14.89%)
    [env1] Culled 'env1_seed_0' (norm, Δacc +14.89%)
[12:03:06] env1_seed_1 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_1' (attention, 2.0K params)
[12:03:06] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[12:03:10] env0_seed_1 | Stage transition: BLENDING → SHADOWING
[12:03:13] env0_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[12:03:13] env1_seed_1 | Stage transition: TRAINING → BLENDING
[12:03:18] env1_seed_1 | Stage transition: BLENDING → CULLED
[12:03:18] env1_seed_1 | Culled (attention, Δacc +8.10%)
    [env1] Culled 'env1_seed_1' (attention, Δacc +8.10%)
[12:03:20] env0_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[12:03:20] env0_seed_1 | Fossilized (conv_enhance, Δacc +15.82%)
    [env0] Fossilized 'env0_seed_1' (conv_enhance, Δacc +15.82%)
[12:03:21] env1_seed_2 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_2' (norm, 0.1K params)
[12:03:21] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[12:03:28] env1_seed_2 | Stage transition: TRAINING → BLENDING
[12:03:37] env1_seed_2 | Stage transition: BLENDING → SHADOWING
[12:03:40] env1_seed_2 | Stage transition: SHADOWING → PROBATIONARY
[12:03:42] env1_seed_2 | Stage transition: PROBATIONARY → CULLED
[12:03:42] env1_seed_2 | Culled (norm, Δacc +10.16%)
    [env1] Culled 'env1_seed_2' (norm, Δacc +10.16%)
[12:03:43] env1_seed_3 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_3' (norm, 0.1K params)
[12:03:43] env1_seed_3 | Stage transition: GERMINATED → TRAINING
[12:03:50] env1_seed_3 | Stage transition: TRAINING → BLENDING
[12:03:52] env1_seed_3 | Stage transition: BLENDING → CULLED
[12:03:52] env1_seed_3 | Culled (norm, Δacc +0.58%)
    [env1] Culled 'env1_seed_3' (norm, Δacc +0.58%)
[12:03:58] env1_seed_4 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_4' (depthwise, 4.8K params)
[12:03:58] env1_seed_4 | Stage transition: GERMINATED → TRAINING
[12:04:04] env1_seed_4 | Stage transition: TRAINING → CULLED
[12:04:04] env1_seed_4 | Culled (depthwise, Δacc +1.45%)
    [env1] Culled 'env1_seed_4' (depthwise, Δacc +1.45%)
[12:04:05] env1_seed_5 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_5' (attention, 2.0K params)
[12:04:05] env1_seed_5 | Stage transition: GERMINATED → TRAINING
[12:04:19] env1_seed_5 | Stage transition: TRAINING → BLENDING
[12:04:26] env1_seed_5 | Stage transition: BLENDING → CULLED
[12:04:26] env1_seed_5 | Culled (attention, Δacc -3.82%)
    [env1] Culled 'env1_seed_5' (attention, Δacc -3.82%)
[12:04:27] env1_seed_6 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_6' (norm, 0.1K params)
[12:04:27] env1_seed_6 | Stage transition: GERMINATED → TRAINING
[12:04:31] env1_seed_6 | Stage transition: TRAINING → CULLED
[12:04:31] env1_seed_6 | Culled (norm, Δacc -3.07%)
    [env1] Culled 'env1_seed_6' (norm, Δacc -3.07%)
[12:04:32] env1_seed_7 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_7' (conv_enhance, 74.0K params)
[12:04:32] env1_seed_7 | Stage transition: GERMINATED → TRAINING
[12:04:36] env1_seed_7 | Stage transition: TRAINING → CULLED
[12:04:36] env1_seed_7 | Culled (conv_enhance, Δacc +5.55%)
    [env1] Culled 'env1_seed_7' (conv_enhance, Δacc +5.55%)
[12:04:38] env1_seed_8 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_8' (norm, 0.1K params)
[12:04:38] env1_seed_8 | Stage transition: GERMINATED → TRAINING
[12:04:44] env1_seed_8 | Stage transition: TRAINING → CULLED
[12:04:44] env1_seed_8 | Culled (norm, Δacc +1.49%)
    [env1] Culled 'env1_seed_8' (norm, Δacc +1.49%)
[12:04:48] env1_seed_9 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_9' (depthwise, 4.8K params)
[12:04:48] env1_seed_9 | Stage transition: GERMINATED → TRAINING
Batch 5: Episodes 10/200
  Env accuracies: ['71.5%', '75.9%']
  Avg acc: 73.7% (rolling: 72.5%)
  Avg reward: 63.9
  Actions: {'WAIT': 22, 'GERMINATE_NORM': 30, 'GERMINATE_ATTENTION': 10, 'GERMINATE_DEPTHWISE': 19, 'GERMINATE_CONV_ENHANCE': 27, 'FOSSILIZE': 21, 'CULL': 21}
  Successful: {'WAIT': 22, 'GERMINATE_NORM': 6, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 2, 'GERMINATE_CONV_ENHANCE': 2, 'FOSSILIZE': 1, 'CULL': 15}
  Policy loss: -0.0692, Value loss: 522.9283, Entropy: 1.9251, Entropy coef: 0.1873

Blueprint Stats
  ---------------------------------------------------------------------------

Blueprint       Germ  Foss  Cull   Rate     ΔAcc    Churn
  ---------------------------------------------------------------------------

  attention         12     2    10  16.7%   +0.89%   +0.07%
  conv_enhance      17     1    14   6.7%  +15.82%   -3.52%
  depthwise         13     0    12   0.0%   +0.00%   +2.50%
  norm              31     1    28   3.4%   +8.14%   +3.41%
Seed Scoreboard (env 0):
  Fossilized: 2 (+76.0K params, +80.2% of host)
  Culled: 29
  Avg fossilize age: 16.0 epochs
  Avg cull age: 5.7 epochs
  Compute cost: 1.50x baseline
  Distribution: attention x1, conv_enhance x1
Seed Scoreboard (env 1):
  Fossilized: 2 (+2.2K params, +2.3% of host)
  Culled: 35
  Avg fossilize age: 12.5 epochs
  Avg cull age: 5.1 epochs
  Compute cost: 1.37x baseline
  Distribution: attention x1, norm x1

[12:04:55] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[12:04:55] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[12:04:56] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[12:04:56] env1_seed_0 | Stage transition: TRAINING → CULLED
[12:04:56] env1_seed_0 | Culled (norm, Δacc +0.00%)
    [env1] Culled 'env1_seed_0' (norm, Δacc +0.00%)
[12:04:56] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[12:04:58] env0_seed_0 | Stage transition: TRAINING → CULLED
[12:04:58] env0_seed_0 | Culled (norm, Δacc +0.00%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +0.00%)
[12:04:58] env1_seed_1 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_1' (norm, 0.1K params)
[12:04:58] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[12:05:03] env0_seed_1 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_1' (depthwise, 4.8K params)
[12:05:03] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[12:05:05] env1_seed_1 | Stage transition: TRAINING → BLENDING
[12:05:06] env1_seed_1 | Stage transition: BLENDING → CULLED
[12:05:06] env1_seed_1 | Culled (norm, Δacc +12.01%)
    [env1] Culled 'env1_seed_1' (norm, Δacc +12.01%)
[12:05:08] env1_seed_2 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_2' (norm, 0.1K params)
[12:05:08] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[12:05:11] env0_seed_1 | Stage transition: TRAINING → BLENDING
[12:05:15] env1_seed_2 | Stage transition: TRAINING → BLENDING
[12:05:15] env1_seed_2 | Stage transition: BLENDING → CULLED
[12:05:15] env1_seed_2 | Culled (norm, Δacc +3.66%)
    [env1] Culled 'env1_seed_2' (norm, Δacc +3.66%)
[12:05:16] env1_seed_3 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_3' (conv_enhance, 74.0K params)
[12:05:16] env1_seed_3 | Stage transition: GERMINATED → TRAINING
[12:05:18] env1_seed_3 | Stage transition: TRAINING → CULLED
[12:05:18] env1_seed_3 | Culled (conv_enhance, Δacc +0.00%)
    [env1] Culled 'env1_seed_3' (conv_enhance, Δacc +0.00%)
[12:05:20] env0_seed_1 | Stage transition: BLENDING → SHADOWING
[12:05:20] env1_seed_4 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_4' (norm, 0.1K params)
[12:05:20] env1_seed_4 | Stage transition: GERMINATED → TRAINING
[12:05:23] env0_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[12:05:27] env1_seed_4 | Stage transition: TRAINING → BLENDING
[12:05:30] env0_seed_1 | Stage transition: PROBATIONARY → CULLED
[12:05:30] env0_seed_1 | Culled (depthwise, Δacc +0.86%)
    [env0] Culled 'env0_seed_1' (depthwise, Δacc +0.86%)
[12:05:33] env0_seed_2 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_2' (norm, 0.1K params)
[12:05:33] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[12:05:35] env1_seed_4 | Stage transition: BLENDING → SHADOWING
[12:05:35] env0_seed_2 | Stage transition: TRAINING → CULLED
[12:05:35] env0_seed_2 | Culled (norm, Δacc +0.00%)
    [env0] Culled 'env0_seed_2' (norm, Δacc +0.00%)
[12:05:37] env1_seed_4 | Stage transition: SHADOWING → CULLED
[12:05:37] env1_seed_4 | Culled (norm, Δacc +10.88%)
    [env1] Culled 'env1_seed_4' (norm, Δacc +10.88%)
[12:05:40] env1_seed_5 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_5' (norm, 0.1K params)
[12:05:40] env1_seed_5 | Stage transition: GERMINATED → TRAINING
[12:05:45] env1_seed_5 | Stage transition: TRAINING → CULLED
[12:05:45] env1_seed_5 | Culled (norm, Δacc -0.27%)
    [env1] Culled 'env1_seed_5' (norm, Δacc -0.27%)
[12:05:48] env1_seed_6 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_6' (norm, 0.1K params)
[12:05:48] env1_seed_6 | Stage transition: GERMINATED → TRAINING
[12:05:50] env1_seed_6 | Stage transition: TRAINING → CULLED
[12:05:50] env1_seed_6 | Culled (norm, Δacc +0.00%)
    [env1] Culled 'env1_seed_6' (norm, Δacc +0.00%)
[12:05:51] env1_seed_7 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_7' (norm, 0.1K params)
[12:05:52] env1_seed_7 | Stage transition: GERMINATED → TRAINING
[12:05:53] env0_seed_3 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_3' (depthwise, 4.8K params)
[12:05:53] env0_seed_3 | Stage transition: GERMINATED → TRAINING
[12:05:57] env1_seed_7 | Stage transition: TRAINING → CULLED
[12:05:57] env1_seed_7 | Culled (norm, Δacc -1.10%)
    [env1] Culled 'env1_seed_7' (norm, Δacc -1.10%)
[12:06:00] env1_seed_8 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_8' (norm, 0.1K params)
[12:06:00] env1_seed_8 | Stage transition: GERMINATED → TRAINING
[12:06:03] env0_seed_3 | Stage transition: TRAINING → BLENDING
[12:06:07] env0_seed_3 | Stage transition: BLENDING → CULLED
[12:06:07] env0_seed_3 | Culled (depthwise, Δacc +0.94%)
    [env0] Culled 'env0_seed_3' (depthwise, Δacc +0.94%)
[12:06:08] env0_seed_4 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_4' (depthwise, 4.8K params)
[12:06:08] env0_seed_4 | Stage transition: GERMINATED → TRAINING
[12:06:10] env1_seed_8 | Stage transition: TRAINING → CULLED
[12:06:10] env1_seed_8 | Culled (norm, Δacc -2.41%)
    [env1] Culled 'env1_seed_8' (norm, Δacc -2.41%)
[12:06:12] env0_seed_4 | Stage transition: TRAINING → CULLED
[12:06:12] env0_seed_4 | Culled (depthwise, Δacc -0.18%)
    [env0] Culled 'env0_seed_4' (depthwise, Δacc -0.18%)
[12:06:13] env0_seed_5 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_5' (attention, 2.0K params)
[12:06:13] env1_seed_9 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_9' (attention, 2.0K params)
[12:06:13] env0_seed_5 | Stage transition: GERMINATED → TRAINING
[12:06:13] env1_seed_9 | Stage transition: GERMINATED → TRAINING
[12:06:20] env0_seed_5 | Stage transition: TRAINING → BLENDING
[12:06:20] env1_seed_9 | Stage transition: TRAINING → CULLED
[12:06:20] env1_seed_9 | Culled (attention, Δacc -1.40%)
    [env1] Culled 'env1_seed_9' (attention, Δacc -1.40%)
[12:06:21] env1_seed_10 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_10' (attention, 2.0K params)
[12:06:22] env1_seed_10 | Stage transition: GERMINATED → TRAINING
[12:06:26] env0_seed_5 | Stage transition: BLENDING → CULLED
[12:06:26] env0_seed_5 | Culled (attention, Δacc +4.39%)
    [env0] Culled 'env0_seed_5' (attention, Δacc +4.39%)
[12:06:28] env1_seed_10 | Stage transition: TRAINING → BLENDING
[12:06:28] env0_seed_6 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_6' (norm, 0.1K params)
[12:06:28] env1_seed_10 | Stage transition: BLENDING → CULLED
[12:06:28] env1_seed_10 | Culled (attention, Δacc +1.00%)
    [env1] Culled 'env1_seed_10' (attention, Δacc +1.00%)
[12:06:28] env0_seed_6 | Stage transition: GERMINATED → TRAINING
[12:06:31] env1_seed_11 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_11' (attention, 2.0K params)
[12:06:31] env1_seed_11 | Stage transition: GERMINATED → TRAINING
[12:06:33] env1_seed_11 | Stage transition: TRAINING → CULLED
[12:06:33] env1_seed_11 | Culled (attention, Δacc +0.00%)
    [env1] Culled 'env1_seed_11' (attention, Δacc +0.00%)
[12:06:35] env0_seed_6 | Stage transition: TRAINING → BLENDING
[12:06:38] env1_seed_12 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_12' (attention, 2.0K params)
[12:06:38] env1_seed_12 | Stage transition: GERMINATED → TRAINING
[12:06:43] env0_seed_6 | Stage transition: BLENDING → SHADOWING
[12:06:43] env1_seed_12 | Stage transition: TRAINING → CULLED
[12:06:43] env1_seed_12 | Culled (attention, Δacc +2.20%)
    [env1] Culled 'env1_seed_12' (attention, Δacc +2.20%)
[12:06:46] env0_seed_6 | Stage transition: SHADOWING → PROBATIONARY
[12:06:48] env1_seed_13 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_13' (conv_enhance, 74.0K params)
[12:06:48] env1_seed_13 | Stage transition: GERMINATED → TRAINING
[12:06:50] env1_seed_13 | Stage transition: TRAINING → CULLED
[12:06:50] env1_seed_13 | Culled (conv_enhance, Δacc +0.00%)
    [env1] Culled 'env1_seed_13' (conv_enhance, Δacc +0.00%)
[12:06:51] env1_seed_14 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_14' (attention, 2.0K params)
[12:06:51] env1_seed_14 | Stage transition: GERMINATED → TRAINING
[12:06:56] env1_seed_14 | Stage transition: TRAINING → CULLED
[12:06:56] env1_seed_14 | Culled (attention, Δacc -0.17%)
    [env1] Culled 'env1_seed_14' (attention, Δacc -0.17%)
Batch 6: Episodes 12/200
  Env accuracies: ['77.4%', '72.8%']
  Avg acc: 75.1% (rolling: 72.9%)
  Avg reward: 138.2
  Actions: {'WAIT': 22, 'GERMINATE_NORM': 27, 'GERMINATE_ATTENTION': 20, 'GERMINATE_DEPTHWISE': 14, 'GERMINATE_CONV_ENHANCE': 19, 'FOSSILIZE': 24, 'CULL': 24}
  Successful: {'WAIT': 22, 'GERMINATE_NORM': 11, 'GERMINATE_ATTENTION': 6, 'GERMINATE_DEPTHWISE': 3, 'GERMINATE_CONV_ENHANCE': 2, 'FOSSILIZE': 0, 'CULL': 21}
  Policy loss: -0.0234, Value loss: 286.8647, Entropy: 1.9292, Entropy coef: 0.1848
[12:07:00] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[12:07:00] env1_seed_0 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_0' (conv_enhance, 74.0K params)
[12:07:00] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[12:07:00] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[12:07:01] env1_seed_0 | Stage transition: TRAINING → CULLED
[12:07:01] env1_seed_0 | Culled (conv_enhance, Δacc +0.00%)
    [env1] Culled 'env1_seed_0' (conv_enhance, Δacc +0.00%)
[12:07:03] env0_seed_0 | Stage transition: TRAINING → CULLED
[12:07:03] env0_seed_0 | Culled (norm, Δacc +5.59%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +5.59%)
[12:07:05] env0_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_1' (conv_enhance, 74.0K params)
[12:07:05] env1_seed_1 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_1' (norm, 0.1K params)
[12:07:05] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[12:07:05] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[12:07:11] env0_seed_1 | Stage transition: TRAINING → BLENDING
[12:07:11] env1_seed_1 | Stage transition: TRAINING → BLENDING
[12:07:20] env0_seed_1 | Stage transition: BLENDING → SHADOWING
[12:07:20] env1_seed_1 | Stage transition: BLENDING → SHADOWING
[12:07:20] env1_seed_1 | Stage transition: SHADOWING → CULLED
[12:07:20] env1_seed_1 | Culled (norm, Δacc +10.07%)
    [env1] Culled 'env1_seed_1' (norm, Δacc +10.07%)
[12:07:21] env1_seed_2 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_2' (conv_enhance, 74.0K params)
[12:07:21] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[12:07:23] env0_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[12:07:34] env1_seed_2 | Stage transition: TRAINING → BLENDING
[12:07:34] env0_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[12:07:34] env0_seed_1 | Fossilized (conv_enhance, Δacc +17.38%)
    [env0] Fossilized 'env0_seed_1' (conv_enhance, Δacc +17.38%)
[12:07:40] env1_seed_2 | Stage transition: BLENDING → CULLED
[12:07:40] env1_seed_2 | Culled (conv_enhance, Δacc +5.46%)
    [env1] Culled 'env1_seed_2' (conv_enhance, Δacc +5.46%)
[12:07:41] env1_seed_3 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_3' (depthwise, 4.8K params)
[12:07:41] env1_seed_3 | Stage transition: GERMINATED → TRAINING
[12:07:48] env1_seed_3 | Stage transition: TRAINING → CULLED
[12:07:48] env1_seed_3 | Culled (depthwise, Δacc -0.45%)
    [env1] Culled 'env1_seed_3' (depthwise, Δacc -0.45%)
[12:07:55] env1_seed_4 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_4' (attention, 2.0K params)
[12:07:55] env1_seed_4 | Stage transition: GERMINATED → TRAINING
[12:08:02] env1_seed_4 | Stage transition: TRAINING → CULLED
[12:08:02] env1_seed_4 | Culled (attention, Δacc +0.67%)
    [env1] Culled 'env1_seed_4' (attention, Δacc +0.67%)
[12:08:03] env1_seed_5 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_5' (norm, 0.1K params)
[12:08:03] env1_seed_5 | Stage transition: GERMINATED → TRAINING
[12:08:10] env1_seed_5 | Stage transition: TRAINING → BLENDING
[12:08:12] env1_seed_5 | Stage transition: BLENDING → CULLED
[12:08:12] env1_seed_5 | Culled (norm, Δacc +6.08%)
    [env1] Culled 'env1_seed_5' (norm, Δacc +6.08%)
[12:08:13] env1_seed_6 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_6' (attention, 2.0K params)
[12:08:14] env1_seed_6 | Stage transition: GERMINATED → TRAINING
[12:08:17] env1_seed_6 | Stage transition: TRAINING → CULLED
[12:08:17] env1_seed_6 | Culled (attention, Δacc -5.86%)
    [env1] Culled 'env1_seed_6' (attention, Δacc -5.86%)
[12:08:19] env1_seed_7 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_7' (attention, 2.0K params)
[12:08:19] env1_seed_7 | Stage transition: GERMINATED → TRAINING
[12:08:25] env1_seed_7 | Stage transition: TRAINING → BLENDING
[12:08:29] env1_seed_7 | Stage transition: BLENDING → CULLED
[12:08:29] env1_seed_7 | Culled (attention, Δacc +2.26%)
    [env1] Culled 'env1_seed_7' (attention, Δacc +2.26%)
[12:08:30] env1_seed_8 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_8' (attention, 2.0K params)
[12:08:30] env1_seed_8 | Stage transition: GERMINATED → TRAINING
[12:08:32] env1_seed_8 | Stage transition: TRAINING → CULLED
[12:08:32] env1_seed_8 | Culled (attention, Δacc +0.00%)
    [env1] Culled 'env1_seed_8' (attention, Δacc +0.00%)
[12:08:37] env1_seed_9 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_9' (depthwise, 4.8K params)
[12:08:37] env1_seed_9 | Stage transition: GERMINATED → TRAINING
[12:08:44] env1_seed_9 | Stage transition: TRAINING → BLENDING
[12:08:46] env1_seed_9 | Stage transition: BLENDING → CULLED
[12:08:46] env1_seed_9 | Culled (depthwise, Δacc +7.15%)
    [env1] Culled 'env1_seed_9' (depthwise, Δacc +7.15%)
[12:08:49] env1_seed_10 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_10' (attention, 2.0K params)
[12:08:49] env1_seed_10 | Stage transition: GERMINATED → TRAINING
[12:09:04] env1_seed_10 | Stage transition: TRAINING → CULLED
[12:09:04] env1_seed_10 | Culled (attention, Δacc -2.06%)
    [env1] Culled 'env1_seed_10' (attention, Δacc -2.06%)
Batch 7: Episodes 14/200
  Env accuracies: ['66.7%', '70.5%']
  Avg acc: 68.6% (rolling: 72.3%)
  Avg reward: 52.0
  Actions: {'WAIT': 29, 'GERMINATE_NORM': 22, 'GERMINATE_ATTENTION': 19, 'GERMINATE_DEPTHWISE': 25, 'GERMINATE_CONV_ENHANCE': 20, 'FOSSILIZE': 18, 'CULL': 17}
  Successful: {'WAIT': 29, 'GERMINATE_NORM': 3, 'GERMINATE_ATTENTION': 5, 'GERMINATE_DEPTHWISE': 2, 'GERMINATE_CONV_ENHANCE': 3, 'FOSSILIZE': 1, 'CULL': 16}
  Policy loss: -0.0107, Value loss: 417.1883, Entropy: 1.9175, Entropy coef: 0.1823
[12:09:08] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[12:09:08] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[12:09:08] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[12:09:08] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[12:09:09] env1_seed_0 | Stage transition: TRAINING → CULLED
[12:09:09] env1_seed_0 | Culled (norm, Δacc +0.00%)
    [env1] Culled 'env1_seed_0' (norm, Δacc +0.00%)
[12:09:11] env1_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_1' (conv_enhance, 74.0K params)
[12:09:11] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[12:09:15] env0_seed_0 | Stage transition: TRAINING → BLENDING
[12:09:18] env1_seed_1 | Stage transition: TRAINING → BLENDING
[12:09:20] env0_seed_0 | Stage transition: BLENDING → CULLED
[12:09:20] env0_seed_0 | Culled (norm, Δacc +17.09%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +17.09%)
[12:09:24] env0_seed_1 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_1' (depthwise, 4.8K params)
[12:09:24] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[12:09:25] env1_seed_1 | Stage transition: BLENDING → CULLED
[12:09:25] env1_seed_1 | Culled (conv_enhance, Δacc +10.15%)
    [env1] Culled 'env1_seed_1' (conv_enhance, Δacc +10.15%)
[12:09:27] env1_seed_2 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_2' (conv_enhance, 74.0K params)
[12:09:27] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[12:09:31] env0_seed_1 | Stage transition: TRAINING → BLENDING
[12:09:32] env1_seed_2 | Stage transition: TRAINING → CULLED
[12:09:32] env1_seed_2 | Culled (conv_enhance, Δacc -1.19%)
    [env1] Culled 'env1_seed_2' (conv_enhance, Δacc -1.19%)
[12:09:34] env1_seed_3 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_3' (depthwise, 4.8K params)
[12:09:34] env1_seed_3 | Stage transition: GERMINATED → TRAINING
[12:09:38] env1_seed_3 | Stage transition: TRAINING → CULLED
[12:09:38] env1_seed_3 | Culled (depthwise, Δacc +2.98%)
    [env1] Culled 'env1_seed_3' (depthwise, Δacc +2.98%)
[12:09:39] env0_seed_1 | Stage transition: BLENDING → SHADOWING
[12:09:41] env0_seed_1 | Stage transition: SHADOWING → CULLED
[12:09:41] env0_seed_1 | Culled (depthwise, Δacc +9.01%)
    [env0] Culled 'env0_seed_1' (depthwise, Δacc +9.01%)
[12:09:44] env1_seed_4 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_4' (depthwise, 4.8K params)
[12:09:44] env1_seed_4 | Stage transition: GERMINATED → TRAINING
[12:09:46] env0_seed_2 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_2' (norm, 0.1K params)
[12:09:46] env1_seed_4 | Stage transition: TRAINING → CULLED
[12:09:46] env1_seed_4 | Culled (depthwise, Δacc +0.00%)
    [env1] Culled 'env1_seed_4' (depthwise, Δacc +0.00%)
[12:09:46] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[12:09:48] env1_seed_5 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_5' (depthwise, 4.8K params)
[12:09:48] env1_seed_5 | Stage transition: GERMINATED → TRAINING
[12:09:55] env1_seed_5 | Stage transition: TRAINING → BLENDING
[12:09:56] env0_seed_2 | Stage transition: TRAINING → BLENDING
[12:09:56] env1_seed_5 | Stage transition: BLENDING → CULLED
[12:09:56] env1_seed_5 | Culled (depthwise, Δacc -1.44%)
    [env1] Culled 'env1_seed_5' (depthwise, Δacc -1.44%)
[12:09:58] env1_seed_6 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_6' (norm, 0.1K params)
[12:09:58] env1_seed_6 | Stage transition: GERMINATED → TRAINING
[12:10:05] env0_seed_2 | Stage transition: BLENDING → SHADOWING
[12:10:05] env1_seed_6 | Stage transition: TRAINING → BLENDING
[12:10:05] env1_seed_6 | Stage transition: BLENDING → CULLED
[12:10:05] env1_seed_6 | Culled (norm, Δacc -4.86%)
    [env1] Culled 'env1_seed_6' (norm, Δacc -4.86%)
[12:10:08] env0_seed_2 | Stage transition: SHADOWING → PROBATIONARY
[12:10:08] env0_seed_2 | Stage transition: PROBATIONARY → CULLED
[12:10:08] env0_seed_2 | Culled (norm, Δacc +0.83%)
    [env0] Culled 'env0_seed_2' (norm, Δacc +0.83%)
[12:10:10] env0_seed_3 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_3' (depthwise, 4.8K params)
[12:10:10] env0_seed_3 | Stage transition: GERMINATED → TRAINING
[12:10:13] env0_seed_3 | Stage transition: TRAINING → CULLED
[12:10:13] env0_seed_3 | Culled (depthwise, Δacc -1.94%)
    [env0] Culled 'env0_seed_3' (depthwise, Δacc -1.94%)
[12:10:14] env1_seed_7 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_7' (conv_enhance, 74.0K params)
[12:10:15] env1_seed_7 | Stage transition: GERMINATED → TRAINING
[12:10:20] env0_seed_4 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_4' (conv_enhance, 74.0K params)
[12:10:20] env0_seed_4 | Stage transition: GERMINATED → TRAINING
[12:10:22] env0_seed_4 | Stage transition: TRAINING → CULLED
[12:10:22] env0_seed_4 | Culled (conv_enhance, Δacc +0.00%)
    [env0] Culled 'env0_seed_4' (conv_enhance, Δacc +0.00%)
[12:10:27] env1_seed_7 | Stage transition: TRAINING → BLENDING
[12:10:31] env0_seed_5 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_5' (depthwise, 4.8K params)
[12:10:31] env0_seed_5 | Stage transition: GERMINATED → TRAINING
[12:10:36] env1_seed_7 | Stage transition: BLENDING → SHADOWING
[12:10:36] env0_seed_5 | Stage transition: TRAINING → CULLED
[12:10:36] env0_seed_5 | Culled (depthwise, Δacc -0.69%)
    [env0] Culled 'env0_seed_5' (depthwise, Δacc -0.69%)
[12:10:38] env0_seed_6 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_6' (depthwise, 4.8K params)
[12:10:38] env1_seed_7 | Stage transition: SHADOWING → CULLED
[12:10:38] env1_seed_7 | Culled (conv_enhance, Δacc -9.47%)
    [env1] Culled 'env1_seed_7' (conv_enhance, Δacc -9.47%)
[12:10:38] env0_seed_6 | Stage transition: GERMINATED → TRAINING
[12:10:40] env1_seed_8 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_8' (depthwise, 4.8K params)
[12:10:40] env1_seed_8 | Stage transition: GERMINATED → TRAINING
[12:10:42] env1_seed_8 | Stage transition: TRAINING → CULLED
[12:10:42] env1_seed_8 | Culled (depthwise, Δacc +0.00%)
    [env1] Culled 'env1_seed_8' (depthwise, Δacc +0.00%)
[12:10:45] env0_seed_6 | Stage transition: TRAINING → BLENDING
[12:10:48] env1_seed_9 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_9' (norm, 0.1K params)
[12:10:48] env1_seed_9 | Stage transition: GERMINATED → TRAINING
[12:10:50] env0_seed_6 | Stage transition: BLENDING → CULLED
[12:10:50] env0_seed_6 | Culled (depthwise, Δacc +3.27%)
    [env0] Culled 'env0_seed_6' (depthwise, Δacc +3.27%)
[12:10:51] env0_seed_7 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_7' (norm, 0.1K params)
[12:10:51] env0_seed_7 | Stage transition: GERMINATED → TRAINING
[12:11:03] env1_seed_9 | Stage transition: TRAINING → CULLED
[12:11:03] env1_seed_9 | Culled (norm, Δacc -2.78%)
    [env1] Culled 'env1_seed_9' (norm, Δacc -2.78%)
[12:11:05] env0_seed_7 | Stage transition: TRAINING → BLENDING
[12:11:05] env1_seed_10 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_10' (norm, 0.1K params)
[12:11:05] env1_seed_10 | Stage transition: GERMINATED → TRAINING
[12:11:10] env0_seed_7 | Stage transition: BLENDING → CULLED
[12:11:10] env0_seed_7 | Culled (norm, Δacc +0.69%)
    [env0] Culled 'env0_seed_7' (norm, Δacc +0.69%)
[12:11:11] env1_seed_10 | Stage transition: TRAINING → BLENDING
[12:11:13] env0_seed_8 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_8' (depthwise, 4.8K params)
[12:11:13] env0_seed_8 | Stage transition: GERMINATED → TRAINING
Batch 8: Episodes 16/200
  Env accuracies: ['69.1%', '74.7%']
  Avg acc: 71.9% (rolling: 72.2%)
  Avg reward: 126.8
  Actions: {'WAIT': 19, 'GERMINATE_NORM': 25, 'GERMINATE_ATTENTION': 14, 'GERMINATE_DEPTHWISE': 23, 'GERMINATE_CONV_ENHANCE': 24, 'FOSSILIZE': 18, 'CULL': 27}
  Successful: {'WAIT': 19, 'GERMINATE_NORM': 7, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 9, 'GERMINATE_CONV_ENHANCE': 4, 'FOSSILIZE': 0, 'CULL': 18}
  Policy loss: -0.0138, Value loss: 253.4950, Entropy: 1.9058, Entropy coef: 0.1797
[12:11:17] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[12:11:17] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[12:11:18] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[12:11:18] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[12:11:22] env0_seed_0 | Stage transition: TRAINING → CULLED
[12:11:22] env0_seed_0 | Culled (norm, Δacc +3.37%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +3.37%)
[12:11:23] env1_seed_0 | Stage transition: TRAINING → BLENDING
[12:11:23] env0_seed_1 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_1' (norm, 0.1K params)
[12:11:23] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[12:11:25] env0_seed_1 | Stage transition: TRAINING → CULLED
[12:11:25] env0_seed_1 | Culled (norm, Δacc +0.00%)
    [env0] Culled 'env0_seed_1' (norm, Δacc +0.00%)
[12:11:27] env0_seed_2 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_2' (conv_enhance, 74.0K params)
[12:11:27] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[12:11:30] env0_seed_2 | Stage transition: TRAINING → CULLED
[12:11:30] env0_seed_2 | Culled (conv_enhance, Δacc +1.70%)
    [env0] Culled 'env0_seed_2' (conv_enhance, Δacc +1.70%)
[12:11:32] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[12:11:32] env0_seed_3 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_3' (attention, 2.0K params)
[12:11:32] env0_seed_3 | Stage transition: GERMINATED → TRAINING
[12:11:35] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[12:11:35] env1_seed_0 | Stage transition: PROBATIONARY → CULLED
[12:11:35] env1_seed_0 | Culled (norm, Δacc +13.75%)
    [env1] Culled 'env1_seed_0' (norm, Δacc +13.75%)
[12:11:37] env0_seed_3 | Stage transition: TRAINING → CULLED
[12:11:37] env0_seed_3 | Culled (attention, Δacc +0.83%)
    [env0] Culled 'env0_seed_3' (attention, Δacc +0.83%)
[12:11:37] env1_seed_1 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_1' (depthwise, 4.8K params)
[12:11:37] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[12:11:38] env0_seed_4 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_4' (attention, 2.0K params)
[12:11:38] env0_seed_4 | Stage transition: GERMINATED → TRAINING
[12:11:42] env1_seed_1 | Stage transition: TRAINING → CULLED
[12:11:42] env1_seed_1 | Culled (depthwise, Δacc -10.44%)
    [env1] Culled 'env1_seed_1' (depthwise, Δacc -10.44%)
[12:11:45] env1_seed_2 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_2' (attention, 2.0K params)
[12:11:45] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[12:11:47] env0_seed_4 | Stage transition: TRAINING → BLENDING
[12:11:53] env1_seed_2 | Stage transition: TRAINING → BLENDING
[12:11:53] env1_seed_2 | Stage transition: BLENDING → CULLED
[12:11:53] env1_seed_2 | Culled (attention, Δacc -2.12%)
    [env1] Culled 'env1_seed_2' (attention, Δacc -2.12%)
[12:11:55] env0_seed_4 | Stage transition: BLENDING → SHADOWING
[12:11:55] env1_seed_3 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_3' (conv_enhance, 74.0K params)
[12:11:55] env1_seed_3 | Stage transition: GERMINATED → TRAINING
[12:11:59] env0_seed_4 | Stage transition: SHADOWING → PROBATIONARY
[12:11:59] env1_seed_3 | Stage transition: TRAINING → CULLED
[12:11:59] env1_seed_3 | Culled (conv_enhance, Δacc -0.70%)
    [env1] Culled 'env1_seed_3' (conv_enhance, Δacc -0.70%)
[12:12:00] env1_seed_4 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_4' (attention, 2.0K params)
[12:12:00] env1_seed_4 | Stage transition: GERMINATED → TRAINING
[12:12:07] env0_seed_4 | Stage transition: PROBATIONARY → CULLED
[12:12:07] env0_seed_4 | Culled (attention, Δacc -0.15%)
    [env0] Culled 'env0_seed_4' (attention, Δacc -0.15%)
[12:12:09] env0_seed_5 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_5' (norm, 0.1K params)
[12:12:09] env0_seed_5 | Stage transition: GERMINATED → TRAINING
[12:12:10] env0_seed_5 | Stage transition: TRAINING → CULLED
[12:12:10] env0_seed_5 | Culled (norm, Δacc +0.00%)
    [env0] Culled 'env0_seed_5' (norm, Δacc +0.00%)
[12:12:13] env0_seed_6 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_6' (conv_enhance, 74.0K params)
[12:12:13] env1_seed_4 | Stage transition: TRAINING → CULLED
[12:12:13] env1_seed_4 | Culled (attention, Δacc -1.82%)
    [env1] Culled 'env1_seed_4' (attention, Δacc -1.82%)
[12:12:14] env0_seed_6 | Stage transition: GERMINATED → TRAINING
[12:12:15] env1_seed_5 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_5' (attention, 2.0K params)
[12:12:15] env1_seed_5 | Stage transition: GERMINATED → TRAINING
[12:12:20] env0_seed_6 | Stage transition: TRAINING → BLENDING
[12:12:20] env1_seed_5 | Stage transition: TRAINING → CULLED
[12:12:20] env1_seed_5 | Culled (attention, Δacc -0.47%)
    [env1] Culled 'env1_seed_5' (attention, Δacc -0.47%)
[12:12:22] env1_seed_6 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_6' (attention, 2.0K params)
[12:12:22] env1_seed_6 | Stage transition: GERMINATED → TRAINING
[12:12:23] env1_seed_6 | Stage transition: TRAINING → CULLED
[12:12:23] env1_seed_6 | Culled (attention, Δacc +0.00%)
    [env1] Culled 'env1_seed_6' (attention, Δacc +0.00%)
[12:12:25] env1_seed_7 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_7' (norm, 0.1K params)
[12:12:25] env1_seed_7 | Stage transition: GERMINATED → TRAINING
[12:12:27] env0_seed_6 | Stage transition: BLENDING → CULLED
[12:12:27] env0_seed_6 | Culled (conv_enhance, Δacc -2.08%)
    [env0] Culled 'env0_seed_6' (conv_enhance, Δacc -2.08%)
[12:12:29] env0_seed_7 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_7' (conv_enhance, 74.0K params)
[12:12:29] env0_seed_7 | Stage transition: GERMINATED → TRAINING
[12:12:30] env1_seed_7 | Stage transition: TRAINING → CULLED
[12:12:30] env1_seed_7 | Culled (norm, Δacc +2.40%)
    [env1] Culled 'env1_seed_7' (norm, Δacc +2.40%)
[12:12:32] env1_seed_8 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_8' (conv_enhance, 74.0K params)
[12:12:32] env1_seed_8 | Stage transition: GERMINATED → TRAINING
[12:12:34] env0_seed_7 | Stage transition: TRAINING → CULLED
[12:12:34] env0_seed_7 | Culled (conv_enhance, Δacc -0.76%)
    [env0] Culled 'env0_seed_7' (conv_enhance, Δacc -0.76%)
[12:12:35] env0_seed_8 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_8' (conv_enhance, 74.0K params)
[12:12:36] env0_seed_8 | Stage transition: GERMINATED → TRAINING
[12:12:44] env0_seed_8 | Stage transition: TRAINING → BLENDING
[12:12:44] env1_seed_8 | Stage transition: TRAINING → BLENDING
[12:12:44] env1_seed_8 | Stage transition: BLENDING → CULLED
[12:12:44] env1_seed_8 | Culled (conv_enhance, Δacc -3.08%)
    [env1] Culled 'env1_seed_8' (conv_enhance, Δacc -3.08%)
[12:12:46] env1_seed_9 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_9' (conv_enhance, 74.0K params)
[12:12:46] env1_seed_9 | Stage transition: GERMINATED → TRAINING
[12:12:48] env1_seed_9 | Stage transition: TRAINING → CULLED
[12:12:48] env1_seed_9 | Culled (conv_enhance, Δacc +0.00%)
    [env1] Culled 'env1_seed_9' (conv_enhance, Δacc +0.00%)
[12:12:50] env1_seed_10 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_10' (attention, 2.0K params)
[12:12:50] env1_seed_10 | Stage transition: GERMINATED → TRAINING
[12:12:53] env0_seed_8 | Stage transition: BLENDING → SHADOWING
[12:12:55] env0_seed_8 | Stage transition: SHADOWING → CULLED
[12:12:55] env0_seed_8 | Culled (conv_enhance, Δacc -0.88%)
    [env0] Culled 'env0_seed_8' (conv_enhance, Δacc -0.88%)
[12:12:55] env1_seed_10 | Stage transition: TRAINING → CULLED
[12:12:55] env1_seed_10 | Culled (attention, Δacc -0.78%)
    [env1] Culled 'env1_seed_10' (attention, Δacc -0.78%)
[12:12:56] env0_seed_9 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_9' (attention, 2.0K params)
[12:12:56] env1_seed_11 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_11' (conv_enhance, 74.0K params)
[12:12:56] env0_seed_9 | Stage transition: GERMINATED → TRAINING
[12:12:56] env1_seed_11 | Stage transition: GERMINATED → TRAINING
[12:13:04] env1_seed_11 | Stage transition: TRAINING → BLENDING
[12:13:04] env0_seed_9 | Stage transition: TRAINING → CULLED
[12:13:04] env0_seed_9 | Culled (attention, Δacc -0.23%)
    [env0] Culled 'env0_seed_9' (attention, Δacc -0.23%)
[12:13:07] env0_seed_10 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_10' (norm, 0.1K params)
[12:13:07] env0_seed_10 | Stage transition: GERMINATED → TRAINING
[12:13:13] env1_seed_11 | Stage transition: BLENDING → SHADOWING
[12:13:13] env1_seed_11 | Stage transition: SHADOWING → CULLED
[12:13:13] env1_seed_11 | Culled (conv_enhance, Δacc -9.32%)
    [env1] Culled 'env1_seed_11' (conv_enhance, Δacc -9.32%)
[12:13:16] env0_seed_10 | Stage transition: TRAINING → BLENDING
[12:13:21] env1_seed_12 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_12' (norm, 0.1K params)
[12:13:21] env1_seed_12 | Stage transition: GERMINATED → TRAINING
Batch 9: Episodes 18/200
  Env accuracies: ['75.4%', '70.2%']
  Avg acc: 72.8% (rolling: 72.3%)
  Avg reward: 117.0
  Actions: {'WAIT': 18, 'GERMINATE_NORM': 21, 'GERMINATE_ATTENTION': 30, 'GERMINATE_DEPTHWISE': 19, 'GERMINATE_CONV_ENHANCE': 24, 'FOSSILIZE': 15, 'CULL': 23}
  Successful: {'WAIT': 18, 'GERMINATE_NORM': 7, 'GERMINATE_ATTENTION': 8, 'GERMINATE_DEPTHWISE': 1, 'GERMINATE_CONV_ENHANCE': 8, 'FOSSILIZE': 0, 'CULL': 22}
  Policy loss: -0.0243, Value loss: 281.1423, Entropy: 1.9025, Entropy coef: 0.1772
[12:13:24] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[12:13:24] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[12:13:24] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[12:13:24] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[12:13:29] env1_seed_0 | Stage transition: TRAINING → CULLED
[12:13:29] env1_seed_0 | Culled (norm, Δacc +7.80%)
    [env1] Culled 'env1_seed_0' (norm, Δacc +7.80%)
[12:13:31] env0_seed_0 | Stage transition: TRAINING → BLENDING
[12:13:31] env0_seed_0 | Stage transition: BLENDING → CULLED
[12:13:31] env0_seed_0 | Culled (norm, Δacc -2.90%)
    [env0] Culled 'env0_seed_0' (norm, Δacc -2.90%)
[12:13:31] env1_seed_1 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_1' (norm, 0.1K params)
[12:13:31] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[12:13:33] env0_seed_1 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_1' (norm, 0.1K params)
[12:13:33] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[12:13:34] env0_seed_1 | Stage transition: TRAINING → CULLED
[12:13:34] env0_seed_1 | Culled (norm, Δacc +0.00%)
    [env0] Culled 'env0_seed_1' (norm, Δacc +0.00%)
[12:13:36] env0_seed_2 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_2' (norm, 0.1K params)
[12:13:36] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[12:13:39] env0_seed_2 | Stage transition: TRAINING → CULLED
[12:13:39] env0_seed_2 | Culled (norm, Δacc -0.59%)
    [env0] Culled 'env0_seed_2' (norm, Δacc -0.59%)
[12:13:39] env1_seed_1 | Stage transition: TRAINING → CULLED
[12:13:39] env1_seed_1 | Culled (norm, Δacc +4.99%)
    [env1] Culled 'env1_seed_1' (norm, Δacc +4.99%)
[12:13:41] env1_seed_2 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_2' (conv_enhance, 74.0K params)
[12:13:41] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[12:13:43] env0_seed_3 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_3' (attention, 2.0K params)
[12:13:43] env0_seed_3 | Stage transition: GERMINATED → TRAINING
[12:13:52] env0_seed_3 | Stage transition: TRAINING → BLENDING
[12:13:52] env1_seed_2 | Stage transition: TRAINING → CULLED
[12:13:52] env1_seed_2 | Culled (conv_enhance, Δacc -3.49%)
    [env1] Culled 'env1_seed_2' (conv_enhance, Δacc -3.49%)
[12:13:53] env1_seed_3 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_3' (attention, 2.0K params)
[12:13:53] env1_seed_3 | Stage transition: GERMINATED → TRAINING
[12:14:00] env0_seed_3 | Stage transition: BLENDING → SHADOWING
[12:14:02] env0_seed_3 | Stage transition: SHADOWING → CULLED
[12:14:02] env0_seed_3 | Culled (attention, Δacc +4.00%)
    [env0] Culled 'env0_seed_3' (attention, Δacc +4.00%)
[12:14:03] env0_seed_4 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_4' (norm, 0.1K params)
[12:14:03] env0_seed_4 | Stage transition: GERMINATED → TRAINING
[12:14:07] env1_seed_3 | Stage transition: TRAINING → BLENDING
[12:14:10] env0_seed_4 | Stage transition: TRAINING → BLENDING
[12:14:15] env1_seed_3 | Stage transition: BLENDING → SHADOWING
[12:14:18] env0_seed_4 | Stage transition: BLENDING → SHADOWING
[12:14:18] env1_seed_3 | Stage transition: SHADOWING → PROBATIONARY
[12:14:20] env0_seed_4 | Stage transition: SHADOWING → CULLED
[12:14:20] env0_seed_4 | Culled (norm, Δacc +3.26%)
    [env0] Culled 'env0_seed_4' (norm, Δacc +3.26%)
[12:14:23] env0_seed_5 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_5' (norm, 0.1K params)
[12:14:23] env1_seed_3 | Stage transition: PROBATIONARY → CULLED
[12:14:23] env1_seed_3 | Culled (attention, Δacc +4.03%)
    [env1] Culled 'env1_seed_3' (attention, Δacc +4.03%)
[12:14:23] env0_seed_5 | Stage transition: GERMINATED → TRAINING
[12:14:25] env0_seed_5 | Stage transition: TRAINING → CULLED
[12:14:25] env0_seed_5 | Culled (norm, Δacc +0.00%)
    [env0] Culled 'env0_seed_5' (norm, Δacc +0.00%)
[12:14:25] env1_seed_4 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_4' (attention, 2.0K params)
[12:14:25] env1_seed_4 | Stage transition: GERMINATED → TRAINING
[12:14:28] env0_seed_6 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_6' (conv_enhance, 74.0K params)
[12:14:28] env1_seed_4 | Stage transition: TRAINING → CULLED
[12:14:28] env1_seed_4 | Culled (attention, Δacc +0.94%)
    [env1] Culled 'env1_seed_4' (attention, Δacc +0.94%)
[12:14:28] env0_seed_6 | Stage transition: GERMINATED → TRAINING
[12:14:30] env1_seed_5 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_5' (depthwise, 4.8K params)
[12:14:30] env1_seed_5 | Stage transition: GERMINATED → TRAINING
[12:14:35] env0_seed_6 | Stage transition: TRAINING → BLENDING
[12:14:35] env0_seed_6 | Stage transition: BLENDING → CULLED
[12:14:35] env0_seed_6 | Culled (conv_enhance, Δacc -1.92%)
    [env0] Culled 'env0_seed_6' (conv_enhance, Δacc -1.92%)
[12:14:37] env0_seed_7 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_7' (depthwise, 4.8K params)
[12:14:37] env0_seed_7 | Stage transition: GERMINATED → TRAINING
[12:14:39] env0_seed_7 | Stage transition: TRAINING → CULLED
[12:14:39] env0_seed_7 | Culled (depthwise, Δacc +0.00%)
    [env0] Culled 'env0_seed_7' (depthwise, Δacc +0.00%)
[12:14:41] env1_seed_5 | Stage transition: TRAINING → BLENDING
[12:14:44] env0_seed_8 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_8' (attention, 2.0K params)
[12:14:44] env0_seed_8 | Stage transition: GERMINATED → TRAINING
[12:14:49] env1_seed_5 | Stage transition: BLENDING → SHADOWING
[12:14:51] env0_seed_8 | Stage transition: TRAINING → BLENDING
[12:14:53] env1_seed_5 | Stage transition: SHADOWING → PROBATIONARY
[12:15:00] env0_seed_8 | Stage transition: BLENDING → SHADOWING
[12:15:03] env0_seed_8 | Stage transition: SHADOWING → PROBATIONARY
[12:15:03] env0_seed_8 | Stage transition: PROBATIONARY → FOSSILIZED
[12:15:03] env0_seed_8 | Fossilized (attention, Δacc +2.92%)
    [env0] Fossilized 'env0_seed_8' (attention, Δacc +2.92%)
[12:15:23] env1_seed_5 | Stage transition: PROBATIONARY → CULLED
[12:15:23] env1_seed_5 | Culled (depthwise, Δacc -4.87%)
    [env1] Culled 'env1_seed_5' (depthwise, Δacc -4.87%)
[12:15:26] env1_seed_6 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_6' (depthwise, 4.8K params)
[12:15:26] env1_seed_6 | Stage transition: GERMINATED → TRAINING
[12:15:31] env1_seed_6 | Stage transition: TRAINING → CULLED
[12:15:31] env1_seed_6 | Culled (depthwise, Δacc -3.38%)
    [env1] Culled 'env1_seed_6' (depthwise, Δacc -3.38%)
Batch 10: Episodes 20/200
  Env accuracies: ['68.3%', '72.6%']
  Avg acc: 70.5% (rolling: 72.1%)
  Avg reward: 129.0
  Actions: {'WAIT': 17, 'GERMINATE_NORM': 30, 'GERMINATE_ATTENTION': 24, 'GERMINATE_DEPTHWISE': 25, 'GERMINATE_CONV_ENHANCE': 16, 'FOSSILIZE': 17, 'CULL': 21}
  Successful: {'WAIT': 17, 'GERMINATE_NORM': 7, 'GERMINATE_ATTENTION': 4, 'GERMINATE_DEPTHWISE': 3, 'GERMINATE_CONV_ENHANCE': 2, 'FOSSILIZE': 1, 'CULL': 17}
  Policy loss: -0.0120, Value loss: 241.6056, Entropy: 1.8754, Entropy coef: 0.1747

Blueprint Stats
  ---------------------------------------------------------------------------

Blueprint       Germ  Foss  Cull   Rate     ΔAcc    Churn
  ---------------------------------------------------------------------------

  attention         35     3    32   8.6%   +1.56%   +0.19%
  conv_enhance      36     2    32   5.9%  +16.60%   -2.03%
  depthwise         31     0    29   0.0%   +0.00%   +1.06%
  norm              66     1    59   1.7%   +8.14%   +3.10%
Seed Scoreboard (env 0):
  Fossilized: 4 (+152.1K params, +160.5% of host)
  Culled: 62
  Avg fossilize age: 15.0 epochs
  Avg cull age: 5.5 epochs
  Compute cost: 2.00x baseline
  Distribution: attention x2, conv_enhance x2
Seed Scoreboard (env 1):
  Fossilized: 2 (+2.2K params, +2.3% of host)
  Culled: 90
  Avg fossilize age: 12.5 epochs
  Avg cull age: 5.1 epochs
  Compute cost: 1.37x baseline
  Distribution: attention x1, norm x1

[12:15:33] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[12:15:33] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[12:15:33] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[12:15:33] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[12:15:40] env0_seed_0 | Stage transition: TRAINING → BLENDING
[12:15:40] env1_seed_0 | Stage transition: TRAINING → BLENDING
[12:15:45] env0_seed_0 | Stage transition: BLENDING → CULLED
[12:15:45] env0_seed_0 | Culled (norm, Δacc +13.81%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +13.81%)
[12:15:48] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[12:15:48] env0_seed_1 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_1' (norm, 0.1K params)
[12:15:48] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[12:15:52] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[12:15:52] env0_seed_1 | Stage transition: TRAINING → CULLED
[12:15:52] env0_seed_1 | Culled (norm, Δacc -1.53%)
    [env0] Culled 'env0_seed_1' (norm, Δacc -1.53%)
[12:15:52] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[12:15:52] env1_seed_0 | Fossilized (norm, Δacc +17.20%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +17.20%)
[12:15:53] env0_seed_2 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_2' (norm, 0.1K params)
[12:15:53] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[12:15:57] env0_seed_2 | Stage transition: TRAINING → CULLED
[12:15:57] env0_seed_2 | Culled (norm, Δacc +2.79%)
    [env0] Culled 'env0_seed_2' (norm, Δacc +2.79%)
[12:15:58] env0_seed_3 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_3' (depthwise, 4.8K params)
[12:15:58] env0_seed_3 | Stage transition: GERMINATED → TRAINING
[12:16:00] env0_seed_3 | Stage transition: TRAINING → CULLED
[12:16:00] env0_seed_3 | Culled (depthwise, Δacc +0.00%)
    [env0] Culled 'env0_seed_3' (depthwise, Δacc +0.00%)
[12:16:02] env0_seed_4 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_4' (norm, 0.1K params)
[12:16:02] env0_seed_4 | Stage transition: GERMINATED → TRAINING
[12:16:08] env0_seed_4 | Stage transition: TRAINING → BLENDING
[12:16:12] env0_seed_4 | Stage transition: BLENDING → CULLED
[12:16:12] env0_seed_4 | Culled (norm, Δacc +9.98%)
    [env0] Culled 'env0_seed_4' (norm, Δacc +9.98%)
[12:16:13] env0_seed_5 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_5' (depthwise, 4.8K params)
[12:16:13] env0_seed_5 | Stage transition: GERMINATED → TRAINING
[12:16:22] env0_seed_5 | Stage transition: TRAINING → BLENDING
[12:16:30] env0_seed_5 | Stage transition: BLENDING → SHADOWING
[12:16:32] env0_seed_5 | Stage transition: SHADOWING → CULLED
[12:16:32] env0_seed_5 | Culled (depthwise, Δacc -0.89%)
    [env0] Culled 'env0_seed_5' (depthwise, Δacc -0.89%)
[12:16:33] env0_seed_6 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_6' (norm, 0.1K params)
[12:16:34] env0_seed_6 | Stage transition: GERMINATED → TRAINING
[12:16:35] env0_seed_6 | Stage transition: TRAINING → CULLED
[12:16:35] env0_seed_6 | Culled (norm, Δacc +0.00%)
    [env0] Culled 'env0_seed_6' (norm, Δacc +0.00%)
[12:16:37] env0_seed_7 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_7' (norm, 0.1K params)
[12:16:37] env0_seed_7 | Stage transition: GERMINATED → TRAINING
[12:16:43] env0_seed_7 | Stage transition: TRAINING → BLENDING
[12:16:47] env0_seed_7 | Stage transition: BLENDING → CULLED
[12:16:47] env0_seed_7 | Culled (norm, Δacc +5.51%)
    [env0] Culled 'env0_seed_7' (norm, Δacc +5.51%)
[12:16:50] env0_seed_8 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_8' (norm, 0.1K params)
[12:16:50] env0_seed_8 | Stage transition: GERMINATED → TRAINING
[12:16:57] env0_seed_8 | Stage transition: TRAINING → BLENDING
[12:17:05] env0_seed_8 | Stage transition: BLENDING → SHADOWING
[12:17:09] env0_seed_8 | Stage transition: SHADOWING → PROBATIONARY
[12:17:09] env0_seed_8 | Stage transition: PROBATIONARY → CULLED
[12:17:09] env0_seed_8 | Culled (norm, Δacc +6.11%)
    [env0] Culled 'env0_seed_8' (norm, Δacc +6.11%)
[12:17:10] env0_seed_9 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_9' (attention, 2.0K params)
[12:17:10] env0_seed_9 | Stage transition: GERMINATED → TRAINING
[12:17:17] env0_seed_9 | Stage transition: TRAINING → BLENDING
[12:17:24] env0_seed_9 | Stage transition: BLENDING → CULLED
[12:17:24] env0_seed_9 | Culled (attention, Δacc +4.21%)
    [env0] Culled 'env0_seed_9' (attention, Δacc +4.21%)
[12:17:25] env0_seed_10 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_10' (attention, 2.0K params)
[12:17:25] env0_seed_10 | Stage transition: GERMINATED → TRAINING
[12:17:35] env0_seed_10 | Stage transition: TRAINING → BLENDING
Batch 11: Episodes 22/200
  Env accuracies: ['63.6%', '75.2%']
  Avg acc: 69.4% (rolling: 71.7%)
  Avg reward: 138.9
  Actions: {'WAIT': 9, 'GERMINATE_NORM': 42, 'GERMINATE_ATTENTION': 24, 'GERMINATE_DEPTHWISE': 27, 'GERMINATE_CONV_ENHANCE': 10, 'FOSSILIZE': 19, 'CULL': 19}
  Successful: {'WAIT': 9, 'GERMINATE_NORM': 8, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 2, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 1, 'CULL': 18}
  Policy loss: -0.0062, Value loss: 220.5394, Entropy: 1.8680, Entropy coef: 0.1721
[12:17:39] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[12:17:39] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[12:17:39] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[12:17:39] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[12:17:46] env1_seed_0 | Stage transition: TRAINING → BLENDING
[12:17:47] env0_seed_0 | Stage transition: TRAINING → BLENDING
[12:17:51] env1_seed_0 | Stage transition: BLENDING → CULLED
[12:17:51] env1_seed_0 | Culled (norm, Δacc +13.02%)
    [env1] Culled 'env1_seed_0' (norm, Δacc +13.02%)
[12:17:52] env1_seed_1 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_1' (norm, 0.1K params)
[12:17:52] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[12:17:56] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[12:17:59] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[12:17:59] env1_seed_1 | Stage transition: TRAINING → BLENDING
[12:18:04] env0_seed_0 | Stage transition: PROBATIONARY → CULLED
[12:18:04] env0_seed_0 | Culled (norm, Δacc +16.30%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +16.30%)
[12:18:06] env0_seed_1 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_1' (norm, 0.1K params)
[12:18:06] env1_seed_1 | Stage transition: BLENDING → CULLED
[12:18:06] env1_seed_1 | Culled (norm, Δacc +4.25%)
    [env1] Culled 'env1_seed_1' (norm, Δacc +4.25%)
[12:18:06] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[12:18:07] env1_seed_2 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_2' (depthwise, 4.8K params)
[12:18:07] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[12:18:14] env1_seed_2 | Stage transition: TRAINING → BLENDING
[12:18:20] env0_seed_1 | Stage transition: TRAINING → BLENDING
[12:18:23] env1_seed_2 | Stage transition: BLENDING → SHADOWING
[12:18:27] env1_seed_2 | Stage transition: SHADOWING → PROBATIONARY
[12:18:28] env0_seed_1 | Stage transition: BLENDING → SHADOWING
[12:18:32] env0_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[12:18:32] env0_seed_1 | Stage transition: PROBATIONARY → CULLED
[12:18:32] env0_seed_1 | Culled (norm, Δacc +2.49%)
    [env0] Culled 'env0_seed_1' (norm, Δacc +2.49%)
[12:18:34] env0_seed_2 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_2' (norm, 0.1K params)
[12:18:34] env1_seed_2 | Stage transition: PROBATIONARY → CULLED
[12:18:34] env1_seed_2 | Culled (depthwise, Δacc +7.62%)
    [env1] Culled 'env1_seed_2' (depthwise, Δacc +7.62%)
[12:18:34] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[12:18:35] env1_seed_3 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_3' (depthwise, 4.8K params)
[12:18:35] env1_seed_3 | Stage transition: GERMINATED → TRAINING
[12:18:42] env0_seed_2 | Stage transition: TRAINING → BLENDING
[12:18:44] env0_seed_2 | Stage transition: BLENDING → CULLED
[12:18:44] env0_seed_2 | Culled (norm, Δacc +0.01%)
    [env0] Culled 'env0_seed_2' (norm, Δacc +0.01%)
[12:18:46] env0_seed_3 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_3' (depthwise, 4.8K params)
[12:18:46] env1_seed_3 | Stage transition: TRAINING → CULLED
[12:18:46] env1_seed_3 | Culled (depthwise, Δacc -4.18%)
    [env1] Culled 'env1_seed_3' (depthwise, Δacc -4.18%)
[12:18:46] env0_seed_3 | Stage transition: GERMINATED → TRAINING
[12:18:51] env1_seed_4 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_4' (norm, 0.1K params)
[12:18:51] env1_seed_4 | Stage transition: GERMINATED → TRAINING
[12:18:52] env0_seed_3 | Stage transition: TRAINING → BLENDING
[12:18:52] env0_seed_3 | Stage transition: BLENDING → CULLED
[12:18:52] env0_seed_3 | Culled (depthwise, Δacc -3.29%)
    [env0] Culled 'env0_seed_3' (depthwise, Δacc -3.29%)
[12:18:54] env0_seed_4 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_4' (attention, 2.0K params)
[12:18:54] env0_seed_4 | Stage transition: GERMINATED → TRAINING
[12:18:59] env1_seed_4 | Stage transition: TRAINING → CULLED
[12:18:59] env1_seed_4 | Culled (norm, Δacc -6.36%)
    [env1] Culled 'env1_seed_4' (norm, Δacc -6.36%)
[12:19:04] env1_seed_5 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_5' (norm, 0.1K params)
[12:19:04] env1_seed_5 | Stage transition: GERMINATED → TRAINING
[12:19:07] env0_seed_4 | Stage transition: TRAINING → BLENDING
[12:19:12] env0_seed_4 | Stage transition: BLENDING → CULLED
[12:19:12] env0_seed_4 | Culled (attention, Δacc +4.29%)
    [env0] Culled 'env0_seed_4' (attention, Δacc +4.29%)
[12:19:14] env1_seed_5 | Stage transition: TRAINING → CULLED
[12:19:14] env1_seed_5 | Culled (norm, Δacc -1.67%)
    [env1] Culled 'env1_seed_5' (norm, Δacc -1.67%)
[12:19:16] env1_seed_6 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_6' (conv_enhance, 74.0K params)
[12:19:16] env1_seed_6 | Stage transition: GERMINATED → TRAINING
[12:19:18] env0_seed_5 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_5' (depthwise, 4.8K params)
[12:19:18] env0_seed_5 | Stage transition: GERMINATED → TRAINING
[12:19:23] env1_seed_6 | Stage transition: TRAINING → BLENDING
[12:19:23] env1_seed_6 | Stage transition: BLENDING → CULLED
[12:19:23] env1_seed_6 | Culled (conv_enhance, Δacc -0.47%)
    [env1] Culled 'env1_seed_6' (conv_enhance, Δacc -0.47%)
[12:19:25] env0_seed_5 | Stage transition: TRAINING → BLENDING
[12:19:25] env1_seed_7 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_7' (norm, 0.1K params)
[12:19:25] env1_seed_7 | Stage transition: GERMINATED → TRAINING
[12:19:28] env1_seed_7 | Stage transition: TRAINING → CULLED
[12:19:28] env1_seed_7 | Culled (norm, Δacc +3.19%)
    [env1] Culled 'env1_seed_7' (norm, Δacc +3.19%)
[12:19:30] env0_seed_5 | Stage transition: BLENDING → CULLED
[12:19:30] env0_seed_5 | Culled (depthwise, Δacc -3.94%)
    [env0] Culled 'env0_seed_5' (depthwise, Δacc -3.94%)
[12:19:31] env0_seed_6 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_6' (norm, 0.1K params)
[12:19:31] env0_seed_6 | Stage transition: GERMINATED → TRAINING
[12:19:36] env1_seed_8 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_8' (attention, 2.0K params)
[12:19:36] env1_seed_8 | Stage transition: GERMINATED → TRAINING
[12:19:43] env1_seed_8 | Stage transition: TRAINING → BLENDING
Batch 12: Episodes 24/200
  Env accuracies: ['71.1%', '72.7%']
  Avg acc: 71.9% (rolling: 71.9%)
  Avg reward: 134.0
  Actions: {'WAIT': 23, 'GERMINATE_NORM': 38, 'GERMINATE_ATTENTION': 24, 'GERMINATE_DEPTHWISE': 20, 'GERMINATE_CONV_ENHANCE': 14, 'FOSSILIZE': 15, 'CULL': 16}
  Successful: {'WAIT': 23, 'GERMINATE_NORM': 9, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 4, 'GERMINATE_CONV_ENHANCE': 1, 'FOSSILIZE': 0, 'CULL': 14}
  Policy loss: 0.0109, Value loss: 194.5006, Entropy: 1.8515, Entropy coef: 0.1696
[12:19:46] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[12:19:46] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[12:19:46] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[12:19:46] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[12:19:53] env0_seed_0 | Stage transition: TRAINING → BLENDING
[12:19:53] env1_seed_0 | Stage transition: TRAINING → BLENDING
[12:19:56] env0_seed_0 | Stage transition: BLENDING → CULLED
[12:19:56] env0_seed_0 | Culled (norm, Δacc +16.08%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +16.08%)
[12:19:58] env0_seed_1 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_1' (depthwise, 4.8K params)
[12:19:58] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[12:20:01] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[12:20:05] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[12:20:05] env1_seed_0 | Stage transition: PROBATIONARY → CULLED
[12:20:05] env1_seed_0 | Culled (norm, Δacc +17.90%)
    [env1] Culled 'env1_seed_0' (norm, Δacc +17.90%)
[12:20:06] env0_seed_1 | Stage transition: TRAINING → BLENDING
[12:20:06] env1_seed_1 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_1' (attention, 2.0K params)
[12:20:06] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[12:20:15] env0_seed_1 | Stage transition: BLENDING → SHADOWING
[12:20:16] env1_seed_1 | Stage transition: TRAINING → BLENDING
[12:20:16] env0_seed_1 | Stage transition: SHADOWING → CULLED
[12:20:16] env0_seed_1 | Culled (depthwise, Δacc +10.80%)
    [env0] Culled 'env0_seed_1' (depthwise, Δacc +10.80%)
[12:20:18] env0_seed_2 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_2' (norm, 0.1K params)
[12:20:18] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[12:20:23] env1_seed_1 | Stage transition: BLENDING → CULLED
[12:20:23] env1_seed_1 | Culled (attention, Δacc +5.19%)
    [env1] Culled 'env1_seed_1' (attention, Δacc +5.19%)
[12:20:26] env0_seed_2 | Stage transition: TRAINING → BLENDING
[12:20:26] env1_seed_2 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_2' (norm, 0.1K params)
[12:20:27] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[12:20:33] env1_seed_2 | Stage transition: TRAINING → BLENDING
[12:20:33] env0_seed_2 | Stage transition: BLENDING → CULLED
[12:20:33] env0_seed_2 | Culled (norm, Δacc +4.46%)
    [env0] Culled 'env0_seed_2' (norm, Δacc +4.46%)
[12:20:36] env0_seed_3 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_3' (attention, 2.0K params)
[12:20:37] env0_seed_3 | Stage transition: GERMINATED → TRAINING
[12:20:40] env0_seed_3 | Stage transition: TRAINING → CULLED
[12:20:40] env0_seed_3 | Culled (attention, Δacc -4.15%)
    [env0] Culled 'env0_seed_3' (attention, Δacc -4.15%)
[12:20:42] env1_seed_2 | Stage transition: BLENDING → SHADOWING
[12:20:43] env0_seed_4 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_4' (attention, 2.0K params)
[12:20:43] env0_seed_4 | Stage transition: GERMINATED → TRAINING
[12:20:45] env1_seed_2 | Stage transition: SHADOWING → PROBATIONARY
[12:20:47] env0_seed_4 | Stage transition: TRAINING → CULLED
[12:20:47] env0_seed_4 | Culled (attention, Δacc +0.87%)
    [env0] Culled 'env0_seed_4' (attention, Δacc +0.87%)
[12:20:52] env0_seed_5 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_5' (conv_enhance, 74.0K params)
[12:20:52] env0_seed_5 | Stage transition: GERMINATED → TRAINING
[12:20:58] env0_seed_5 | Stage transition: TRAINING → BLENDING
[12:21:03] env1_seed_2 | Stage transition: PROBATIONARY → FOSSILIZED
[12:21:03] env1_seed_2 | Fossilized (norm, Δacc +11.18%)
    [env1] Fossilized 'env1_seed_2' (norm, Δacc +11.18%)
[12:21:07] env0_seed_5 | Stage transition: BLENDING → SHADOWING
[12:21:10] env0_seed_5 | Stage transition: SHADOWING → PROBATIONARY
[12:21:24] env0_seed_5 | Stage transition: PROBATIONARY → FOSSILIZED
[12:21:24] env0_seed_5 | Fossilized (conv_enhance, Δacc +0.36%)
    [env0] Fossilized 'env0_seed_5' (conv_enhance, Δacc +0.36%)
Batch 13: Episodes 26/200
  Env accuracies: ['60.2%', '76.2%']
  Avg acc: 68.2% (rolling: 71.5%)
  Avg reward: 107.4
  Actions: {'WAIT': 19, 'GERMINATE_NORM': 41, 'GERMINATE_ATTENTION': 24, 'GERMINATE_DEPTHWISE': 15, 'GERMINATE_CONV_ENHANCE': 20, 'FOSSILIZE': 16, 'CULL': 15}
  Successful: {'WAIT': 19, 'GERMINATE_NORM': 4, 'GERMINATE_ATTENTION': 3, 'GERMINATE_DEPTHWISE': 1, 'GERMINATE_CONV_ENHANCE': 1, 'FOSSILIZE': 2, 'CULL': 12}
  Policy loss: -0.0033, Value loss: 227.3327, Entropy: 1.8202, Entropy coef: 0.1671
[12:21:52] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[12:21:52] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[12:21:52] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[12:21:52] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[12:21:59] env1_seed_0 | Stage transition: TRAINING → BLENDING
[12:22:00] env0_seed_0 | Stage transition: TRAINING → BLENDING
[12:22:07] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[12:22:07] env0_seed_0 | Stage transition: BLENDING → CULLED
[12:22:07] env0_seed_0 | Culled (norm, Δacc +10.97%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +10.97%)
[12:22:09] env0_seed_1 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_1' (norm, 0.1K params)
[12:22:09] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[12:22:11] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[12:22:16] env0_seed_1 | Stage transition: TRAINING → CULLED
[12:22:16] env0_seed_1 | Culled (norm, Δacc +3.12%)
    [env0] Culled 'env0_seed_1' (norm, Δacc +3.12%)
[12:22:19] env0_seed_2 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_2' (conv_enhance, 74.0K params)
[12:22:19] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[12:22:21] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[12:22:21] env1_seed_0 | Fossilized (norm, Δacc +20.64%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +20.64%)
[12:22:34] env0_seed_2 | Stage transition: TRAINING → BLENDING
[12:22:34] env0_seed_2 | Stage transition: BLENDING → CULLED
[12:22:34] env0_seed_2 | Culled (conv_enhance, Δacc +1.79%)
    [env0] Culled 'env0_seed_2' (conv_enhance, Δacc +1.79%)
[12:22:37] env0_seed_3 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_3' (attention, 2.0K params)
[12:22:38] env0_seed_3 | Stage transition: GERMINATED → TRAINING
[12:22:44] env0_seed_3 | Stage transition: TRAINING → BLENDING
[12:22:53] env0_seed_3 | Stage transition: BLENDING → SHADOWING
[12:22:56] env0_seed_3 | Stage transition: SHADOWING → PROBATIONARY
[12:22:59] env0_seed_3 | Stage transition: PROBATIONARY → CULLED
[12:22:59] env0_seed_3 | Culled (attention, Δacc +2.83%)
    [env0] Culled 'env0_seed_3' (attention, Δacc +2.83%)
[12:23:01] env0_seed_4 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_4' (norm, 0.1K params)
[12:23:01] env0_seed_4 | Stage transition: GERMINATED → TRAINING
[12:23:08] env0_seed_4 | Stage transition: TRAINING → BLENDING
[12:23:16] env0_seed_4 | Stage transition: BLENDING → SHADOWING
[12:23:20] env0_seed_4 | Stage transition: SHADOWING → PROBATIONARY
[12:23:20] env0_seed_4 | Stage transition: PROBATIONARY → CULLED
[12:23:20] env0_seed_4 | Culled (norm, Δacc +5.39%)
    [env0] Culled 'env0_seed_4' (norm, Δacc +5.39%)
[12:23:23] env0_seed_5 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_5' (norm, 0.1K params)
[12:23:23] env0_seed_5 | Stage transition: GERMINATED → TRAINING
[12:23:25] env0_seed_5 | Stage transition: TRAINING → CULLED
[12:23:25] env0_seed_5 | Culled (norm, Δacc +0.00%)
    [env0] Culled 'env0_seed_5' (norm, Δacc +0.00%)
[12:23:26] env0_seed_6 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_6' (attention, 2.0K params)
[12:23:26] env0_seed_6 | Stage transition: GERMINATED → TRAINING
[12:23:33] env0_seed_6 | Stage transition: TRAINING → BLENDING
[12:23:41] env0_seed_6 | Stage transition: BLENDING → SHADOWING
[12:23:45] env0_seed_6 | Stage transition: SHADOWING → PROBATIONARY
[12:23:51] env0_seed_6 | Stage transition: PROBATIONARY → CULLED
[12:23:51] env0_seed_6 | Culled (attention, Δacc +4.15%)
    [env0] Culled 'env0_seed_6' (attention, Δacc +4.15%)
[12:23:53] env0_seed_7 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_7' (depthwise, 4.8K params)
[12:23:53] env0_seed_7 | Stage transition: GERMINATED → TRAINING
Batch 14: Episodes 28/200
  Env accuracies: ['74.7%', '76.6%']
  Avg acc: 75.7% (rolling: 71.8%)
  Avg reward: 150.1
  Actions: {'WAIT': 22, 'GERMINATE_NORM': 43, 'GERMINATE_ATTENTION': 23, 'GERMINATE_DEPTHWISE': 19, 'GERMINATE_CONV_ENHANCE': 18, 'FOSSILIZE': 9, 'CULL': 16}
  Successful: {'WAIT': 22, 'GERMINATE_NORM': 5, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 1, 'GERMINATE_CONV_ENHANCE': 1, 'FOSSILIZE': 1, 'CULL': 16}
  Policy loss: 0.0068, Value loss: 187.2879, Entropy: 1.8465, Entropy coef: 0.1645
[12:23:58] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[12:23:58] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[12:23:58] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[12:23:58] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[12:24:05] env0_seed_0 | Stage transition: TRAINING → BLENDING
[12:24:05] env1_seed_0 | Stage transition: TRAINING → BLENDING
[12:24:13] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[12:24:13] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[12:24:17] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[12:24:17] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[12:24:17] env1_seed_0 | Stage transition: PROBATIONARY → CULLED
[12:24:17] env1_seed_0 | Culled (norm, Δacc +16.96%)
    [env1] Culled 'env1_seed_0' (norm, Δacc +16.96%)
[12:24:18] env1_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_1' (conv_enhance, 74.0K params)
[12:24:18] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[12:24:24] env1_seed_1 | Stage transition: TRAINING → CULLED
[12:24:24] env1_seed_1 | Culled (conv_enhance, Δacc -1.34%)
    [env1] Culled 'env1_seed_1' (conv_enhance, Δacc -1.34%)
[12:24:25] env0_seed_0 | Stage transition: PROBATIONARY → CULLED
[12:24:25] env0_seed_0 | Culled (norm, Δacc +16.25%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +16.25%)
[12:24:25] env1_seed_2 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_2' (depthwise, 4.8K params)
[12:24:25] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[12:24:29] env0_seed_1 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_1' (depthwise, 4.8K params)
[12:24:29] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[12:24:31] env0_seed_1 | Stage transition: TRAINING → CULLED
[12:24:31] env0_seed_1 | Culled (depthwise, Δacc +0.00%)
    [env0] Culled 'env0_seed_1' (depthwise, Δacc +0.00%)
[12:24:32] env1_seed_2 | Stage transition: TRAINING → BLENDING
[12:24:32] env0_seed_2 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_2' (attention, 2.0K params)
[12:24:32] env1_seed_2 | Stage transition: BLENDING → CULLED
[12:24:32] env1_seed_2 | Culled (depthwise, Δacc -0.17%)
    [env1] Culled 'env1_seed_2' (depthwise, Δacc -0.17%)
[12:24:32] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[12:24:34] env1_seed_3 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_3' (depthwise, 4.8K params)
[12:24:34] env1_seed_3 | Stage transition: GERMINATED → TRAINING
[12:24:37] env0_seed_2 | Stage transition: TRAINING → CULLED
[12:24:37] env0_seed_2 | Culled (attention, Δacc +1.01%)
    [env0] Culled 'env0_seed_2' (attention, Δacc +1.01%)
[12:24:39] env0_seed_3 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_3' (depthwise, 4.8K params)
[12:24:39] env0_seed_3 | Stage transition: GERMINATED → TRAINING
[12:24:46] env0_seed_3 | Stage transition: TRAINING → BLENDING
[12:24:46] env1_seed_3 | Stage transition: TRAINING → CULLED
[12:24:46] env1_seed_3 | Culled (depthwise, Δacc -0.53%)
    [env1] Culled 'env1_seed_3' (depthwise, Δacc -0.53%)
[12:24:48] env1_seed_4 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_4' (conv_enhance, 74.0K params)
[12:24:48] env1_seed_4 | Stage transition: GERMINATED → TRAINING
[12:24:53] env1_seed_4 | Stage transition: TRAINING → CULLED
[12:24:53] env1_seed_4 | Culled (conv_enhance, Δacc -4.60%)
    [env1] Culled 'env1_seed_4' (conv_enhance, Δacc -4.60%)
[12:24:55] env0_seed_3 | Stage transition: BLENDING → SHADOWING
[12:24:55] env1_seed_5 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_5' (norm, 0.1K params)
[12:24:55] env1_seed_5 | Stage transition: GERMINATED → TRAINING
[12:24:58] env0_seed_3 | Stage transition: SHADOWING → PROBATIONARY
[12:25:05] env0_seed_3 | Stage transition: PROBATIONARY → CULLED
[12:25:05] env0_seed_3 | Culled (depthwise, Δacc +2.36%)
    [env0] Culled 'env0_seed_3' (depthwise, Δacc +2.36%)
[12:25:06] env1_seed_5 | Stage transition: TRAINING → BLENDING
[12:25:06] env1_seed_5 | Stage transition: BLENDING → CULLED
[12:25:06] env1_seed_5 | Culled (norm, Δacc -3.73%)
    [env1] Culled 'env1_seed_5' (norm, Δacc -3.73%)
[12:25:08] env0_seed_4 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_4' (attention, 2.0K params)
[12:25:08] env1_seed_6 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_6' (attention, 2.0K params)
[12:25:08] env0_seed_4 | Stage transition: GERMINATED → TRAINING
[12:25:08] env1_seed_6 | Stage transition: GERMINATED → TRAINING
[12:25:15] env0_seed_4 | Stage transition: TRAINING → BLENDING
[12:25:15] env1_seed_6 | Stage transition: TRAINING → CULLED
[12:25:15] env1_seed_6 | Culled (attention, Δacc +0.29%)
    [env1] Culled 'env1_seed_6' (attention, Δacc +0.29%)
[12:25:16] env1_seed_7 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_7' (norm, 0.1K params)
[12:25:16] env1_seed_7 | Stage transition: GERMINATED → TRAINING
[12:25:18] env1_seed_7 | Stage transition: TRAINING → CULLED
[12:25:18] env1_seed_7 | Culled (norm, Δacc +0.00%)
    [env1] Culled 'env1_seed_7' (norm, Δacc +0.00%)
[12:25:21] env1_seed_8 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_8' (attention, 2.0K params)
[12:25:21] env1_seed_8 | Stage transition: GERMINATED → TRAINING
[12:25:23] env0_seed_4 | Stage transition: BLENDING → SHADOWING
[12:25:25] env1_seed_8 | Stage transition: TRAINING → CULLED
[12:25:25] env1_seed_8 | Culled (attention, Δacc -6.29%)
    [env1] Culled 'env1_seed_8' (attention, Δacc -6.29%)
[12:25:26] env0_seed_4 | Stage transition: SHADOWING → PROBATIONARY
[12:25:26] env1_seed_9 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_9' (attention, 2.0K params)
[12:25:26] env1_seed_9 | Stage transition: GERMINATED → TRAINING
[12:25:31] env1_seed_9 | Stage transition: TRAINING → CULLED
[12:25:31] env1_seed_9 | Culled (attention, Δacc -5.93%)
    [env1] Culled 'env1_seed_9' (attention, Δacc -5.93%)
[12:25:35] env1_seed_10 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_10' (conv_enhance, 74.0K params)
[12:25:35] env1_seed_10 | Stage transition: GERMINATED → TRAINING
[12:25:42] env1_seed_10 | Stage transition: TRAINING → BLENDING
[12:25:51] env1_seed_10 | Stage transition: BLENDING → SHADOWING
[12:25:55] env1_seed_10 | Stage transition: SHADOWING → PROBATIONARY
[12:25:58] env1_seed_10 | Stage transition: PROBATIONARY → CULLED
[12:25:58] env1_seed_10 | Culled (conv_enhance, Δacc -10.95%)
    [env1] Culled 'env1_seed_10' (conv_enhance, Δacc -10.95%)
[12:26:00] env1_seed_11 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_11' (norm, 0.1K params)
[12:26:00] env1_seed_11 | Stage transition: GERMINATED → TRAINING
[12:26:03] env0_seed_4 | Stage transition: PROBATIONARY → FOSSILIZED
[12:26:03] env0_seed_4 | Fossilized (attention, Δacc +1.86%)
    [env0] Fossilized 'env0_seed_4' (attention, Δacc +1.86%)
[12:26:03] env1_seed_11 | Stage transition: TRAINING → CULLED
[12:26:03] env1_seed_11 | Culled (norm, Δacc -3.40%)
    [env1] Culled 'env1_seed_11' (norm, Δacc -3.40%)
[12:26:05] env1_seed_12 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_12' (norm, 0.1K params)
Batch 15: Episodes 30/200
  Env accuracies: ['76.5%', '73.7%']
  Avg acc: 75.1% (rolling: 71.9%)
  Avg reward: 132.8
  Actions: {'WAIT': 21, 'GERMINATE_NORM': 44, 'GERMINATE_ATTENTION': 25, 'GERMINATE_DEPTHWISE': 16, 'GERMINATE_CONV_ENHANCE': 14, 'FOSSILIZE': 13, 'CULL': 17}
  Successful: {'WAIT': 21, 'GERMINATE_NORM': 6, 'GERMINATE_ATTENTION': 5, 'GERMINATE_DEPTHWISE': 4, 'GERMINATE_CONV_ENHANCE': 3, 'FOSSILIZE': 1, 'CULL': 16}
  Policy loss: -0.0068, Value loss: 196.6744, Entropy: 1.8495, Entropy coef: 0.1620

Blueprint Stats
  ---------------------------------------------------------------------------

Blueprint       Germ  Foss  Cull   Rate     ΔAcc    Churn
  ---------------------------------------------------------------------------

  attention         49     4    43   8.5%   +1.64%   +0.29%
  conv_enhance      42     3    37   7.5%  +11.19%   -2.17%
  depthwise         43     0    40   0.0%   +0.00%   +0.97%
  norm              98     4    86   4.4%  +14.29%   +3.90%
Seed Scoreboard (env 0):
  Fossilized: 6 (+228.1K params, +240.7% of host)
  Culled: 94
  Avg fossilize age: 18.5 epochs
  Avg cull age: 6.2 epochs
  Compute cost: 2.50x baseline
  Distribution: attention x3, conv_enhance x3
Seed Scoreboard (env 1):
  Fossilized: 5 (+2.6K params, +2.7% of host)
  Culled: 112
  Avg fossilize age: 15.0 epochs
  Avg cull age: 5.3 epochs
  Compute cost: 1.43x baseline
  Distribution: attention x1, norm x4

[12:26:07] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[12:26:07] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[12:26:07] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[12:26:07] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[12:26:12] env1_seed_0 | Stage transition: TRAINING → CULLED
[12:26:12] env1_seed_0 | Culled (norm, Δacc +7.21%)
    [env1] Culled 'env1_seed_0' (norm, Δacc +7.21%)
[12:26:13] env0_seed_0 | Stage transition: TRAINING → BLENDING
[12:26:13] env0_seed_0 | Stage transition: BLENDING → CULLED
[12:26:13] env0_seed_0 | Culled (norm, Δacc +8.14%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +8.14%)
[12:26:13] env1_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_1' (conv_enhance, 74.0K params)
[12:26:13] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[12:26:15] env0_seed_1 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_1' (attention, 2.0K params)
[12:26:15] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[12:26:20] env1_seed_1 | Stage transition: TRAINING → BLENDING
[12:26:20] env0_seed_1 | Stage transition: TRAINING → CULLED
[12:26:20] env0_seed_1 | Culled (attention, Δacc +0.05%)
    [env0] Culled 'env0_seed_1' (attention, Δacc +0.05%)
[12:26:22] env0_seed_2 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_2' (norm, 0.1K params)
[12:26:22] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[12:26:30] env0_seed_2 | Stage transition: TRAINING → BLENDING
[12:26:30] env1_seed_1 | Stage transition: BLENDING → SHADOWING
[12:26:31] env0_seed_2 | Stage transition: BLENDING → CULLED
[12:26:31] env0_seed_2 | Culled (norm, Δacc +1.82%)
    [env0] Culled 'env0_seed_2' (norm, Δacc +1.82%)
[12:26:33] env1_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[12:26:33] env0_seed_3 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_3' (conv_enhance, 74.0K params)
[12:26:33] env0_seed_3 | Stage transition: GERMINATED → TRAINING
[12:26:44] env0_seed_3 | Stage transition: TRAINING → BLENDING
[12:26:44] env0_seed_3 | Stage transition: BLENDING → CULLED
[12:26:44] env0_seed_3 | Culled (conv_enhance, Δacc +0.54%)
    [env0] Culled 'env0_seed_3' (conv_enhance, Δacc +0.54%)
[12:26:44] env1_seed_1 | Stage transition: PROBATIONARY → CULLED
[12:26:44] env1_seed_1 | Culled (conv_enhance, Δacc +11.40%)
    [env1] Culled 'env1_seed_1' (conv_enhance, Δacc +11.40%)
[12:26:46] env0_seed_4 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_4' (attention, 2.0K params)
[12:26:46] env0_seed_4 | Stage transition: GERMINATED → TRAINING
[12:26:49] env0_seed_4 | Stage transition: TRAINING → CULLED
[12:26:49] env0_seed_4 | Culled (attention, Δacc +1.54%)
    [env0] Culled 'env0_seed_4' (attention, Δacc +1.54%)
[12:26:51] env0_seed_5 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_5' (attention, 2.0K params)
[12:26:51] env0_seed_5 | Stage transition: GERMINATED → TRAINING
[12:26:52] env1_seed_2 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_2' (depthwise, 4.8K params)
[12:26:52] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[12:26:54] env1_seed_2 | Stage transition: TRAINING → CULLED
[12:26:54] env1_seed_2 | Culled (depthwise, Δacc +0.00%)
    [env1] Culled 'env1_seed_2' (depthwise, Δacc +0.00%)
[12:26:56] env1_seed_3 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_3' (attention, 2.0K params)
[12:26:56] env1_seed_3 | Stage transition: GERMINATED → TRAINING
[12:26:57] env0_seed_5 | Stage transition: TRAINING → BLENDING
[12:26:57] env0_seed_5 | Stage transition: BLENDING → CULLED
[12:26:57] env0_seed_5 | Culled (attention, Δacc +4.87%)
    [env0] Culled 'env0_seed_5' (attention, Δacc +4.87%)
[12:26:57] env1_seed_3 | Stage transition: TRAINING → CULLED
[12:26:57] env1_seed_3 | Culled (attention, Δacc +0.00%)
    [env1] Culled 'env1_seed_3' (attention, Δacc +0.00%)
[12:26:59] env0_seed_6 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_6' (conv_enhance, 74.0K params)
[12:26:59] env1_seed_4 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_4' (norm, 0.1K params)
[12:26:59] env0_seed_6 | Stage transition: GERMINATED → TRAINING
[12:26:59] env1_seed_4 | Stage transition: GERMINATED → TRAINING
[12:27:06] env1_seed_4 | Stage transition: TRAINING → BLENDING
[12:27:09] env0_seed_6 | Stage transition: TRAINING → BLENDING
[12:27:14] env1_seed_4 | Stage transition: BLENDING → SHADOWING
[12:27:14] env0_seed_6 | Stage transition: BLENDING → CULLED
[12:27:14] env0_seed_6 | Culled (conv_enhance, Δacc -5.64%)
    [env0] Culled 'env0_seed_6' (conv_enhance, Δacc -5.64%)
[12:27:16] env0_seed_7 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_7' (attention, 2.0K params)
[12:27:16] env0_seed_7 | Stage transition: GERMINATED → TRAINING
[12:27:17] env1_seed_4 | Stage transition: SHADOWING → PROBATIONARY
[12:27:17] env1_seed_4 | Stage transition: PROBATIONARY → CULLED
[12:27:17] env1_seed_4 | Culled (norm, Δacc +10.45%)
    [env1] Culled 'env1_seed_4' (norm, Δacc +10.45%)
[12:27:21] env1_seed_5 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_5' (norm, 0.1K params)
[12:27:21] env1_seed_5 | Stage transition: GERMINATED → TRAINING
[12:27:22] env0_seed_7 | Stage transition: TRAINING → BLENDING
[12:27:29] env1_seed_5 | Stage transition: TRAINING → BLENDING
[12:27:31] env0_seed_7 | Stage transition: BLENDING → SHADOWING
[12:27:32] env1_seed_5 | Stage transition: BLENDING → CULLED
[12:27:32] env1_seed_5 | Culled (norm, Δacc +2.28%)
    [env1] Culled 'env1_seed_5' (norm, Δacc +2.28%)
[12:27:34] env0_seed_7 | Stage transition: SHADOWING → PROBATIONARY
[12:27:34] env1_seed_6 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_6' (norm, 0.1K params)
[12:27:34] env1_seed_6 | Stage transition: GERMINATED → TRAINING
[12:27:36] env0_seed_7 | Stage transition: PROBATIONARY → CULLED
[12:27:36] env0_seed_7 | Culled (attention, Δacc +2.20%)
    [env0] Culled 'env0_seed_7' (attention, Δacc +2.20%)
[12:27:36] env1_seed_6 | Stage transition: TRAINING → CULLED
[12:27:36] env1_seed_6 | Culled (norm, Δacc +0.00%)
    [env1] Culled 'env1_seed_6' (norm, Δacc +0.00%)
[12:27:37] env0_seed_8 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_8' (conv_enhance, 74.0K params)
[12:27:37] env1_seed_7 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_7' (conv_enhance, 74.0K params)
[12:27:37] env0_seed_8 | Stage transition: GERMINATED → TRAINING
[12:27:37] env1_seed_7 | Stage transition: GERMINATED → TRAINING
[12:27:41] env0_seed_8 | Stage transition: TRAINING → CULLED
[12:27:41] env0_seed_8 | Culled (conv_enhance, Δacc +0.41%)
    [env0] Culled 'env0_seed_8' (conv_enhance, Δacc +0.41%)
[12:27:43] env0_seed_9 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_9' (attention, 2.0K params)
[12:27:43] env0_seed_9 | Stage transition: GERMINATED → TRAINING
[12:27:45] env1_seed_7 | Stage transition: TRAINING → BLENDING
[12:27:48] env0_seed_9 | Stage transition: TRAINING → CULLED
[12:27:48] env0_seed_9 | Culled (attention, Δacc -10.72%)
    [env0] Culled 'env0_seed_9' (attention, Δacc -10.72%)
[12:27:54] env1_seed_7 | Stage transition: BLENDING → SHADOWING
[12:27:56] env0_seed_10 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_10' (norm, 0.1K params)
[12:27:56] env1_seed_7 | Stage transition: SHADOWING → CULLED
[12:27:56] env1_seed_7 | Culled (conv_enhance, Δacc -3.21%)
    [env1] Culled 'env1_seed_7' (conv_enhance, Δacc -3.21%)
[12:27:56] env0_seed_10 | Stage transition: GERMINATED → TRAINING
[12:27:57] env1_seed_8 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_8' (depthwise, 4.8K params)
[12:27:57] env1_seed_8 | Stage transition: GERMINATED → TRAINING
[12:27:59] env0_seed_10 | Stage transition: TRAINING → CULLED
[12:27:59] env0_seed_10 | Culled (norm, Δacc +0.18%)
    [env0] Culled 'env0_seed_10' (norm, Δacc +0.18%)
[12:28:04] env1_seed_8 | Stage transition: TRAINING → BLENDING
[12:28:04] env0_seed_11 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_11' (depthwise, 4.8K params)
[12:28:04] env0_seed_11 | Stage transition: GERMINATED → TRAINING
[12:28:09] env1_seed_8 | Stage transition: BLENDING → CULLED
[12:28:09] env1_seed_8 | Culled (depthwise, Δacc -16.51%)
    [env1] Culled 'env1_seed_8' (depthwise, Δacc -16.51%)
[12:28:11] env1_seed_9 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_9' (norm, 0.1K params)
[12:28:11] env1_seed_9 | Stage transition: GERMINATED → TRAINING
[12:28:13] env1_seed_9 | Stage transition: TRAINING → CULLED
[12:28:13] env1_seed_9 | Culled (norm, Δacc +0.00%)
    [env1] Culled 'env1_seed_9' (norm, Δacc +0.00%)
[12:28:14] env1_seed_10 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_10' (depthwise, 4.8K params)
Batch 16: Episodes 32/200
  Env accuracies: ['70.0%', '73.9%']
  Avg acc: 72.0% (rolling: 71.6%)
  Avg reward: 113.7
  Actions: {'WAIT': 24, 'GERMINATE_NORM': 36, 'GERMINATE_ATTENTION': 20, 'GERMINATE_DEPTHWISE': 19, 'GERMINATE_CONV_ENHANCE': 13, 'FOSSILIZE': 12, 'CULL': 26}
  Successful: {'WAIT': 24, 'GERMINATE_NORM': 8, 'GERMINATE_ATTENTION': 6, 'GERMINATE_DEPTHWISE': 4, 'GERMINATE_CONV_ENHANCE': 5, 'FOSSILIZE': 0, 'CULL': 21}
  Policy loss: -0.0149, Value loss: 184.8661, Entropy: 1.8492, Entropy coef: 0.1595
[12:28:16] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[12:28:16] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[12:28:16] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[12:28:16] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[12:28:23] env1_seed_0 | Stage transition: TRAINING → BLENDING
[12:28:24] env0_seed_0 | Stage transition: TRAINING → BLENDING
[12:28:26] env1_seed_0 | Stage transition: BLENDING → CULLED
[12:28:26] env1_seed_0 | Culled (norm, Δacc +13.86%)
    [env1] Culled 'env1_seed_0' (norm, Δacc +13.86%)
[12:28:28] env1_seed_1 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_1' (depthwise, 4.8K params)
[12:28:28] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[12:28:33] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[12:28:35] env1_seed_1 | Stage transition: TRAINING → BLENDING
[12:28:37] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[12:28:37] env0_seed_0 | Stage transition: PROBATIONARY → CULLED
[12:28:37] env0_seed_0 | Culled (norm, Δacc +14.26%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +14.26%)
[12:28:37] env1_seed_1 | Stage transition: BLENDING → CULLED
[12:28:37] env1_seed_1 | Culled (depthwise, Δacc +4.51%)
    [env1] Culled 'env1_seed_1' (depthwise, Δacc +4.51%)
[12:28:40] env0_seed_1 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_1' (norm, 0.1K params)
[12:28:40] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[12:28:41] env1_seed_2 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_2' (norm, 0.1K params)
[12:28:41] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[12:28:45] env1_seed_2 | Stage transition: TRAINING → CULLED
[12:28:45] env1_seed_2 | Culled (norm, Δacc +2.19%)
    [env1] Culled 'env1_seed_2' (norm, Δacc +2.19%)
[12:28:46] env0_seed_1 | Stage transition: TRAINING → CULLED
[12:28:46] env0_seed_1 | Culled (norm, Δacc +0.42%)
    [env0] Culled 'env0_seed_1' (norm, Δacc +0.42%)
[12:28:46] env1_seed_3 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_3' (norm, 0.1K params)
[12:28:46] env1_seed_3 | Stage transition: GERMINATED → TRAINING
[12:28:48] env0_seed_2 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_2' (depthwise, 4.8K params)
[12:28:48] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[12:28:51] env1_seed_3 | Stage transition: TRAINING → CULLED
[12:28:51] env1_seed_3 | Culled (norm, Δacc -2.64%)
    [env1] Culled 'env1_seed_3' (norm, Δacc -2.64%)
[12:28:55] env0_seed_2 | Stage transition: TRAINING → BLENDING
[12:28:55] env0_seed_2 | Stage transition: BLENDING → CULLED
[12:28:55] env0_seed_2 | Culled (depthwise, Δacc -1.54%)
    [env0] Culled 'env0_seed_2' (depthwise, Δacc -1.54%)
[12:28:55] env1_seed_4 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_4' (depthwise, 4.8K params)
[12:28:55] env1_seed_4 | Stage transition: GERMINATED → TRAINING
[12:28:58] env0_seed_3 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_3' (norm, 0.1K params)
[12:28:58] env0_seed_3 | Stage transition: GERMINATED → TRAINING
[12:29:00] env1_seed_4 | Stage transition: TRAINING → CULLED
[12:29:00] env1_seed_4 | Culled (depthwise, Δacc -0.82%)
    [env1] Culled 'env1_seed_4' (depthwise, Δacc -0.82%)
[12:29:02] env1_seed_5 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_5' (norm, 0.1K params)
[12:29:02] env1_seed_5 | Stage transition: GERMINATED → TRAINING
[12:29:07] env0_seed_3 | Stage transition: TRAINING → CULLED
[12:29:07] env0_seed_3 | Culled (norm, Δacc -7.23%)
    [env0] Culled 'env0_seed_3' (norm, Δacc -7.23%)
[12:29:08] env1_seed_5 | Stage transition: TRAINING → BLENDING
[12:29:08] env0_seed_4 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_4' (conv_enhance, 74.0K params)
[12:29:08] env0_seed_4 | Stage transition: GERMINATED → TRAINING
[12:29:17] env1_seed_5 | Stage transition: BLENDING → SHADOWING
[12:29:18] env0_seed_4 | Stage transition: TRAINING → BLENDING
[12:29:18] env1_seed_5 | Stage transition: SHADOWING → CULLED
[12:29:18] env1_seed_5 | Culled (norm, Δacc +11.21%)
    [env1] Culled 'env1_seed_5' (norm, Δacc +11.21%)
[12:29:23] env1_seed_6 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_6' (depthwise, 4.8K params)
[12:29:23] env1_seed_6 | Stage transition: GERMINATED → TRAINING
[12:29:27] env0_seed_4 | Stage transition: BLENDING → SHADOWING
[12:29:30] env0_seed_4 | Stage transition: SHADOWING → PROBATIONARY
[12:29:34] env1_seed_6 | Stage transition: TRAINING → CULLED
[12:29:34] env1_seed_6 | Culled (depthwise, Δacc -2.54%)
    [env1] Culled 'env1_seed_6' (depthwise, Δacc -2.54%)
[12:29:39] env1_seed_7 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_7' (attention, 2.0K params)
[12:29:39] env1_seed_7 | Stage transition: GERMINATED → TRAINING
[12:29:41] env1_seed_7 | Stage transition: TRAINING → CULLED
[12:29:41] env1_seed_7 | Culled (attention, Δacc +0.00%)
    [env1] Culled 'env1_seed_7' (attention, Δacc +0.00%)
[12:29:42] env1_seed_8 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_8' (conv_enhance, 74.0K params)
[12:29:42] env1_seed_8 | Stage transition: GERMINATED → TRAINING
[12:29:53] env0_seed_4 | Stage transition: PROBATIONARY → CULLED
[12:29:53] env0_seed_4 | Culled (conv_enhance, Δacc +1.84%)
    [env0] Culled 'env0_seed_4' (conv_enhance, Δacc +1.84%)
[12:29:55] env1_seed_8 | Stage transition: TRAINING → BLENDING
[12:30:04] env1_seed_8 | Stage transition: BLENDING → SHADOWING
[12:30:04] env0_seed_5 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_5' (depthwise, 4.8K params)
[12:30:04] env0_seed_5 | Stage transition: GERMINATED → TRAINING
[12:30:08] env1_seed_8 | Stage transition: SHADOWING → PROBATIONARY
[GOVERNOR] CRITICAL INSTABILITY DETECTED. INITIATING ROLLBACK.
[12:30:13] env1_seed_8 | Stage transition: PROBATIONARY → CULLED
[12:30:13] env1_seed_8 | Culled (conv_enhance, Δacc -27.52%) (governor_rollback)
    [env1] Culled 'env1_seed_8' (conv_enhance, Δacc -27.52%) (governor_rollback)
  [ENV 1] Governor rollback: Structural Collapse (threshold=2.3874, panics=3)
  [ENV 1] Punishment reward: -10.0 (final reward: -43.8)
[12:30:15] env0_seed_5 | Stage transition: TRAINING → BLENDING
[12:30:15] env1_seed_9 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_9' (depthwise, 4.8K params)
[12:30:15] env1_seed_9 | Stage transition: GERMINATED → TRAINING
[12:30:23] env0_seed_5 | Stage transition: BLENDING → SHADOWING
Batch 17: Episodes 34/200
  Env accuracies: ['58.2%', '71.2%']
  Avg acc: 64.7% (rolling: 71.2%)
  Avg reward: 92.9
  Actions: {'WAIT': 20, 'GERMINATE_NORM': 36, 'GERMINATE_ATTENTION': 16, 'GERMINATE_DEPTHWISE': 27, 'GERMINATE_CONV_ENHANCE': 18, 'FOSSILIZE': 14, 'CULL': 19}
  Successful: {'WAIT': 20, 'GERMINATE_NORM': 7, 'GERMINATE_ATTENTION': 1, 'GERMINATE_DEPTHWISE': 6, 'GERMINATE_CONV_ENHANCE': 2, 'FOSSILIZE': 0, 'CULL': 13}
  Policy loss: -0.0281, Value loss: 249.7752, Entropy: 1.8677, Entropy coef: 0.1569
[12:30:25] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[12:30:25] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[12:30:25] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[12:30:25] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[12:30:30] env1_seed_0 | Stage transition: TRAINING → CULLED
[12:30:30] env1_seed_0 | Culled (norm, Δacc +8.71%)
    [env1] Culled 'env1_seed_0' (norm, Δacc +8.71%)
[12:30:32] env0_seed_0 | Stage transition: TRAINING → BLENDING
[12:30:32] env1_seed_1 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_1' (depthwise, 4.8K params)
[12:30:32] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[12:30:40] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[12:30:44] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[12:30:44] env1_seed_1 | Stage transition: TRAINING → BLENDING
[12:30:47] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[12:30:47] env0_seed_0 | Fossilized (norm, Δacc +17.61%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +17.61%)
[12:30:53] env1_seed_1 | Stage transition: BLENDING → SHADOWING
[12:30:56] env1_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[12:30:56] env1_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[12:30:56] env1_seed_1 | Fossilized (depthwise, Δacc +4.35%)
    [env1] Fossilized 'env1_seed_1' (depthwise, Δacc +4.35%)
Batch 18: Episodes 36/200
  Env accuracies: ['75.8%', '66.5%']
  Avg acc: 71.2% (rolling: 71.1%)
  Avg reward: 150.2
  Actions: {'WAIT': 15, 'GERMINATE_NORM': 27, 'GERMINATE_ATTENTION': 31, 'GERMINATE_DEPTHWISE': 21, 'GERMINATE_CONV_ENHANCE': 24, 'FOSSILIZE': 17, 'CULL': 15}
  Successful: {'WAIT': 15, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 1, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 15}
  Policy loss: 0.0001, Value loss: 139.3068, Entropy: 1.8850, Entropy coef: 0.1544
[12:32:37] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[12:32:37] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[12:32:37] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[12:32:37] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[12:32:43] env0_seed_0 | Stage transition: TRAINING → BLENDING
[12:32:43] env1_seed_0 | Stage transition: TRAINING → BLENDING
[12:32:45] env0_seed_0 | Stage transition: BLENDING → CULLED
[12:32:45] env0_seed_0 | Culled (norm, Δacc +9.94%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +9.94%)
[12:32:47] env0_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_1' (conv_enhance, 74.0K params)
[12:32:47] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[12:32:52] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[12:32:53] env0_seed_1 | Stage transition: TRAINING → BLENDING
[12:32:55] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[12:33:02] env0_seed_1 | Stage transition: BLENDING → SHADOWING
[12:33:02] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[12:33:02] env1_seed_0 | Fossilized (norm, Δacc +11.19%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +11.19%)
[12:33:05] env0_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[12:33:05] env0_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[12:33:05] env0_seed_1 | Fossilized (conv_enhance, Δacc +10.87%)
    [env0] Fossilized 'env0_seed_1' (conv_enhance, Δacc +10.87%)
Batch 19: Episodes 38/200
  Env accuracies: ['64.6%', '76.6%']
  Avg acc: 70.6% (rolling: 70.9%)
  Avg reward: 69.3
  Actions: {'WAIT': 21, 'GERMINATE_NORM': 35, 'GERMINATE_ATTENTION': 24, 'GERMINATE_DEPTHWISE': 21, 'GERMINATE_CONV_ENHANCE': 23, 'FOSSILIZE': 12, 'CULL': 14}
  Successful: {'WAIT': 21, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 1, 'FOSSILIZE': 2, 'CULL': 14}
  Policy loss: -0.0085, Value loss: 238.4007, Entropy: 1.8742, Entropy coef: 0.1519
[12:34:43] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[12:34:43] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[12:34:43] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[12:34:43] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[12:34:49] env0_seed_0 | Stage transition: TRAINING → BLENDING
[12:34:49] env1_seed_0 | Stage transition: TRAINING → BLENDING
[12:34:58] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[12:34:58] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[12:35:01] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[12:35:01] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[12:35:01] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[12:35:01] env1_seed_0 | Fossilized (norm, Δacc +16.45%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +16.45%)
[12:35:11] env0_seed_0 | Stage transition: PROBATIONARY → CULLED
[12:35:11] env0_seed_0 | Culled (norm, Δacc +20.44%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +20.44%)
[12:35:13] env0_seed_1 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_1' (depthwise, 4.8K params)
[12:35:13] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[12:35:20] env0_seed_1 | Stage transition: TRAINING → BLENDING
[12:35:20] env0_seed_1 | Stage transition: BLENDING → CULLED
[12:35:20] env0_seed_1 | Culled (depthwise, Δacc +3.25%)
    [env0] Culled 'env0_seed_1' (depthwise, Δacc +3.25%)
[12:35:21] env0_seed_2 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_2' (norm, 0.1K params)
[12:35:21] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[12:35:33] env0_seed_2 | Stage transition: TRAINING → BLENDING
[12:35:41] env0_seed_2 | Stage transition: BLENDING → SHADOWING
[12:35:45] env0_seed_2 | Stage transition: SHADOWING → PROBATIONARY
[12:35:53] env0_seed_2 | Stage transition: PROBATIONARY → FOSSILIZED
[12:35:53] env0_seed_2 | Fossilized (norm, Δacc +2.62%)
    [env0] Fossilized 'env0_seed_2' (norm, Δacc +2.62%)
Batch 20: Episodes 40/200
  Env accuracies: ['77.2%', '74.0%']
  Avg acc: 75.6% (rolling: 71.4%)
  Avg reward: 161.7
  Actions: {'WAIT': 22, 'GERMINATE_NORM': 31, 'GERMINATE_ATTENTION': 23, 'GERMINATE_DEPTHWISE': 32, 'GERMINATE_CONV_ENHANCE': 17, 'FOSSILIZE': 12, 'CULL': 13}
  Successful: {'WAIT': 22, 'GERMINATE_NORM': 3, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 1, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 13}
  Policy loss: -0.0026, Value loss: 124.3114, Entropy: 1.8686, Entropy coef: 0.1493

Blueprint Stats
  ---------------------------------------------------------------------------

Blueprint       Germ  Foss  Cull   Rate     ΔAcc    Churn
  ---------------------------------------------------------------------------

  attention         56     4    50   7.4%   +1.64%   +0.21%
  conv_enhance      50     4    44   8.3%  +11.11%   -2.33%
  depthwise         55     1    47   2.1%   +4.35%   +0.53%
  norm             120     8   104   7.1%  +13.13%   +4.19%
Seed Scoreboard (env 0):
  Fossilized: 9 (+302.3K params, +319.0% of host)
  Culled: 113
  Avg fossilize age: 17.1 epochs
  Avg cull age: 6.3 epochs
  Compute cost: 2.69x baseline
  Distribution: attention x3, conv_enhance x4, norm x2
Seed Scoreboard (env 1):
  Fossilized: 8 (+7.6K params, +8.0% of host)
  Culled: 132
  Avg fossilize age: 14.4 epochs
  Avg cull age: 5.3 epochs
  Compute cost: 1.55x baseline
  Distribution: attention x1, norm x6, depthwise x1

[12:36:49] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[12:36:49] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[12:36:49] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[12:36:49] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[12:36:55] env0_seed_0 | Stage transition: TRAINING → BLENDING
[12:36:55] env1_seed_0 | Stage transition: TRAINING → BLENDING
[12:37:04] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[12:37:04] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[12:37:07] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[12:37:07] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[12:37:10] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[12:37:10] env0_seed_0 | Fossilized (norm, Δacc +16.48%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +16.48%)
[12:37:22] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[12:37:22] env1_seed_0 | Fossilized (norm, Δacc +23.36%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +23.36%)
Batch 21: Episodes 42/200
  Env accuracies: ['74.9%', '74.1%']
  Avg acc: 74.5% (rolling: 71.9%)
  Avg reward: 163.6
  Actions: {'WAIT': 22, 'GERMINATE_NORM': 33, 'GERMINATE_ATTENTION': 17, 'GERMINATE_DEPTHWISE': 19, 'GERMINATE_CONV_ENHANCE': 17, 'FOSSILIZE': 32, 'CULL': 10}
  Successful: {'WAIT': 22, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 10}
  Policy loss: 0.0059, Value loss: 128.6294, Entropy: 1.8769, Entropy coef: 0.1468
[12:38:54] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[12:38:54] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[12:38:54] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[12:38:54] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[12:39:01] env0_seed_0 | Stage transition: TRAINING → BLENDING
[12:39:01] env1_seed_0 | Stage transition: TRAINING → BLENDING
[12:39:09] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[12:39:09] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[12:39:13] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[12:39:13] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[12:39:14] env1_seed_0 | Stage transition: PROBATIONARY → CULLED
[12:39:14] env1_seed_0 | Culled (norm, Δacc +13.47%)
    [env1] Culled 'env1_seed_0' (norm, Δacc +13.47%)
[12:39:16] env1_seed_1 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_1' (attention, 2.0K params)
[12:39:16] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[12:39:18] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[12:39:18] env0_seed_0 | Fossilized (norm, Δacc +18.42%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +18.42%)
[12:39:23] env1_seed_1 | Stage transition: TRAINING → BLENDING
[12:39:31] env1_seed_1 | Stage transition: BLENDING → SHADOWING
[12:39:34] env1_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[12:39:43] env1_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[12:39:43] env1_seed_1 | Fossilized (attention, Δacc +5.52%)
    [env1] Fossilized 'env1_seed_1' (attention, Δacc +5.52%)
Batch 22: Episodes 44/200
  Env accuracies: ['75.9%', '70.3%']
  Avg acc: 73.1% (rolling: 72.1%)
  Avg reward: 157.3
  Actions: {'WAIT': 16, 'GERMINATE_NORM': 39, 'GERMINATE_ATTENTION': 18, 'GERMINATE_DEPTHWISE': 16, 'GERMINATE_CONV_ENHANCE': 30, 'FOSSILIZE': 19, 'CULL': 12}
  Successful: {'WAIT': 16, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 1, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 12}
  Policy loss: -0.0142, Value loss: 126.3785, Entropy: 1.8614, Entropy coef: 0.1443
[12:41:00] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[12:41:00] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[12:41:00] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[12:41:00] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[12:41:06] env0_seed_0 | Stage transition: TRAINING → BLENDING
[12:41:06] env1_seed_0 | Stage transition: TRAINING → BLENDING
[12:41:13] env1_seed_0 | Stage transition: BLENDING → CULLED
[12:41:13] env1_seed_0 | Culled (norm, Δacc +19.78%)
    [env1] Culled 'env1_seed_0' (norm, Δacc +19.78%)
[12:41:15] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[12:41:16] env0_seed_0 | Stage transition: SHADOWING → CULLED
[12:41:16] env0_seed_0 | Culled (norm, Δacc +11.15%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +11.15%)
[12:41:16] env1_seed_1 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_1' (norm, 0.1K params)
[12:41:16] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[12:41:18] env0_seed_1 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_1' (depthwise, 4.8K params)
[12:41:18] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[12:41:23] env1_seed_1 | Stage transition: TRAINING → BLENDING
[12:41:25] env0_seed_1 | Stage transition: TRAINING → BLENDING
[12:41:31] env1_seed_1 | Stage transition: BLENDING → SHADOWING
[12:41:31] env1_seed_1 | Stage transition: SHADOWING → CULLED
[12:41:31] env1_seed_1 | Culled (norm, Δacc +12.06%)
    [env1] Culled 'env1_seed_1' (norm, Δacc +12.06%)
[12:41:33] env0_seed_1 | Stage transition: BLENDING → SHADOWING
[12:41:33] env1_seed_2 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_2' (conv_enhance, 74.0K params)
[12:41:33] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[12:41:36] env0_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[12:41:40] env1_seed_2 | Stage transition: TRAINING → BLENDING
[12:41:40] env0_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[12:41:40] env0_seed_1 | Fossilized (depthwise, Δacc +3.96%)
    [env0] Fossilized 'env0_seed_1' (depthwise, Δacc +3.96%)
[12:41:44] env1_seed_2 | Stage transition: BLENDING → CULLED
[12:41:44] env1_seed_2 | Culled (conv_enhance, Δacc +2.32%)
    [env1] Culled 'env1_seed_2' (conv_enhance, Δacc +2.32%)
[12:41:47] env1_seed_3 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_3' (conv_enhance, 74.0K params)
[12:41:47] env1_seed_3 | Stage transition: GERMINATED → TRAINING
[12:41:54] env1_seed_3 | Stage transition: TRAINING → BLENDING
[12:42:01] env1_seed_3 | Stage transition: BLENDING → CULLED
[12:42:01] env1_seed_3 | Culled (conv_enhance, Δacc -2.16%)
    [env1] Culled 'env1_seed_3' (conv_enhance, Δacc -2.16%)
[12:42:06] env1_seed_4 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_4' (attention, 2.0K params)
[12:42:06] env1_seed_4 | Stage transition: GERMINATED → TRAINING
[12:42:08] env1_seed_4 | Stage transition: TRAINING → CULLED
[12:42:08] env1_seed_4 | Culled (attention, Δacc +0.00%)
    [env1] Culled 'env1_seed_4' (attention, Δacc +0.00%)
[12:42:10] env1_seed_5 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_5' (norm, 0.1K params)
[12:42:10] env1_seed_5 | Stage transition: GERMINATED → TRAINING
[12:42:13] env1_seed_5 | Stage transition: TRAINING → CULLED
[12:42:13] env1_seed_5 | Culled (norm, Δacc -5.04%)
    [env1] Culled 'env1_seed_5' (norm, Δacc -5.04%)
[12:42:16] env1_seed_6 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_6' (norm, 0.1K params)
[12:42:16] env1_seed_6 | Stage transition: GERMINATED → TRAINING
[12:42:23] env1_seed_6 | Stage transition: TRAINING → BLENDING
[12:42:28] env1_seed_6 | Stage transition: BLENDING → CULLED
[12:42:28] env1_seed_6 | Culled (norm, Δacc +4.47%)
    [env1] Culled 'env1_seed_6' (norm, Δacc +4.47%)
[12:42:30] env1_seed_7 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_7' (depthwise, 4.8K params)
[12:42:30] env1_seed_7 | Stage transition: GERMINATED → TRAINING
[12:42:42] env1_seed_7 | Stage transition: TRAINING → CULLED
[12:42:42] env1_seed_7 | Culled (depthwise, Δacc -0.24%)
    [env1] Culled 'env1_seed_7' (depthwise, Δacc -0.24%)
[12:42:45] env1_seed_8 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_8' (norm, 0.1K params)
[12:42:45] env1_seed_8 | Stage transition: GERMINATED → TRAINING
[12:42:50] env1_seed_8 | Stage transition: TRAINING → CULLED
[12:42:50] env1_seed_8 | Culled (norm, Δacc +2.53%)
    [env1] Culled 'env1_seed_8' (norm, Δacc +2.53%)
[12:42:52] env1_seed_9 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_9' (attention, 2.0K params)
[12:42:52] env1_seed_9 | Stage transition: GERMINATED → TRAINING
[12:42:58] env1_seed_9 | Stage transition: TRAINING → BLENDING
[12:43:03] env1_seed_9 | Stage transition: BLENDING → CULLED
[12:43:03] env1_seed_9 | Culled (attention, Δacc +5.16%)
    [env1] Culled 'env1_seed_9' (attention, Δacc +5.16%)
Batch 23: Episodes 46/200
  Env accuracies: ['67.8%', '74.3%']
  Avg acc: 71.1% (rolling: 72.3%)
  Avg reward: 130.8
  Actions: {'WAIT': 20, 'GERMINATE_NORM': 12, 'GERMINATE_ATTENTION': 15, 'GERMINATE_DEPTHWISE': 31, 'GERMINATE_CONV_ENHANCE': 28, 'FOSSILIZE': 25, 'CULL': 19}
  Successful: {'WAIT': 20, 'GERMINATE_NORM': 6, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 2, 'GERMINATE_CONV_ENHANCE': 2, 'FOSSILIZE': 1, 'CULL': 16}
  Policy loss: 0.0093, Value loss: 96.9917, Entropy: 1.8818, Entropy coef: 0.1417
[12:43:07] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[12:43:07] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[12:43:07] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[12:43:07] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[12:43:10] env1_seed_0 | Stage transition: TRAINING → CULLED
[12:43:10] env1_seed_0 | Culled (norm, Δacc +2.66%)
    [env1] Culled 'env1_seed_0' (norm, Δacc +2.66%)
[12:43:12] env1_seed_1 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_1' (norm, 0.1K params)
[12:43:12] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[12:43:13] env0_seed_0 | Stage transition: TRAINING → BLENDING
[12:43:18] env1_seed_1 | Stage transition: TRAINING → BLENDING
[12:43:22] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[12:43:25] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[12:43:27] env1_seed_1 | Stage transition: BLENDING → SHADOWING
[12:43:28] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[12:43:28] env0_seed_0 | Fossilized (norm, Δacc +21.33%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +21.33%)
[12:43:30] env1_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[12:43:32] env1_seed_1 | Stage transition: PROBATIONARY → CULLED
[12:43:32] env1_seed_1 | Culled (norm, Δacc +11.17%)
    [env1] Culled 'env1_seed_1' (norm, Δacc +11.17%)
[12:43:33] env1_seed_2 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_2' (attention, 2.0K params)
[12:43:33] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[12:43:38] env1_seed_2 | Stage transition: TRAINING → CULLED
[12:43:38] env1_seed_2 | Culled (attention, Δacc -3.47%)
    [env1] Culled 'env1_seed_2' (attention, Δacc -3.47%)
[12:43:40] env1_seed_3 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_3' (norm, 0.1K params)
[12:43:40] env1_seed_3 | Stage transition: GERMINATED → TRAINING
[12:43:48] env1_seed_3 | Stage transition: TRAINING → BLENDING
[12:43:57] env1_seed_3 | Stage transition: BLENDING → SHADOWING
[12:44:00] env1_seed_3 | Stage transition: SHADOWING → PROBATIONARY
[12:44:00] env1_seed_3 | Stage transition: PROBATIONARY → FOSSILIZED
[12:44:00] env1_seed_3 | Fossilized (norm, Δacc +6.36%)
    [env1] Fossilized 'env1_seed_3' (norm, Δacc +6.36%)
Batch 24: Episodes 48/200
  Env accuracies: ['75.8%', '76.2%']
  Avg acc: 76.0% (rolling: 72.4%)
  Avg reward: 166.3
  Actions: {'WAIT': 21, 'GERMINATE_NORM': 25, 'GERMINATE_ATTENTION': 27, 'GERMINATE_DEPTHWISE': 23, 'GERMINATE_CONV_ENHANCE': 18, 'FOSSILIZE': 28, 'CULL': 8}
  Successful: {'WAIT': 21, 'GERMINATE_NORM': 4, 'GERMINATE_ATTENTION': 1, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 8}
  Policy loss: 0.0092, Value loss: 90.7794, Entropy: 1.8768, Entropy coef: 0.1392
[12:45:12] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[12:45:12] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[12:45:12] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[12:45:12] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[12:45:17] env0_seed_0 | Stage transition: TRAINING → CULLED
[12:45:17] env0_seed_0 | Culled (norm, Δacc +11.93%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +11.93%)
[12:45:19] env1_seed_0 | Stage transition: TRAINING → BLENDING
[12:45:19] env0_seed_1 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_1' (attention, 2.0K params)
[12:45:19] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[12:45:22] env1_seed_0 | Stage transition: BLENDING → CULLED
[12:45:22] env1_seed_0 | Culled (norm, Δacc +15.80%)
    [env1] Culled 'env1_seed_0' (norm, Δacc +15.80%)
[12:45:24] env1_seed_1 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_1' (norm, 0.1K params)
[12:45:24] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[12:45:25] env0_seed_1 | Stage transition: TRAINING → BLENDING
[12:45:34] env0_seed_1 | Stage transition: BLENDING → SHADOWING
[12:45:35] env1_seed_1 | Stage transition: TRAINING → BLENDING
[12:45:37] env0_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[12:45:39] env0_seed_1 | Stage transition: PROBATIONARY → CULLED
[12:45:39] env0_seed_1 | Culled (attention, Δacc +5.18%)
    [env0] Culled 'env0_seed_1' (attention, Δacc +5.18%)
[12:45:39] env1_seed_1 | Stage transition: BLENDING → CULLED
[12:45:39] env1_seed_1 | Culled (norm, Δacc +4.98%)
    [env1] Culled 'env1_seed_1' (norm, Δacc +4.98%)
[12:45:40] env0_seed_2 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_2' (attention, 2.0K params)
[12:45:40] env1_seed_2 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_2' (depthwise, 4.8K params)
[12:45:40] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[12:45:40] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[12:45:42] env0_seed_2 | Stage transition: TRAINING → CULLED
[12:45:42] env0_seed_2 | Culled (attention, Δacc +0.00%)
    [env0] Culled 'env0_seed_2' (attention, Δacc +0.00%)
[12:45:44] env0_seed_3 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_3' (attention, 2.0K params)
[12:45:44] env0_seed_3 | Stage transition: GERMINATED → TRAINING
[12:45:47] env1_seed_2 | Stage transition: TRAINING → BLENDING
[12:45:51] env0_seed_3 | Stage transition: TRAINING → BLENDING
[12:45:51] env0_seed_3 | Stage transition: BLENDING → CULLED
[12:45:51] env0_seed_3 | Culled (attention, Δacc +0.73%)
    [env0] Culled 'env0_seed_3' (attention, Δacc +0.73%)
[12:45:54] env0_seed_4 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_4' (norm, 0.1K params)
[12:45:54] env0_seed_4 | Stage transition: GERMINATED → TRAINING
[12:45:56] env1_seed_2 | Stage transition: BLENDING → SHADOWING
[12:46:00] env1_seed_2 | Stage transition: SHADOWING → PROBATIONARY
[12:46:00] env0_seed_4 | Stage transition: TRAINING → CULLED
[12:46:00] env0_seed_4 | Culled (norm, Δacc +0.56%)
    [env0] Culled 'env0_seed_4' (norm, Δacc +0.56%)
[12:46:03] env1_seed_2 | Stage transition: PROBATIONARY → CULLED
[12:46:03] env1_seed_2 | Culled (depthwise, Δacc +0.03%)
    [env1] Culled 'env1_seed_2' (depthwise, Δacc +0.03%)
[12:46:05] env1_seed_3 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_3' (conv_enhance, 74.0K params)
[12:46:05] env1_seed_3 | Stage transition: GERMINATED → TRAINING
[12:46:06] env0_seed_5 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_5' (conv_enhance, 74.0K params)
[12:46:07] env0_seed_5 | Stage transition: GERMINATED → TRAINING
[12:46:10] env1_seed_3 | Stage transition: TRAINING → CULLED
[12:46:10] env1_seed_3 | Culled (conv_enhance, Δacc -2.59%)
    [env1] Culled 'env1_seed_3' (conv_enhance, Δacc -2.59%)
[12:46:13] env0_seed_5 | Stage transition: TRAINING → BLENDING
[12:46:13] env1_seed_4 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_4' (conv_enhance, 74.0K params)
[12:46:14] env1_seed_4 | Stage transition: GERMINATED → TRAINING
[12:46:15] env1_seed_4 | Stage transition: TRAINING → CULLED
[12:46:15] env1_seed_4 | Culled (conv_enhance, Δacc +0.00%)
    [env1] Culled 'env1_seed_4' (conv_enhance, Δacc +0.00%)
[12:46:17] env1_seed_5 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_5' (attention, 2.0K params)
[12:46:17] env1_seed_5 | Stage transition: GERMINATED → TRAINING
[12:46:22] env0_seed_5 | Stage transition: BLENDING → SHADOWING
[12:46:24] env1_seed_5 | Stage transition: TRAINING → BLENDING
[12:46:25] env0_seed_5 | Stage transition: SHADOWING → PROBATIONARY
[12:46:32] env1_seed_5 | Stage transition: BLENDING → SHADOWING
[12:46:35] env1_seed_5 | Stage transition: SHADOWING → PROBATIONARY
[12:46:37] env1_seed_5 | Stage transition: PROBATIONARY → CULLED
[12:46:37] env1_seed_5 | Culled (attention, Δacc -5.75%)
    [env1] Culled 'env1_seed_5' (attention, Δacc -5.75%)
[12:46:39] env0_seed_5 | Stage transition: PROBATIONARY → FOSSILIZED
[12:46:39] env0_seed_5 | Fossilized (conv_enhance, Δacc +0.88%)
    [env0] Fossilized 'env0_seed_5' (conv_enhance, Δacc +0.88%)
[12:46:39] env1_seed_6 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_6' (attention, 2.0K params)
[12:46:39] env1_seed_6 | Stage transition: GERMINATED → TRAINING
[12:46:46] env1_seed_6 | Stage transition: TRAINING → CULLED
[12:46:46] env1_seed_6 | Culled (attention, Δacc -0.66%)
    [env1] Culled 'env1_seed_6' (attention, Δacc -0.66%)
[12:46:47] env1_seed_7 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_7' (norm, 0.1K params)
[12:46:47] env1_seed_7 | Stage transition: GERMINATED → TRAINING
[12:47:08] env1_seed_7 | Stage transition: TRAINING → BLENDING
[12:47:08] env1_seed_7 | Stage transition: BLENDING → CULLED
[12:47:08] env1_seed_7 | Culled (norm, Δacc -1.22%)
    [env1] Culled 'env1_seed_7' (norm, Δacc -1.22%)
[12:47:11] env1_seed_8 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_8' (norm, 0.1K params)
[12:47:11] env1_seed_8 | Stage transition: GERMINATED → TRAINING
Batch 25: Episodes 50/200
  Env accuracies: ['64.5%', '71.6%']
  Avg acc: 68.0% (rolling: 71.7%)
  Avg reward: 86.6
  Actions: {'WAIT': 18, 'GERMINATE_NORM': 32, 'GERMINATE_ATTENTION': 23, 'GERMINATE_DEPTHWISE': 23, 'GERMINATE_CONV_ENHANCE': 15, 'FOSSILIZE': 21, 'CULL': 18}
  Successful: {'WAIT': 18, 'GERMINATE_NORM': 6, 'GERMINATE_ATTENTION': 5, 'GERMINATE_DEPTHWISE': 1, 'GERMINATE_CONV_ENHANCE': 3, 'FOSSILIZE': 1, 'CULL': 18}
  Policy loss: 0.0052, Value loss: 206.0910, Entropy: 1.8716, Entropy coef: 0.1367

Blueprint Stats
  ---------------------------------------------------------------------------

Blueprint       Germ  Foss  Cull   Rate     ΔAcc    Churn
  ---------------------------------------------------------------------------

  attention         65     5    58   7.9%   +2.41%   +0.20%
  conv_enhance      55     5    48   9.4%   +9.06%   -2.19%
  depthwise         58     2    49   3.9%   +4.15%   +0.51%
  norm             140    13   118   9.9%  +14.69%   +4.58%
Seed Scoreboard (env 0):
  Fossilized: 14 (+381.5K params, +402.6% of host)
  Culled: 119
  Avg fossilize age: 16.1 epochs
  Avg cull age: 6.3 epochs
  Compute cost: 2.98x baseline
  Distribution: attention x3, conv_enhance x5, norm x5, depthwise x1
Seed Scoreboard (env 1):
  Fossilized: 11 (+9.9K params, +10.5% of host)
  Culled: 154
  Avg fossilize age: 14.8 epochs
  Avg cull age: 5.5 epochs
  Compute cost: 1.94x baseline
  Distribution: attention x2, norm x8, depthwise x1

[12:47:20] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[12:47:20] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[12:47:20] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[12:47:20] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[12:47:26] env0_seed_0 | Stage transition: TRAINING → BLENDING
[12:47:26] env1_seed_0 | Stage transition: TRAINING → BLENDING
[12:47:31] env1_seed_0 | Stage transition: BLENDING → CULLED
[12:47:31] env1_seed_0 | Culled (norm, Δacc +12.24%)
    [env1] Culled 'env1_seed_0' (norm, Δacc +12.24%)
[12:47:33] env1_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_1' (conv_enhance, 74.0K params)
[12:47:33] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[12:47:35] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[12:47:38] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[12:47:40] env1_seed_1 | Stage transition: TRAINING → BLENDING
[12:47:40] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[12:47:40] env0_seed_0 | Fossilized (norm, Δacc +23.76%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +23.76%)
[12:47:49] env1_seed_1 | Stage transition: BLENDING → SHADOWING
[12:47:51] env1_seed_1 | Stage transition: SHADOWING → CULLED
[12:47:51] env1_seed_1 | Culled (conv_enhance, Δacc +10.92%)
    [env1] Culled 'env1_seed_1' (conv_enhance, Δacc +10.92%)
[12:47:53] env1_seed_2 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_2' (attention, 2.0K params)
[12:47:53] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[12:48:04] env1_seed_2 | Stage transition: TRAINING → BLENDING
[12:48:13] env1_seed_2 | Stage transition: BLENDING → SHADOWING
[12:48:16] env1_seed_2 | Stage transition: SHADOWING → PROBATIONARY
[12:48:16] env1_seed_2 | Stage transition: PROBATIONARY → CULLED
[12:48:16] env1_seed_2 | Culled (attention, Δacc +0.23%)
    [env1] Culled 'env1_seed_2' (attention, Δacc +0.23%)
[12:48:18] env1_seed_3 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_3' (conv_enhance, 74.0K params)
[12:48:18] env1_seed_3 | Stage transition: GERMINATED → TRAINING
[12:48:21] env1_seed_3 | Stage transition: TRAINING → CULLED
[12:48:21] env1_seed_3 | Culled (conv_enhance, Δacc -0.49%)
    [env1] Culled 'env1_seed_3' (conv_enhance, Δacc -0.49%)
[12:48:25] env1_seed_4 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_4' (norm, 0.1K params)
[12:48:25] env1_seed_4 | Stage transition: GERMINATED → TRAINING
[12:48:31] env1_seed_4 | Stage transition: TRAINING → BLENDING
[12:48:40] env1_seed_4 | Stage transition: BLENDING → SHADOWING
[12:48:43] env1_seed_4 | Stage transition: SHADOWING → PROBATIONARY
[12:48:45] env1_seed_4 | Stage transition: PROBATIONARY → FOSSILIZED
[12:48:45] env1_seed_4 | Fossilized (norm, Δacc +8.02%)
    [env1] Fossilized 'env1_seed_4' (norm, Δacc +8.02%)
Batch 26: Episodes 52/200
  Env accuracies: ['75.7%', '76.5%']
  Avg acc: 76.1% (rolling: 72.1%)
  Avg reward: 154.5
  Actions: {'WAIT': 20, 'GERMINATE_NORM': 24, 'GERMINATE_ATTENTION': 25, 'GERMINATE_DEPTHWISE': 25, 'GERMINATE_CONV_ENHANCE': 24, 'FOSSILIZE': 16, 'CULL': 16}
  Successful: {'WAIT': 20, 'GERMINATE_NORM': 3, 'GERMINATE_ATTENTION': 1, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 2, 'FOSSILIZE': 2, 'CULL': 16}
  Policy loss: -0.0229, Value loss: 97.0650, Entropy: 1.8795, Entropy coef: 0.1341
[12:49:27] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[12:49:27] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[12:49:27] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[12:49:27] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[12:49:33] env0_seed_0 | Stage transition: TRAINING → BLENDING
[12:49:33] env1_seed_0 | Stage transition: TRAINING → BLENDING
[12:49:33] env0_seed_0 | Stage transition: BLENDING → CULLED
[12:49:33] env0_seed_0 | Culled (norm, Δacc +13.61%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +13.61%)
[12:49:37] env0_seed_1 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_1' (norm, 0.1K params)
[12:49:37] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[12:49:40] env0_seed_1 | Stage transition: TRAINING → CULLED
[12:49:40] env0_seed_1 | Culled (norm, Δacc -1.28%)
    [env0] Culled 'env0_seed_1' (norm, Δacc -1.28%)
[12:49:42] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[12:49:42] env0_seed_2 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_2' (attention, 2.0K params)
[12:49:42] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[12:49:43] env0_seed_2 | Stage transition: TRAINING → CULLED
[12:49:43] env0_seed_2 | Culled (attention, Δacc +0.00%)
    [env0] Culled 'env0_seed_2' (attention, Δacc +0.00%)
[12:49:45] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[12:49:45] env0_seed_3 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_3' (norm, 0.1K params)
[12:49:45] env1_seed_0 | Stage transition: PROBATIONARY → CULLED
[12:49:45] env1_seed_0 | Culled (norm, Δacc +19.27%)
    [env1] Culled 'env1_seed_0' (norm, Δacc +19.27%)
[12:49:45] env0_seed_3 | Stage transition: GERMINATED → TRAINING
[12:49:50] env1_seed_1 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_1' (depthwise, 4.8K params)
[12:49:50] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[12:49:57] env1_seed_1 | Stage transition: TRAINING → BLENDING
[12:50:04] env0_seed_3 | Stage transition: TRAINING → BLENDING
[12:50:06] env1_seed_1 | Stage transition: BLENDING → SHADOWING
[12:50:09] env1_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[12:50:11] env1_seed_1 | Stage transition: PROBATIONARY → CULLED
[12:50:11] env1_seed_1 | Culled (depthwise, Δacc -1.64%)
    [env1] Culled 'env1_seed_1' (depthwise, Δacc -1.64%)
[12:50:13] env0_seed_3 | Stage transition: BLENDING → SHADOWING
[12:50:13] env1_seed_2 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_2' (attention, 2.0K params)
[12:50:13] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[12:50:14] env0_seed_3 | Stage transition: SHADOWING → CULLED
[12:50:14] env0_seed_3 | Culled (norm, Δacc +2.76%)
    [env0] Culled 'env0_seed_3' (norm, Δacc +2.76%)
[12:50:18] env0_seed_4 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_4' (norm, 0.1K params)
[12:50:18] env0_seed_4 | Stage transition: GERMINATED → TRAINING
[12:50:19] env1_seed_2 | Stage transition: TRAINING → BLENDING
[12:50:21] env1_seed_2 | Stage transition: BLENDING → CULLED
[12:50:21] env1_seed_2 | Culled (attention, Δacc +5.89%)
    [env1] Culled 'env1_seed_2' (attention, Δacc +5.89%)
[12:50:24] env1_seed_3 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_3' (depthwise, 4.8K params)
[12:50:24] env1_seed_3 | Stage transition: GERMINATED → TRAINING
[12:50:28] env0_seed_4 | Stage transition: TRAINING → BLENDING
[12:50:37] env0_seed_4 | Stage transition: BLENDING → SHADOWING
[12:50:38] env1_seed_3 | Stage transition: TRAINING → BLENDING
[12:50:40] env0_seed_4 | Stage transition: SHADOWING → PROBATIONARY
[12:50:42] env1_seed_3 | Stage transition: BLENDING → CULLED
[12:50:42] env1_seed_3 | Culled (depthwise, Δacc +1.74%)
    [env1] Culled 'env1_seed_3' (depthwise, Δacc +1.74%)
[12:50:43] env1_seed_4 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_4' (norm, 0.1K params)
[12:50:43] env1_seed_4 | Stage transition: GERMINATED → TRAINING
[12:50:45] env0_seed_4 | Stage transition: PROBATIONARY → CULLED
[12:50:45] env0_seed_4 | Culled (norm, Δacc +4.98%)
    [env0] Culled 'env0_seed_4' (norm, Δacc +4.98%)
[12:50:47] env1_seed_4 | Stage transition: TRAINING → CULLED
[12:50:47] env1_seed_4 | Culled (norm, Δacc +1.54%)
    [env1] Culled 'env1_seed_4' (norm, Δacc +1.54%)
[12:50:48] env1_seed_5 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_5' (conv_enhance, 74.0K params)
[12:50:48] env1_seed_5 | Stage transition: GERMINATED → TRAINING
[12:50:52] env0_seed_5 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_5' (attention, 2.0K params)
[12:50:52] env0_seed_5 | Stage transition: GERMINATED → TRAINING
[12:50:56] env1_seed_5 | Stage transition: TRAINING → BLENDING
[12:50:56] env0_seed_5 | Stage transition: TRAINING → CULLED
[12:50:56] env0_seed_5 | Culled (attention, Δacc -4.63%)
    [env0] Culled 'env0_seed_5' (attention, Δacc -4.63%)
[12:50:57] env0_seed_6 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_6' (norm, 0.1K params)
[12:50:57] env0_seed_6 | Stage transition: GERMINATED → TRAINING
[12:51:03] env0_seed_6 | Stage transition: TRAINING → CULLED
[12:51:03] env0_seed_6 | Culled (norm, Δacc -2.90%)
    [env0] Culled 'env0_seed_6' (norm, Δacc -2.90%)
[12:51:05] env1_seed_5 | Stage transition: BLENDING → SHADOWING
[12:51:05] env0_seed_7 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_7' (depthwise, 4.8K params)
[12:51:05] env0_seed_7 | Stage transition: GERMINATED → TRAINING
[12:51:08] env1_seed_5 | Stage transition: SHADOWING → PROBATIONARY
[12:51:08] env0_seed_7 | Stage transition: TRAINING → CULLED
[12:51:08] env0_seed_7 | Culled (depthwise, Δacc -1.39%)
    [env0] Culled 'env0_seed_7' (depthwise, Δacc -1.39%)
[12:51:10] env1_seed_5 | Stage transition: PROBATIONARY → CULLED
[12:51:10] env1_seed_5 | Culled (conv_enhance, Δacc -8.81%)
    [env1] Culled 'env1_seed_5' (conv_enhance, Δacc -8.81%)
[12:51:12] env1_seed_6 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_6' (norm, 0.1K params)
[12:51:12] env1_seed_6 | Stage transition: GERMINATED → TRAINING
[12:51:13] env0_seed_8 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_8' (norm, 0.1K params)
[12:51:14] env0_seed_8 | Stage transition: GERMINATED → TRAINING
[12:51:17] env0_seed_8 | Stage transition: TRAINING → CULLED
[12:51:17] env0_seed_8 | Culled (norm, Δacc +0.32%)
    [env0] Culled 'env0_seed_8' (norm, Δacc +0.32%)
[12:51:18] env1_seed_6 | Stage transition: TRAINING → BLENDING
[12:51:18] env0_seed_9 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_9' (norm, 0.1K params)
[12:51:18] env1_seed_6 | Stage transition: BLENDING → CULLED
[12:51:18] env1_seed_6 | Culled (norm, Δacc -4.24%)
    [env1] Culled 'env1_seed_6' (norm, Δacc -4.24%)
[12:51:18] env0_seed_9 | Stage transition: GERMINATED → TRAINING
[12:51:22] env1_seed_7 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_7' (conv_enhance, 74.0K params)
[12:51:22] env1_seed_7 | Stage transition: GERMINATED → TRAINING
[12:51:33] env1_seed_7 | Stage transition: TRAINING → BLENDING
Batch 27: Episodes 54/200
  Env accuracies: ['73.7%', '73.6%']
  Avg acc: 73.7% (rolling: 73.0%)
  Avg reward: 131.6
  Actions: {'WAIT': 22, 'GERMINATE_NORM': 31, 'GERMINATE_ATTENTION': 21, 'GERMINATE_DEPTHWISE': 26, 'GERMINATE_CONV_ENHANCE': 15, 'FOSSILIZE': 17, 'CULL': 18}
  Successful: {'WAIT': 22, 'GERMINATE_NORM': 10, 'GERMINATE_ATTENTION': 3, 'GERMINATE_DEPTHWISE': 3, 'GERMINATE_CONV_ENHANCE': 2, 'FOSSILIZE': 0, 'CULL': 16}
  Policy loss: -0.0167, Value loss: 90.2091, Entropy: 1.8693, Entropy coef: 0.1316
[12:51:36] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[12:51:36] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[12:51:36] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[12:51:36] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[12:51:43] env0_seed_0 | Stage transition: TRAINING → BLENDING
[12:51:43] env1_seed_0 | Stage transition: TRAINING → BLENDING
[12:51:51] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[12:51:51] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[12:51:55] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[12:51:55] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[12:51:55] env0_seed_0 | Stage transition: PROBATIONARY → CULLED
[12:51:55] env0_seed_0 | Culled (norm, Δacc +17.41%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +17.41%)
[12:51:56] env0_seed_1 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_1' (norm, 0.1K params)
[12:51:56] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[12:52:00] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[12:52:00] env1_seed_0 | Fossilized (norm, Δacc +19.70%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +19.70%)
[12:52:06] env0_seed_1 | Stage transition: TRAINING → BLENDING
[12:52:06] env0_seed_1 | Stage transition: BLENDING → CULLED
[12:52:06] env0_seed_1 | Culled (norm, Δacc -0.70%)
    [env0] Culled 'env0_seed_1' (norm, Δacc -0.70%)
[12:52:11] env0_seed_2 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_2' (depthwise, 4.8K params)
[12:52:12] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[12:52:16] env0_seed_2 | Stage transition: TRAINING → CULLED
[12:52:16] env0_seed_2 | Culled (depthwise, Δacc -5.05%)
    [env0] Culled 'env0_seed_2' (depthwise, Δacc -5.05%)
[12:52:18] env0_seed_3 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_3' (depthwise, 4.8K params)
[12:52:18] env0_seed_3 | Stage transition: GERMINATED → TRAINING
[12:52:25] env0_seed_3 | Stage transition: TRAINING → BLENDING
[12:52:33] env0_seed_3 | Stage transition: BLENDING → SHADOWING
[12:52:37] env0_seed_3 | Stage transition: SHADOWING → PROBATIONARY
[12:52:40] env0_seed_3 | Stage transition: PROBATIONARY → CULLED
[12:52:40] env0_seed_3 | Culled (depthwise, Δacc +0.53%)
    [env0] Culled 'env0_seed_3' (depthwise, Δacc +0.53%)
[12:52:42] env0_seed_4 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_4' (depthwise, 4.8K params)
[12:52:42] env0_seed_4 | Stage transition: GERMINATED → TRAINING
[12:52:50] env0_seed_4 | Stage transition: TRAINING → BLENDING
[12:52:55] env0_seed_4 | Stage transition: BLENDING → CULLED
[12:52:55] env0_seed_4 | Culled (depthwise, Δacc -1.14%)
    [env0] Culled 'env0_seed_4' (depthwise, Δacc -1.14%)
[12:53:00] env0_seed_5 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_5' (attention, 2.0K params)
[12:53:00] env0_seed_5 | Stage transition: GERMINATED → TRAINING
[12:53:03] env0_seed_5 | Stage transition: TRAINING → CULLED
[12:53:03] env0_seed_5 | Culled (attention, Δacc -1.54%)
    [env0] Culled 'env0_seed_5' (attention, Δacc -1.54%)
[12:53:05] env0_seed_6 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_6' (depthwise, 4.8K params)
[12:53:05] env0_seed_6 | Stage transition: GERMINATED → TRAINING
[12:53:25] env0_seed_6 | Stage transition: TRAINING → BLENDING
[12:53:34] env0_seed_6 | Stage transition: BLENDING → SHADOWING
[12:53:37] env0_seed_6 | Stage transition: SHADOWING → PROBATIONARY
Batch 28: Episodes 56/200
  Env accuracies: ['57.2%', '75.9%']
  Avg acc: 66.5% (rolling: 72.5%)
  Avg reward: 130.9
  Actions: {'WAIT': 12, 'GERMINATE_NORM': 39, 'GERMINATE_ATTENTION': 18, 'GERMINATE_DEPTHWISE': 16, 'GERMINATE_CONV_ENHANCE': 17, 'FOSSILIZE': 30, 'CULL': 18}
  Successful: {'WAIT': 12, 'GERMINATE_NORM': 3, 'GERMINATE_ATTENTION': 1, 'GERMINATE_DEPTHWISE': 4, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 1, 'CULL': 18}
  Policy loss: -0.0051, Value loss: 88.0894, Entropy: 1.8831, Entropy coef: 0.1291
[12:53:42] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[12:53:42] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[12:53:42] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[12:53:42] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[12:53:47] env0_seed_0 | Stage transition: TRAINING → CULLED
[12:53:47] env0_seed_0 | Culled (norm, Δacc +1.70%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +1.70%)
[12:53:49] env0_seed_1 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_1' (norm, 0.1K params)
[12:53:49] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[12:53:50] env1_seed_0 | Stage transition: TRAINING → BLENDING
[12:53:55] env0_seed_1 | Stage transition: TRAINING → BLENDING
[12:53:59] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[12:54:00] env1_seed_0 | Stage transition: SHADOWING → CULLED
[12:54:00] env1_seed_0 | Culled (norm, Δacc +11.44%)
    [env1] Culled 'env1_seed_0' (norm, Δacc +11.44%)
[12:54:02] env1_seed_1 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_1' (attention, 2.0K params)
[12:54:02] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[12:54:04] env0_seed_1 | Stage transition: BLENDING → SHADOWING
[12:54:05] env0_seed_1 | Stage transition: SHADOWING → CULLED
[12:54:05] env0_seed_1 | Culled (norm, Δacc +7.89%)
    [env0] Culled 'env0_seed_1' (norm, Δacc +7.89%)
[12:54:05] env1_seed_1 | Stage transition: TRAINING → CULLED
[12:54:05] env1_seed_1 | Culled (attention, Δacc +2.66%)
    [env1] Culled 'env1_seed_1' (attention, Δacc +2.66%)
[12:54:07] env0_seed_2 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_2' (conv_enhance, 74.0K params)
[12:54:07] env1_seed_2 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_2' (conv_enhance, 74.0K params)
[12:54:07] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[12:54:07] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[12:54:14] env1_seed_2 | Stage transition: TRAINING → BLENDING
[12:54:14] env1_seed_2 | Stage transition: BLENDING → CULLED
[12:54:14] env1_seed_2 | Culled (conv_enhance, Δacc +1.48%)
    [env1] Culled 'env1_seed_2' (conv_enhance, Δacc +1.48%)
[12:54:16] env0_seed_2 | Stage transition: TRAINING → CULLED
[12:54:16] env0_seed_2 | Culled (conv_enhance, Δacc -2.20%)
    [env0] Culled 'env0_seed_2' (conv_enhance, Δacc -2.20%)
[12:54:18] env0_seed_3 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_3' (norm, 0.1K params)
[12:54:18] env0_seed_3 | Stage transition: GERMINATED → TRAINING
[12:54:21] env1_seed_3 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_3' (depthwise, 4.8K params)
[12:54:21] env1_seed_3 | Stage transition: GERMINATED → TRAINING
[12:54:23] env0_seed_3 | Stage transition: TRAINING → CULLED
[12:54:23] env0_seed_3 | Culled (norm, Δacc -0.34%)
    [env0] Culled 'env0_seed_3' (norm, Δacc -0.34%)
[12:54:24] env0_seed_4 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_4' (norm, 0.1K params)
[12:54:24] env0_seed_4 | Stage transition: GERMINATED → TRAINING
[12:54:28] env1_seed_3 | Stage transition: TRAINING → CULLED
[12:54:28] env1_seed_3 | Culled (depthwise, Δacc +0.53%)
    [env1] Culled 'env1_seed_3' (depthwise, Δacc +0.53%)
[12:54:29] env0_seed_4 | Stage transition: TRAINING → CULLED
[12:54:29] env0_seed_4 | Culled (norm, Δacc -7.25%)
    [env0] Culled 'env0_seed_4' (norm, Δacc -7.25%)
[12:54:31] env0_seed_5 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_5' (norm, 0.1K params)
[12:54:31] env0_seed_5 | Stage transition: GERMINATED → TRAINING
[12:54:33] env1_seed_4 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_4' (attention, 2.0K params)
[12:54:33] env1_seed_4 | Stage transition: GERMINATED → TRAINING
[12:54:38] env0_seed_5 | Stage transition: TRAINING → BLENDING
[12:54:39] env1_seed_4 | Stage transition: TRAINING → BLENDING
[12:54:46] env0_seed_5 | Stage transition: BLENDING → SHADOWING
[12:54:48] env1_seed_4 | Stage transition: BLENDING → SHADOWING
[12:54:49] env0_seed_5 | Stage transition: SHADOWING → PROBATIONARY
[12:54:51] env1_seed_4 | Stage transition: SHADOWING → PROBATIONARY
[12:54:54] env1_seed_4 | Stage transition: PROBATIONARY → CULLED
[12:54:54] env1_seed_4 | Culled (attention, Δacc -2.27%)
    [env1] Culled 'env1_seed_4' (attention, Δacc -2.27%)
[12:54:56] env1_seed_5 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_5' (norm, 0.1K params)
[12:54:56] env1_seed_5 | Stage transition: GERMINATED → TRAINING
[12:55:01] env1_seed_5 | Stage transition: TRAINING → CULLED
[12:55:01] env1_seed_5 | Culled (norm, Δacc +8.34%)
    [env1] Culled 'env1_seed_5' (norm, Δacc +8.34%)
[12:55:03] env1_seed_6 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_6' (norm, 0.1K params)
[12:55:03] env1_seed_6 | Stage transition: GERMINATED → TRAINING
[12:55:04] env0_seed_5 | Stage transition: PROBATIONARY → FOSSILIZED
[12:55:04] env0_seed_5 | Fossilized (norm, Δacc +6.55%)
    [env0] Fossilized 'env0_seed_5' (norm, Δacc +6.55%)
[12:55:11] env1_seed_6 | Stage transition: TRAINING → BLENDING
[12:55:14] env1_seed_6 | Stage transition: BLENDING → CULLED
[12:55:14] env1_seed_6 | Culled (norm, Δacc +2.48%)
    [env1] Culled 'env1_seed_6' (norm, Δacc +2.48%)
[12:55:18] env1_seed_7 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_7' (attention, 2.0K params)
[12:55:18] env1_seed_7 | Stage transition: GERMINATED → TRAINING
[12:55:24] env1_seed_7 | Stage transition: TRAINING → BLENDING
[12:55:33] env1_seed_7 | Stage transition: BLENDING → SHADOWING
[12:55:36] env1_seed_7 | Stage transition: SHADOWING → PROBATIONARY
[12:55:36] env1_seed_7 | Stage transition: PROBATIONARY → CULLED
[12:55:36] env1_seed_7 | Culled (attention, Δacc +5.73%)
    [env1] Culled 'env1_seed_7' (attention, Δacc +5.73%)
[12:55:38] env1_seed_8 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_8' (norm, 0.1K params)
[12:55:38] env1_seed_8 | Stage transition: GERMINATED → TRAINING
[12:55:44] env1_seed_8 | Stage transition: TRAINING → BLENDING
Batch 29: Episodes 58/200
  Env accuracies: ['74.4%', '68.1%']
  Avg acc: 71.2% (rolling: 72.6%)
  Avg reward: 132.6
  Actions: {'WAIT': 15, 'GERMINATE_NORM': 35, 'GERMINATE_ATTENTION': 19, 'GERMINATE_DEPTHWISE': 24, 'GERMINATE_CONV_ENHANCE': 22, 'FOSSILIZE': 18, 'CULL': 17}
  Successful: {'WAIT': 15, 'GERMINATE_NORM': 9, 'GERMINATE_ATTENTION': 3, 'GERMINATE_DEPTHWISE': 1, 'GERMINATE_CONV_ENHANCE': 2, 'FOSSILIZE': 1, 'CULL': 17}
  Policy loss: 0.0046, Value loss: 71.1452, Entropy: 1.8747, Entropy coef: 0.1265
[12:55:48] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[12:55:48] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[12:55:48] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[12:55:48] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[12:55:54] env0_seed_0 | Stage transition: TRAINING → BLENDING
[12:55:54] env1_seed_0 | Stage transition: TRAINING → BLENDING
[12:56:03] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[12:56:03] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[12:56:06] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[12:56:06] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[12:56:08] env1_seed_0 | Stage transition: PROBATIONARY → CULLED
[12:56:08] env1_seed_0 | Culled (norm, Δacc +13.87%)
    [env1] Culled 'env1_seed_0' (norm, Δacc +13.87%)
[12:56:09] env1_seed_1 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_1' (norm, 0.1K params)
[12:56:09] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[12:56:14] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[12:56:14] env0_seed_0 | Fossilized (norm, Δacc +20.90%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +20.90%)
[12:56:16] env1_seed_1 | Stage transition: TRAINING → BLENDING
[12:56:19] env1_seed_1 | Stage transition: BLENDING → CULLED
[12:56:19] env1_seed_1 | Culled (norm, Δacc +6.95%)
    [env1] Culled 'env1_seed_1' (norm, Δacc +6.95%)
[12:56:23] env1_seed_2 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_2' (attention, 2.0K params)
[12:56:23] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[12:56:29] env1_seed_2 | Stage transition: TRAINING → BLENDING
[12:56:37] env1_seed_2 | Stage transition: BLENDING → SHADOWING
[12:56:39] env1_seed_2 | Stage transition: SHADOWING → CULLED
[12:56:39] env1_seed_2 | Culled (attention, Δacc -3.91%)
    [env1] Culled 'env1_seed_2' (attention, Δacc -3.91%)
[12:56:46] env1_seed_3 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_3' (attention, 2.0K params)
[12:56:46] env1_seed_3 | Stage transition: GERMINATED → TRAINING
[12:56:59] env1_seed_3 | Stage transition: TRAINING → CULLED
[12:56:59] env1_seed_3 | Culled (attention, Δacc -0.77%)
    [env1] Culled 'env1_seed_3' (attention, Δacc -0.77%)
[12:57:02] env1_seed_4 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_4' (depthwise, 4.8K params)
[12:57:02] env1_seed_4 | Stage transition: GERMINATED → TRAINING
[12:57:06] env1_seed_4 | Stage transition: TRAINING → CULLED
[12:57:06] env1_seed_4 | Culled (depthwise, Δacc +5.07%)
    [env1] Culled 'env1_seed_4' (depthwise, Δacc +5.07%)
[12:57:12] env1_seed_5 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_5' (norm, 0.1K params)
[12:57:12] env1_seed_5 | Stage transition: GERMINATED → TRAINING
[12:57:17] env1_seed_5 | Stage transition: TRAINING → CULLED
[12:57:17] env1_seed_5 | Culled (norm, Δacc -1.01%)
    [env1] Culled 'env1_seed_5' (norm, Δacc -1.01%)
[12:57:19] env1_seed_6 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_6' (attention, 2.0K params)
[12:57:19] env1_seed_6 | Stage transition: GERMINATED → TRAINING
[12:57:20] env1_seed_6 | Stage transition: TRAINING → CULLED
[12:57:20] env1_seed_6 | Culled (attention, Δacc +0.00%)
    [env1] Culled 'env1_seed_6' (attention, Δacc +0.00%)
[12:57:25] env1_seed_7 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_7' (attention, 2.0K params)
[12:57:25] env1_seed_7 | Stage transition: GERMINATED → TRAINING
[12:57:27] env1_seed_7 | Stage transition: TRAINING → CULLED
[12:57:27] env1_seed_7 | Culled (attention, Δacc +0.00%)
    [env1] Culled 'env1_seed_7' (attention, Δacc +0.00%)
[12:57:28] env1_seed_8 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_8' (depthwise, 4.8K params)
[12:57:29] env1_seed_8 | Stage transition: GERMINATED → TRAINING
Batch 30: Episodes 60/200
  Env accuracies: ['77.0%', '71.4%']
  Avg acc: 74.2% (rolling: 72.4%)
  Avg reward: 151.7
  Actions: {'WAIT': 22, 'GERMINATE_NORM': 24, 'GERMINATE_ATTENTION': 34, 'GERMINATE_DEPTHWISE': 19, 'GERMINATE_CONV_ENHANCE': 10, 'FOSSILIZE': 24, 'CULL': 17}
  Successful: {'WAIT': 22, 'GERMINATE_NORM': 4, 'GERMINATE_ATTENTION': 4, 'GERMINATE_DEPTHWISE': 2, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 1, 'CULL': 16}
  Policy loss: 0.0150, Value loss: 61.9848, Entropy: 1.8834, Entropy coef: 0.1240

Blueprint Stats
  ---------------------------------------------------------------------------

Blueprint       Germ  Foss  Cull   Rate     ΔAcc    Churn
  ---------------------------------------------------------------------------

  attention         77     5    70   6.7%   +2.41%   +0.19%
  conv_enhance      61     5    53   8.6%   +9.06%   -1.97%
  depthwise         68     2    57   3.4%   +4.15%   +0.41%
  norm             169    18   140  11.4%  +14.99%   +4.63%
Seed Scoreboard (env 0):
  Fossilized: 17 (+381.9K params, +403.0% of host)
  Culled: 139
  Avg fossilize age: 16.1 epochs
  Avg cull age: 6.2 epochs
  Compute cost: 3.04x baseline
  Distribution: attention x3, conv_enhance x5, norm x8, depthwise x1
Seed Scoreboard (env 1):
  Fossilized: 13 (+10.2K params, +10.7% of host)
  Culled: 181
  Avg fossilize age: 14.5 epochs
  Avg cull age: 5.7 epochs
  Compute cost: 1.98x baseline
  Distribution: attention x2, norm x10, depthwise x1

[12:57:53] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[12:57:53] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[12:57:53] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[12:57:53] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[12:57:54] env1_seed_0 | Stage transition: TRAINING → CULLED
[12:57:54] env1_seed_0 | Culled (norm, Δacc +0.00%)
    [env1] Culled 'env1_seed_0' (norm, Δacc +0.00%)
[12:57:56] env1_seed_1 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_1' (norm, 0.1K params)
[12:57:56] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[12:57:59] env0_seed_0 | Stage transition: TRAINING → BLENDING
[12:58:03] env1_seed_1 | Stage transition: TRAINING → BLENDING
[12:58:03] env0_seed_0 | Stage transition: BLENDING → CULLED
[12:58:03] env0_seed_0 | Culled (norm, Δacc +11.31%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +11.31%)
[12:58:04] env0_seed_1 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_1' (attention, 2.0K params)
[12:58:05] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[12:58:11] env0_seed_1 | Stage transition: TRAINING → BLENDING
[12:58:11] env1_seed_1 | Stage transition: BLENDING → SHADOWING
[12:58:15] env1_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[12:58:16] env1_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[12:58:16] env1_seed_1 | Fossilized (norm, Δacc +15.19%)
    [env1] Fossilized 'env1_seed_1' (norm, Δacc +15.19%)
[12:58:20] env0_seed_1 | Stage transition: BLENDING → SHADOWING
[12:58:23] env0_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[12:58:36] env0_seed_1 | Stage transition: PROBATIONARY → CULLED
[12:58:36] env0_seed_1 | Culled (attention, Δacc +7.64%)
    [env0] Culled 'env0_seed_1' (attention, Δacc +7.64%)
[12:58:38] env0_seed_2 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_2' (norm, 0.1K params)
[12:58:38] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[12:58:50] env0_seed_2 | Stage transition: TRAINING → BLENDING
[12:58:58] env0_seed_2 | Stage transition: BLENDING → SHADOWING
[12:59:01] env0_seed_2 | Stage transition: SHADOWING → PROBATIONARY
[12:59:06] env0_seed_2 | Stage transition: PROBATIONARY → CULLED
[12:59:06] env0_seed_2 | Culled (norm, Δacc +6.38%)
    [env0] Culled 'env0_seed_2' (norm, Δacc +6.38%)
[12:59:08] env0_seed_3 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_3' (depthwise, 4.8K params)
[12:59:08] env0_seed_3 | Stage transition: GERMINATED → TRAINING
[12:59:15] env0_seed_3 | Stage transition: TRAINING → CULLED
[12:59:15] env0_seed_3 | Culled (depthwise, Δacc +0.82%)
    [env0] Culled 'env0_seed_3' (depthwise, Δacc +0.82%)
[12:59:17] env0_seed_4 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_4' (norm, 0.1K params)
[12:59:17] env0_seed_4 | Stage transition: GERMINATED → TRAINING
[12:59:18] env0_seed_4 | Stage transition: TRAINING → CULLED
[12:59:18] env0_seed_4 | Culled (norm, Δacc +0.00%)
    [env0] Culled 'env0_seed_4' (norm, Δacc +0.00%)
[12:59:20] env0_seed_5 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_5' (depthwise, 4.8K params)
[12:59:20] env0_seed_5 | Stage transition: GERMINATED → TRAINING
[12:59:28] env0_seed_5 | Stage transition: TRAINING → CULLED
[12:59:28] env0_seed_5 | Culled (depthwise, Δacc -6.94%)
    [env0] Culled 'env0_seed_5' (depthwise, Δacc -6.94%)
[12:59:30] env0_seed_6 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_6' (attention, 2.0K params)
[12:59:30] env0_seed_6 | Stage transition: GERMINATED → TRAINING
[12:59:32] env0_seed_6 | Stage transition: TRAINING → CULLED
[12:59:32] env0_seed_6 | Culled (attention, Δacc +0.00%)
    [env0] Culled 'env0_seed_6' (attention, Δacc +0.00%)
[12:59:35] env0_seed_7 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_7' (attention, 2.0K params)
[12:59:35] env0_seed_7 | Stage transition: GERMINATED → TRAINING
[12:59:42] env0_seed_7 | Stage transition: TRAINING → CULLED
[12:59:42] env0_seed_7 | Culled (attention, Δacc +2.25%)
    [env0] Culled 'env0_seed_7' (attention, Δacc +2.25%)
[12:59:43] env0_seed_8 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_8' (conv_enhance, 74.0K params)
[12:59:43] env0_seed_8 | Stage transition: GERMINATED → TRAINING
[12:59:48] env0_seed_8 | Stage transition: TRAINING → CULLED
[12:59:48] env0_seed_8 | Culled (conv_enhance, Δacc -1.45%)
    [env0] Culled 'env0_seed_8' (conv_enhance, Δacc -1.45%)
[12:59:52] env0_seed_9 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_9' (norm, 0.1K params)
[12:59:52] env0_seed_9 | Stage transition: GERMINATED → TRAINING
Batch 31: Episodes 62/200
  Env accuracies: ['72.4%', '74.3%']
  Avg acc: 73.4% (rolling: 72.3%)
  Avg reward: 142.2
  Actions: {'WAIT': 17, 'GERMINATE_NORM': 33, 'GERMINATE_ATTENTION': 20, 'GERMINATE_DEPTHWISE': 21, 'GERMINATE_CONV_ENHANCE': 15, 'FOSSILIZE': 23, 'CULL': 21}
  Successful: {'WAIT': 17, 'GERMINATE_NORM': 6, 'GERMINATE_ATTENTION': 3, 'GERMINATE_DEPTHWISE': 2, 'GERMINATE_CONV_ENHANCE': 1, 'FOSSILIZE': 1, 'CULL': 19}
  Policy loss: -0.0062, Value loss: 67.1726, Entropy: 1.8763, Entropy coef: 0.1215
[12:59:59] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[12:59:59] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[12:59:59] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[12:59:59] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[13:00:05] env0_seed_0 | Stage transition: TRAINING → BLENDING
[13:00:05] env1_seed_0 | Stage transition: TRAINING → BLENDING
[13:00:10] env0_seed_0 | Stage transition: BLENDING → CULLED
[13:00:10] env0_seed_0 | Culled (norm, Δacc +14.78%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +14.78%)
[13:00:14] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[13:00:14] env0_seed_1 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_1' (attention, 2.0K params)
[13:00:14] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[13:00:17] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:00:17] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:00:17] env1_seed_0 | Fossilized (norm, Δacc +15.87%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +15.87%)
[13:00:20] env0_seed_1 | Stage transition: TRAINING → BLENDING
[13:00:29] env0_seed_1 | Stage transition: BLENDING → SHADOWING
[13:00:32] env0_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[13:00:32] env0_seed_1 | Stage transition: PROBATIONARY → CULLED
[13:00:32] env0_seed_1 | Culled (attention, Δacc +2.61%)
    [env0] Culled 'env0_seed_1' (attention, Δacc +2.61%)
[13:00:39] env0_seed_2 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_2' (depthwise, 4.8K params)
[13:00:39] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[13:00:57] env0_seed_2 | Stage transition: TRAINING → BLENDING
[13:00:59] env0_seed_2 | Stage transition: BLENDING → CULLED
[13:00:59] env0_seed_2 | Culled (depthwise, Δacc +0.38%)
    [env0] Culled 'env0_seed_2' (depthwise, Δacc +0.38%)
[13:01:01] env0_seed_3 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_3' (depthwise, 4.8K params)
[13:01:01] env0_seed_3 | Stage transition: GERMINATED → TRAINING
[13:01:07] env0_seed_3 | Stage transition: TRAINING → CULLED
[13:01:07] env0_seed_3 | Culled (depthwise, Δacc -5.08%)
    [env0] Culled 'env0_seed_3' (depthwise, Δacc -5.08%)
[13:01:11] env0_seed_4 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_4' (conv_enhance, 74.0K params)
[13:01:11] env0_seed_4 | Stage transition: GERMINATED → TRAINING
[13:01:17] env0_seed_4 | Stage transition: TRAINING → BLENDING
[13:01:23] env0_seed_4 | Stage transition: BLENDING → CULLED
[13:01:23] env0_seed_4 | Culled (conv_enhance, Δacc +7.33%)
    [env0] Culled 'env0_seed_4' (conv_enhance, Δacc +7.33%)
[13:01:28] env0_seed_5 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_5' (norm, 0.1K params)
[13:01:28] env0_seed_5 | Stage transition: GERMINATED → TRAINING
[13:01:38] env0_seed_5 | Stage transition: TRAINING → BLENDING
[13:01:44] env0_seed_5 | Stage transition: BLENDING → CULLED
[13:01:44] env0_seed_5 | Culled (norm, Δacc -0.10%)
    [env0] Culled 'env0_seed_5' (norm, Δacc -0.10%)
[13:01:46] env0_seed_6 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_6' (depthwise, 4.8K params)
[13:01:46] env0_seed_6 | Stage transition: GERMINATED → TRAINING
[13:01:49] env0_seed_6 | Stage transition: TRAINING → CULLED
[13:01:49] env0_seed_6 | Culled (depthwise, Δacc -4.51%)
    [env0] Culled 'env0_seed_6' (depthwise, Δacc -4.51%)
[13:01:53] env0_seed_7 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_7' (conv_enhance, 74.0K params)
[13:01:53] env0_seed_7 | Stage transition: GERMINATED → TRAINING
Batch 32: Episodes 64/200
  Env accuracies: ['71.7%', '76.8%']
  Avg acc: 74.3% (rolling: 72.4%)
  Avg reward: 148.1
  Actions: {'WAIT': 27, 'GERMINATE_NORM': 34, 'GERMINATE_ATTENTION': 15, 'GERMINATE_DEPTHWISE': 19, 'GERMINATE_CONV_ENHANCE': 23, 'FOSSILIZE': 18, 'CULL': 14}
  Successful: {'WAIT': 27, 'GERMINATE_NORM': 3, 'GERMINATE_ATTENTION': 1, 'GERMINATE_DEPTHWISE': 3, 'GERMINATE_CONV_ENHANCE': 2, 'FOSSILIZE': 1, 'CULL': 12}
  Policy loss: -0.0199, Value loss: 65.7274, Entropy: 1.8790, Entropy coef: 0.1189
[13:02:05] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[13:02:05] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[13:02:05] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[13:02:05] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[13:02:11] env0_seed_0 | Stage transition: TRAINING → BLENDING
[13:02:11] env1_seed_0 | Stage transition: TRAINING → BLENDING
[13:02:20] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[13:02:20] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[13:02:23] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:02:23] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:02:25] env0_seed_0 | Stage transition: PROBATIONARY → CULLED
[13:02:25] env0_seed_0 | Culled (norm, Δacc +13.09%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +13.09%)
[13:02:28] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:02:28] env1_seed_0 | Fossilized (norm, Δacc +21.59%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +21.59%)
[13:02:35] env0_seed_1 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_1' (norm, 0.1K params)
[13:02:35] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[13:02:36] env0_seed_1 | Stage transition: TRAINING → CULLED
[13:02:36] env0_seed_1 | Culled (norm, Δacc +0.00%)
    [env0] Culled 'env0_seed_1' (norm, Δacc +0.00%)
[13:02:38] env0_seed_2 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_2' (norm, 0.1K params)
[13:02:38] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[13:02:45] env0_seed_2 | Stage transition: TRAINING → BLENDING
[13:02:45] env0_seed_2 | Stage transition: BLENDING → CULLED
[13:02:45] env0_seed_2 | Culled (norm, Δacc +5.76%)
    [env0] Culled 'env0_seed_2' (norm, Δacc +5.76%)
[13:02:46] env0_seed_3 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_3' (norm, 0.1K params)
[13:02:46] env0_seed_3 | Stage transition: GERMINATED → TRAINING
[13:02:53] env0_seed_3 | Stage transition: TRAINING → BLENDING
[13:02:56] env0_seed_3 | Stage transition: BLENDING → CULLED
[13:02:56] env0_seed_3 | Culled (norm, Δacc +4.82%)
    [env0] Culled 'env0_seed_3' (norm, Δacc +4.82%)
[13:02:58] env0_seed_4 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_4' (norm, 0.1K params)
[13:02:58] env0_seed_4 | Stage transition: GERMINATED → TRAINING
[13:03:05] env0_seed_4 | Stage transition: TRAINING → BLENDING
[13:03:13] env0_seed_4 | Stage transition: BLENDING → SHADOWING
[13:03:17] env0_seed_4 | Stage transition: SHADOWING → PROBATIONARY
[13:03:17] env0_seed_4 | Stage transition: PROBATIONARY → FOSSILIZED
[13:03:17] env0_seed_4 | Fossilized (norm, Δacc +5.25%)
    [env0] Fossilized 'env0_seed_4' (norm, Δacc +5.25%)
Batch 33: Episodes 66/200
  Env accuracies: ['76.8%', '74.4%']
  Avg acc: 75.6% (rolling: 72.9%)
  Avg reward: 159.3
  Actions: {'WAIT': 30, 'GERMINATE_NORM': 32, 'GERMINATE_ATTENTION': 21, 'GERMINATE_DEPTHWISE': 10, 'GERMINATE_CONV_ENHANCE': 13, 'FOSSILIZE': 25, 'CULL': 19}
  Successful: {'WAIT': 30, 'GERMINATE_NORM': 6, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 18}
  Policy loss: -0.0240, Value loss: 79.5583, Entropy: 1.8570, Entropy coef: 0.1164
[13:04:10] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[13:04:10] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[13:04:10] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[13:04:10] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[13:04:17] env0_seed_0 | Stage transition: TRAINING → BLENDING
[13:04:17] env1_seed_0 | Stage transition: TRAINING → BLENDING
[13:04:25] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[13:04:25] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[13:04:29] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:04:29] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:04:30] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:04:30] env0_seed_0 | Fossilized (norm, Δacc +14.44%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +14.44%)
[13:04:34] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:04:34] env1_seed_0 | Fossilized (norm, Δacc +15.93%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +15.93%)
Batch 34: Episodes 68/200
  Env accuracies: ['76.7%', '73.6%']
  Avg acc: 75.2% (rolling: 72.8%)
  Avg reward: 160.6
  Actions: {'WAIT': 19, 'GERMINATE_NORM': 41, 'GERMINATE_ATTENTION': 17, 'GERMINATE_DEPTHWISE': 6, 'GERMINATE_CONV_ENHANCE': 15, 'FOSSILIZE': 34, 'CULL': 18}
  Successful: {'WAIT': 19, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 18}
  Policy loss: -0.0154, Value loss: 68.9647, Entropy: 1.8488, Entropy coef: 0.1139
[13:06:16] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[13:06:16] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[13:06:16] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[13:06:16] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[13:06:23] env0_seed_0 | Stage transition: TRAINING → BLENDING
[13:06:23] env1_seed_0 | Stage transition: TRAINING → BLENDING
[13:06:31] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[13:06:31] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[13:06:33] env1_seed_0 | Stage transition: SHADOWING → CULLED
[13:06:33] env1_seed_0 | Culled (norm, Δacc +11.58%)
    [env1] Culled 'env1_seed_0' (norm, Δacc +11.58%)
[13:06:34] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:06:34] env1_seed_1 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_1' (depthwise, 4.8K params)
[13:06:34] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[13:06:40] env0_seed_0 | Stage transition: PROBATIONARY → CULLED
[13:06:40] env0_seed_0 | Culled (norm, Δacc +17.53%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +17.53%)
[13:06:41] env1_seed_1 | Stage transition: TRAINING → BLENDING
[13:06:41] env0_seed_1 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_1' (norm, 0.1K params)
[13:06:41] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[13:06:48] env0_seed_1 | Stage transition: TRAINING → BLENDING
[13:06:48] env0_seed_1 | Stage transition: BLENDING → CULLED
[13:06:48] env0_seed_1 | Culled (norm, Δacc +0.75%)
    [env0] Culled 'env0_seed_1' (norm, Δacc +0.75%)
[13:06:50] env1_seed_1 | Stage transition: BLENDING → SHADOWING
[13:06:50] env0_seed_2 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_2' (attention, 2.0K params)
[13:06:50] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[13:06:52] env1_seed_1 | Stage transition: SHADOWING → CULLED
[13:06:52] env1_seed_1 | Culled (depthwise, Δacc +8.36%)
    [env1] Culled 'env1_seed_1' (depthwise, Δacc +8.36%)
[13:06:54] env1_seed_2 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_2' (attention, 2.0K params)
[13:06:54] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[13:06:57] env0_seed_2 | Stage transition: TRAINING → BLENDING
[13:06:57] env0_seed_2 | Stage transition: BLENDING → CULLED
[13:06:57] env0_seed_2 | Culled (attention, Δacc -5.44%)
    [env0] Culled 'env0_seed_2' (attention, Δacc -5.44%)
[13:07:00] env1_seed_2 | Stage transition: TRAINING → BLENDING
[13:07:00] env0_seed_3 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_3' (depthwise, 4.8K params)
[13:07:00] env0_seed_3 | Stage transition: GERMINATED → TRAINING
[13:07:07] env0_seed_3 | Stage transition: TRAINING → BLENDING
[13:07:08] env1_seed_2 | Stage transition: BLENDING → SHADOWING
[13:07:10] env0_seed_3 | Stage transition: BLENDING → CULLED
[13:07:10] env0_seed_3 | Culled (depthwise, Δacc +6.08%)
    [env0] Culled 'env0_seed_3' (depthwise, Δacc +6.08%)
[13:07:10] env1_seed_2 | Stage transition: SHADOWING → CULLED
[13:07:10] env1_seed_2 | Culled (attention, Δacc +1.31%)
    [env1] Culled 'env1_seed_2' (attention, Δacc +1.31%)
[13:07:12] env0_seed_4 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_4' (depthwise, 4.8K params)
[13:07:12] env1_seed_3 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_3' (norm, 0.1K params)
[13:07:12] env0_seed_4 | Stage transition: GERMINATED → TRAINING
[13:07:12] env1_seed_3 | Stage transition: GERMINATED → TRAINING
[13:07:13] env0_seed_4 | Stage transition: TRAINING → CULLED
[13:07:13] env0_seed_4 | Culled (depthwise, Δacc +0.00%)
    [env0] Culled 'env0_seed_4' (depthwise, Δacc +0.00%)
[13:07:15] env1_seed_3 | Stage transition: TRAINING → CULLED
[13:07:15] env1_seed_3 | Culled (norm, Δacc +0.74%)
    [env1] Culled 'env1_seed_3' (norm, Δacc +0.74%)
[13:07:17] env0_seed_5 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_5' (norm, 0.1K params)
[13:07:17] env1_seed_4 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_4' (attention, 2.0K params)
[13:07:17] env0_seed_5 | Stage transition: GERMINATED → TRAINING
[13:07:17] env1_seed_4 | Stage transition: GERMINATED → TRAINING
[13:07:23] env0_seed_5 | Stage transition: TRAINING → BLENDING
[13:07:23] env1_seed_4 | Stage transition: TRAINING → BLENDING
[13:07:30] env0_seed_5 | Stage transition: BLENDING → CULLED
[13:07:30] env0_seed_5 | Culled (norm, Δacc +5.85%)
    [env0] Culled 'env0_seed_5' (norm, Δacc +5.85%)
[13:07:32] env1_seed_4 | Stage transition: BLENDING → SHADOWING
[13:07:35] env1_seed_4 | Stage transition: SHADOWING → PROBATIONARY
[13:07:35] env0_seed_6 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_6' (conv_enhance, 74.0K params)
[13:07:35] env0_seed_6 | Stage transition: GERMINATED → TRAINING
[13:07:42] env0_seed_6 | Stage transition: TRAINING → BLENDING
[13:07:43] env1_seed_4 | Stage transition: PROBATIONARY → FOSSILIZED
[13:07:43] env1_seed_4 | Fossilized (attention, Δacc +5.13%)
    [env1] Fossilized 'env1_seed_4' (attention, Δacc +5.13%)
[13:07:50] env0_seed_6 | Stage transition: BLENDING → SHADOWING
[13:07:53] env0_seed_6 | Stage transition: SHADOWING → PROBATIONARY
[13:07:53] env0_seed_6 | Stage transition: PROBATIONARY → CULLED
[13:07:53] env0_seed_6 | Culled (conv_enhance, Δacc -35.68%)
    [env0] Culled 'env0_seed_6' (conv_enhance, Δacc -35.68%)
[13:07:55] env0_seed_7 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_7' (norm, 0.1K params)
[13:07:55] env0_seed_7 | Stage transition: GERMINATED → TRAINING
[13:08:03] env0_seed_7 | Stage transition: TRAINING → BLENDING
[13:08:03] env0_seed_7 | Stage transition: BLENDING → CULLED
[13:08:03] env0_seed_7 | Culled (norm, Δacc -4.06%)
    [env0] Culled 'env0_seed_7' (norm, Δacc -4.06%)
[13:08:05] env0_seed_8 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_8' (attention, 2.0K params)
[13:08:05] env0_seed_8 | Stage transition: GERMINATED → TRAINING
[13:08:08] env0_seed_8 | Stage transition: TRAINING → CULLED
[13:08:08] env0_seed_8 | Culled (attention, Δacc +1.63%)
    [env0] Culled 'env0_seed_8' (attention, Δacc +1.63%)
[13:08:10] env0_seed_9 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_9' (norm, 0.1K params)
[13:08:10] env0_seed_9 | Stage transition: GERMINATED → TRAINING
Batch 35: Episodes 70/200
  Env accuracies: ['73.9%', '70.4%']
  Avg acc: 72.2% (rolling: 73.2%)
  Avg reward: 133.1
  Actions: {'WAIT': 19, 'GERMINATE_NORM': 26, 'GERMINATE_ATTENTION': 25, 'GERMINATE_DEPTHWISE': 15, 'GERMINATE_CONV_ENHANCE': 16, 'FOSSILIZE': 33, 'CULL': 16}
  Successful: {'WAIT': 19, 'GERMINATE_NORM': 7, 'GERMINATE_ATTENTION': 4, 'GERMINATE_DEPTHWISE': 3, 'GERMINATE_CONV_ENHANCE': 1, 'FOSSILIZE': 1, 'CULL': 15}
  Policy loss: -0.0081, Value loss: 80.1663, Entropy: 1.8253, Entropy coef: 0.1113

Blueprint Stats
  ---------------------------------------------------------------------------

Blueprint       Germ  Foss  Cull   Rate     ΔAcc    Churn
  ---------------------------------------------------------------------------

  attention         85     6    77   7.2%   +2.87%   +0.30%
  conv_enhance      65     5    56   8.2%   +9.06%   -2.39%
  depthwise         76     2    65   3.0%   +4.15%   +0.35%
  norm             193    24   156  13.3%  +14.92%   +4.72%
Seed Scoreboard (env 0):
  Fossilized: 19 (+382.2K params, +403.3% of host)
  Culled: 168
  Avg fossilize age: 15.6 epochs
  Avg cull age: 6.3 epochs
  Compute cost: 3.08x baseline
  Distribution: attention x3, conv_enhance x5, norm x10, depthwise x1
Seed Scoreboard (env 1):
  Fossilized: 18 (+12.8K params, +13.5% of host)
  Culled: 186
  Avg fossilize age: 14.2 epochs
  Avg cull age: 5.8 epochs
  Compute cost: 2.41x baseline
  Distribution: attention x3, norm x14, depthwise x1

[13:08:22] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[13:08:22] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[13:08:22] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[13:08:22] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[13:08:28] env0_seed_0 | Stage transition: TRAINING → BLENDING
[13:08:28] env1_seed_0 | Stage transition: TRAINING → BLENDING
[13:08:35] env0_seed_0 | Stage transition: BLENDING → CULLED
[13:08:35] env0_seed_0 | Culled (norm, Δacc +14.44%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +14.44%)
[13:08:37] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[13:08:37] env0_seed_1 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_1' (norm, 0.1K params)
[13:08:37] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[13:08:40] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:08:42] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:08:42] env1_seed_0 | Fossilized (norm, Δacc +22.61%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +22.61%)
[13:08:44] env0_seed_1 | Stage transition: TRAINING → BLENDING
[13:08:49] env0_seed_1 | Stage transition: BLENDING → CULLED
[13:08:49] env0_seed_1 | Culled (norm, Δacc +5.46%)
    [env0] Culled 'env0_seed_1' (norm, Δacc +5.46%)
[13:08:52] env0_seed_2 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_2' (conv_enhance, 74.0K params)
[13:08:52] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[13:09:02] env0_seed_2 | Stage transition: TRAINING → BLENDING
[13:09:10] env0_seed_2 | Stage transition: BLENDING → SHADOWING
[13:09:14] env0_seed_2 | Stage transition: SHADOWING → PROBATIONARY
[13:09:19] env0_seed_2 | Stage transition: PROBATIONARY → FOSSILIZED
[13:09:19] env0_seed_2 | Fossilized (conv_enhance, Δacc +2.95%)
    [env0] Fossilized 'env0_seed_2' (conv_enhance, Δacc +2.95%)
Batch 36: Episodes 72/200
  Env accuracies: ['67.2%', '75.8%']
  Avg acc: 71.5% (rolling: 72.8%)
  Avg reward: 90.6
  Actions: {'WAIT': 22, 'GERMINATE_NORM': 32, 'GERMINATE_ATTENTION': 20, 'GERMINATE_DEPTHWISE': 8, 'GERMINATE_CONV_ENHANCE': 26, 'FOSSILIZE': 32, 'CULL': 10}
  Successful: {'WAIT': 22, 'GERMINATE_NORM': 3, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 1, 'FOSSILIZE': 2, 'CULL': 9}
  Policy loss: 0.0108, Value loss: 154.6435, Entropy: 1.8326, Entropy coef: 0.1088
[13:10:28] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[13:10:28] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[13:10:28] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[13:10:28] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[13:10:34] env0_seed_0 | Stage transition: TRAINING → BLENDING
[13:10:34] env1_seed_0 | Stage transition: TRAINING → BLENDING
[13:10:43] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[13:10:43] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[13:10:46] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:10:46] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:10:48] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:10:48] env1_seed_0 | Fossilized (norm, Δacc +14.66%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +14.66%)
[13:11:05] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:11:05] env0_seed_0 | Fossilized (norm, Δacc +18.51%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +18.51%)
Batch 37: Episodes 74/200
  Env accuracies: ['77.4%', '76.5%']
  Avg acc: 77.0% (rolling: 73.1%)
  Avg reward: 168.7
  Actions: {'WAIT': 25, 'GERMINATE_NORM': 38, 'GERMINATE_ATTENTION': 20, 'GERMINATE_DEPTHWISE': 24, 'GERMINATE_CONV_ENHANCE': 14, 'FOSSILIZE': 20, 'CULL': 9}
  Successful: {'WAIT': 25, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 9}
  Policy loss: 0.0062, Value loss: 59.3818, Entropy: 1.8277, Entropy coef: 0.1063
[13:12:34] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[13:12:34] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[13:12:34] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[13:12:34] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[13:12:40] env0_seed_0 | Stage transition: TRAINING → BLENDING
[13:12:40] env1_seed_0 | Stage transition: TRAINING → BLENDING
[13:12:49] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[13:12:49] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[13:12:52] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:12:52] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:12:55] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:12:55] env1_seed_0 | Fossilized (norm, Δacc +14.69%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +14.69%)
[13:12:57] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:12:57] env0_seed_0 | Fossilized (norm, Δacc +15.68%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +15.68%)
Batch 38: Episodes 76/200
  Env accuracies: ['76.3%', '77.2%']
  Avg acc: 76.7% (rolling: 74.1%)
  Avg reward: 168.9
  Actions: {'WAIT': 28, 'GERMINATE_NORM': 39, 'GERMINATE_ATTENTION': 16, 'GERMINATE_DEPTHWISE': 12, 'GERMINATE_CONV_ENHANCE': 17, 'FOSSILIZE': 36, 'CULL': 2}
  Successful: {'WAIT': 28, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 2}
  Policy loss: 0.0110, Value loss: 51.1720, Entropy: 1.8250, Entropy coef: 0.1037
[13:14:39] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[13:14:39] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[13:14:39] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[13:14:39] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[13:14:46] env0_seed_0 | Stage transition: TRAINING → BLENDING
[13:14:46] env1_seed_0 | Stage transition: TRAINING → BLENDING
[13:14:54] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[13:14:54] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[13:14:58] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:14:58] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:14:58] env0_seed_0 | Stage transition: PROBATIONARY → CULLED
[13:14:58] env0_seed_0 | Culled (norm, Δacc +11.77%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +11.77%)
[13:14:59] env0_seed_1 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_1' (depthwise, 4.8K params)
[13:14:59] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:14:59] env1_seed_0 | Fossilized (norm, Δacc +19.67%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +19.67%)
[13:14:59] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[13:15:08] env0_seed_1 | Stage transition: TRAINING → BLENDING
[13:15:16] env0_seed_1 | Stage transition: BLENDING → SHADOWING
[13:15:19] env0_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[13:15:19] env0_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[13:15:19] env0_seed_1 | Fossilized (depthwise, Δacc +3.83%)
    [env0] Fossilized 'env0_seed_1' (depthwise, Δacc +3.83%)
Batch 39: Episodes 78/200
  Env accuracies: ['66.8%', '75.8%']
  Avg acc: 71.3% (rolling: 74.1%)
  Avg reward: 150.9
  Actions: {'WAIT': 22, 'GERMINATE_NORM': 25, 'GERMINATE_ATTENTION': 27, 'GERMINATE_DEPTHWISE': 11, 'GERMINATE_CONV_ENHANCE': 21, 'FOSSILIZE': 30, 'CULL': 14}
  Successful: {'WAIT': 22, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 1, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 14}
  Policy loss: 0.0181, Value loss: 53.7073, Entropy: 1.8659, Entropy coef: 0.1012
[13:16:45] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[13:16:45] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[13:16:45] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[13:16:45] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[13:16:52] env0_seed_0 | Stage transition: TRAINING → BLENDING
[13:16:52] env1_seed_0 | Stage transition: TRAINING → BLENDING
[13:17:00] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[13:17:00] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[13:17:03] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:17:03] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:17:03] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:17:03] env1_seed_0 | Fossilized (norm, Δacc +15.82%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +15.82%)
[13:17:05] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:17:05] env0_seed_0 | Fossilized (norm, Δacc +15.36%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +15.36%)
Batch 40: Episodes 80/200
  Env accuracies: ['75.0%', '75.5%']
  Avg acc: 75.3% (rolling: 74.2%)
  Avg reward: 164.4
  Actions: {'WAIT': 24, 'GERMINATE_NORM': 20, 'GERMINATE_ATTENTION': 24, 'GERMINATE_DEPTHWISE': 17, 'GERMINATE_CONV_ENHANCE': 22, 'FOSSILIZE': 28, 'CULL': 15}
  Successful: {'WAIT': 24, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 15}
  Policy loss: 0.0057, Value loss: 49.5163, Entropy: 1.8582, Entropy coef: 0.0987

Blueprint Stats
  ---------------------------------------------------------------------------

Blueprint       Germ  Foss  Cull   Rate     ΔAcc    Churn
  ---------------------------------------------------------------------------

  attention         85     6    77   7.2%   +2.87%   +0.30%
  conv_enhance      66     6    56   9.7%   +8.04%   -2.39%
  depthwise         77     3    65   4.4%   +4.05%   +0.35%
  norm             204    32   159  16.8%  +15.47%   +4.83%
Seed Scoreboard (env 0):
  Fossilized: 24 (+461.3K params, +486.8% of host)
  Culled: 171
  Avg fossilize age: 15.5 epochs
  Avg cull age: 6.3 epochs
  Compute cost: 3.37x baseline
  Distribution: attention x3, conv_enhance x6, norm x13, depthwise x2
Seed Scoreboard (env 1):
  Fossilized: 23 (+13.4K params, +14.1% of host)
  Culled: 186
  Avg fossilize age: 13.7 epochs
  Avg cull age: 5.8 epochs
  Compute cost: 2.51x baseline
  Distribution: attention x3, norm x19, depthwise x1

[13:18:51] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[13:18:51] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[13:18:51] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[13:18:51] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[13:18:57] env0_seed_0 | Stage transition: TRAINING → BLENDING
[13:18:57] env1_seed_0 | Stage transition: TRAINING → BLENDING
[13:19:04] env0_seed_0 | Stage transition: BLENDING → CULLED
[13:19:04] env0_seed_0 | Culled (norm, Δacc +9.40%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +9.40%)
[13:19:06] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[13:19:06] env0_seed_1 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_1' (norm, 0.1K params)
[13:19:06] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[13:19:09] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:19:11] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:19:11] env1_seed_0 | Fossilized (norm, Δacc +13.31%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +13.31%)
[13:19:12] env0_seed_1 | Stage transition: TRAINING → BLENDING
[13:19:21] env0_seed_1 | Stage transition: BLENDING → SHADOWING
[13:19:22] env0_seed_1 | Stage transition: SHADOWING → CULLED
[13:19:22] env0_seed_1 | Culled (norm, Δacc +8.17%)
    [env0] Culled 'env0_seed_1' (norm, Δacc +8.17%)
[13:19:28] env0_seed_2 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_2' (depthwise, 4.8K params)
[13:19:28] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[13:19:43] env0_seed_2 | Stage transition: TRAINING → BLENDING
[13:19:43] env0_seed_2 | Stage transition: BLENDING → CULLED
[13:19:43] env0_seed_2 | Culled (depthwise, Δacc +1.16%)
    [env0] Culled 'env0_seed_2' (depthwise, Δacc +1.16%)
[13:19:44] env0_seed_3 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_3' (norm, 0.1K params)
[13:19:44] env0_seed_3 | Stage transition: GERMINATED → TRAINING
[13:19:51] env0_seed_3 | Stage transition: TRAINING → BLENDING
[13:19:59] env0_seed_3 | Stage transition: BLENDING → SHADOWING
[13:20:03] env0_seed_3 | Stage transition: SHADOWING → PROBATIONARY
[13:20:09] env0_seed_3 | Stage transition: PROBATIONARY → CULLED
[13:20:09] env0_seed_3 | Culled (norm, Δacc +6.88%)
    [env0] Culled 'env0_seed_3' (norm, Δacc +6.88%)
[13:20:13] env0_seed_4 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_4' (norm, 0.1K params)
[13:20:13] env0_seed_4 | Stage transition: GERMINATED → TRAINING
[13:20:19] env0_seed_4 | Stage transition: TRAINING → BLENDING
[13:20:28] env0_seed_4 | Stage transition: BLENDING → SHADOWING
[13:20:31] env0_seed_4 | Stage transition: SHADOWING → PROBATIONARY
[13:20:33] env0_seed_4 | Stage transition: PROBATIONARY → FOSSILIZED
[13:20:33] env0_seed_4 | Fossilized (norm, Δacc +10.02%)
    [env0] Fossilized 'env0_seed_4' (norm, Δacc +10.02%)
Batch 41: Episodes 82/200
  Env accuracies: ['76.2%', '74.2%']
  Avg acc: 75.2% (rolling: 74.4%)
  Avg reward: 158.7
  Actions: {'WAIT': 29, 'GERMINATE_NORM': 26, 'GERMINATE_ATTENTION': 21, 'GERMINATE_DEPTHWISE': 16, 'GERMINATE_CONV_ENHANCE': 20, 'FOSSILIZE': 28, 'CULL': 10}
  Successful: {'WAIT': 29, 'GERMINATE_NORM': 5, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 1, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 10}
  Policy loss: 0.0001, Value loss: 42.9875, Entropy: 1.8490, Entropy coef: 0.0961
[13:20:56] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[13:20:56] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[13:20:56] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[13:20:56] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[13:21:03] env0_seed_0 | Stage transition: TRAINING → BLENDING
[13:21:03] env1_seed_0 | Stage transition: TRAINING → BLENDING
[13:21:11] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[13:21:11] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[13:21:15] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:21:15] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:21:15] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:21:15] env0_seed_0 | Fossilized (norm, Δacc +12.29%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +12.29%)
[13:21:16] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:21:16] env1_seed_0 | Fossilized (norm, Δacc +18.07%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +18.07%)
Batch 42: Episodes 84/200
  Env accuracies: ['75.4%', '75.6%']
  Avg acc: 75.5% (rolling: 74.5%)
  Avg reward: 166.7
  Actions: {'WAIT': 34, 'GERMINATE_NORM': 14, 'GERMINATE_ATTENTION': 15, 'GERMINATE_DEPTHWISE': 18, 'GERMINATE_CONV_ENHANCE': 30, 'FOSSILIZE': 26, 'CULL': 13}
  Successful: {'WAIT': 34, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 13}
  Policy loss: 0.0011, Value loss: 41.9750, Entropy: 1.8770, Entropy coef: 0.0936
[13:23:02] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[13:23:02] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[13:23:02] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[13:23:02] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[13:23:09] env0_seed_0 | Stage transition: TRAINING → BLENDING
[13:23:09] env1_seed_0 | Stage transition: TRAINING → BLENDING
[13:23:17] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[13:23:17] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[13:23:20] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:23:20] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:23:22] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:23:22] env1_seed_0 | Fossilized (norm, Δacc +20.31%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +20.31%)
[13:23:24] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:23:24] env0_seed_0 | Fossilized (norm, Δacc +24.07%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +24.07%)
Batch 43: Episodes 86/200
  Env accuracies: ['76.8%', '76.0%']
  Avg acc: 76.4% (rolling: 74.6%)
  Avg reward: 165.4
  Actions: {'WAIT': 29, 'GERMINATE_NORM': 21, 'GERMINATE_ATTENTION': 17, 'GERMINATE_DEPTHWISE': 11, 'GERMINATE_CONV_ENHANCE': 22, 'FOSSILIZE': 36, 'CULL': 14}
  Successful: {'WAIT': 29, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 14}
  Policy loss: -0.0096, Value loss: 40.7996, Entropy: 1.8573, Entropy coef: 0.0911
[13:25:08] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[13:25:08] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[13:25:08] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[13:25:08] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[13:25:14] env0_seed_0 | Stage transition: TRAINING → BLENDING
[13:25:14] env1_seed_0 | Stage transition: TRAINING → BLENDING
[13:25:18] env0_seed_0 | Stage transition: BLENDING → CULLED
[13:25:18] env0_seed_0 | Culled (norm, Δacc +15.12%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +15.12%)
[13:25:19] env0_seed_1 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_1' (norm, 0.1K params)
[13:25:19] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[13:25:23] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[13:25:26] env0_seed_1 | Stage transition: TRAINING → BLENDING
[13:25:26] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:25:26] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:25:26] env1_seed_0 | Fossilized (norm, Δacc +21.61%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +21.61%)
[13:25:31] env0_seed_1 | Stage transition: BLENDING → CULLED
[13:25:31] env0_seed_1 | Culled (norm, Δacc +2.34%)
    [env0] Culled 'env0_seed_1' (norm, Δacc +2.34%)
[13:25:33] env0_seed_2 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_2' (norm, 0.1K params)
[13:25:33] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[13:25:38] env0_seed_2 | Stage transition: TRAINING → CULLED
[13:25:38] env0_seed_2 | Culled (norm, Δacc -1.78%)
    [env0] Culled 'env0_seed_2' (norm, Δacc -1.78%)
[13:25:40] env0_seed_3 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_3' (norm, 0.1K params)
[13:25:40] env0_seed_3 | Stage transition: GERMINATED → TRAINING
[13:25:41] env0_seed_3 | Stage transition: TRAINING → CULLED
[13:25:41] env0_seed_3 | Culled (norm, Δacc +0.00%)
    [env0] Culled 'env0_seed_3' (norm, Δacc +0.00%)
[13:25:43] env0_seed_4 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_4' (norm, 0.1K params)
[13:25:43] env0_seed_4 | Stage transition: GERMINATED → TRAINING
[13:25:50] env0_seed_4 | Stage transition: TRAINING → CULLED
[13:25:50] env0_seed_4 | Culled (norm, Δacc -2.80%)
    [env0] Culled 'env0_seed_4' (norm, Δacc -2.80%)
[13:25:51] env0_seed_5 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_5' (norm, 0.1K params)
[13:25:51] env0_seed_5 | Stage transition: GERMINATED → TRAINING
[13:26:00] env0_seed_5 | Stage transition: TRAINING → BLENDING
[13:26:01] env0_seed_5 | Stage transition: BLENDING → CULLED
[13:26:01] env0_seed_5 | Culled (norm, Δacc +3.11%)
    [env0] Culled 'env0_seed_5' (norm, Δacc +3.11%)
[13:26:03] env0_seed_6 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_6' (norm, 0.1K params)
[13:26:03] env0_seed_6 | Stage transition: GERMINATED → TRAINING
[13:26:10] env0_seed_6 | Stage transition: TRAINING → BLENDING
[13:26:18] env0_seed_6 | Stage transition: BLENDING → SHADOWING
[13:26:20] env0_seed_6 | Stage transition: SHADOWING → CULLED
[13:26:20] env0_seed_6 | Culled (norm, Δacc +5.04%)
    [env0] Culled 'env0_seed_6' (norm, Δacc +5.04%)
[13:26:21] env0_seed_7 | Germinated (conv_enhance, 74.0K params)
    [env0] Germinated 'env0_seed_7' (conv_enhance, 74.0K params)
[13:26:22] env0_seed_7 | Stage transition: GERMINATED → TRAINING
[13:26:30] env0_seed_7 | Stage transition: TRAINING → BLENDING
[13:26:38] env0_seed_7 | Stage transition: BLENDING → SHADOWING
[13:26:42] env0_seed_7 | Stage transition: SHADOWING → PROBATIONARY
[13:26:43] env0_seed_7 | Stage transition: PROBATIONARY → CULLED
[13:26:43] env0_seed_7 | Culled (conv_enhance, Δacc -9.23%)
    [env0] Culled 'env0_seed_7' (conv_enhance, Δacc -9.23%)
[13:26:47] env0_seed_8 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_8' (depthwise, 4.8K params)
[13:26:47] env0_seed_8 | Stage transition: GERMINATED → TRAINING
[13:26:48] env0_seed_8 | Stage transition: TRAINING → CULLED
[13:26:48] env0_seed_8 | Culled (depthwise, Δacc +0.00%)
    [env0] Culled 'env0_seed_8' (depthwise, Δacc +0.00%)
[13:26:53] env0_seed_9 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_9' (norm, 0.1K params)
[13:26:53] env0_seed_9 | Stage transition: GERMINATED → TRAINING
[13:26:57] env0_seed_9 | Stage transition: TRAINING → CULLED
[13:26:57] env0_seed_9 | Culled (norm, Δacc -1.75%)
    [env0] Culled 'env0_seed_9' (norm, Δacc -1.75%)
[13:26:58] env0_seed_10 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_10' (depthwise, 4.8K params)
[13:26:58] env0_seed_10 | Stage transition: GERMINATED → TRAINING
[13:27:00] env0_seed_10 | Stage transition: TRAINING → CULLED
[13:27:00] env0_seed_10 | Culled (depthwise, Δacc +0.00%)
    [env0] Culled 'env0_seed_10' (depthwise, Δacc +0.00%)
[13:27:05] env0_seed_11 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_11' (norm, 0.1K params)
[13:27:05] env0_seed_11 | Stage transition: GERMINATED → TRAINING
[13:27:12] env0_seed_11 | Stage transition: TRAINING → CULLED
[13:27:12] env0_seed_11 | Culled (norm, Δacc -7.94%)
    [env0] Culled 'env0_seed_11' (norm, Δacc -7.94%)
Batch 44: Episodes 88/200
  Env accuracies: ['65.4%', '75.3%']
  Avg acc: 70.4% (rolling: 74.2%)
  Avg reward: 133.8
  Actions: {'WAIT': 28, 'GERMINATE_NORM': 29, 'GERMINATE_ATTENTION': 11, 'GERMINATE_DEPTHWISE': 16, 'GERMINATE_CONV_ENHANCE': 19, 'FOSSILIZE': 27, 'CULL': 20}
  Successful: {'WAIT': 28, 'GERMINATE_NORM': 10, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 2, 'GERMINATE_CONV_ENHANCE': 1, 'FOSSILIZE': 1, 'CULL': 19}
  Policy loss: -0.0207, Value loss: 46.1896, Entropy: 1.8548, Entropy coef: 0.0885
[13:27:14] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[13:27:14] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[13:27:14] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[13:27:14] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[13:27:20] env0_seed_0 | Stage transition: TRAINING → BLENDING
[13:27:20] env1_seed_0 | Stage transition: TRAINING → BLENDING
[13:27:29] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[13:27:29] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[13:27:32] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:27:32] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:27:34] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:27:34] env1_seed_0 | Fossilized (norm, Δacc +12.25%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +12.25%)
[13:27:37] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:27:37] env0_seed_0 | Fossilized (norm, Δacc +17.64%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +17.64%)
Batch 45: Episodes 90/200
  Env accuracies: ['74.3%', '76.7%']
  Avg acc: 75.5% (rolling: 74.5%)
  Avg reward: 161.8
  Actions: {'WAIT': 28, 'GERMINATE_NORM': 22, 'GERMINATE_ATTENTION': 24, 'GERMINATE_DEPTHWISE': 24, 'GERMINATE_CONV_ENHANCE': 9, 'FOSSILIZE': 22, 'CULL': 21}
  Successful: {'WAIT': 28, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 21}
  Policy loss: 0.0013, Value loss: 38.8212, Entropy: 1.8801, Entropy coef: 0.0860

Blueprint Stats
  ---------------------------------------------------------------------------

Blueprint       Germ  Foss  Cull   Rate     ΔAcc    Churn
  ---------------------------------------------------------------------------

  attention         85     6    77   7.2%   +2.87%   +0.30%
  conv_enhance      67     6    57   9.5%   +8.04%   -2.51%
  depthwise         80     3    68   4.2%   +4.05%   +0.35%
  norm             225    41   171  19.3%  +15.73%   +4.70%
Seed Scoreboard (env 0):
  Fossilized: 28 (+461.8K params, +487.4% of host)
  Culled: 187
  Avg fossilize age: 15.1 epochs
  Avg cull age: 6.3 epochs
  Compute cost: 3.45x baseline
  Distribution: attention x3, conv_enhance x6, norm x17, depthwise x2
Seed Scoreboard (env 1):
  Fossilized: 28 (+14.0K params, +14.8% of host)
  Culled: 186
  Avg fossilize age: 13.4 epochs
  Avg cull age: 5.8 epochs
  Compute cost: 2.61x baseline
  Distribution: attention x3, norm x24, depthwise x1

[13:29:19] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[13:29:19] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[13:29:19] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[13:29:19] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[13:29:26] env0_seed_0 | Stage transition: TRAINING → BLENDING
[13:29:26] env1_seed_0 | Stage transition: TRAINING → BLENDING
[13:29:34] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[13:29:34] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[13:29:34] env0_seed_0 | Stage transition: SHADOWING → CULLED
[13:29:34] env0_seed_0 | Culled (norm, Δacc +19.89%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +19.89%)
[13:29:36] env0_seed_1 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_1' (attention, 2.0K params)
[13:29:36] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[13:29:38] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:29:38] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:29:38] env1_seed_0 | Fossilized (norm, Δacc +18.20%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +18.20%)
[13:29:44] env0_seed_1 | Stage transition: TRAINING → BLENDING
[13:29:46] env0_seed_1 | Stage transition: BLENDING → CULLED
[13:29:46] env0_seed_1 | Culled (attention, Δacc +1.51%)
    [env0] Culled 'env0_seed_1' (attention, Δacc +1.51%)
[13:29:51] env0_seed_2 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_2' (attention, 2.0K params)
[13:29:51] env0_seed_2 | Stage transition: GERMINATED → TRAINING
[13:29:53] env0_seed_2 | Stage transition: TRAINING → CULLED
[13:29:53] env0_seed_2 | Culled (attention, Δacc +0.00%)
    [env0] Culled 'env0_seed_2' (attention, Δacc +0.00%)
[13:29:56] env0_seed_3 | Germinated (depthwise, 4.8K params)
    [env0] Germinated 'env0_seed_3' (depthwise, 4.8K params)
[13:29:56] env0_seed_3 | Stage transition: GERMINATED → TRAINING
[13:30:08] env0_seed_3 | Stage transition: TRAINING → BLENDING
[13:30:08] env0_seed_3 | Stage transition: BLENDING → CULLED
[13:30:08] env0_seed_3 | Culled (depthwise, Δacc -6.67%)
    [env0] Culled 'env0_seed_3' (depthwise, Δacc -6.67%)
[13:30:09] env0_seed_4 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_4' (norm, 0.1K params)
[13:30:10] env0_seed_4 | Stage transition: GERMINATED → TRAINING
[13:30:26] env0_seed_4 | Stage transition: TRAINING → CULLED
[13:30:26] env0_seed_4 | Culled (norm, Δacc -3.06%)
    [env0] Culled 'env0_seed_4' (norm, Δacc -3.06%)
[13:30:28] env0_seed_5 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_5' (norm, 0.1K params)
[13:30:28] env0_seed_5 | Stage transition: GERMINATED → TRAINING
[13:30:36] env0_seed_5 | Stage transition: TRAINING → BLENDING
[13:30:36] env0_seed_5 | Stage transition: BLENDING → CULLED
[13:30:36] env0_seed_5 | Culled (norm, Δacc -2.29%)
    [env0] Culled 'env0_seed_5' (norm, Δacc -2.29%)
[13:30:43] env0_seed_6 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_6' (norm, 0.1K params)
[13:30:43] env0_seed_6 | Stage transition: GERMINATED → TRAINING
[13:30:50] env0_seed_6 | Stage transition: TRAINING → BLENDING
[13:30:58] env0_seed_6 | Stage transition: BLENDING → SHADOWING
[13:31:02] env0_seed_6 | Stage transition: SHADOWING → PROBATIONARY
[13:31:02] env0_seed_6 | Stage transition: PROBATIONARY → FOSSILIZED
[13:31:02] env0_seed_6 | Fossilized (norm, Δacc +4.68%)
    [env0] Fossilized 'env0_seed_6' (norm, Δacc +4.68%)
Batch 46: Episodes 92/200
  Env accuracies: ['76.0%', '75.8%']
  Avg acc: 75.9% (rolling: 74.9%)
  Avg reward: 160.0
  Actions: {'WAIT': 30, 'GERMINATE_NORM': 27, 'GERMINATE_ATTENTION': 20, 'GERMINATE_DEPTHWISE': 18, 'GERMINATE_CONV_ENHANCE': 8, 'FOSSILIZE': 30, 'CULL': 17}
  Successful: {'WAIT': 30, 'GERMINATE_NORM': 5, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 1, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 16}
  Policy loss: -0.0139, Value loss: 40.6096, Entropy: 1.8688, Entropy coef: 0.0835
[13:31:25] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[13:31:25] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[13:31:25] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[13:31:25] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[13:31:32] env0_seed_0 | Stage transition: TRAINING → BLENDING
[13:31:32] env1_seed_0 | Stage transition: TRAINING → BLENDING
[13:31:37] env1_seed_0 | Stage transition: BLENDING → CULLED
[13:31:37] env1_seed_0 | Culled (norm, Δacc +13.88%)
    [env1] Culled 'env1_seed_0' (norm, Δacc +13.88%)
[13:31:40] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[13:31:42] env1_seed_1 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_1' (norm, 0.1K params)
[13:31:42] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[13:31:43] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:31:47] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:31:47] env0_seed_0 | Fossilized (norm, Δacc +16.65%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +16.65%)
[13:31:50] env1_seed_1 | Stage transition: TRAINING → BLENDING
[13:31:58] env1_seed_1 | Stage transition: BLENDING → SHADOWING
[13:32:02] env1_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[13:32:02] env1_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[13:32:02] env1_seed_1 | Fossilized (norm, Δacc +5.91%)
    [env1] Fossilized 'env1_seed_1' (norm, Δacc +5.91%)
Batch 47: Episodes 94/200
  Env accuracies: ['75.6%', '77.3%']
  Avg acc: 76.4% (rolling: 74.9%)
  Avg reward: 163.9
  Actions: {'WAIT': 22, 'GERMINATE_NORM': 22, 'GERMINATE_ATTENTION': 18, 'GERMINATE_DEPTHWISE': 30, 'GERMINATE_CONV_ENHANCE': 20, 'FOSSILIZE': 28, 'CULL': 10}
  Successful: {'WAIT': 22, 'GERMINATE_NORM': 3, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 10}
  Policy loss: -0.0016, Value loss: 39.2340, Entropy: 1.8840, Entropy coef: 0.0809
[13:33:30] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[13:33:30] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[13:33:30] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[13:33:30] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[13:33:37] env1_seed_0 | Stage transition: TRAINING → BLENDING
[13:33:39] env0_seed_0 | Stage transition: TRAINING → BLENDING
[13:33:44] env1_seed_0 | Stage transition: BLENDING → CULLED
[13:33:44] env1_seed_0 | Culled (norm, Δacc +12.72%)
    [env1] Culled 'env1_seed_0' (norm, Δacc +12.72%)
[13:33:45] env1_seed_1 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_1' (norm, 0.1K params)
[13:33:45] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[13:33:47] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[13:33:50] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:33:50] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:33:50] env0_seed_0 | Fossilized (norm, Δacc +11.28%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +11.28%)
[13:33:52] env1_seed_1 | Stage transition: TRAINING → BLENDING
[13:33:55] env1_seed_1 | Stage transition: BLENDING → CULLED
[13:33:55] env1_seed_1 | Culled (norm, Δacc +7.69%)
    [env1] Culled 'env1_seed_1' (norm, Δacc +7.69%)
[13:34:02] env1_seed_2 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_2' (norm, 0.1K params)
[13:34:02] env1_seed_2 | Stage transition: GERMINATED → TRAINING
[13:34:09] env1_seed_2 | Stage transition: TRAINING → BLENDING
[13:34:15] env1_seed_2 | Stage transition: BLENDING → CULLED
[13:34:15] env1_seed_2 | Culled (norm, Δacc +11.67%)
    [env1] Culled 'env1_seed_2' (norm, Δacc +11.67%)
[13:34:19] env1_seed_3 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_3' (norm, 0.1K params)
[13:34:19] env1_seed_3 | Stage transition: GERMINATED → TRAINING
[13:34:25] env1_seed_3 | Stage transition: TRAINING → BLENDING
[13:34:29] env1_seed_3 | Stage transition: BLENDING → CULLED
[13:34:29] env1_seed_3 | Culled (norm, Δacc +2.71%)
    [env1] Culled 'env1_seed_3' (norm, Δacc +2.71%)
[13:34:30] env1_seed_4 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_4' (conv_enhance, 74.0K params)
[13:34:30] env1_seed_4 | Stage transition: GERMINATED → TRAINING
[13:34:38] env1_seed_4 | Stage transition: TRAINING → BLENDING
[13:34:45] env1_seed_4 | Stage transition: BLENDING → CULLED
[13:34:45] env1_seed_4 | Culled (conv_enhance, Δacc -9.26%)
    [env1] Culled 'env1_seed_4' (conv_enhance, Δacc -9.26%)
[13:34:46] env1_seed_5 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_5' (conv_enhance, 74.0K params)
[13:34:47] env1_seed_5 | Stage transition: GERMINATED → TRAINING
[13:34:54] env1_seed_5 | Stage transition: TRAINING → CULLED
[13:34:54] env1_seed_5 | Culled (conv_enhance, Δacc +1.95%)
    [env1] Culled 'env1_seed_5' (conv_enhance, Δacc +1.95%)
[13:34:57] env1_seed_6 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_6' (attention, 2.0K params)
[13:34:57] env1_seed_6 | Stage transition: GERMINATED → TRAINING
[13:35:04] env1_seed_6 | Stage transition: TRAINING → BLENDING
[13:35:07] env1_seed_6 | Stage transition: BLENDING → CULLED
[13:35:07] env1_seed_6 | Culled (attention, Δacc +3.58%)
    [env1] Culled 'env1_seed_6' (attention, Δacc +3.58%)
[13:35:09] env1_seed_7 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_7' (conv_enhance, 74.0K params)
[13:35:09] env1_seed_7 | Stage transition: GERMINATED → TRAINING
[13:35:16] env1_seed_7 | Stage transition: TRAINING → BLENDING
[13:35:21] env1_seed_7 | Stage transition: BLENDING → CULLED
[13:35:21] env1_seed_7 | Culled (conv_enhance, Δacc +3.45%)
    [env1] Culled 'env1_seed_7' (conv_enhance, Δacc +3.45%)
[13:35:26] env1_seed_8 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_8' (depthwise, 4.8K params)
[13:35:26] env1_seed_8 | Stage transition: GERMINATED → TRAINING
[13:35:28] env1_seed_8 | Stage transition: TRAINING → CULLED
[13:35:28] env1_seed_8 | Culled (depthwise, Δacc +0.00%)
    [env1] Culled 'env1_seed_8' (depthwise, Δacc +0.00%)
[13:35:33] env1_seed_9 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_9' (depthwise, 4.8K params)
[13:35:33] env1_seed_9 | Stage transition: GERMINATED → TRAINING
Batch 48: Episodes 96/200
  Env accuracies: ['76.2%', '73.3%']
  Avg acc: 74.7% (rolling: 74.7%)
  Avg reward: 141.5
  Actions: {'WAIT': 21, 'GERMINATE_NORM': 33, 'GERMINATE_ATTENTION': 24, 'GERMINATE_DEPTHWISE': 10, 'GERMINATE_CONV_ENHANCE': 15, 'FOSSILIZE': 31, 'CULL': 16}
  Successful: {'WAIT': 21, 'GERMINATE_NORM': 5, 'GERMINATE_ATTENTION': 1, 'GERMINATE_DEPTHWISE': 2, 'GERMINATE_CONV_ENHANCE': 3, 'FOSSILIZE': 1, 'CULL': 13}
  Policy loss: 0.0020, Value loss: 41.5773, Entropy: 1.8779, Entropy coef: 0.0784
[13:35:38] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[13:35:38] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[13:35:38] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[13:35:38] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[13:35:45] env0_seed_0 | Stage transition: TRAINING → BLENDING
[13:35:45] env1_seed_0 | Stage transition: TRAINING → BLENDING
[13:35:48] env0_seed_0 | Stage transition: BLENDING → CULLED
[13:35:48] env0_seed_0 | Culled (norm, Δacc +15.33%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +15.33%)
[13:35:53] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[13:35:53] env0_seed_1 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_1' (norm, 0.1K params)
[13:35:53] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[13:35:56] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:36:00] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:36:00] env1_seed_0 | Fossilized (norm, Δacc +15.97%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +15.97%)
[13:36:03] env0_seed_1 | Stage transition: TRAINING → BLENDING
[13:36:11] env0_seed_1 | Stage transition: BLENDING → SHADOWING
[13:36:15] env0_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[13:36:15] env0_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[13:36:15] env0_seed_1 | Fossilized (norm, Δacc +6.97%)
    [env0] Fossilized 'env0_seed_1' (norm, Δacc +6.97%)
Batch 49: Episodes 98/200
  Env accuracies: ['75.4%', '76.8%']
  Avg acc: 76.1% (rolling: 75.1%)
  Avg reward: 161.4
  Actions: {'WAIT': 27, 'GERMINATE_NORM': 15, 'GERMINATE_ATTENTION': 24, 'GERMINATE_DEPTHWISE': 22, 'GERMINATE_CONV_ENHANCE': 13, 'FOSSILIZE': 25, 'CULL': 24}
  Successful: {'WAIT': 27, 'GERMINATE_NORM': 3, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 24}
  Policy loss: 0.0100, Value loss: 37.6339, Entropy: 1.8812, Entropy coef: 0.0759
[13:37:44] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[13:37:44] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[13:37:44] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[13:37:44] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[13:37:51] env0_seed_0 | Stage transition: TRAINING → BLENDING
[13:37:51] env1_seed_0 | Stage transition: TRAINING → BLENDING
[13:37:59] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[13:37:59] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[13:38:02] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:38:02] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:38:02] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:38:02] env0_seed_0 | Fossilized (norm, Δacc +23.01%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +23.01%)
[13:38:07] env1_seed_0 | Stage transition: PROBATIONARY → CULLED
[13:38:07] env1_seed_0 | Culled (norm, Δacc +29.99%)
    [env1] Culled 'env1_seed_0' (norm, Δacc +29.99%)
[13:38:12] env1_seed_1 | Germinated (depthwise, 4.8K params)
    [env1] Germinated 'env1_seed_1' (depthwise, 4.8K params)
[13:38:12] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[13:38:19] env1_seed_1 | Stage transition: TRAINING → BLENDING
[13:38:28] env1_seed_1 | Stage transition: BLENDING → SHADOWING
[13:38:32] env1_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[13:38:42] env1_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[13:38:42] env1_seed_1 | Fossilized (depthwise, Δacc +3.31%)
    [env1] Fossilized 'env1_seed_1' (depthwise, Δacc +3.31%)
Batch 50: Episodes 100/200
  Env accuracies: ['75.6%', '74.9%']
  Avg acc: 75.2% (rolling: 75.1%)
  Avg reward: 158.8
  Actions: {'WAIT': 17, 'GERMINATE_NORM': 23, 'GERMINATE_ATTENTION': 23, 'GERMINATE_DEPTHWISE': 23, 'GERMINATE_CONV_ENHANCE': 27, 'FOSSILIZE': 25, 'CULL': 12}
  Successful: {'WAIT': 17, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 1, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 12}
  Policy loss: -0.0185, Value loss: 54.8178, Entropy: 1.8674, Entropy coef: 0.0733

Blueprint Stats
  ---------------------------------------------------------------------------

Blueprint       Germ  Foss  Cull   Rate     ΔAcc    Churn
  ---------------------------------------------------------------------------

  attention         88     6    80   7.0%   +2.87%   +0.35%
  conv_enhance      70     6    60   9.1%   +8.04%   -2.45%
  depthwise         84     4    70   5.4%   +3.86%   +0.24%
  norm             243    49   181  21.3%  +15.25%   +5.04%
Seed Scoreboard (env 0):
  Fossilized: 33 (+462.5K params, +488.1% of host)
  Culled: 194
  Avg fossilize age: 14.6 epochs
  Avg cull age: 6.3 epochs
  Compute cost: 3.55x baseline
  Distribution: attention x3, conv_enhance x6, norm x22, depthwise x2
Seed Scoreboard (env 1):
  Fossilized: 32 (+19.2K params, +20.3% of host)
  Culled: 197
  Avg fossilize age: 13.4 epochs
  Avg cull age: 5.8 epochs
  Compute cost: 2.75x baseline
  Distribution: attention x3, norm x27, depthwise x2

[13:39:54] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[13:39:54] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[13:39:54] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[13:39:54] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[13:39:57] env0_seed_0 | Stage transition: TRAINING → CULLED
[13:39:57] env0_seed_0 | Culled (norm, Δacc +4.83%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +4.83%)
[13:39:59] env0_seed_1 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_1' (norm, 0.1K params)
[13:39:59] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[13:40:01] env1_seed_0 | Stage transition: TRAINING → BLENDING
[13:40:06] env0_seed_1 | Stage transition: TRAINING → BLENDING
[13:40:09] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[13:40:13] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:40:14] env0_seed_1 | Stage transition: BLENDING → SHADOWING
[13:40:18] env0_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[13:40:23] env0_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[13:40:23] env0_seed_1 | Fossilized (norm, Δacc +11.63%)
    [env0] Fossilized 'env0_seed_1' (norm, Δacc +11.63%)
[13:40:23] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:40:23] env1_seed_0 | Fossilized (norm, Δacc +16.40%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +16.40%)
Batch 51: Episodes 102/200
  Env accuracies: ['77.0%', '75.0%']
  Avg acc: 76.0% (rolling: 75.2%)
  Avg reward: 166.0
  Actions: {'WAIT': 28, 'GERMINATE_NORM': 26, 'GERMINATE_ATTENTION': 23, 'GERMINATE_DEPTHWISE': 21, 'GERMINATE_CONV_ENHANCE': 12, 'FOSSILIZE': 27, 'CULL': 13}
  Successful: {'WAIT': 28, 'GERMINATE_NORM': 3, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 13}
  Policy loss: 0.0069, Value loss: 38.0350, Entropy: 1.8734, Entropy coef: 0.0708
[13:42:00] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[13:42:00] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[13:42:00] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[13:42:00] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[13:42:07] env0_seed_0 | Stage transition: TRAINING → BLENDING
[13:42:07] env1_seed_0 | Stage transition: TRAINING → BLENDING
[13:42:15] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[13:42:15] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[13:42:18] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:42:18] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:42:20] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:42:20] env0_seed_0 | Fossilized (norm, Δacc +17.72%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +17.72%)
[13:42:28] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:42:28] env1_seed_0 | Fossilized (norm, Δacc +15.94%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +15.94%)
Batch 52: Episodes 104/200
  Env accuracies: ['74.6%', '77.0%']
  Avg acc: 75.8% (rolling: 75.2%)
  Avg reward: 166.8
  Actions: {'WAIT': 28, 'GERMINATE_NORM': 25, 'GERMINATE_ATTENTION': 21, 'GERMINATE_DEPTHWISE': 25, 'GERMINATE_CONV_ENHANCE': 23, 'FOSSILIZE': 13, 'CULL': 15}
  Successful: {'WAIT': 28, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 15}
  Policy loss: -0.0070, Value loss: 33.2897, Entropy: 1.8696, Entropy coef: 0.0683
[13:44:06] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[13:44:06] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[13:44:06] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[13:44:06] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[13:44:13] env0_seed_0 | Stage transition: TRAINING → BLENDING
[13:44:13] env1_seed_0 | Stage transition: TRAINING → BLENDING
[13:44:22] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[13:44:22] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[13:44:25] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:44:25] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:44:35] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:44:35] env0_seed_0 | Fossilized (norm, Δacc +20.30%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +20.30%)
[13:45:00] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:45:00] env1_seed_0 | Fossilized (norm, Δacc +22.97%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +22.97%)
Batch 53: Episodes 106/200
  Env accuracies: ['75.5%', '76.8%']
  Avg acc: 76.2% (rolling: 75.2%)
  Avg reward: 166.9
  Actions: {'WAIT': 35, 'GERMINATE_NORM': 23, 'GERMINATE_ATTENTION': 23, 'GERMINATE_DEPTHWISE': 24, 'GERMINATE_CONV_ENHANCE': 18, 'FOSSILIZE': 17, 'CULL': 10}
  Successful: {'WAIT': 35, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 10}
  Policy loss: -0.0090, Value loss: 38.7544, Entropy: 1.8477, Entropy coef: 0.0657
[13:46:12] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[13:46:12] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[13:46:12] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[13:46:12] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[13:46:19] env0_seed_0 | Stage transition: TRAINING → BLENDING
[13:46:19] env1_seed_0 | Stage transition: TRAINING → BLENDING
[13:46:27] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[13:46:27] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[13:46:31] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:46:31] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:46:31] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:46:31] env1_seed_0 | Fossilized (norm, Δacc +16.03%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +16.03%)
[13:46:37] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:46:37] env0_seed_0 | Fossilized (norm, Δacc +21.54%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +21.54%)
Batch 54: Episodes 108/200
  Env accuracies: ['76.6%', '76.0%']
  Avg acc: 76.3% (rolling: 75.8%)
  Avg reward: 168.0
  Actions: {'WAIT': 23, 'GERMINATE_NORM': 25, 'GERMINATE_ATTENTION': 25, 'GERMINATE_DEPTHWISE': 22, 'GERMINATE_CONV_ENHANCE': 19, 'FOSSILIZE': 22, 'CULL': 14}
  Successful: {'WAIT': 23, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 14}
  Policy loss: -0.0055, Value loss: 33.8065, Entropy: 1.8417, Entropy coef: 0.0632
[13:48:18] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[13:48:18] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[13:48:18] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[13:48:18] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[13:48:25] env0_seed_0 | Stage transition: TRAINING → BLENDING
[13:48:25] env1_seed_0 | Stage transition: TRAINING → BLENDING
[13:48:33] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[13:48:33] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[13:48:36] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:48:36] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:48:36] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:48:36] env1_seed_0 | Fossilized (norm, Δacc +10.60%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +10.60%)
[13:48:40] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:48:40] env0_seed_0 | Fossilized (norm, Δacc +17.31%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +17.31%)
Batch 55: Episodes 110/200
  Env accuracies: ['75.8%', '74.6%']
  Avg acc: 75.2% (rolling: 75.8%)
  Avg reward: 166.1
  Actions: {'WAIT': 22, 'GERMINATE_NORM': 27, 'GERMINATE_ATTENTION': 26, 'GERMINATE_DEPTHWISE': 17, 'GERMINATE_CONV_ENHANCE': 25, 'FOSSILIZE': 26, 'CULL': 7}
  Successful: {'WAIT': 22, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 7}
  Policy loss: -0.0257, Value loss: 43.1504, Entropy: 1.8071, Entropy coef: 0.0607

Blueprint Stats
  ---------------------------------------------------------------------------

Blueprint       Germ  Foss  Cull   Rate     ΔAcc    Churn
  ---------------------------------------------------------------------------

  attention         88     6    80   7.0%   +2.87%   +0.35%
  conv_enhance      70     6    60   9.1%   +8.04%   -2.45%
  depthwise         84     4    70   5.4%   +3.86%   +0.24%
  norm             254    59   182  24.5%  +15.56%   +5.04%
Seed Scoreboard (env 0):
  Fossilized: 38 (+463.1K params, +488.7% of host)
  Culled: 195
  Avg fossilize age: 14.6 epochs
  Avg cull age: 6.3 epochs
  Compute cost: 3.65x baseline
  Distribution: attention x3, conv_enhance x6, norm x27, depthwise x2
Seed Scoreboard (env 1):
  Fossilized: 37 (+19.9K params, +21.0% of host)
  Culled: 197
  Avg fossilize age: 13.9 epochs
  Avg cull age: 5.8 epochs
  Compute cost: 2.85x baseline
  Distribution: attention x3, norm x32, depthwise x2

[13:50:24] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[13:50:24] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[13:50:24] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[13:50:24] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[13:50:30] env0_seed_0 | Stage transition: TRAINING → BLENDING
[13:50:30] env1_seed_0 | Stage transition: TRAINING → BLENDING
[13:50:39] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[13:50:39] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[13:50:42] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:50:42] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:50:44] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:50:44] env0_seed_0 | Fossilized (norm, Δacc +16.90%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +16.90%)
[13:50:49] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:50:49] env1_seed_0 | Fossilized (norm, Δacc +17.84%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +17.84%)
Batch 56: Episodes 112/200
  Env accuracies: ['75.3%', '75.5%']
  Avg acc: 75.4% (rolling: 75.7%)
  Avg reward: 167.2
  Actions: {'WAIT': 22, 'GERMINATE_NORM': 27, 'GERMINATE_ATTENTION': 28, 'GERMINATE_DEPTHWISE': 16, 'GERMINATE_CONV_ENHANCE': 21, 'FOSSILIZE': 29, 'CULL': 7}
  Successful: {'WAIT': 22, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 7}
  Policy loss: 0.0055, Value loss: 31.0103, Entropy: 1.8213, Entropy coef: 0.0581
[13:52:30] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[13:52:30] env1_seed_0 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_0' (conv_enhance, 74.0K params)
[13:52:30] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[13:52:30] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[13:52:37] env0_seed_0 | Stage transition: TRAINING → BLENDING
[13:52:37] env1_seed_0 | Stage transition: TRAINING → BLENDING
[13:52:46] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[13:52:46] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[13:52:50] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:52:50] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:52:57] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:52:57] env0_seed_0 | Fossilized (norm, Δacc +20.15%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +20.15%)
[13:53:04] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:53:04] env1_seed_0 | Fossilized (conv_enhance, Δacc +21.50%)
    [env1] Fossilized 'env1_seed_0' (conv_enhance, Δacc +21.50%)
Batch 57: Episodes 114/200
  Env accuracies: ['75.8%', '58.8%']
  Avg acc: 67.3% (rolling: 74.8%)
  Avg reward: 64.9
  Actions: {'WAIT': 31, 'GERMINATE_NORM': 18, 'GERMINATE_ATTENTION': 28, 'GERMINATE_DEPTHWISE': 15, 'GERMINATE_CONV_ENHANCE': 29, 'FOSSILIZE': 22, 'CULL': 7}
  Successful: {'WAIT': 31, 'GERMINATE_NORM': 1, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 1, 'FOSSILIZE': 2, 'CULL': 7}
  Policy loss: -0.0310, Value loss: 224.4908, Entropy: 1.7879, Entropy coef: 0.0556
[13:54:46] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[13:54:46] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[13:54:46] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[13:54:46] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[13:54:53] env0_seed_0 | Stage transition: TRAINING → BLENDING
[13:54:53] env1_seed_0 | Stage transition: TRAINING → BLENDING
[13:55:01] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[13:55:01] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[13:55:04] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:55:04] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:55:04] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:55:04] env0_seed_0 | Fossilized (norm, Δacc +15.55%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +15.55%)
[13:55:09] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:55:09] env1_seed_0 | Fossilized (norm, Δacc +19.39%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +19.39%)
Batch 58: Episodes 116/200
  Env accuracies: ['76.8%', '74.8%']
  Avg acc: 75.8% (rolling: 74.9%)
  Avg reward: 168.2
  Actions: {'WAIT': 25, 'GERMINATE_NORM': 26, 'GERMINATE_ATTENTION': 23, 'GERMINATE_DEPTHWISE': 15, 'GERMINATE_CONV_ENHANCE': 28, 'FOSSILIZE': 24, 'CULL': 9}
  Successful: {'WAIT': 25, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 9}
  Policy loss: 0.0084, Value loss: 30.6884, Entropy: 1.8129, Entropy coef: 0.0531
[13:56:52] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[13:56:52] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[13:56:52] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[13:56:52] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[13:56:59] env0_seed_0 | Stage transition: TRAINING → BLENDING
[13:56:59] env1_seed_0 | Stage transition: TRAINING → BLENDING
[13:57:07] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[13:57:07] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[13:57:10] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:57:10] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:57:29] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:57:29] env0_seed_0 | Fossilized (norm, Δacc +21.78%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +21.78%)
[13:57:49] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:57:49] env1_seed_0 | Fossilized (norm, Δacc +23.87%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +23.87%)
Batch 59: Episodes 118/200
  Env accuracies: ['76.9%', '76.3%']
  Avg acc: 76.6% (rolling: 75.0%)
  Avg reward: 168.1
  Actions: {'WAIT': 40, 'GERMINATE_NORM': 25, 'GERMINATE_ATTENTION': 18, 'GERMINATE_DEPTHWISE': 20, 'GERMINATE_CONV_ENHANCE': 21, 'FOSSILIZE': 17, 'CULL': 9}
  Successful: {'WAIT': 40, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 9}
  Policy loss: 0.0078, Value loss: 28.7946, Entropy: 1.8023, Entropy coef: 0.0505
[13:58:58] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[13:58:58] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[13:58:58] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[13:58:58] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[13:59:05] env0_seed_0 | Stage transition: TRAINING → BLENDING
[13:59:05] env1_seed_0 | Stage transition: TRAINING → BLENDING
[13:59:13] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[13:59:13] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[13:59:16] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:59:16] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[13:59:23] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[13:59:23] env1_seed_0 | Fossilized (norm, Δacc +17.58%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +17.58%)
[13:59:32] env0_seed_0 | Stage transition: PROBATIONARY → CULLED
[13:59:32] env0_seed_0 | Culled (norm, Δacc +20.32%)
    [env0] Culled 'env0_seed_0' (norm, Δacc +20.32%)
[13:59:33] env0_seed_1 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_1' (norm, 0.1K params)
[13:59:33] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[13:59:40] env0_seed_1 | Stage transition: TRAINING → BLENDING
[13:59:48] env0_seed_1 | Stage transition: BLENDING → SHADOWING
[13:59:52] env0_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[13:59:55] env0_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[13:59:55] env0_seed_1 | Fossilized (norm, Δacc +4.78%)
    [env0] Fossilized 'env0_seed_1' (norm, Δacc +4.78%)
Batch 60: Episodes 120/200
  Env accuracies: ['75.9%', '77.3%']
  Avg acc: 76.6% (rolling: 75.1%)
  Avg reward: 168.0
  Actions: {'WAIT': 39, 'GERMINATE_NORM': 27, 'GERMINATE_ATTENTION': 25, 'GERMINATE_DEPTHWISE': 17, 'GERMINATE_CONV_ENHANCE': 15, 'FOSSILIZE': 14, 'CULL': 13}
  Successful: {'WAIT': 39, 'GERMINATE_NORM': 3, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 13}
  Policy loss: 0.0189, Value loss: 33.5888, Entropy: 1.8242, Entropy coef: 0.0480

Blueprint Stats
  ---------------------------------------------------------------------------

Blueprint       Germ  Foss  Cull   Rate     ΔAcc    Churn
  ---------------------------------------------------------------------------

  attention         88     6    80   7.0%   +2.87%   +0.35%
  conv_enhance      71     7    60  10.4%   +9.97%   -2.45%
  depthwise         84     4    70   5.4%   +3.86%   +0.24%
  norm             264    68   183  27.1%  +15.82%   +5.12%
Seed Scoreboard (env 0):
  Fossilized: 43 (+463.8K params, +489.4% of host)
  Culled: 196
  Avg fossilize age: 14.6 epochs
  Avg cull age: 6.3 epochs
  Compute cost: 3.75x baseline
  Distribution: attention x3, conv_enhance x6, norm x32, depthwise x2
Seed Scoreboard (env 1):
  Fossilized: 42 (+94.4K params, +99.6% of host)
  Culled: 197
  Avg fossilize age: 14.6 epochs
  Avg cull age: 5.8 epochs
  Compute cost: 3.08x baseline
  Distribution: attention x3, norm x36, depthwise x2, conv_enhance x1

[14:01:04] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[14:01:04] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[14:01:04] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[14:01:04] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[14:01:11] env0_seed_0 | Stage transition: TRAINING → BLENDING
[14:01:11] env1_seed_0 | Stage transition: TRAINING → BLENDING
[14:01:19] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[14:01:19] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[14:01:22] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:01:22] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:01:26] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:01:26] env0_seed_0 | Fossilized (norm, Δacc +18.14%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +18.14%)
[14:01:29] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:01:29] env1_seed_0 | Fossilized (norm, Δacc +17.08%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +17.08%)
Batch 61: Episodes 122/200
  Env accuracies: ['75.2%', '74.6%']
  Avg acc: 74.9% (rolling: 75.0%)
  Avg reward: 168.4
  Actions: {'WAIT': 38, 'GERMINATE_NORM': 23, 'GERMINATE_ATTENTION': 26, 'GERMINATE_DEPTHWISE': 17, 'GERMINATE_CONV_ENHANCE': 16, 'FOSSILIZE': 20, 'CULL': 10}
  Successful: {'WAIT': 38, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 10}
  Policy loss: -0.0357, Value loss: 43.7299, Entropy: 1.8127, Entropy coef: 0.0455
[14:03:10] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[14:03:10] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[14:03:10] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[14:03:10] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[14:03:16] env0_seed_0 | Stage transition: TRAINING → BLENDING
[14:03:16] env1_seed_0 | Stage transition: TRAINING → BLENDING
[14:03:25] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[14:03:25] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[14:03:28] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:03:28] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:03:32] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:03:32] env1_seed_0 | Fossilized (norm, Δacc +15.99%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +15.99%)
[14:03:33] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:03:33] env0_seed_0 | Fossilized (norm, Δacc +19.97%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +19.97%)
Batch 62: Episodes 124/200
  Env accuracies: ['76.7%', '76.6%']
  Avg acc: 76.6% (rolling: 75.1%)
  Avg reward: 170.6
  Actions: {'WAIT': 26, 'GERMINATE_NORM': 24, 'GERMINATE_ATTENTION': 29, 'GERMINATE_DEPTHWISE': 17, 'GERMINATE_CONV_ENHANCE': 25, 'FOSSILIZE': 20, 'CULL': 9}
  Successful: {'WAIT': 26, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 9}
  Policy loss: -0.0029, Value loss: 31.3772, Entropy: 1.8152, Entropy coef: 0.0429
[14:05:16] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[14:05:16] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[14:05:16] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[14:05:16] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[14:05:22] env0_seed_0 | Stage transition: TRAINING → BLENDING
[14:05:22] env1_seed_0 | Stage transition: TRAINING → BLENDING
[14:05:31] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[14:05:31] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[14:05:34] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:05:34] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:05:36] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:05:36] env0_seed_0 | Fossilized (norm, Δacc +13.31%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +13.31%)
[14:05:37] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:05:37] env1_seed_0 | Fossilized (norm, Δacc +15.23%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +15.23%)
Batch 63: Episodes 126/200
  Env accuracies: ['74.4%', '76.2%']
  Avg acc: 75.3% (rolling: 75.0%)
  Avg reward: 165.1
  Actions: {'WAIT': 22, 'GERMINATE_NORM': 23, 'GERMINATE_ATTENTION': 25, 'GERMINATE_DEPTHWISE': 23, 'GERMINATE_CONV_ENHANCE': 17, 'FOSSILIZE': 34, 'CULL': 6}
  Successful: {'WAIT': 22, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 6}
  Policy loss: -0.0025, Value loss: 30.4139, Entropy: 1.8233, Entropy coef: 0.0404
[14:07:21] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[14:07:21] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[14:07:21] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[14:07:21] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[14:07:28] env0_seed_0 | Stage transition: TRAINING → BLENDING
[14:07:28] env1_seed_0 | Stage transition: TRAINING → BLENDING
[14:07:36] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[14:07:36] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[14:07:39] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:07:39] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:07:39] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:07:39] env1_seed_0 | Fossilized (norm, Δacc +17.37%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +17.37%)
[14:07:43] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:07:43] env0_seed_0 | Fossilized (norm, Δacc +14.04%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +14.04%)
Batch 64: Episodes 128/200
  Env accuracies: ['75.1%', '75.6%']
  Avg acc: 75.3% (rolling: 74.9%)
  Avg reward: 169.5
  Actions: {'WAIT': 34, 'GERMINATE_NORM': 25, 'GERMINATE_ATTENTION': 24, 'GERMINATE_DEPTHWISE': 16, 'GERMINATE_CONV_ENHANCE': 17, 'FOSSILIZE': 31, 'CULL': 3}
  Successful: {'WAIT': 34, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 3}
  Policy loss: 0.0014, Value loss: 35.1676, Entropy: 1.8088, Entropy coef: 0.0379
[14:09:27] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[14:09:27] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[14:09:27] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[14:09:27] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[14:09:33] env0_seed_0 | Stage transition: TRAINING → BLENDING
[14:09:33] env1_seed_0 | Stage transition: TRAINING → BLENDING
[14:09:42] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[14:09:42] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[14:09:45] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:09:45] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:09:50] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:09:50] env1_seed_0 | Fossilized (norm, Δacc +18.50%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +18.50%)
[14:09:52] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:09:52] env0_seed_0 | Fossilized (norm, Δacc +21.18%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +21.18%)
Batch 65: Episodes 130/200
  Env accuracies: ['75.2%', '77.1%']
  Avg acc: 76.1% (rolling: 75.0%)
  Avg reward: 169.7
  Actions: {'WAIT': 34, 'GERMINATE_NORM': 12, 'GERMINATE_ATTENTION': 28, 'GERMINATE_DEPTHWISE': 17, 'GERMINATE_CONV_ENHANCE': 24, 'FOSSILIZE': 31, 'CULL': 4}
  Successful: {'WAIT': 34, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 4}
  Policy loss: 0.0126, Value loss: 33.0383, Entropy: 1.8108, Entropy coef: 0.0353

Blueprint Stats
  ---------------------------------------------------------------------------

Blueprint       Germ  Foss  Cull   Rate     ΔAcc    Churn
  ---------------------------------------------------------------------------

  attention         88     6    80   7.0%   +2.87%   +0.35%
  conv_enhance      71     7    60  10.4%   +9.97%   -2.45%
  depthwise         84     4    70   5.4%   +3.86%   +0.24%
  norm             274    78   183  29.9%  +15.98%   +5.12%
Seed Scoreboard (env 0):
  Fossilized: 48 (+464.4K params, +490.1% of host)
  Culled: 196
  Avg fossilize age: 14.5 epochs
  Avg cull age: 6.3 epochs
  Compute cost: 3.85x baseline
  Distribution: attention x3, conv_enhance x6, norm x37, depthwise x2
Seed Scoreboard (env 1):
  Fossilized: 47 (+95.0K params, +100.3% of host)
  Culled: 197
  Avg fossilize age: 14.4 epochs
  Avg cull age: 5.8 epochs
  Compute cost: 3.18x baseline
  Distribution: attention x3, norm x41, depthwise x2, conv_enhance x1

[14:11:33] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[14:11:33] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[14:11:33] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[14:11:33] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[14:11:39] env0_seed_0 | Stage transition: TRAINING → BLENDING
[14:11:41] env1_seed_0 | Stage transition: TRAINING → BLENDING
[14:11:48] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[14:11:49] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[14:11:51] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:11:51] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:11:51] env0_seed_0 | Fossilized (norm, Δacc +16.65%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +16.65%)
[14:11:53] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:11:56] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:11:56] env1_seed_0 | Fossilized (norm, Δacc +18.85%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +18.85%)
Batch 66: Episodes 132/200
  Env accuracies: ['75.2%', '76.3%']
  Avg acc: 75.7% (rolling: 75.0%)
  Avg reward: 167.1
  Actions: {'WAIT': 27, 'GERMINATE_NORM': 21, 'GERMINATE_ATTENTION': 21, 'GERMINATE_DEPTHWISE': 22, 'GERMINATE_CONV_ENHANCE': 22, 'FOSSILIZE': 27, 'CULL': 10}
  Successful: {'WAIT': 27, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 10}
  Policy loss: -0.0118, Value loss: 32.6592, Entropy: 1.8122, Entropy coef: 0.0328
[14:13:38] env0_seed_0 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_0' (attention, 2.0K params)
[14:13:38] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[14:13:38] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[14:13:38] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[14:13:45] env0_seed_0 | Stage transition: TRAINING → BLENDING
[14:13:45] env1_seed_0 | Stage transition: TRAINING → BLENDING
[14:13:53] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[14:13:53] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[14:13:57] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:13:57] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:13:57] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:13:57] env0_seed_0 | Fossilized (attention, Δacc +12.45%)
    [env0] Fossilized 'env0_seed_0' (attention, Δacc +12.45%)
[14:14:00] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:14:00] env1_seed_0 | Fossilized (norm, Δacc +23.48%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +23.48%)
Batch 67: Episodes 134/200
  Env accuracies: ['64.1%', '72.0%']
  Avg acc: 68.1% (rolling: 75.1%)
  Avg reward: 153.3
  Actions: {'WAIT': 35, 'GERMINATE_NORM': 27, 'GERMINATE_ATTENTION': 24, 'GERMINATE_DEPTHWISE': 18, 'GERMINATE_CONV_ENHANCE': 14, 'FOSSILIZE': 27, 'CULL': 5}
  Successful: {'WAIT': 35, 'GERMINATE_NORM': 1, 'GERMINATE_ATTENTION': 1, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 5}
  Policy loss: -0.0100, Value loss: 41.5201, Entropy: 1.7865, Entropy coef: 0.0303
[14:15:44] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[14:15:44] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[14:15:44] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[14:15:44] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[14:15:51] env0_seed_0 | Stage transition: TRAINING → BLENDING
[14:15:51] env1_seed_0 | Stage transition: TRAINING → BLENDING
[14:15:59] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[14:15:59] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[14:16:02] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:16:02] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:16:02] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:16:02] env1_seed_0 | Fossilized (attention, Δacc +11.83%)
    [env1] Fossilized 'env1_seed_0' (attention, Δacc +11.83%)
[14:16:14] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:16:14] env0_seed_0 | Fossilized (norm, Δacc +20.27%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +20.27%)
Batch 68: Episodes 136/200
  Env accuracies: ['76.4%', '72.5%']
  Avg acc: 74.5% (rolling: 75.0%)
  Avg reward: 164.0
  Actions: {'WAIT': 29, 'GERMINATE_NORM': 17, 'GERMINATE_ATTENTION': 31, 'GERMINATE_DEPTHWISE': 13, 'GERMINATE_CONV_ENHANCE': 23, 'FOSSILIZE': 32, 'CULL': 5}
  Successful: {'WAIT': 29, 'GERMINATE_NORM': 1, 'GERMINATE_ATTENTION': 1, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 5}
  Policy loss: -0.0067, Value loss: 32.0382, Entropy: 1.7767, Entropy coef: 0.0277
[14:17:49] env0_seed_0 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_0' (attention, 2.0K params)
[14:17:49] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[14:17:49] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[14:17:49] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[14:17:56] env1_seed_0 | Stage transition: TRAINING → BLENDING
[14:17:57] env0_seed_0 | Stage transition: TRAINING → BLENDING
[14:18:04] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[14:18:06] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[14:18:07] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:18:09] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:18:17] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:18:17] env0_seed_0 | Fossilized (attention, Δacc +15.70%)
    [env0] Fossilized 'env0_seed_0' (attention, Δacc +15.70%)
[14:18:24] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:18:24] env1_seed_0 | Fossilized (norm, Δacc +23.59%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +23.59%)
Batch 69: Episodes 138/200
  Env accuracies: ['68.8%', '76.0%']
  Avg acc: 72.4% (rolling: 74.6%)
  Avg reward: 161.2
  Actions: {'WAIT': 29, 'GERMINATE_NORM': 14, 'GERMINATE_ATTENTION': 41, 'GERMINATE_DEPTHWISE': 17, 'GERMINATE_CONV_ENHANCE': 22, 'FOSSILIZE': 23, 'CULL': 4}
  Successful: {'WAIT': 29, 'GERMINATE_NORM': 1, 'GERMINATE_ATTENTION': 1, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 4}
  Policy loss: -0.0066, Value loss: 40.1078, Entropy: 1.7442, Entropy coef: 0.0252
[14:19:55] env0_seed_0 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_0' (attention, 2.0K params)
[14:19:55] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[14:19:55] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[14:19:55] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[14:20:01] env0_seed_0 | Stage transition: TRAINING → BLENDING
[14:20:01] env1_seed_0 | Stage transition: TRAINING → BLENDING
[14:20:10] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[14:20:10] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[14:20:13] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:20:13] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:20:18] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:20:18] env0_seed_0 | Fossilized (attention, Δacc +17.45%)
    [env0] Fossilized 'env0_seed_0' (attention, Δacc +17.45%)
[14:20:23] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:20:23] env1_seed_0 | Fossilized (attention, Δacc +10.64%)
    [env1] Fossilized 'env1_seed_0' (attention, Δacc +10.64%)
Batch 70: Episodes 140/200
  Env accuracies: ['73.5%', '73.7%']
  Avg acc: 73.6% (rolling: 74.2%)
  Avg reward: 163.3
  Actions: {'WAIT': 35, 'GERMINATE_NORM': 6, 'GERMINATE_ATTENTION': 42, 'GERMINATE_DEPTHWISE': 30, 'GERMINATE_CONV_ENHANCE': 13, 'FOSSILIZE': 20, 'CULL': 4}
  Successful: {'WAIT': 35, 'GERMINATE_NORM': 0, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 4}
  Policy loss: -0.0149, Value loss: 41.4212, Entropy: 1.6699, Entropy coef: 0.0227

Blueprint Stats
  ---------------------------------------------------------------------------

Blueprint       Germ  Foss  Cull   Rate     ΔAcc    Churn
  ---------------------------------------------------------------------------

  attention         93    11    80  12.1%   +7.75%   +0.35%
  conv_enhance      71     7    60  10.4%   +9.97%   -2.45%
  depthwise         84     4    70   5.4%   +3.86%   +0.24%
  norm             279    83   183  31.2%  +16.26%   +5.12%
Seed Scoreboard (env 0):
  Fossilized: 53 (+470.8K params, +496.8% of host)
  Culled: 196
  Avg fossilize age: 14.4 epochs
  Avg cull age: 6.3 epochs
  Compute cost: 4.94x baseline
  Distribution: attention x6, conv_enhance x6, norm x39, depthwise x2
Seed Scoreboard (env 1):
  Fossilized: 52 (+99.5K params, +105.0% of host)
  Culled: 197
  Avg fossilize age: 14.5 epochs
  Avg cull age: 5.8 epochs
  Compute cost: 3.94x baseline
  Distribution: attention x5, norm x44, depthwise x2, conv_enhance x1

[14:21:59] env0_seed_0 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_0' (attention, 2.0K params)
[14:21:59] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[14:21:59] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[14:21:59] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[14:22:06] env1_seed_0 | Stage transition: TRAINING → BLENDING
[14:22:08] env0_seed_0 | Stage transition: TRAINING → BLENDING
[14:22:14] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[14:22:16] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[14:22:18] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:22:19] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:22:21] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:22:21] env1_seed_0 | Fossilized (attention, Δacc +12.89%)
    [env1] Fossilized 'env1_seed_0' (attention, Δacc +12.89%)
[14:22:23] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:22:23] env0_seed_0 | Fossilized (attention, Δacc +3.93%)
    [env0] Fossilized 'env0_seed_0' (attention, Δacc +3.93%)
Batch 71: Episodes 142/200
  Env accuracies: ['68.9%', '75.9%']
  Avg acc: 72.4% (rolling: 74.0%)
  Avg reward: 159.7
  Actions: {'WAIT': 24, 'GERMINATE_NORM': 16, 'GERMINATE_ATTENTION': 52, 'GERMINATE_DEPTHWISE': 16, 'GERMINATE_CONV_ENHANCE': 18, 'FOSSILIZE': 23, 'CULL': 1}
  Successful: {'WAIT': 24, 'GERMINATE_NORM': 0, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 1}
  Policy loss: -0.0016, Value loss: 38.8102, Entropy: 1.6228, Entropy coef: 0.0201
[14:24:04] env0_seed_0 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_0' (attention, 2.0K params)
[14:24:04] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[14:24:04] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[14:24:04] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[14:24:11] env0_seed_0 | Stage transition: TRAINING → BLENDING
[14:24:11] env1_seed_0 | Stage transition: TRAINING → BLENDING
[14:24:19] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[14:24:19] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[14:24:22] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:24:22] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:24:24] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:24:24] env0_seed_0 | Fossilized (attention, Δacc +11.32%)
    [env0] Fossilized 'env0_seed_0' (attention, Δacc +11.32%)
[14:24:51] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:24:51] env1_seed_0 | Fossilized (attention, Δacc +23.92%)
    [env1] Fossilized 'env1_seed_0' (attention, Δacc +23.92%)
Batch 72: Episodes 144/200
  Env accuracies: ['70.0%', '73.7%']
  Avg acc: 71.9% (rolling: 73.5%)
  Avg reward: 154.8
  Actions: {'WAIT': 19, 'GERMINATE_NORM': 11, 'GERMINATE_ATTENTION': 47, 'GERMINATE_DEPTHWISE': 21, 'GERMINATE_CONV_ENHANCE': 26, 'FOSSILIZE': 19, 'CULL': 7}
  Successful: {'WAIT': 19, 'GERMINATE_NORM': 0, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 7}
  Policy loss: -0.0072, Value loss: 35.2323, Entropy: 1.6273, Entropy coef: 0.0176
[14:26:09] env0_seed_0 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_0' (attention, 2.0K params)
[14:26:09] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[14:26:09] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[14:26:09] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[14:26:16] env0_seed_0 | Stage transition: TRAINING → BLENDING
[14:26:16] env1_seed_0 | Stage transition: TRAINING → BLENDING
[14:26:24] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[14:26:24] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[14:26:27] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:26:27] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:26:40] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:26:40] env0_seed_0 | Fossilized (attention, Δacc +21.50%)
    [env0] Fossilized 'env0_seed_0' (attention, Δacc +21.50%)
[14:26:40] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:26:40] env1_seed_0 | Fossilized (attention, Δacc +22.48%)
    [env1] Fossilized 'env1_seed_0' (attention, Δacc +22.48%)
Batch 73: Episodes 146/200
  Env accuracies: ['70.5%', '71.1%']
  Avg acc: 70.8% (rolling: 73.1%)
  Avg reward: 158.8
  Actions: {'WAIT': 28, 'GERMINATE_NORM': 9, 'GERMINATE_ATTENTION': 53, 'GERMINATE_DEPTHWISE': 20, 'GERMINATE_CONV_ENHANCE': 22, 'FOSSILIZE': 16, 'CULL': 2}
  Successful: {'WAIT': 28, 'GERMINATE_NORM': 0, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 2}
  Policy loss: 0.0027, Value loss: 37.3798, Entropy: 1.6378, Entropy coef: 0.0151
[14:28:14] env0_seed_0 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_0' (attention, 2.0K params)
[14:28:14] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[14:28:14] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[14:28:14] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[14:28:20] env0_seed_0 | Stage transition: TRAINING → BLENDING
[14:28:20] env1_seed_0 | Stage transition: TRAINING → BLENDING
[14:28:29] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[14:28:29] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[14:28:32] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:28:32] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:28:52] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:28:52] env1_seed_0 | Fossilized (attention, Δacc +16.57%)
    [env1] Fossilized 'env1_seed_0' (attention, Δacc +16.57%)
[14:28:57] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:28:57] env0_seed_0 | Fossilized (attention, Δacc +24.53%)
    [env0] Fossilized 'env0_seed_0' (attention, Δacc +24.53%)
Batch 74: Episodes 148/200
  Env accuracies: ['68.3%', '64.7%']
  Avg acc: 66.5% (rolling: 72.2%)
  Avg reward: 146.3
  Actions: {'WAIT': 30, 'GERMINATE_NORM': 19, 'GERMINATE_ATTENTION': 47, 'GERMINATE_DEPTHWISE': 17, 'GERMINATE_CONV_ENHANCE': 17, 'FOSSILIZE': 14, 'CULL': 6}
  Successful: {'WAIT': 30, 'GERMINATE_NORM': 0, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 6}
  Policy loss: 0.0033, Value loss: 33.4875, Entropy: 1.6089, Entropy coef: 0.0125
[14:30:18] env0_seed_0 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_0' (attention, 2.0K params)
[14:30:18] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[14:30:18] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[14:30:18] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[14:30:25] env0_seed_0 | Stage transition: TRAINING → BLENDING
[14:30:25] env1_seed_0 | Stage transition: TRAINING → BLENDING
[14:30:33] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[14:30:33] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[14:30:37] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:30:37] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:31:00] env0_seed_0 | Stage transition: PROBATIONARY → CULLED
[14:31:00] env0_seed_0 | Culled (attention, Δacc +29.13%)
    [env0] Culled 'env0_seed_0' (attention, Δacc +29.13%)
[14:31:01] env0_seed_1 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_1' (attention, 2.0K params)
[14:31:02] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[14:31:08] env0_seed_1 | Stage transition: TRAINING → BLENDING
[14:31:16] env0_seed_1 | Stage transition: BLENDING → SHADOWING
[14:31:16] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:31:16] env1_seed_0 | Fossilized (attention, Δacc +13.64%)
    [env1] Fossilized 'env1_seed_0' (attention, Δacc +13.64%)
[14:31:20] env0_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[14:31:46] env0_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[14:31:46] env0_seed_1 | Fossilized (attention, Δacc +4.63%)
    [env0] Fossilized 'env0_seed_1' (attention, Δacc +4.63%)
Batch 75: Episodes 150/200
  Env accuracies: ['74.8%', '71.1%']
  Avg acc: 72.9% (rolling: 71.9%)
  Avg reward: 152.3
  Actions: {'WAIT': 32, 'GERMINATE_NORM': 20, 'GERMINATE_ATTENTION': 55, 'GERMINATE_DEPTHWISE': 11, 'GERMINATE_CONV_ENHANCE': 20, 'FOSSILIZE': 9, 'CULL': 3}
  Successful: {'WAIT': 32, 'GERMINATE_NORM': 0, 'GERMINATE_ATTENTION': 3, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 3}
  Policy loss: -0.0181, Value loss: 51.1849, Entropy: 1.6001, Entropy coef: 0.0100

Blueprint Stats
  ---------------------------------------------------------------------------

Blueprint       Germ  Foss  Cull   Rate     ΔAcc    Churn
  ---------------------------------------------------------------------------

  attention        104    21    81  20.6%  +11.46%   +0.71%
  conv_enhance      71     7    60  10.4%   +9.97%   -2.45%
  depthwise         84     4    70   5.4%   +3.86%   +0.24%
  norm             279    83   183  31.2%  +16.26%   +5.12%
Seed Scoreboard (env 0):
  Fossilized: 58 (+481.1K params, +507.7% of host)
  Culled: 197
  Avg fossilize age: 14.9 epochs
  Avg cull age: 6.4 epochs
  Compute cost: 6.69x baseline
  Distribution: attention x11, conv_enhance x6, norm x39, depthwise x2
Seed Scoreboard (env 1):
  Fossilized: 57 (+109.7K params, +115.8% of host)
  Culled: 197
  Avg fossilize age: 15.3 epochs
  Avg cull age: 5.8 epochs
  Compute cost: 5.69x baseline
  Distribution: attention x10, norm x44, depthwise x2, conv_enhance x1

[14:32:23] env0_seed_0 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_0' (attention, 2.0K params)
[14:32:23] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[14:32:23] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[14:32:23] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[14:32:30] env0_seed_0 | Stage transition: TRAINING → BLENDING
[14:32:30] env1_seed_0 | Stage transition: TRAINING → BLENDING
[14:32:38] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[14:32:38] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[14:32:41] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:32:41] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:32:41] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:32:41] env1_seed_0 | Fossilized (attention, Δacc +12.46%)
    [env1] Fossilized 'env1_seed_0' (attention, Δacc +12.46%)
[14:33:18] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:33:18] env0_seed_0 | Fossilized (attention, Δacc +18.80%)
    [env0] Fossilized 'env0_seed_0' (attention, Δacc +18.80%)
Batch 76: Episodes 152/200
  Env accuracies: ['73.0%', '73.5%']
  Avg acc: 73.3% (rolling: 71.6%)
  Avg reward: 160.5
  Actions: {'WAIT': 30, 'GERMINATE_NORM': 11, 'GERMINATE_ATTENTION': 67, 'GERMINATE_DEPTHWISE': 14, 'GERMINATE_CONV_ENHANCE': 9, 'FOSSILIZE': 16, 'CULL': 3}
  Successful: {'WAIT': 30, 'GERMINATE_NORM': 0, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 3}
  Policy loss: -0.0063, Value loss: 35.6468, Entropy: 1.5651, Entropy coef: 0.0100
[14:34:28] env0_seed_0 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_0' (attention, 2.0K params)
[14:34:28] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[14:34:28] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[14:34:28] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[14:34:34] env0_seed_0 | Stage transition: TRAINING → BLENDING
[14:34:34] env1_seed_0 | Stage transition: TRAINING → BLENDING
[14:34:43] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[14:34:43] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[14:34:46] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:34:46] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:34:54] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:34:54] env0_seed_0 | Fossilized (attention, Δacc +14.64%)
    [env0] Fossilized 'env0_seed_0' (attention, Δacc +14.64%)
[14:34:54] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:34:54] env1_seed_0 | Fossilized (attention, Δacc +12.69%)
    [env1] Fossilized 'env1_seed_0' (attention, Δacc +12.69%)
Batch 77: Episodes 154/200
  Env accuracies: ['73.6%', '70.3%']
  Avg acc: 72.0% (rolling: 72.0%)
  Avg reward: 161.7
  Actions: {'WAIT': 36, 'GERMINATE_NORM': 13, 'GERMINATE_ATTENTION': 63, 'GERMINATE_DEPTHWISE': 13, 'GERMINATE_CONV_ENHANCE': 12, 'FOSSILIZE': 12, 'CULL': 1}
  Successful: {'WAIT': 36, 'GERMINATE_NORM': 0, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 1}
  Policy loss: 0.0237, Value loss: 34.0801, Entropy: 1.5623, Entropy coef: 0.0100
[14:36:33] env0_seed_0 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_0' (attention, 2.0K params)
[14:36:33] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[14:36:33] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[14:36:33] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[14:36:40] env0_seed_0 | Stage transition: TRAINING → BLENDING
[14:36:40] env1_seed_0 | Stage transition: TRAINING → BLENDING
[14:36:48] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[14:36:48] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[14:36:51] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:36:51] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:37:30] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:37:30] env0_seed_0 | Fossilized (attention, Δacc +20.91%)
    [env0] Fossilized 'env0_seed_0' (attention, Δacc +20.91%)
[14:37:43] env1_seed_0 | Stage transition: PROBATIONARY → CULLED
[14:37:43] env1_seed_0 | Culled (attention, Δacc +20.06%)
    [env1] Culled 'env1_seed_0' (attention, Δacc +20.06%)
[14:37:46] env1_seed_1 | Germinated (conv_enhance, 74.0K params)
    [env1] Germinated 'env1_seed_1' (conv_enhance, 74.0K params)
[14:37:46] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[14:37:57] env1_seed_1 | Stage transition: TRAINING → BLENDING
[14:38:06] env1_seed_1 | Stage transition: BLENDING → SHADOWING
[14:38:10] env1_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[14:38:35] env1_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[14:38:35] env1_seed_1 | Fossilized (conv_enhance, Δacc +0.20%)
    [env1] Fossilized 'env1_seed_1' (conv_enhance, Δacc +0.20%)
Batch 78: Episodes 156/200
  Env accuracies: ['74.0%', '62.6%']
  Avg acc: 68.3% (rolling: 71.4%)
  Avg reward: 121.4
  Actions: {'WAIT': 42, 'GERMINATE_NORM': 16, 'GERMINATE_ATTENTION': 48, 'GERMINATE_DEPTHWISE': 14, 'GERMINATE_CONV_ENHANCE': 16, 'FOSSILIZE': 12, 'CULL': 2}
  Successful: {'WAIT': 42, 'GERMINATE_NORM': 0, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 1, 'FOSSILIZE': 2, 'CULL': 2}
  Policy loss: -0.0151, Value loss: 64.9357, Entropy: 1.5761, Entropy coef: 0.0100
[14:38:42] env0_seed_0 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_0' (attention, 2.0K params)
[14:38:42] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[14:38:42] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[14:38:42] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[14:38:49] env0_seed_0 | Stage transition: TRAINING → BLENDING
[14:38:49] env1_seed_0 | Stage transition: TRAINING → BLENDING
[14:38:57] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[14:38:57] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[14:39:01] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:39:01] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:39:01] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:39:01] env0_seed_0 | Fossilized (attention, Δacc +8.60%)
    [env0] Fossilized 'env0_seed_0' (attention, Δacc +8.60%)
[14:39:14] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:39:14] env1_seed_0 | Fossilized (attention, Δacc +17.44%)
    [env1] Fossilized 'env1_seed_0' (attention, Δacc +17.44%)
Batch 79: Episodes 158/200
  Env accuracies: ['74.6%', '71.9%']
  Avg acc: 73.3% (rolling: 71.5%)
  Avg reward: 162.1
  Actions: {'WAIT': 27, 'GERMINATE_NORM': 14, 'GERMINATE_ATTENTION': 66, 'GERMINATE_DEPTHWISE': 14, 'GERMINATE_CONV_ENHANCE': 8, 'FOSSILIZE': 18, 'CULL': 3}
  Successful: {'WAIT': 27, 'GERMINATE_NORM': 0, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 3}
  Policy loss: -0.0127, Value loss: 38.6722, Entropy: 1.6169, Entropy coef: 0.0100
[14:40:48] env0_seed_0 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_0' (attention, 2.0K params)
[14:40:48] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[14:40:48] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[14:40:48] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[14:40:54] env0_seed_0 | Stage transition: TRAINING → BLENDING
[14:40:54] env1_seed_0 | Stage transition: TRAINING → BLENDING
[14:41:02] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[14:41:02] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[14:41:06] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:41:06] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:41:22] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:41:22] env1_seed_0 | Fossilized (attention, Δacc +16.41%)
    [env1] Fossilized 'env1_seed_0' (attention, Δacc +16.41%)
[14:41:29] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:41:29] env0_seed_0 | Fossilized (attention, Δacc +18.86%)
    [env0] Fossilized 'env0_seed_0' (attention, Δacc +18.86%)
Batch 80: Episodes 160/200
  Env accuracies: ['71.3%', '74.7%']
  Avg acc: 73.0% (rolling: 71.4%)
  Avg reward: 159.9
  Actions: {'WAIT': 32, 'GERMINATE_NORM': 14, 'GERMINATE_ATTENTION': 58, 'GERMINATE_DEPTHWISE': 14, 'GERMINATE_CONV_ENHANCE': 11, 'FOSSILIZE': 18, 'CULL': 3}
  Successful: {'WAIT': 32, 'GERMINATE_NORM': 0, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 3}
  Policy loss: -0.0208, Value loss: 37.6519, Entropy: 1.6013, Entropy coef: 0.0100

Blueprint Stats
  ---------------------------------------------------------------------------

Blueprint       Germ  Foss  Cull   Rate     ΔAcc    Churn
  ---------------------------------------------------------------------------

  attention        114    30    82  26.8%  +12.72%   +0.94%
  conv_enhance      72     8    60  11.8%   +8.75%   -2.45%
  depthwise         84     4    70   5.4%   +3.86%   +0.24%
  norm             279    83   183  31.2%  +16.26%   +5.12%
Seed Scoreboard (env 0):
  Fossilized: 63 (+491.3K params, +518.5% of host)
  Culled: 197
  Avg fossilize age: 15.6 epochs
  Avg cull age: 6.4 epochs
  Compute cost: 8.44x baseline
  Distribution: attention x16, conv_enhance x6, norm x39, depthwise x2
Seed Scoreboard (env 1):
  Fossilized: 62 (+191.9K params, +202.5% of host)
  Culled: 198
  Avg fossilize age: 15.6 epochs
  Avg cull age: 6.0 epochs
  Compute cost: 7.24x baseline
  Distribution: attention x14, norm x44, depthwise x2, conv_enhance x2

[14:42:52] env0_seed_0 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_0' (attention, 2.0K params)
[14:42:52] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[14:42:52] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[14:42:52] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[14:42:59] env1_seed_0 | Stage transition: TRAINING → BLENDING
[14:43:00] env0_seed_0 | Stage transition: TRAINING → BLENDING
[14:43:07] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[14:43:09] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[14:43:11] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:43:12] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:43:24] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:43:24] env1_seed_0 | Fossilized (attention, Δacc +15.49%)
    [env1] Fossilized 'env1_seed_0' (attention, Δacc +15.49%)
[14:43:27] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:43:27] env0_seed_0 | Fossilized (attention, Δacc +12.96%)
    [env0] Fossilized 'env0_seed_0' (attention, Δacc +12.96%)
Batch 81: Episodes 162/200
  Env accuracies: ['73.5%', '74.6%']
  Avg acc: 74.0% (rolling: 71.6%)
  Avg reward: 164.1
  Actions: {'WAIT': 28, 'GERMINATE_NORM': 18, 'GERMINATE_ATTENTION': 51, 'GERMINATE_DEPTHWISE': 10, 'GERMINATE_CONV_ENHANCE': 22, 'FOSSILIZE': 20, 'CULL': 1}
  Successful: {'WAIT': 28, 'GERMINATE_NORM': 0, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 1}
  Policy loss: -0.0050, Value loss: 37.2187, Entropy: 1.6407, Entropy coef: 0.0100
[14:44:57] env0_seed_0 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_0' (attention, 2.0K params)
[14:44:57] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[14:44:57] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[14:44:57] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[14:45:04] env0_seed_0 | Stage transition: TRAINING → BLENDING
[14:45:04] env1_seed_0 | Stage transition: TRAINING → BLENDING
[14:45:12] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[14:45:12] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[14:45:15] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:45:15] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:45:17] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:45:17] env1_seed_0 | Fossilized (attention, Δacc +9.47%)
    [env1] Fossilized 'env1_seed_0' (attention, Δacc +9.47%)
[14:45:37] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:45:37] env0_seed_0 | Fossilized (attention, Δacc +15.14%)
    [env0] Fossilized 'env0_seed_0' (attention, Δacc +15.14%)
Batch 82: Episodes 164/200
  Env accuracies: ['70.2%', '73.1%']
  Avg acc: 71.7% (rolling: 71.6%)
  Avg reward: 156.0
  Actions: {'WAIT': 30, 'GERMINATE_NORM': 10, 'GERMINATE_ATTENTION': 50, 'GERMINATE_DEPTHWISE': 13, 'GERMINATE_CONV_ENHANCE': 21, 'FOSSILIZE': 19, 'CULL': 7}
  Successful: {'WAIT': 30, 'GERMINATE_NORM': 0, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 7}
  Policy loss: 0.0055, Value loss: 31.9935, Entropy: 1.6490, Entropy coef: 0.0100
[14:47:02] env0_seed_0 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_0' (attention, 2.0K params)
[14:47:02] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[14:47:02] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[14:47:02] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[14:47:08] env0_seed_0 | Stage transition: TRAINING → BLENDING
[14:47:08] env1_seed_0 | Stage transition: TRAINING → BLENDING
[14:47:17] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[14:47:17] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[14:47:20] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:47:20] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:47:25] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:47:25] env1_seed_0 | Fossilized (attention, Δacc +18.25%)
    [env1] Fossilized 'env1_seed_0' (attention, Δacc +18.25%)
[14:47:28] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:47:28] env0_seed_0 | Fossilized (attention, Δacc +19.51%)
    [env0] Fossilized 'env0_seed_0' (attention, Δacc +19.51%)
Batch 83: Episodes 166/200
  Env accuracies: ['74.2%', '73.8%']
  Avg acc: 74.0% (rolling: 71.9%)
  Avg reward: 165.1
  Actions: {'WAIT': 29, 'GERMINATE_NORM': 19, 'GERMINATE_ATTENTION': 52, 'GERMINATE_DEPTHWISE': 4, 'GERMINATE_CONV_ENHANCE': 16, 'FOSSILIZE': 27, 'CULL': 3}
  Successful: {'WAIT': 29, 'GERMINATE_NORM': 0, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 3}
  Policy loss: 0.0060, Value loss: 30.5191, Entropy: 1.6478, Entropy coef: 0.0100
[14:49:07] env0_seed_0 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_0' (attention, 2.0K params)
[14:49:07] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[14:49:07] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[14:49:07] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[14:49:14] env0_seed_0 | Stage transition: TRAINING → BLENDING
[14:49:14] env1_seed_0 | Stage transition: TRAINING → BLENDING
[14:49:22] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[14:49:22] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[14:49:25] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:49:25] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:49:30] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:49:30] env0_seed_0 | Fossilized (attention, Δacc +11.49%)
    [env0] Fossilized 'env0_seed_0' (attention, Δacc +11.49%)
[14:50:15] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:50:15] env1_seed_0 | Fossilized (attention, Δacc +23.68%)
    [env1] Fossilized 'env1_seed_0' (attention, Δacc +23.68%)
Batch 84: Episodes 168/200
  Env accuracies: ['71.7%', '67.2%']
  Avg acc: 69.5% (rolling: 72.2%)
  Avg reward: 152.5
  Actions: {'WAIT': 34, 'GERMINATE_NORM': 20, 'GERMINATE_ATTENTION': 41, 'GERMINATE_DEPTHWISE': 17, 'GERMINATE_CONV_ENHANCE': 24, 'FOSSILIZE': 12, 'CULL': 2}
  Successful: {'WAIT': 34, 'GERMINATE_NORM': 0, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 2}
  Policy loss: 0.0056, Value loss: 34.4680, Entropy: 1.6449, Entropy coef: 0.0100
[14:51:12] env0_seed_0 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_0' (attention, 2.0K params)
[14:51:12] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[14:51:12] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[14:51:12] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[14:51:19] env0_seed_0 | Stage transition: TRAINING → BLENDING
[14:51:19] env1_seed_0 | Stage transition: TRAINING → BLENDING
[14:51:27] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[14:51:27] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[14:51:30] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:51:30] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:51:42] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:51:42] env1_seed_0 | Fossilized (attention, Δacc +14.32%)
    [env1] Fossilized 'env1_seed_0' (attention, Δacc +14.32%)
[14:51:54] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:51:54] env0_seed_0 | Fossilized (attention, Δacc +19.13%)
    [env0] Fossilized 'env0_seed_0' (attention, Δacc +19.13%)
Batch 85: Episodes 170/200
  Env accuracies: ['75.6%', '72.7%']
  Avg acc: 74.2% (rolling: 72.3%)
  Avg reward: 166.3
  Actions: {'WAIT': 44, 'GERMINATE_NORM': 10, 'GERMINATE_ATTENTION': 41, 'GERMINATE_DEPTHWISE': 13, 'GERMINATE_CONV_ENHANCE': 24, 'FOSSILIZE': 17, 'CULL': 1}
  Successful: {'WAIT': 44, 'GERMINATE_NORM': 0, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 1}
  Policy loss: 0.0065, Value loss: 33.1637, Entropy: 1.6261, Entropy coef: 0.0100

Blueprint Stats
  ---------------------------------------------------------------------------

Blueprint       Germ  Foss  Cull   Rate     ΔAcc    Churn
  ---------------------------------------------------------------------------

  attention        124    40    82  32.8%  +13.52%   +0.94%
  conv_enhance      72     8    60  11.8%   +8.75%   -2.45%
  depthwise         84     4    70   5.4%   +3.86%   +0.24%
  norm             279    83   183  31.2%  +16.26%   +5.12%
Seed Scoreboard (env 0):
  Fossilized: 68 (+501.5K params, +529.3% of host)
  Culled: 197
  Avg fossilize age: 15.9 epochs
  Avg cull age: 6.4 epochs
  Compute cost: 10.19x baseline
  Distribution: attention x21, conv_enhance x6, norm x39, depthwise x2
Seed Scoreboard (env 1):
  Fossilized: 67 (+202.2K params, +213.3% of host)
  Culled: 198
  Avg fossilize age: 16.0 epochs
  Avg cull age: 6.0 epochs
  Compute cost: 8.99x baseline
  Distribution: attention x19, norm x44, depthwise x2, conv_enhance x2

[14:53:17] env0_seed_0 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_0' (attention, 2.0K params)
[14:53:17] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[14:53:17] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[14:53:17] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[14:53:24] env0_seed_0 | Stage transition: TRAINING → BLENDING
[14:53:24] env1_seed_0 | Stage transition: TRAINING → BLENDING
[14:53:32] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[14:53:32] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[14:53:35] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:53:35] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:53:50] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:53:50] env1_seed_0 | Fossilized (attention, Δacc +15.86%)
    [env1] Fossilized 'env1_seed_0' (attention, Δacc +15.86%)
[14:54:00] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:54:00] env0_seed_0 | Fossilized (attention, Δacc +23.12%)
    [env0] Fossilized 'env0_seed_0' (attention, Δacc +23.12%)
Batch 86: Episodes 172/200
  Env accuracies: ['66.8%', '71.2%']
  Avg acc: 69.0% (rolling: 71.9%)
  Avg reward: 153.8
  Actions: {'WAIT': 43, 'GERMINATE_NORM': 9, 'GERMINATE_ATTENTION': 45, 'GERMINATE_DEPTHWISE': 10, 'GERMINATE_CONV_ENHANCE': 23, 'FOSSILIZE': 15, 'CULL': 5}
  Successful: {'WAIT': 43, 'GERMINATE_NORM': 0, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 5}
  Policy loss: 0.0234, Value loss: 34.4996, Entropy: 1.6086, Entropy coef: 0.0100
[14:55:22] env0_seed_0 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_0' (attention, 2.0K params)
[14:55:22] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[14:55:22] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[14:55:22] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[14:55:28] env0_seed_0 | Stage transition: TRAINING → BLENDING
[14:55:28] env1_seed_0 | Stage transition: TRAINING → BLENDING
[14:55:37] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[14:55:37] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[14:55:40] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:55:40] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:55:43] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:55:43] env1_seed_0 | Fossilized (attention, Δacc +11.33%)
    [env1] Fossilized 'env1_seed_0' (attention, Δacc +11.33%)
[14:56:02] env0_seed_0 | Stage transition: PROBATIONARY → CULLED
[14:56:02] env0_seed_0 | Culled (attention, Δacc +14.80%)
    [env0] Culled 'env0_seed_0' (attention, Δacc +14.80%)
[14:56:03] env0_seed_1 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_1' (norm, 0.1K params)
[14:56:03] env0_seed_1 | Stage transition: GERMINATED → TRAINING
[14:56:13] env0_seed_1 | Stage transition: TRAINING → BLENDING
[14:56:22] env0_seed_1 | Stage transition: BLENDING → SHADOWING
[14:56:25] env0_seed_1 | Stage transition: SHADOWING → PROBATIONARY
[14:56:30] env0_seed_1 | Stage transition: PROBATIONARY → FOSSILIZED
[14:56:30] env0_seed_1 | Fossilized (norm, Δacc +5.74%)
    [env0] Fossilized 'env0_seed_1' (norm, Δacc +5.74%)
Batch 87: Episodes 174/200
  Env accuracies: ['75.8%', '71.6%']
  Avg acc: 73.7% (rolling: 72.1%)
  Avg reward: 160.2
  Actions: {'WAIT': 32, 'GERMINATE_NORM': 19, 'GERMINATE_ATTENTION': 49, 'GERMINATE_DEPTHWISE': 9, 'GERMINATE_CONV_ENHANCE': 21, 'FOSSILIZE': 16, 'CULL': 4}
  Successful: {'WAIT': 32, 'GERMINATE_NORM': 1, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 4}
  Policy loss: 0.0046, Value loss: 41.5747, Entropy: 1.5801, Entropy coef: 0.0100
[14:57:26] env0_seed_0 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_0' (attention, 2.0K params)
[14:57:26] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[14:57:26] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[14:57:26] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[14:57:33] env0_seed_0 | Stage transition: TRAINING → BLENDING
[14:57:33] env1_seed_0 | Stage transition: TRAINING → BLENDING
[14:57:41] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[14:57:41] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[14:57:45] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:57:45] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:58:16] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:58:16] env1_seed_0 | Fossilized (attention, Δacc +23.79%)
    [env1] Fossilized 'env1_seed_0' (attention, Δacc +23.79%)
[14:58:20] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:58:20] env0_seed_0 | Fossilized (attention, Δacc +22.01%)
    [env0] Fossilized 'env0_seed_0' (attention, Δacc +22.01%)
Batch 88: Episodes 176/200
  Env accuracies: ['69.9%', '72.7%']
  Avg acc: 71.3% (rolling: 72.4%)
  Avg reward: 155.9
  Actions: {'WAIT': 40, 'GERMINATE_NORM': 12, 'GERMINATE_ATTENTION': 57, 'GERMINATE_DEPTHWISE': 9, 'GERMINATE_CONV_ENHANCE': 16, 'FOSSILIZE': 15, 'CULL': 1}
  Successful: {'WAIT': 40, 'GERMINATE_NORM': 0, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 1}
  Policy loss: -0.0107, Value loss: 32.3911, Entropy: 1.5884, Entropy coef: 0.0100
[14:59:31] env0_seed_0 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_0' (attention, 2.0K params)
[14:59:31] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[14:59:31] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[14:59:31] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[14:59:38] env0_seed_0 | Stage transition: TRAINING → BLENDING
[14:59:38] env1_seed_0 | Stage transition: TRAINING → BLENDING
[14:59:46] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[14:59:46] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[14:59:49] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:59:49] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[14:59:58] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[14:59:58] env1_seed_0 | Fossilized (attention, Δacc +19.78%)
    [env1] Fossilized 'env1_seed_0' (attention, Δacc +19.78%)
[15:00:43] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[15:00:43] env0_seed_0 | Fossilized (attention, Δacc +17.57%)
    [env0] Fossilized 'env0_seed_0' (attention, Δacc +17.57%)
Batch 89: Episodes 178/200
  Env accuracies: ['65.7%', '72.9%']
  Avg acc: 69.3% (rolling: 72.0%)
  Avg reward: 150.6
  Actions: {'WAIT': 33, 'GERMINATE_NORM': 18, 'GERMINATE_ATTENTION': 41, 'GERMINATE_DEPTHWISE': 10, 'GERMINATE_CONV_ENHANCE': 30, 'FOSSILIZE': 16, 'CULL': 2}
  Successful: {'WAIT': 33, 'GERMINATE_NORM': 0, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 2}
  Policy loss: 0.0134, Value loss: 45.4161, Entropy: 1.6153, Entropy coef: 0.0100
[15:01:36] env0_seed_0 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_0' (attention, 2.0K params)
[15:01:36] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[15:01:36] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[15:01:36] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[15:01:43] env0_seed_0 | Stage transition: TRAINING → BLENDING
[15:01:44] env1_seed_0 | Stage transition: TRAINING → BLENDING
[15:01:51] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[15:01:53] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[15:01:54] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[15:01:56] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[15:02:09] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[15:02:09] env0_seed_0 | Fossilized (attention, Δacc +23.83%)
    [env0] Fossilized 'env0_seed_0' (attention, Δacc +23.83%)
[15:02:18] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[15:02:18] env1_seed_0 | Fossilized (attention, Δacc +19.91%)
    [env1] Fossilized 'env1_seed_0' (attention, Δacc +19.91%)
Batch 90: Episodes 180/200
  Env accuracies: ['74.5%', '72.4%']
  Avg acc: 73.5% (rolling: 72.0%)
  Avg reward: 163.9
  Actions: {'WAIT': 41, 'GERMINATE_NORM': 6, 'GERMINATE_ATTENTION': 44, 'GERMINATE_DEPTHWISE': 12, 'GERMINATE_CONV_ENHANCE': 25, 'FOSSILIZE': 19, 'CULL': 3}
  Successful: {'WAIT': 41, 'GERMINATE_NORM': 0, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 3}
  Policy loss: -0.0029, Value loss: 39.3127, Entropy: 1.6013, Entropy coef: 0.0100

Blueprint Stats
  ---------------------------------------------------------------------------

Blueprint       Germ  Foss  Cull   Rate     ΔAcc    Churn
  ---------------------------------------------------------------------------

  attention        134    49    83  37.1%  +14.66%   +1.11%
  conv_enhance      72     8    60  11.8%   +8.75%   -2.45%
  depthwise         84     4    70   5.4%   +3.86%   +0.24%
  norm             280    84   183  31.5%  +16.13%   +5.12%
Seed Scoreboard (env 0):
  Fossilized: 73 (+509.9K params, +538.0% of host)
  Culled: 198
  Avg fossilize age: 16.7 epochs
  Avg cull age: 6.5 epochs
  Compute cost: 11.61x baseline
  Distribution: attention x25, conv_enhance x6, norm x40, depthwise x2
Seed Scoreboard (env 1):
  Fossilized: 72 (+212.4K params, +224.1% of host)
  Culled: 198
  Avg fossilize age: 16.3 epochs
  Avg cull age: 6.0 epochs
  Compute cost: 10.74x baseline
  Distribution: attention x24, norm x44, depthwise x2, conv_enhance x2

[15:03:41] env0_seed_0 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_0' (attention, 2.0K params)
[15:03:41] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[15:03:41] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[15:03:41] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[15:03:48] env0_seed_0 | Stage transition: TRAINING → BLENDING
[15:03:48] env1_seed_0 | Stage transition: TRAINING → BLENDING
[15:03:56] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[15:03:56] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[15:03:59] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[15:03:59] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[15:04:36] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[15:04:36] env1_seed_0 | Fossilized (attention, Δacc +15.23%)
    [env1] Fossilized 'env1_seed_0' (attention, Δacc +15.23%)
[15:04:38] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[15:04:38] env0_seed_0 | Fossilized (attention, Δacc +17.22%)
    [env0] Fossilized 'env0_seed_0' (attention, Δacc +17.22%)
Batch 91: Episodes 182/200
  Env accuracies: ['74.3%', '72.8%']
  Avg acc: 73.5% (rolling: 72.0%)
  Avg reward: 159.2
  Actions: {'WAIT': 36, 'GERMINATE_NORM': 13, 'GERMINATE_ATTENTION': 58, 'GERMINATE_DEPTHWISE': 10, 'GERMINATE_CONV_ENHANCE': 21, 'FOSSILIZE': 9, 'CULL': 3}
  Successful: {'WAIT': 36, 'GERMINATE_NORM': 0, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 3}
  Policy loss: -0.0148, Value loss: 39.4585, Entropy: 1.6219, Entropy coef: 0.0100
[15:05:46] env0_seed_0 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_0' (attention, 2.0K params)
[15:05:46] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[15:05:46] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[15:05:46] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[15:05:53] env0_seed_0 | Stage transition: TRAINING → BLENDING
[15:05:53] env1_seed_0 | Stage transition: TRAINING → BLENDING
[15:06:01] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[15:06:01] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[15:06:04] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[15:06:04] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[15:06:09] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[15:06:09] env1_seed_0 | Fossilized (attention, Δacc +7.37%)
    [env1] Fossilized 'env1_seed_0' (attention, Δacc +7.37%)
[15:06:33] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[15:06:33] env0_seed_0 | Fossilized (attention, Δacc +17.38%)
    [env0] Fossilized 'env0_seed_0' (attention, Δacc +17.38%)
Batch 92: Episodes 184/200
  Env accuracies: ['69.5%', '75.5%']
  Avg acc: 72.5% (rolling: 72.0%)
  Avg reward: 161.4
  Actions: {'WAIT': 42, 'GERMINATE_NORM': 16, 'GERMINATE_ATTENTION': 45, 'GERMINATE_DEPTHWISE': 11, 'GERMINATE_CONV_ENHANCE': 17, 'FOSSILIZE': 16, 'CULL': 3}
  Successful: {'WAIT': 42, 'GERMINATE_NORM': 0, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 3}
  Policy loss: 0.0083, Value loss: 34.7435, Entropy: 1.6666, Entropy coef: 0.0100
[15:07:51] env0_seed_0 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_0' (attention, 2.0K params)
[15:07:51] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[15:07:51] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[15:07:51] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[15:07:57] env0_seed_0 | Stage transition: TRAINING → BLENDING
[15:07:57] env1_seed_0 | Stage transition: TRAINING → BLENDING
[15:08:06] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[15:08:06] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[15:08:09] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[15:08:09] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[15:08:19] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[15:08:19] env0_seed_0 | Fossilized (attention, Δacc +16.35%)
    [env0] Fossilized 'env0_seed_0' (attention, Δacc +16.35%)
[15:08:39] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[15:08:39] env1_seed_0 | Fossilized (attention, Δacc +24.85%)
    [env1] Fossilized 'env1_seed_0' (attention, Δacc +24.85%)
Batch 93: Episodes 186/200
  Env accuracies: ['63.6%', '74.5%']
  Avg acc: 69.0% (rolling: 71.5%)
  Avg reward: 155.6
  Actions: {'WAIT': 43, 'GERMINATE_NORM': 12, 'GERMINATE_ATTENTION': 43, 'GERMINATE_DEPTHWISE': 13, 'GERMINATE_CONV_ENHANCE': 28, 'FOSSILIZE': 9, 'CULL': 2}
  Successful: {'WAIT': 43, 'GERMINATE_NORM': 0, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 2}
  Policy loss: 0.0061, Value loss: 33.4560, Entropy: 1.6938, Entropy coef: 0.0100
[15:09:55] env0_seed_0 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_0' (attention, 2.0K params)
[15:09:55] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[15:09:55] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[15:09:55] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[15:10:02] env0_seed_0 | Stage transition: TRAINING → BLENDING
[15:10:02] env1_seed_0 | Stage transition: TRAINING → BLENDING
[15:10:10] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[15:10:10] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[15:10:14] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[15:10:14] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[15:10:17] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[15:10:17] env1_seed_0 | Fossilized (attention, Δacc +14.28%)
    [env1] Fossilized 'env1_seed_0' (attention, Δacc +14.28%)
[15:10:30] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[15:10:30] env0_seed_0 | Fossilized (attention, Δacc +18.16%)
    [env0] Fossilized 'env0_seed_0' (attention, Δacc +18.16%)
Batch 94: Episodes 188/200
  Env accuracies: ['74.5%', '76.8%']
  Avg acc: 75.7% (rolling: 72.2%)
  Avg reward: 167.6
  Actions: {'WAIT': 32, 'GERMINATE_NORM': 16, 'GERMINATE_ATTENTION': 37, 'GERMINATE_DEPTHWISE': 16, 'GERMINATE_CONV_ENHANCE': 29, 'FOSSILIZE': 18, 'CULL': 2}
  Successful: {'WAIT': 32, 'GERMINATE_NORM': 0, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 2}
  Policy loss: 0.0063, Value loss: 33.4514, Entropy: 1.6843, Entropy coef: 0.0100
[15:12:00] env0_seed_0 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_0' (attention, 2.0K params)
[15:12:00] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[15:12:00] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[15:12:00] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[15:12:07] env0_seed_0 | Stage transition: TRAINING → BLENDING
[15:12:07] env1_seed_0 | Stage transition: TRAINING → BLENDING
[15:12:15] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[15:12:15] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[15:12:18] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[15:12:18] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[15:12:18] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[15:12:18] env1_seed_0 | Fossilized (attention, Δacc +15.86%)
    [env1] Fossilized 'env1_seed_0' (attention, Δacc +15.86%)
[15:12:32] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[15:12:32] env0_seed_0 | Fossilized (attention, Δacc +14.02%)
    [env0] Fossilized 'env0_seed_0' (attention, Δacc +14.02%)
Batch 95: Episodes 190/200
  Env accuracies: ['73.3%', '73.6%']
  Avg acc: 73.4% (rolling: 72.1%)
  Avg reward: 166.3
  Actions: {'WAIT': 42, 'GERMINATE_NORM': 16, 'GERMINATE_ATTENTION': 35, 'GERMINATE_DEPTHWISE': 14, 'GERMINATE_CONV_ENHANCE': 25, 'FOSSILIZE': 16, 'CULL': 2}
  Successful: {'WAIT': 42, 'GERMINATE_NORM': 0, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 2}
  Policy loss: 0.0033, Value loss: 32.1085, Entropy: 1.6798, Entropy coef: 0.0100

Blueprint Stats
  ---------------------------------------------------------------------------

Blueprint       Germ  Foss  Cull   Rate     ΔAcc    Churn
  ---------------------------------------------------------------------------

  attention        144    59    83  41.5%  +14.90%   +1.11%
  conv_enhance      72     8    60  11.8%   +8.75%   -2.45%
  depthwise         84     4    70   5.4%   +3.86%   +0.24%
  norm             280    84   183  31.5%  +16.13%   +5.12%
Seed Scoreboard (env 0):
  Fossilized: 78 (+520.1K params, +548.9% of host)
  Culled: 198
  Avg fossilize age: 17.2 epochs
  Avg cull age: 6.5 epochs
  Compute cost: 13.36x baseline
  Distribution: attention x30, conv_enhance x6, norm x40, depthwise x2
Seed Scoreboard (env 1):
  Fossilized: 77 (+222.6K params, +234.9% of host)
  Culled: 198
  Avg fossilize age: 16.6 epochs
  Avg cull age: 6.0 epochs
  Compute cost: 12.49x baseline
  Distribution: attention x29, norm x44, depthwise x2, conv_enhance x2

[15:14:05] env0_seed_0 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_0' (attention, 2.0K params)
[15:14:05] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[15:14:05] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[15:14:05] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[15:14:12] env0_seed_0 | Stage transition: TRAINING → BLENDING
[15:14:12] env1_seed_0 | Stage transition: TRAINING → BLENDING
[15:14:20] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[15:14:20] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[15:14:23] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[15:14:23] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[15:14:25] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[15:14:25] env1_seed_0 | Fossilized (attention, Δacc +17.03%)
    [env1] Fossilized 'env1_seed_0' (attention, Δacc +17.03%)
[15:14:27] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[15:14:27] env0_seed_0 | Fossilized (attention, Δacc +16.30%)
    [env0] Fossilized 'env0_seed_0' (attention, Δacc +16.30%)
Batch 96: Episodes 192/200
  Env accuracies: ['72.2%', '74.7%']
  Avg acc: 73.4% (rolling: 72.5%)
  Avg reward: 166.6
  Actions: {'WAIT': 40, 'GERMINATE_NORM': 14, 'GERMINATE_ATTENTION': 40, 'GERMINATE_DEPTHWISE': 14, 'GERMINATE_CONV_ENHANCE': 15, 'FOSSILIZE': 24, 'CULL': 3}
  Successful: {'WAIT': 40, 'GERMINATE_NORM': 0, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 3}
  Policy loss: -0.0200, Value loss: 35.8602, Entropy: 1.6707, Entropy coef: 0.0100
[15:16:10] env0_seed_0 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_0' (attention, 2.0K params)
[15:16:10] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[15:16:10] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[15:16:10] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[15:16:16] env0_seed_0 | Stage transition: TRAINING → BLENDING
[15:16:16] env1_seed_0 | Stage transition: TRAINING → BLENDING
[15:16:25] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[15:16:25] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[15:16:28] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[15:16:28] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[15:16:35] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[15:16:35] env0_seed_0 | Fossilized (attention, Δacc +12.08%)
    [env0] Fossilized 'env0_seed_0' (attention, Δacc +12.08%)
[15:16:50] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[15:16:50] env1_seed_0 | Fossilized (attention, Δacc +20.52%)
    [env1] Fossilized 'env1_seed_0' (attention, Δacc +20.52%)
Batch 97: Episodes 194/200
  Env accuracies: ['71.5%', '74.9%']
  Avg acc: 73.2% (rolling: 72.5%)
  Avg reward: 163.4
  Actions: {'WAIT': 40, 'GERMINATE_NORM': 14, 'GERMINATE_ATTENTION': 39, 'GERMINATE_DEPTHWISE': 8, 'GERMINATE_CONV_ENHANCE': 28, 'FOSSILIZE': 18, 'CULL': 3}
  Successful: {'WAIT': 40, 'GERMINATE_NORM': 0, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 3}
  Policy loss: 0.0096, Value loss: 28.5042, Entropy: 1.6571, Entropy coef: 0.0100
[15:18:14] env0_seed_0 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_0' (attention, 2.0K params)
[15:18:14] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[15:18:14] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[15:18:14] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[15:18:21] env0_seed_0 | Stage transition: TRAINING → BLENDING
[15:18:21] env1_seed_0 | Stage transition: TRAINING → BLENDING
[15:18:29] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[15:18:29] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[15:18:33] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[15:18:33] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[15:18:36] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[15:18:36] env1_seed_0 | Fossilized (attention, Δacc +22.84%)
    [env1] Fossilized 'env1_seed_0' (attention, Δacc +22.84%)
[15:18:46] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[15:18:46] env0_seed_0 | Fossilized (attention, Δacc +14.59%)
    [env0] Fossilized 'env0_seed_0' (attention, Δacc +14.59%)
Batch 98: Episodes 196/200
  Env accuracies: ['69.8%', '75.4%']
  Avg acc: 72.6% (rolling: 72.6%)
  Avg reward: 164.3
  Actions: {'WAIT': 36, 'GERMINATE_NORM': 10, 'GERMINATE_ATTENTION': 51, 'GERMINATE_DEPTHWISE': 15, 'GERMINATE_CONV_ENHANCE': 13, 'FOSSILIZE': 24, 'CULL': 1}
  Successful: {'WAIT': 36, 'GERMINATE_NORM': 0, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 1}
  Policy loss: 0.0050, Value loss: 51.9454, Entropy: 1.6404, Entropy coef: 0.0100
[15:20:19] env0_seed_0 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_0' (attention, 2.0K params)
[15:20:19] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[15:20:19] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[15:20:19] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[15:20:26] env0_seed_0 | Stage transition: TRAINING → BLENDING
[15:20:26] env1_seed_0 | Stage transition: TRAINING → BLENDING
[15:20:34] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[15:20:34] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[15:20:38] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[15:20:38] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[15:20:39] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[15:20:39] env0_seed_0 | Fossilized (attention, Δacc +14.24%)
    [env0] Fossilized 'env0_seed_0' (attention, Δacc +14.24%)
[15:20:51] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[15:20:51] env1_seed_0 | Fossilized (attention, Δacc +17.94%)
    [env1] Fossilized 'env1_seed_0' (attention, Δacc +17.94%)
Batch 99: Episodes 198/200
  Env accuracies: ['71.2%', '73.7%']
  Avg acc: 72.4% (rolling: 72.9%)
  Avg reward: 163.6
  Actions: {'WAIT': 40, 'GERMINATE_NORM': 10, 'GERMINATE_ATTENTION': 46, 'GERMINATE_DEPTHWISE': 14, 'GERMINATE_CONV_ENHANCE': 21, 'FOSSILIZE': 16, 'CULL': 3}
  Successful: {'WAIT': 40, 'GERMINATE_NORM': 0, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 3}
  Policy loss: -0.0287, Value loss: 35.9935, Entropy: 1.6194, Entropy coef: 0.0100
[15:22:24] env0_seed_0 | Germinated (attention, 2.0K params)
    [env0] Germinated 'env0_seed_0' (attention, 2.0K params)
[15:22:24] env1_seed_0 | Germinated (attention, 2.0K params)
    [env1] Germinated 'env1_seed_0' (attention, 2.0K params)
[15:22:24] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[15:22:24] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[15:22:30] env0_seed_0 | Stage transition: TRAINING → BLENDING
[15:22:30] env1_seed_0 | Stage transition: TRAINING → BLENDING
[15:22:39] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[15:22:39] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[15:22:42] env0_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[15:22:42] env1_seed_0 | Stage transition: SHADOWING → PROBATIONARY
[15:22:42] env0_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[15:22:42] env0_seed_0 | Fossilized (attention, Δacc +21.80%)
    [env0] Fossilized 'env0_seed_0' (attention, Δacc +21.80%)
[15:22:55] env1_seed_0 | Stage transition: PROBATIONARY → FOSSILIZED
[15:22:55] env1_seed_0 | Fossilized (attention, Δacc +13.08%)
    [env1] Fossilized 'env1_seed_0' (attention, Δacc +13.08%)
Batch 100: Episodes 200/200
  Env accuracies: ['75.5%', '73.3%']
  Avg acc: 74.4% (rolling: 73.0%)
  Avg reward: 167.6
  Actions: {'WAIT': 39, 'GERMINATE_NORM': 6, 'GERMINATE_ATTENTION': 49, 'GERMINATE_DEPTHWISE': 15, 'GERMINATE_CONV_ENHANCE': 19, 'FOSSILIZE': 21, 'CULL': 1}
  Successful: {'WAIT': 39, 'GERMINATE_NORM': 0, 'GERMINATE_ATTENTION': 2, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_ENHANCE': 0, 'FOSSILIZE': 2, 'CULL': 1}
  Policy loss: 0.0059, Value loss: 27.8008, Entropy: 1.5775, Entropy coef: 0.0100

Blueprint Stats
  ---------------------------------------------------------------------------

Blueprint       Germ  Foss  Cull   Rate     ΔAcc    Churn
  ---------------------------------------------------------------------------

  attention        154    69    83  45.4%  +15.21%   +1.11%
  conv_enhance      72     8    60  11.8%   +8.75%   -2.45%
  depthwise         84     4    70   5.4%   +3.86%   +0.24%
  norm             280    84   183  31.5%  +16.13%   +5.12%
Seed Scoreboard (env 0):
  Fossilized: 83 (+530.3K params, +559.7% of host)
  Culled: 198
  Avg fossilize age: 17.0 epochs
  Avg cull age: 6.5 epochs
  Compute cost: 15.11x baseline
  Distribution: attention x35, conv_enhance x6, norm x40, depthwise x2
Seed Scoreboard (env 1):
  Fossilized: 82 (+232.9K params, +245.7% of host)
  Culled: 198
  Avg fossilize age: 16.6 epochs
  Avg cull age: 6.0 epochs
  Compute cost: 14.24x baseline
  Distribution: attention x34, norm x44, depthwise x2, conv_enhance x2

Loaded best weights (avg_acc=75.8%)
(.venv) john@nyx:~/esper-lite$
