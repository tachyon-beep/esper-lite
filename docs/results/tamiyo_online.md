john@nyx:~/esper-lite$ uv run src/esper/scripts/train.py ppo --vectorized --n-envs 4 --devices cuda:0 cuda:1
============================================================

PPO Vectorized Training (INVERTED CONTROL FLOW + CUDA STREAMS)
============================================================

Episodes: 100 (across 4 parallel envs)
Max epochs per episode: 25
Policy device: cuda:0
Env devices: ['cuda:0', 'cuda:1'] (2 envs per device)
Entropy coef: 0.01
Learning rate: 0.0003

Loading CIFAR-10 (4 independent DataLoaders)...
    [Kasmina] Germinated seed 'env0_seed_0' with blueprint 'attention'
    [Kasmina] Germinated seed 'env3_seed_0' with blueprint 'depthwise'
    [Kasmina] Germinated seed 'env1_seed_0' with blueprint 'depthwise'
    [Kasmina] Germinated seed 'env2_seed_0' with blueprint 'depthwise'
    [Kasmina] Culling seed 'env3_seed_0'
    [Kasmina] Germinated seed 'env3_seed_1' with blueprint 'depthwise'
    [Kasmina] Culling seed 'env0_seed_0'
    [Kasmina] Germinated seed 'env0_seed_1' with blueprint 'depthwise'
    [Kasmina] Culling seed 'env1_seed_0'
    [Kasmina] Culling seed 'env0_seed_1'
    [Kasmina] Culling seed 'env2_seed_0'
    [Kasmina] Germinated seed 'env0_seed_2' with blueprint 'norm'
    [Kasmina] Germinated seed 'env1_seed_1' with blueprint 'norm'
    [Kasmina] Germinated seed 'env2_seed_1' with blueprint 'norm'
    [Kasmina] Culling seed 'env0_seed_2'
    [Kasmina] Germinated seed 'env0_seed_3' with blueprint 'attention'
    [Kasmina] Culling seed 'env1_seed_1'
    [Kasmina] Germinated seed 'env1_seed_2' with blueprint 'conv_enhance'
    [Kasmina] Culling seed 'env3_seed_1'
    [Kasmina] Culling seed 'env0_seed_3'
    [Kasmina] Germinated seed 'env0_seed_4' with blueprint 'attention'
    [Kasmina] Culling seed 'env1_seed_2'
    [Kasmina] Culling seed 'env2_seed_1'
    [Kasmina] Germinated seed 'env1_seed_3' with blueprint 'conv_enhance'
    [Kasmina] Germinated seed 'env3_seed_2' with blueprint 'attention'
    [Kasmina] Germinated seed 'env2_seed_2' with blueprint 'norm'
Batch 1: Episodes 4/100
  Env accuracies: ['74.5%', '74.8%', '76.0%', '76.1%']
  Avg acc: 75.4% (rolling: 75.4%)
  Avg reward: 45.3
  Actions: {'WAIT': 13, 'GERMINATE_CONV': 13, 'GERMINATE_ATTENTION': 16, 'GERMINATE_NORM': 14, 'GERMINATE_DEPTHWISE': 15, 'ADVANCE': 15, 'CULL': 14}
  Policy loss: -0.0273, Value loss: 363.5148, Entropy: 1.9455
    [Kasmina] Germinated seed 'env0_seed_0' with blueprint 'attention'
    [Kasmina] Germinated seed 'env2_seed_0' with blueprint 'attention'
    [Kasmina] Germinated seed 'env1_seed_0' with blueprint 'depthwise'
    [Kasmina] Culling seed 'env2_seed_0'
    [Kasmina] Germinated seed 'env3_seed_0' with blueprint 'conv_enhance'
    [Kasmina] Culling seed 'env0_seed_0'
    [Kasmina] Germinated seed 'env2_seed_1' with blueprint 'norm'
    [Kasmina] Germinated seed 'env0_seed_1' with blueprint 'depthwise'
    [Kasmina] Culling seed 'env1_seed_0'
    [Kasmina] Culling seed 'env2_seed_1'
    [Kasmina] Germinated seed 'env2_seed_2' with blueprint 'depthwise'
    [Kasmina] Germinated seed 'env1_seed_1' with blueprint 'norm'
    [Kasmina] Culling seed 'env3_seed_0'
    [Kasmina] Culling seed 'env2_seed_2'
    [Kasmina] Germinated seed 'env2_seed_3' with blueprint 'conv_enhance'
    [Kasmina] Culling seed 'env2_seed_3'
    [Kasmina] Germinated seed 'env3_seed_1' with blueprint 'depthwise'
    [Kasmina] Culling seed 'env0_seed_1'
    [Kasmina] Germinated seed 'env0_seed_2' with blueprint 'conv_enhance'
    [Kasmina] Germinated seed 'env2_seed_4' with blueprint 'attention'
    [Kasmina] Culling seed 'env1_seed_1'
    [Kasmina] Culling seed 'env0_seed_2'
    [Kasmina] Germinated seed 'env0_seed_3' with blueprint 'depthwise'
    [Kasmina] Germinated seed 'env1_seed_2' with blueprint 'conv_enhance'
    [Kasmina] Culling seed 'env3_seed_1'
Batch 2: Episodes 8/100
  Env accuracies: ['76.2%', '70.8%', '68.9%', '72.8%']
  Avg acc: 72.2% (rolling: 73.8%)
  Avg reward: 43.8
  Actions: {'WAIT': 14, 'GERMINATE_CONV': 13, 'GERMINATE_ATTENTION': 10, 'GERMINATE_NORM': 15, 'GERMINATE_DEPTHWISE': 13, 'ADVANCE': 20, 'CULL': 15}
  Policy loss: -0.0291, Value loss: 286.8879, Entropy: 1.9419
    [Kasmina] Germinated seed 'env0_seed_0' with blueprint 'depthwise'
    [Kasmina] Germinated seed 'env1_seed_0' with blueprint 'attention'
    [Kasmina] Germinated seed 'env2_seed_0' with blueprint 'attention'
    [Kasmina] Germinated seed 'env3_seed_0' with blueprint 'attention'
    [Kasmina] Culling seed 'env3_seed_0'
    [Kasmina] Germinated seed 'env3_seed_1' with blueprint 'attention'
    [Kasmina] Culling seed 'env1_seed_0'
    [Kasmina] Culling seed 'env2_seed_0'
    [Kasmina] Germinated seed 'env1_seed_1' with blueprint 'depthwise'
    [Kasmina] Culling seed 'env1_seed_1'
    [Kasmina] Culling seed 'env3_seed_1'
    [Kasmina] Germinated seed 'env1_seed_2' with blueprint 'attention'
    [Kasmina] Germinated seed 'env3_seed_2' with blueprint 'norm'
    [Kasmina] Culling seed 'env1_seed_2'
    [Kasmina] Germinated seed 'env2_seed_1' with blueprint 'depthwise'
    [Kasmina] Culling seed 'env2_seed_1'
    [Kasmina] Germinated seed 'env1_seed_3' with blueprint 'norm'
    [Kasmina] Germinated seed 'env2_seed_2' with blueprint 'attention'
    [Kasmina] Culling seed 'env3_seed_2'
    [Kasmina] Germinated seed 'env3_seed_3' with blueprint 'norm'
    [Kasmina] Culling seed 'env2_seed_2'
    [Kasmina] Culling seed 'env0_seed_0'
    [Kasmina] Germinated seed 'env2_seed_3' with blueprint 'conv_enhance'
    [Kasmina] Germinated seed 'env0_seed_1' with blueprint 'attention'
    [Kasmina] Culling seed 'env0_seed_1'
    [Kasmina] Germinated seed 'env0_seed_2' with blueprint 'depthwise'
Batch 3: Episodes 12/100
  Env accuracies: ['75.3%', '75.3%', '72.5%', '73.6%']
  Avg acc: 74.1% (rolling: 73.9%)
  Avg reward: 44.5
  Actions: {'WAIT': 20, 'GERMINATE_CONV': 8, 'GERMINATE_ATTENTION': 14, 'GERMINATE_NORM': 11, 'GERMINATE_DEPTHWISE': 23, 'ADVANCE': 11, 'CULL': 13}
  Policy loss: -0.0036, Value loss: 138.7499, Entropy: 1.9247
    [Kasmina] Germinated seed 'env0_seed_0' with blueprint 'depthwise'
    [Kasmina] Germinated seed 'env2_seed_0' with blueprint 'depthwise'
    [Kasmina] Germinated seed 'env3_seed_0' with blueprint 'attention'
    [Kasmina] Germinated seed 'env1_seed_0' with blueprint 'norm'
    [Kasmina] Culling seed 'env3_seed_0'
    [Kasmina] Germinated seed 'env3_seed_1' with blueprint 'attention'
    [Kasmina] Culling seed 'env0_seed_0'
    [Kasmina] Culling seed 'env2_seed_0'
    [Kasmina] Culling seed 'env3_seed_1'
    [Kasmina] Germinated seed 'env0_seed_1' with blueprint 'norm'
    [Kasmina] Germinated seed 'env3_seed_2' with blueprint 'conv_enhance'
    [Kasmina] Germinated seed 'env2_seed_1' with blueprint 'conv_enhance'
    [Kasmina] Culling seed 'env3_seed_2'
    [Kasmina] Culling seed 'env0_seed_1'
    [Kasmina] Culling seed 'env2_seed_1'
    [Kasmina] Germinated seed 'env0_seed_2' with blueprint 'depthwise'
    [Kasmina] Germinated seed 'env2_seed_2' with blueprint 'attention'
    [Kasmina] Germinated seed 'env3_seed_3' with blueprint 'conv_enhance'
    [Kasmina] Culling seed 'env1_seed_0'
    [Kasmina] Germinated seed 'env1_seed_1' with blueprint 'depthwise'
    [Kasmina] Culling seed 'env3_seed_3'
    [Kasmina] Germinated seed 'env3_seed_4' with blueprint 'attention'
    [Kasmina] Culling seed 'env3_seed_4'
    [Kasmina] Germinated seed 'env3_seed_5' with blueprint 'depthwise'
Batch 4: Episodes 16/100
  Env accuracies: ['73.4%', '67.6%', '70.5%', '67.1%']
  Avg acc: 69.6% (rolling: 72.8%)
  Avg reward: 42.4
  Actions: {'WAIT': 10, 'GERMINATE_CONV': 11, 'GERMINATE_ATTENTION': 20, 'GERMINATE_NORM': 14, 'GERMINATE_DEPTHWISE': 21, 'ADVANCE': 14, 'CULL': 10}
  Policy loss: -0.0329, Value loss: 91.9899, Entropy: 1.9114
    [Kasmina] Germinated seed 'env1_seed_0' with blueprint 'attention'
    [Kasmina] Germinated seed 'env2_seed_0' with blueprint 'norm'
    [Kasmina] Germinated seed 'env3_seed_0' with blueprint 'depthwise'
    [Kasmina] Culling seed 'env3_seed_0'
    [Kasmina] Germinated seed 'env0_seed_0' with blueprint 'norm'
    [Kasmina] Germinated seed 'env3_seed_1' with blueprint 'attention'
    [Kasmina] Culling seed 'env3_seed_1'
    [Kasmina] Germinated seed 'env3_seed_2' with blueprint 'depthwise'
    [Kasmina] Culling seed 'env1_seed_0'
    [Kasmina] Culling seed 'env2_seed_0'
    [Kasmina] Culling seed 'env0_seed_0'
    [Kasmina] Germinated seed 'env1_seed_1' with blueprint 'attention'
    [Kasmina] Germinated seed 'env2_seed_1' with blueprint 'attention'
    [Kasmina] Culling seed 'env3_seed_2'
    [Kasmina] Germinated seed 'env0_seed_1' with blueprint 'depthwise'
    [Kasmina] Culling seed 'env2_seed_1'
    [Kasmina] Culling seed 'env0_seed_1'
    [Kasmina] Germinated seed 'env0_seed_2' with blueprint 'depthwise'
    [Kasmina] Culling seed 'env1_seed_1'
    [Kasmina] Germinated seed 'env3_seed_3' with blueprint 'conv_enhance'
    [Kasmina] Germinated seed 'env2_seed_2' with blueprint 'conv_enhance'
    [Kasmina] Germinated seed 'env1_seed_2' with blueprint 'conv_enhance'
    [Kasmina] Culling seed 'env0_seed_2'
    [Kasmina] Germinated seed 'env0_seed_3' with blueprint 'attention'
    [Kasmina] Culling seed 'env1_seed_2'
    [Kasmina] Germinated seed 'env1_seed_3' with blueprint 'conv_enhance'
Batch 5: Episodes 20/100
  Env accuracies: ['74.0%', '67.8%', '73.8%', '74.8%']
  Avg acc: 72.6% (rolling: 72.8%)
  Avg reward: 43.0
  Actions: {'WAIT': 12, 'GERMINATE_CONV': 8, 'GERMINATE_ATTENTION': 20, 'GERMINATE_NORM': 8, 'GERMINATE_DEPTHWISE': 20, 'ADVANCE': 16, 'CULL': 16}
  Policy loss: -0.0196, Value loss: 75.7510, Entropy: 1.9160
    [Kasmina] Germinated seed 'env0_seed_0' with blueprint 'attention'
    [Kasmina] Germinated seed 'env2_seed_0' with blueprint 'norm'
    [Kasmina] Germinated seed 'env3_seed_0' with blueprint 'depthwise'
    [Kasmina] Culling seed 'env3_seed_0'
    [Kasmina] Germinated seed 'env1_seed_0' with blueprint 'attention'
    [Kasmina] Germinated seed 'env3_seed_1' with blueprint 'depthwise'
    [Kasmina] Culling seed 'env0_seed_0'
    [Kasmina] Germinated seed 'env0_seed_1' with blueprint 'attention'
    [Kasmina] Culling seed 'env2_seed_0'
    [Kasmina] Culling seed 'env1_seed_0'
    [Kasmina] Germinated seed 'env2_seed_1' with blueprint 'depthwise'
    [Kasmina] Germinated seed 'env1_seed_1' with blueprint 'attention'
    [Kasmina] Culling seed 'env3_seed_1'
    [Kasmina] Germinated seed 'env3_seed_2' with blueprint 'attention'
Batch 6: Episodes 24/100
  Env accuracies: ['75.0%', '73.6%', '70.3%', '74.5%']
  Avg acc: 73.3% (rolling: 72.9%)
  Avg reward: 43.5
  Actions: {'WAIT': 20, 'GERMINATE_CONV': 12, 'GERMINATE_ATTENTION': 21, 'GERMINATE_NORM': 14, 'GERMINATE_DEPTHWISE': 16, 'ADVANCE': 9, 'CULL': 8}
  Policy loss: -0.0287, Value loss: 51.5386, Entropy: 1.9114
    [Kasmina] Germinated seed 'env1_seed_0' with blueprint 'attention'
    [Kasmina] Germinated seed 'env2_seed_0' with blueprint 'attention'
    [Kasmina] Germinated seed 'env3_seed_0' with blueprint 'attention'
    [Kasmina] Germinated seed 'env0_seed_0' with blueprint 'depthwise'
    [Kasmina] Culling seed 'env2_seed_0'
    [Kasmina] Germinated seed 'env2_seed_1' with blueprint 'attention'
    [Kasmina] Culling seed 'env0_seed_0'
    [Kasmina] Germinated seed 'env0_seed_1' with blueprint 'depthwise'
    [Kasmina] Culling seed 'env2_seed_1'
    [Kasmina] Culling seed 'env0_seed_1'
    [Kasmina] Germinated seed 'env2_seed_2' with blueprint 'norm'
    [Kasmina] Culling seed 'env2_seed_2'
    [Kasmina] Germinated seed 'env2_seed_3' with blueprint 'attention'
    [Kasmina] Culling seed 'env1_seed_0'
    [Kasmina] Culling seed 'env3_seed_0'
    [Kasmina] Germinated seed 'env0_seed_2' with blueprint 'attention'
    [Kasmina] Germinated seed 'env1_seed_1' with blueprint 'depthwise'
    [Kasmina] Culling seed 'env1_seed_1'
    [Kasmina] Culling seed 'env2_seed_3'
    [Kasmina] Germinated seed 'env3_seed_1' with blueprint 'depthwise'
    [Kasmina] Germinated seed 'env2_seed_4' with blueprint 'conv_enhance'
    [Kasmina] Culling seed 'env2_seed_4'
    [Kasmina] Germinated seed 'env2_seed_5' with blueprint 'depthwise'
    [Kasmina] Germinated seed 'env1_seed_2' with blueprint 'attention'
    [Kasmina] Culling seed 'env0_seed_2'
Batch 7: Episodes 28/100
  Env accuracies: ['65.8%', '72.6%', '72.4%', '68.2%']
  Avg acc: 69.8% (rolling: 72.4%)
  Avg reward: 39.7
  Actions: {'WAIT': 17, 'GERMINATE_CONV': 9, 'GERMINATE_ATTENTION': 20, 'GERMINATE_NORM': 10, 'GERMINATE_DEPTHWISE': 16, 'ADVANCE': 12, 'CULL': 16}
  Policy loss: -0.0178, Value loss: 60.9470, Entropy: 1.9023
    [Kasmina] Germinated seed 'env2_seed_0' with blueprint 'attention'
    [Kasmina] Germinated seed 'env3_seed_0' with blueprint 'attention'
    [Kasmina] Germinated seed 'env0_seed_0' with blueprint 'depthwise'
    [Kasmina] Germinated seed 'env1_seed_0' with blueprint 'norm'
    [Kasmina] Culling seed 'env3_seed_0'
    [Kasmina] Culling seed 'env0_seed_0'
    [Kasmina] Germinated seed 'env3_seed_1' with blueprint 'conv_enhance'
    [Kasmina] Germinated seed 'env0_seed_1' with blueprint 'conv_enhance'
    [Kasmina] Culling seed 'env3_seed_1'
    [Kasmina] Culling seed 'env0_seed_1'
    [Kasmina] Germinated seed 'env0_seed_2' with blueprint 'attention'
    [Kasmina] Germinated seed 'env3_seed_2' with blueprint 'attention'
    [Kasmina] Culling seed 'env3_seed_2'
    [Kasmina] Culling seed 'env1_seed_0'
    [Kasmina] Germinated seed 'env1_seed_1' with blueprint 'norm'
    [Kasmina] Germinated seed 'env3_seed_3' with blueprint 'depthwise'
    [Kasmina] Culling seed 'env0_seed_2'
    [Kasmina] Germinated seed 'env0_seed_3' with blueprint 'norm'
    [Kasmina] Culling seed 'env0_seed_3'
    [Kasmina] Culling seed 'env3_seed_3'
    [Kasmina] Germinated seed 'env0_seed_4' with blueprint 'depthwise'
    [Kasmina] Germinated seed 'env3_seed_4' with blueprint 'conv_enhance'
Batch 8: Episodes 32/100
  Env accuracies: ['71.5%', '76.7%', '74.0%', '75.5%']
  Avg acc: 74.4% (rolling: 72.7%)
  Avg reward: 46.1
  Actions: {'WAIT': 10, 'GERMINATE_CONV': 8, 'GERMINATE_ATTENTION': 25, 'GERMINATE_NORM': 13, 'GERMINATE_DEPTHWISE': 10, 'ADVANCE': 23, 'CULL': 11}
  Policy loss: -0.0391, Value loss: 60.8759, Entropy: 1.8908
    [Kasmina] Germinated seed 'env1_seed_0' with blueprint 'attention'
    [Kasmina] Germinated seed 'env0_seed_0' with blueprint 'attention'
    [Kasmina] Germinated seed 'env3_seed_0' with blueprint 'attention'
    [Kasmina] Culling seed 'env3_seed_0'
    [Kasmina] Germinated seed 'env2_seed_0' with blueprint 'norm'
    [Kasmina] Germinated seed 'env3_seed_1' with blueprint 'norm'
    [Kasmina] Culling seed 'env0_seed_0'
    [Kasmina] Culling seed 'env2_seed_0'
    [Kasmina] Germinated seed 'env0_seed_1' with blueprint 'attention'
    [Kasmina] Culling seed 'env0_seed_1'
    [Kasmina] Culling seed 'env3_seed_1'
    [Kasmina] Germinated seed 'env0_seed_2' with blueprint 'norm'
    [Kasmina] Germinated seed 'env2_seed_1' with blueprint 'norm'
    [Kasmina] Germinated seed 'env3_seed_2' with blueprint 'norm'
    [Kasmina] Culling seed 'env1_seed_0'
    [Kasmina] Germinated seed 'env1_seed_1' with blueprint 'conv_enhance'
    [Kasmina] Culling seed 'env2_seed_1'
    [Kasmina] Germinated seed 'env2_seed_2' with blueprint 'attention'
    [Kasmina] Culling seed 'env2_seed_2'
    [Kasmina] Germinated seed 'env2_seed_3' with blueprint 'depthwise'
    [Kasmina] Culling seed 'env0_seed_2'
    [Kasmina] Germinated seed 'env0_seed_3' with blueprint 'conv_enhance'
Batch 9: Episodes 36/100
  Env accuracies: ['71.4%', '75.1%', '71.4%', '75.8%']
  Avg acc: 73.4% (rolling: 72.8%)
  Avg reward: 44.2
  Actions: {'WAIT': 15, 'GERMINATE_CONV': 17, 'GERMINATE_ATTENTION': 16, 'GERMINATE_NORM': 11, 'GERMINATE_DEPTHWISE': 5, 'ADVANCE': 19, 'CULL': 17}
  Policy loss: -0.0331, Value loss: 38.4392, Entropy: 1.8893
    [Kasmina] Germinated seed 'env0_seed_0' with blueprint 'attention'
    [Kasmina] Germinated seed 'env3_seed_0' with blueprint 'attention'
    [Kasmina] Germinated seed 'env1_seed_0' with blueprint 'attention'
    [Kasmina] Germinated seed 'env2_seed_0' with blueprint 'depthwise'
    [Kasmina] Culling seed 'env3_seed_0'
    [Kasmina] Culling seed 'env0_seed_0'
    [Kasmina] Culling seed 'env1_seed_0'
    [Kasmina] Culling seed 'env2_seed_0'
    [Kasmina] Germinated seed 'env3_seed_1' with blueprint 'norm'
    [Kasmina] Germinated seed 'env0_seed_1' with blueprint 'norm'
    [Kasmina] Germinated seed 'env1_seed_1' with blueprint 'norm'
    [Kasmina] Germinated seed 'env2_seed_1' with blueprint 'attention'
    [Kasmina] Culling seed 'env3_seed_1'
    [Kasmina] Germinated seed 'env3_seed_2' with blueprint 'conv_enhance'
    [Kasmina] Culling seed 'env1_seed_1'
    [Kasmina] Germinated seed 'env1_seed_2' with blueprint 'conv_enhance'
    [Kasmina] Culling seed 'env1_seed_2'
    [Kasmina] Germinated seed 'env1_seed_3' with blueprint 'attention'
    [Kasmina] Culling seed 'env0_seed_1'
    [Kasmina] Culling seed 'env1_seed_3'
    [Kasmina] Germinated seed 'env1_seed_4' with blueprint 'attention'
    [Kasmina] Germinated seed 'env0_seed_2' with blueprint 'norm'
    [Kasmina] Culling seed 'env3_seed_2'
    [Kasmina] Culling seed 'env2_seed_1'
    [Kasmina] Germinated seed 'env3_seed_3' with blueprint 'norm'
    [Kasmina] Culling seed 'env0_seed_2'
    [Kasmina] Germinated seed 'env2_seed_2' with blueprint 'attention'
Batch 10: Episodes 40/100
  Env accuracies: ['66.0%', '71.5%', '73.5%', '72.5%']
  Avg acc: 70.9% (rolling: 72.6%)
  Avg reward: 41.5
  Actions: {'WAIT': 11, 'GERMINATE_CONV': 15, 'GERMINATE_ATTENTION': 21, 'GERMINATE_NORM': 12, 'GERMINATE_DEPTHWISE': 8, 'ADVANCE': 15, 'CULL': 18}
  Policy loss: -0.0141, Value loss: 40.0132, Entropy: 1.8473
    [Kasmina] Germinated seed 'env3_seed_0' with blueprint 'attention'
    [Kasmina] Germinated seed 'env0_seed_0' with blueprint 'depthwise'
    [Kasmina] Germinated seed 'env1_seed_0' with blueprint 'attention'
    [Kasmina] Germinated seed 'env2_seed_0' with blueprint 'attention'
    [Kasmina] Culling seed 'env2_seed_0'
    [Kasmina] Germinated seed 'env2_seed_1' with blueprint 'conv_enhance'
    [Kasmina] Culling seed 'env3_seed_0'
    [Kasmina] Germinated seed 'env3_seed_1' with blueprint 'attention'
    [Kasmina] Culling seed 'env3_seed_1'
    [Kasmina] Germinated seed 'env3_seed_2' with blueprint 'depthwise'
    [Kasmina] Culling seed 'env0_seed_0'
    [Kasmina] Culling seed 'env3_seed_2'
    [Kasmina] Culling seed 'env1_seed_0'
    [Kasmina] Germinated seed 'env3_seed_3' with blueprint 'attention'
    [Kasmina] Germinated seed 'env0_seed_1' with blueprint 'depthwise'
    [Kasmina] Germinated seed 'env1_seed_1' with blueprint 'depthwise'
    [Kasmina] Culling seed 'env2_seed_1'
    [Kasmina] Culling seed 'env0_seed_1'
    [Kasmina] Culling seed 'env1_seed_1'
    [Kasmina] Germinated seed 'env0_seed_2' with blueprint 'depthwise'
    [Kasmina] Germinated seed 'env2_seed_2' with blueprint 'depthwise'
    [Kasmina] Germinated seed 'env1_seed_2' with blueprint 'norm'
    [Kasmina] Culling seed 'env2_seed_2'
    [Kasmina] Culling seed 'env3_seed_3'
    [Kasmina] Germinated seed 'env2_seed_3' with blueprint 'depthwise'
Batch 11: Episodes 44/100
  Env accuracies: ['76.0%', '76.8%', '71.7%', '72.6%']
  Avg acc: 74.3% (rolling: 72.5%)
  Avg reward: 44.6
  Actions: {'WAIT': 18, 'GERMINATE_CONV': 8, 'GERMINATE_ATTENTION': 17, 'GERMINATE_NORM': 9, 'GERMINATE_DEPTHWISE': 14, 'ADVANCE': 16, 'CULL': 18}
  Policy loss: -0.0141, Value loss: 44.8008, Entropy: 1.8128
    [Kasmina] Germinated seed 'env1_seed_0' with blueprint 'attention'
    [Kasmina] Germinated seed 'env2_seed_0' with blueprint 'attention'
    [Kasmina] Culling seed 'env1_seed_0'
    [Kasmina] Culling seed 'env2_seed_0'
    [Kasmina] Germinated seed 'env3_seed_0' with blueprint 'conv_enhance'
    [Kasmina] Germinated seed 'env0_seed_0' with blueprint 'conv_enhance'
    [Kasmina] Germinated seed 'env1_seed_1' with blueprint 'attention'
    [Kasmina] Culling seed 'env1_seed_1'
    [Kasmina] Germinated seed 'env2_seed_1' with blueprint 'depthwise'
    [Kasmina] Culling seed 'env3_seed_0'
    [Kasmina] Culling seed 'env0_seed_0'
    [Kasmina] Germinated seed 'env3_seed_1' with blueprint 'attention'
    [Kasmina] Germinated seed 'env1_seed_2' with blueprint 'conv_enhance'
    [Kasmina] Germinated seed 'env0_seed_1' with blueprint 'depthwise'
    [Kasmina] Culling seed 'env0_seed_1'
    [Kasmina] Culling seed 'env2_seed_1'
    [Kasmina] Culling seed 'env3_seed_1'
    [Kasmina] Germinated seed 'env0_seed_2' with blueprint 'norm'
    [Kasmina] Culling seed 'env1_seed_2'
    [Kasmina] Germinated seed 'env2_seed_2' with blueprint 'depthwise'
    [Kasmina] Germinated seed 'env3_seed_2' with blueprint 'norm'
    [Kasmina] Culling seed 'env2_seed_2'
    [Kasmina] Culling seed 'env3_seed_2'
    [Kasmina] Germinated seed 'env1_seed_3' with blueprint 'attention'
    [Kasmina] Germinated seed 'env2_seed_3' with blueprint 'norm'
    [Kasmina] Culling seed 'env1_seed_3'
    [Kasmina] Culling seed 'env2_seed_3'
    [Kasmina] Germinated seed 'env3_seed_3' with blueprint 'conv_enhance'
    [Kasmina] Germinated seed 'env1_seed_4' with blueprint 'attention'
    [Kasmina] Germinated seed 'env2_seed_4' with blueprint 'attention'
    [Kasmina] Culling seed 'env3_seed_3'
    [Kasmina] Germinated seed 'env3_seed_4' with blueprint 'attention'
Batch 12: Episodes 48/100
  Env accuracies: ['73.9%', '75.3%', '70.7%', '75.7%']
  Avg acc: 73.9% (rolling: 72.6%)
  Avg reward: 42.2
  Actions: {'WAIT': 7, 'GERMINATE_CONV': 11, 'GERMINATE_ATTENTION': 12, 'GERMINATE_NORM': 15, 'GERMINATE_DEPTHWISE': 11, 'ADVANCE': 11, 'CULL': 33}
  Policy loss: 0.0067, Value loss: 51.2483, Entropy: 1.8009
    [Kasmina] Germinated seed 'env3_seed_0' with blueprint 'attention'
    [Kasmina] Germinated seed 'env0_seed_0' with blueprint 'depthwise'
    [Kasmina] Culling seed 'env0_seed_0'
    [Kasmina] Germinated seed 'env0_seed_1' with blueprint 'conv_enhance'
    [Kasmina] Germinated seed 'env2_seed_0' with blueprint 'attention'
    [Kasmina] Culling seed 'env3_seed_0'
    [Kasmina] Culling seed 'env0_seed_1'
    [Kasmina] Culling seed 'env2_seed_0'
    [Kasmina] Germinated seed 'env1_seed_0' with blueprint 'depthwise'
    [Kasmina] Germinated seed 'env2_seed_1' with blueprint 'attention'
    [Kasmina] Germinated seed 'env3_seed_1' with blueprint 'attention'
    [Kasmina] Culling seed 'env1_seed_0'
    [Kasmina] Culling seed 'env3_seed_1'
    [Kasmina] Germinated seed 'env3_seed_2' with blueprint 'attention'
    [Kasmina] Germinated seed 'env0_seed_2' with blueprint 'depthwise'
    [Kasmina] Germinated seed 'env1_seed_1' with blueprint 'attention'
    [Kasmina] Culling seed 'env2_seed_1'
    [Kasmina] Culling seed 'env1_seed_1'
    [Kasmina] Culling seed 'env3_seed_2'
    [Kasmina] Germinated seed 'env1_seed_2' with blueprint 'norm'
    [Kasmina] Germinated seed 'env3_seed_3' with blueprint 'depthwise'
    [Kasmina] Culling seed 'env0_seed_2'
    [Kasmina] Culling seed 'env1_seed_2'
    [Kasmina] Germinated seed 'env2_seed_2' with blueprint 'conv_enhance'
    [Kasmina] Germinated seed 'env0_seed_3' with blueprint 'depthwise'
    [Kasmina] Germinated seed 'env1_seed_3' with blueprint 'depthwise'
    [Kasmina] Culling seed 'env2_seed_2'
    [Kasmina] Culling seed 'env3_seed_3'
    [Kasmina] Germinated seed 'env2_seed_3' with blueprint 'attention'
    [Kasmina] Germinated seed 'env3_seed_4' with blueprint 'attention'
    [Kasmina] Culling seed 'env0_seed_3'
    [Kasmina] Culling seed 'env3_seed_4'
Batch 13: Episodes 52/100
  Env accuracies: ['72.7%', '64.5%', '74.3%', '75.0%']
  Avg acc: 71.6% (rolling: 72.4%)
  Avg reward: 41.4
  Actions: {'WAIT': 9, 'GERMINATE_CONV': 9, 'GERMINATE_ATTENTION': 16, 'GERMINATE_NORM': 10, 'GERMINATE_DEPTHWISE': 9, 'ADVANCE': 16, 'CULL': 31}
  Policy loss: -0.0043, Value loss: 46.4004, Entropy: 1.8130
    [Kasmina] Germinated seed 'env1_seed_0' with blueprint 'norm'
    [Kasmina] Germinated seed 'env2_seed_0' with blueprint 'norm'
    [Kasmina] Culling seed 'env1_seed_0'
    [Kasmina] Germinated seed 'env1_seed_1' with blueprint 'depthwise'
    [Kasmina] Germinated seed 'env3_seed_0' with blueprint 'norm'
    [Kasmina] Germinated seed 'env0_seed_0' with blueprint 'conv_enhance'
    [Kasmina] Culling seed 'env1_seed_1'
    [Kasmina] Culling seed 'env2_seed_0'
    [Kasmina] Culling seed 'env3_seed_0'
    [Kasmina] Culling seed 'env0_seed_0'
    [Kasmina] Germinated seed 'env2_seed_1' with blueprint 'attention'
    [Kasmina] Germinated seed 'env3_seed_1' with blueprint 'norm'
    [Kasmina] Germinated seed 'env0_seed_1' with blueprint 'attention'
    [Kasmina] Culling seed 'env3_seed_1'
    [Kasmina] Germinated seed 'env3_seed_2' with blueprint 'attention'
    [Kasmina] Culling seed 'env0_seed_1'
    [Kasmina] Germinated seed 'env1_seed_2' with blueprint 'attention'
    [Kasmina] Culling seed 'env1_seed_2'
    [Kasmina] Germinated seed 'env1_seed_3' with blueprint 'attention'
    [Kasmina] Germinated seed 'env0_seed_2' with blueprint 'norm'
    [Kasmina] Culling seed 'env2_seed_1'
    [Kasmina] Culling seed 'env0_seed_2'
    [Kasmina] Germinated seed 'env2_seed_2' with blueprint 'attention'
    [Kasmina] Germinated seed 'env0_seed_3' with blueprint 'attention'
    [Kasmina] Culling seed 'env2_seed_2'
    [Kasmina] Culling seed 'env3_seed_2'
    [Kasmina] Culling seed 'env1_seed_3'
    [Kasmina] Germinated seed 'env3_seed_3' with blueprint 'attention'
    [Kasmina] Germinated seed 'env2_seed_3' with blueprint 'conv_enhance'
    [Kasmina] Culling seed 'env0_seed_3'
    [Kasmina] Culling seed 'env2_seed_3'
    [Kasmina] Germinated seed 'env0_seed_4' with blueprint 'norm'
Batch 14: Episodes 56/100
  Env accuracies: ['70.0%', '69.7%', '70.6%', '75.8%']
  Avg acc: 71.5% (rolling: 72.6%)
  Avg reward: 40.3
  Actions: {'WAIT': 7, 'GERMINATE_CONV': 13, 'GERMINATE_ATTENTION': 17, 'GERMINATE_NORM': 12, 'GERMINATE_DEPTHWISE': 3, 'ADVANCE': 17, 'CULL': 31}
  Policy loss: -0.0065, Value loss: 46.8666, Entropy: 1.8063
    [Kasmina] Germinated seed 'env1_seed_0' with blueprint 'attention'
    [Kasmina] Germinated seed 'env0_seed_0' with blueprint 'attention'
    [Kasmina] Germinated seed 'env2_seed_0' with blueprint 'attention'
    [Kasmina] Germinated seed 'env3_seed_0' with blueprint 'attention'
    [Kasmina] Culling seed 'env1_seed_0'
    [Kasmina] Germinated seed 'env1_seed_1' with blueprint 'conv_enhance'
    [Kasmina] Culling seed 'env3_seed_0'
    [Kasmina] Culling seed 'env0_seed_0'
    [Kasmina] Culling seed 'env1_seed_1'
    [Kasmina] Germinated seed 'env3_seed_1' with blueprint 'attention'
    [Kasmina] Germinated seed 'env0_seed_1' with blueprint 'norm'
    [Kasmina] Culling seed 'env2_seed_0'
    [Kasmina] Culling seed 'env3_seed_1'
    [Kasmina] Germinated seed 'env1_seed_2' with blueprint 'attention'
    [Kasmina] Germinated seed 'env2_seed_1' with blueprint 'norm'
    [Kasmina] Germinated seed 'env3_seed_2' with blueprint 'attention'
    [Kasmina] Culling seed 'env3_seed_2'
    [Kasmina] Culling seed 'env1_seed_2'
    [Kasmina] Germinated seed 'env3_seed_3' with blueprint 'attention'
    [Kasmina] Culling seed 'env0_seed_1'
    [Kasmina] Germinated seed 'env1_seed_3' with blueprint 'conv_enhance'
    [Kasmina] Germinated seed 'env0_seed_2' with blueprint 'attention'
    [Kasmina] Culling seed 'env3_seed_3'
    [Kasmina] Culling seed 'env2_seed_1'
    [Kasmina] Culling seed 'env0_seed_2'
    [Kasmina] Culling seed 'env1_seed_3'
    [Kasmina] Germinated seed 'env2_seed_2' with blueprint 'norm'
    [Kasmina] Germinated seed 'env0_seed_3' with blueprint 'attention'
    [Kasmina] Germinated seed 'env1_seed_4' with blueprint 'conv_enhance'
    [Kasmina] Germinated seed 'env3_seed_4' with blueprint 'attention'
Batch 15: Episodes 60/100
  Env accuracies: ['76.7%', '73.6%', '74.9%', '74.9%']
  Avg acc: 75.0% (rolling: 72.8%)
  Avg reward: 44.5
  Actions: {'WAIT': 10, 'GERMINATE_CONV': 11, 'GERMINATE_ATTENTION': 22, 'GERMINATE_NORM': 9, 'GERMINATE_DEPTHWISE': 11, 'ADVANCE': 14, 'CULL': 23}
  Policy loss: -0.0184, Value loss: 35.9512, Entropy: 1.8140
    [Kasmina] Germinated seed 'env0_seed_0' with blueprint 'attention'
    [Kasmina] Germinated seed 'env2_seed_0' with blueprint 'attention'
    [Kasmina] Culling seed 'env0_seed_0'
    [Kasmina] Germinated seed 'env3_seed_0' with blueprint 'attention'
    [Kasmina] Germinated seed 'env1_seed_0' with blueprint 'norm'
    [Kasmina] Culling seed 'env2_seed_0'
    [Kasmina] Culling seed 'env3_seed_0'
    [Kasmina] Germinated seed 'env0_seed_1' with blueprint 'attention'
    [Kasmina] Culling seed 'env1_seed_0'
    [Kasmina] Germinated seed 'env3_seed_1' with blueprint 'norm'
    [Kasmina] Germinated seed 'env2_seed_1' with blueprint 'norm'
    [Kasmina] Germinated seed 'env1_seed_1' with blueprint 'conv_enhance'
    [Kasmina] Culling seed 'env2_seed_1'
    [Kasmina] Culling seed 'env0_seed_1'
    [Kasmina] Culling seed 'env1_seed_1'
    [Kasmina] Germinated seed 'env0_seed_2' with blueprint 'attention'
    [Kasmina] Germinated seed 'env2_seed_2' with blueprint 'depthwise'
    [Kasmina] Germinated seed 'env1_seed_2' with blueprint 'conv_enhance'
    [Kasmina] Culling seed 'env0_seed_2'
    [Kasmina] Culling seed 'env2_seed_2'
    [Kasmina] Germinated seed 'env0_seed_3' with blueprint 'norm'
    [Kasmina] Germinated seed 'env2_seed_3' with blueprint 'conv_enhance'
    [Kasmina] Culling seed 'env0_seed_3'
    [Kasmina] Germinated seed 'env0_seed_4' with blueprint 'depthwise'
    [Kasmina] Culling seed 'env1_seed_2'
    [Kasmina] Germinated seed 'env1_seed_3' with blueprint 'attention'
    [Kasmina] Culling seed 'env1_seed_3'
    [Kasmina] Germinated seed 'env1_seed_4' with blueprint 'attention'
    [Kasmina] Culling seed 'env3_seed_1'
Batch 16: Episodes 64/100
  Env accuracies: ['72.9%', '74.7%', '68.3%', '75.7%']
  Avg acc: 72.9% (rolling: 72.8%)
  Avg reward: 43.0
  Actions: {'WAIT': 13, 'GERMINATE_CONV': 13, 'GERMINATE_ATTENTION': 20, 'GERMINATE_NORM': 13, 'GERMINATE_DEPTHWISE': 11, 'ADVANCE': 14, 'CULL': 16}
  Policy loss: -0.0019, Value loss: 38.7453, Entropy: 1.8407
    [Kasmina] Germinated seed 'env2_seed_0' with blueprint 'attention'
    [Kasmina] Germinated seed 'env0_seed_0' with blueprint 'attention'
    [Kasmina] Germinated seed 'env1_seed_0' with blueprint 'attention'
    [Kasmina] Culling seed 'env0_seed_0'
    [Kasmina] Culling seed 'env1_seed_0'
    [Kasmina] Germinated seed 'env3_seed_0' with blueprint 'attention'
    [Kasmina] Germinated seed 'env1_seed_1' with blueprint 'attention'
    [Kasmina] Germinated seed 'env0_seed_1' with blueprint 'attention'
    [Kasmina] Culling seed 'env1_seed_1'
    [Kasmina] Germinated seed 'env1_seed_2' with blueprint 'attention'
    [Kasmina] Culling seed 'env1_seed_2'
    [Kasmina] Culling seed 'env2_seed_0'
    [Kasmina] Culling seed 'env3_seed_0'
    [Kasmina] Germinated seed 'env3_seed_1' with blueprint 'norm'
    [Kasmina] Germinated seed 'env1_seed_3' with blueprint 'norm'
    [Kasmina] Culling seed 'env1_seed_3'
    [Kasmina] Germinated seed 'env2_seed_1' with blueprint 'attention'
    [Kasmina] Culling seed 'env3_seed_1'
    [Kasmina] Culling seed 'env0_seed_1'
    [Kasmina] Germinated seed 'env1_seed_4' with blueprint 'depthwise'
    [Kasmina] Germinated seed 'env3_seed_2' with blueprint 'attention'
    [Kasmina] Germinated seed 'env0_seed_2' with blueprint 'norm'
    [Kasmina] Culling seed 'env1_seed_4'
    [Kasmina] Germinated seed 'env1_seed_5' with blueprint 'conv_enhance'
    [Kasmina] Culling seed 'env3_seed_2'
    [Kasmina] Culling seed 'env2_seed_1'
    [Kasmina] Germinated seed 'env3_seed_3' with blueprint 'attention'
    [Kasmina] Culling seed 'env0_seed_2'
    [Kasmina] Germinated seed 'env2_seed_2' with blueprint 'attention'
    [Kasmina] Culling seed 'env3_seed_3'
    [Kasmina] Germinated seed 'env0_seed_3' with blueprint 'norm'
Batch 17: Episodes 68/100
  Env accuracies: ['73.2%', '74.6%', '76.7%', '74.9%']
  Avg acc: 74.8% (rolling: 73.3%)
  Avg reward: 45.2
  Actions: {'WAIT': 13, 'GERMINATE_CONV': 5, 'GERMINATE_ATTENTION': 23, 'GERMINATE_NORM': 9, 'GERMINATE_DEPTHWISE': 5, 'ADVANCE': 23, 'CULL': 22}
  Policy loss: -0.0200, Value loss: 40.6019, Entropy: 1.8423
