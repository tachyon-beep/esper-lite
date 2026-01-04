
======

 Stage transition: TRAINING → BLENDING
[21:00:32] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[21:00:34] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[21:00:36] env1_seed_0 | Stage transition: SHADOWING → HOLDING
[21:00:36] env1_seed_0 | Stage transition: HOLDING → FOSSILIZED
[21:00:36] env1_seed_0 | Fossilized (norm, Δacc +16.77%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +16.77%)
[21:00:37] env0_seed_0 | Stage transition: SHADOWING → HOLDING
[21:00:37] env0_seed_0 | Stage transition: HOLDING → FOSSILIZED
[21:00:37] env0_seed_0 | Fossilized (norm, Δacc +16.30%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +16.30%)
Batch 84: Episodes 168/200
  Env accuracies: ['71.0%', '75.7%']
  Avg acc: 73.3% (rolling: 75.4%)
  Avg reward: 181.4
  Actions: {'WAIT': 130, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_LIGHT': 0, 'GERMINATE_CONV_HEAVY': 0, 'FOSSILIZE': 2, 'CULL': 16}
  Successful: {'WAIT': 130, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_LIGHT': 0, 'GERMINATE_CONV_HEAVY': 0, 'FOSSILIZE': 2, 'CULL': 16}
  Policy loss: 0.0016, Value loss: 31.8347, Entropy: 0.3081, Entropy coef: 0.1000
[21:02:23] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[21:02:23] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[21:02:23] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[21:02:23] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[21:02:30] env0_seed_0 | Stage transition: TRAINING → BLENDING
[21:02:30] env1_seed_0 | Stage transition: TRAINING → BLENDING
[21:02:38] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[21:02:38] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[21:02:38] env1_seed_0 | Stage transition: SHADOWING → PRUNED
[21:02:38] env1_seed_0 | Pruned (norm, Δacc +15.77%)
    [env1] Pruned 'env1_seed_0' (norm, Δacc +15.77%)
[21:02:41] env0_seed_0 | Stage transition: SHADOWING → HOLDING
[21:02:43] env0_seed_0 | Stage transition: HOLDING → FOSSILIZED
[21:02:43] env0_seed_0 | Fossilized (norm, Δacc +16.77%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +16.77%)
[21:02:43] env1_seed_1 | Germinated (conv_heavy, 74.0K params)
    [env1] Germinated 'env1_seed_1' (conv_heavy, 74.0K params)
[21:02:43] env1_seed_1 | Stage transition: GERMINATED → TRAINING
[21:02:50] env1_seed_1 | Stage transition: TRAINING → BLENDING
[21:02:59] env1_seed_1 | Stage transition: BLENDING → SHADOWING
[21:03:03] env1_seed_1 | Stage transition: SHADOWING → HOLDING
[21:03:03] env1_seed_1 | Stage transition: HOLDING → FOSSILIZED
[21:03:03] env1_seed_1 | Fossilized (conv_heavy, Δacc +4.34%)
    [env1] Fossilized 'env1_seed_1' (conv_heavy, Δacc +4.34%)
Batch 85: Episodes 170/200
  Env accuracies: ['74.5%', '67.7%']
  Avg acc: 71.1% (rolling: 74.9%)
  Avg reward: 160.2
  Actions: {'WAIT': 131, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_LIGHT': 0, 'GERMINATE_CONV_HEAVY': 1, 'FOSSILIZE': 2, 'CULL': 14}
  Successful: {'WAIT': 131, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_LIGHT': 0, 'GERMINATE_CONV_HEAVY': 1, 'FOSSILIZE': 2, 'CULL': 14}
  Policy loss: 0.0275, Value loss: 57.9421, Entropy: 0.3000, Entropy coef: 0.1000

Blueprint Stats
  ---------------------------------------------------------------------------

Blueprint       Germ  Foss  Cull   Rate     ΔAcc    Churn
  ---------------------------------------------------------------------------

  attention        250     1   236   0.4%   +0.05%   +0.70%
  conv_heavy       171     2   159   1.2%   +3.45%   +0.78%
  conv_light       133     1   127   0.8%   +4.46%   +0.55%
  depthwise        165     1   153   0.6%   +7.56%   -0.37%
  norm             510    88   400  18.0%  +14.69%   +2.93%
Seed Scoreboard (env 0):
  Fossilized: 48 (+84.7K params, +89.4% of host)
  Pruned: 527
  Avg fossilize age: 11.2 epochs
  Avg cull age: 4.1 epochs
  Compute cost: 2.10x baseline
  Distribution: norm x46, conv_heavy x1, depthwise x1
Seed Scoreboard (env 1):
  Fossilized: 45 (+118.4K params, +125.0% of host)
  Pruned: 548
  Avg fossilize age: 11.3 epochs
  Avg cull age: 4.3 epochs
  Compute cost: 2.39x baseline
  Distribution: norm x42, conv_light x1, attention x1, conv_heavy x1

[21:04:37] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[21:04:37] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[21:04:37] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[21:04:37] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[21:04:44] env0_seed_0 | Stage transition: TRAINING → BLENDING
[21:04:44] env1_seed_0 | Stage transition: TRAINING → BLENDING
[21:04:52] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[21:04:52] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[21:04:56] env0_seed_0 | Stage transition: SHADOWING → HOLDING
[21:04:56] env1_seed_0 | Stage transition: SHADOWING → HOLDING
[21:04:56] env0_seed_0 | Stage transition: HOLDING → FOSSILIZED
[21:04:56] env0_seed_0 | Fossilized (norm, Δacc +11.91%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +11.91%)
[21:04:56] env1_seed_0 | Stage transition: HOLDING → FOSSILIZED
[21:04:56] env1_seed_0 | Fossilized (norm, Δacc +13.07%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +13.07%)
Batch 86: Episodes 172/200
  Env accuracies: ['74.3%', '76.4%']
  Avg acc: 75.3% (rolling: 74.9%)
  Avg reward: 184.2
  Actions: {'WAIT': 131, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_LIGHT': 0, 'GERMINATE_CONV_HEAVY': 0, 'FOSSILIZE': 2, 'CULL': 15}
  Successful: {'WAIT': 131, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_LIGHT': 0, 'GERMINATE_CONV_HEAVY': 0, 'FOSSILIZE': 2, 'CULL': 15}
  Policy loss: -0.0265, Value loss: 46.1750, Entropy: 0.2814, Entropy coef: 0.1000
[21:06:43] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[21:06:43] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[21:06:43] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[21:06:43] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[21:06:50] env0_seed_0 | Stage transition: TRAINING → BLENDING
[21:06:50] env1_seed_0 | Stage transition: TRAINING → BLENDING
[21:06:58] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[21:06:58] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[21:07:01] env0_seed_0 | Stage transition: SHADOWING → HOLDING
[21:07:01] env1_seed_0 | Stage transition: SHADOWING → HOLDING
[21:07:01] env0_seed_0 | Stage transition: HOLDING → FOSSILIZED
[21:07:01] env0_seed_0 | Fossilized (norm, Δacc +14.56%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +14.56%)
[21:07:01] env1_seed_0 | Stage transition: HOLDING → FOSSILIZED
[21:07:01] env1_seed_0 | Fossilized (norm, Δacc +13.88%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +13.88%)
Batch 87: Episodes 174/200
  Env accuracies: ['76.8%', '77.3%']
  Avg acc: 77.1% (rolling: 75.2%)
  Avg reward: 186.7
  Actions: {'WAIT': 129, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_LIGHT': 0, 'GERMINATE_CONV_HEAVY': 0, 'FOSSILIZE': 2, 'CULL': 17}
  Successful: {'WAIT': 129, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_LIGHT': 0, 'GERMINATE_CONV_HEAVY': 0, 'FOSSILIZE': 2, 'CULL': 17}
  Policy loss: 0.0169, Value loss: 27.1253, Entropy: 0.3145, Entropy coef: 0.1000
[21:08:49] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[21:08:49] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[21:08:49] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[21:08:49] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[21:08:55] env0_seed_0 | Stage transition: TRAINING → BLENDING
[21:08:55] env1_seed_0 | Stage transition: TRAINING → BLENDING
[21:09:04] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[21:09:04] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[21:09:07] env0_seed_0 | Stage transition: SHADOWING → HOLDING
[21:09:07] env1_seed_0 | Stage transition: SHADOWING → HOLDING
[21:09:07] env0_seed_0 | Stage transition: HOLDING → FOSSILIZED
[21:09:07] env0_seed_0 | Fossilized (norm, Δacc +17.61%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +17.61%)
[21:09:07] env1_seed_0 | Stage transition: HOLDING → FOSSILIZED
[21:09:07] env1_seed_0 | Fossilized (norm, Δacc +17.50%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +17.50%)
Batch 88: Episodes 176/200
  Env accuracies: ['76.3%', '75.5%']
  Avg acc: 75.9% (rolling: 75.2%)
  Avg reward: 186.5
  Actions: {'WAIT': 130, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_LIGHT': 0, 'GERMINATE_CONV_HEAVY': 0, 'FOSSILIZE': 2, 'CULL': 16}
  Successful: {'WAIT': 130, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_LIGHT': 0, 'GERMINATE_CONV_HEAVY': 0, 'FOSSILIZE': 2, 'CULL': 16}
  Policy loss: 0.0012, Value loss: 31.9900, Entropy: 0.3398, Entropy coef: 0.1000
[21:10:54] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[21:10:54] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[21:10:54] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[21:10:54] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[21:11:01] env1_seed_0 | Stage transition: TRAINING → BLENDING
[21:11:03] env0_seed_0 | Stage transition: TRAINING → BLENDING
[21:11:09] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[21:11:11] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[21:11:13] env1_seed_0 | Stage transition: SHADOWING → HOLDING
[21:11:13] env1_seed_0 | Stage transition: HOLDING → FOSSILIZED
[21:11:13] env1_seed_0 | Fossilized (norm, Δacc +15.49%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +15.49%)
[21:11:14] env0_seed_0 | Stage transition: SHADOWING → HOLDING
[21:11:14] env0_seed_0 | Stage transition: HOLDING → FOSSILIZED
[21:11:14] env0_seed_0 | Fossilized (norm, Δacc +15.67%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +15.67%)
Batch 89: Episodes 178/200
  Env accuracies: ['75.0%', '73.7%']
  Avg acc: 74.4% (rolling: 75.1%)
  Avg reward: 182.8
  Actions: {'WAIT': 132, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_LIGHT': 0, 'GERMINATE_CONV_HEAVY': 0, 'FOSSILIZE': 2, 'CULL': 14}
  Successful: {'WAIT': 132, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_LIGHT': 0, 'GERMINATE_CONV_HEAVY': 0, 'FOSSILIZE': 2, 'CULL': 14}
  Policy loss: 0.0158, Value loss: 27.1107, Entropy: 0.3589, Entropy coef: 0.1000
[21:13:00] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[21:13:00] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[21:13:00] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[21:13:00] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[21:13:07] env0_seed_0 | Stage transition: TRAINING → BLENDING
[21:13:08] env1_seed_0 | Stage transition: TRAINING → BLENDING
[21:13:15] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[21:13:17] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[21:13:18] env0_seed_0 | Stage transition: SHADOWING → HOLDING
[21:13:18] env0_seed_0 | Stage transition: HOLDING → FOSSILIZED
[21:13:18] env0_seed_0 | Fossilized (norm, Δacc +19.23%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +19.23%)
[21:13:20] env1_seed_0 | Stage transition: SHADOWING → HOLDING
[21:13:20] env1_seed_0 | Stage transition: HOLDING → FOSSILIZED
[21:13:20] env1_seed_0 | Fossilized (norm, Δacc +17.17%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +17.17%)
Batch 90: Episodes 180/200
  Env accuracies: ['76.1%', '76.0%']
  Avg acc: 76.0% (rolling: 75.1%)
  Avg reward: 180.4
  Actions: {'WAIT': 122, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_LIGHT': 0, 'GERMINATE_CONV_HEAVY': 0, 'FOSSILIZE': 2, 'CULL': 24}
  Successful: {'WAIT': 122, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_LIGHT': 0, 'GERMINATE_CONV_HEAVY': 0, 'FOSSILIZE': 2, 'CULL': 24}
  Policy loss: -0.0046, Value loss: 32.3598, Entropy: 0.3712, Entropy coef: 0.1000

Blueprint Stats
  ---------------------------------------------------------------------------

Blueprint       Germ  Foss  Cull   Rate     ΔAcc    Churn
  ---------------------------------------------------------------------------

  attention        250     1   236   0.4%   +0.05%   +0.70%
  conv_heavy       171     2   159   1.2%   +3.45%   +0.78%
  conv_light       133     1   127   0.8%   +4.46%   +0.55%
  depthwise        165     1   153   0.6%   +7.56%   -0.37%
  norm             520    98   400  19.7%  +14.78%   +2.93%
Seed Scoreboard (env 0):
  Fossilized: 53 (+85.4K params, +90.1% of host)
  Pruned: 527
  Avg fossilize age: 11.2 epochs
  Avg cull age: 4.1 epochs
  Compute cost: 2.20x baseline
  Distribution: norm x51, conv_heavy x1, depthwise x1
Seed Scoreboard (env 1):
  Fossilized: 50 (+119.1K params, +125.7% of host)
  Pruned: 548
  Avg fossilize age: 11.3 epochs
  Avg cull age: 4.3 epochs
  Compute cost: 2.49x baseline
  Distribution: norm x47, conv_light x1, attention x1, conv_heavy x1

[21:15:05] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[21:15:05] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[21:15:05] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[21:15:05] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[21:15:12] env0_seed_0 | Stage transition: TRAINING → BLENDING
[21:15:12] env1_seed_0 | Stage transition: TRAINING → BLENDING
[21:15:20] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[21:15:20] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[21:15:24] env0_seed_0 | Stage transition: SHADOWING → HOLDING
[21:15:24] env1_seed_0 | Stage transition: SHADOWING → HOLDING
[21:15:24] env0_seed_0 | Stage transition: HOLDING → FOSSILIZED
[21:15:24] env0_seed_0 | Fossilized (norm, Δacc +16.81%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +16.81%)
[21:15:24] env1_seed_0 | Stage transition: HOLDING → FOSSILIZED
[21:15:24] env1_seed_0 | Fossilized (norm, Δacc +11.68%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +11.68%)
Batch 91: Episodes 182/200
  Env accuracies: ['76.8%', '74.9%']
  Avg acc: 75.9% (rolling: 75.1%)
  Avg reward: 183.5
  Actions: {'WAIT': 125, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_LIGHT': 0, 'GERMINATE_CONV_HEAVY': 0, 'FOSSILIZE': 2, 'CULL': 21}
  Successful: {'WAIT': 125, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_LIGHT': 0, 'GERMINATE_CONV_HEAVY': 0, 'FOSSILIZE': 2, 'CULL': 21}
  Policy loss: 0.0016, Value loss: 34.9826, Entropy: 0.3472, Entropy coef: 0.1000
[21:17:11] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[21:17:11] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[21:17:11] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[21:17:11] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[21:17:18] env0_seed_0 | Stage transition: TRAINING → BLENDING
[21:17:18] env1_seed_0 | Stage transition: TRAINING → BLENDING
[21:17:26] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[21:17:26] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[21:17:29] env0_seed_0 | Stage transition: SHADOWING → HOLDING
[21:17:29] env1_seed_0 | Stage transition: SHADOWING → HOLDING
[21:17:29] env0_seed_0 | Stage transition: HOLDING → FOSSILIZED
[21:17:29] env0_seed_0 | Fossilized (norm, Δacc +16.20%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +16.20%)
[21:17:29] env1_seed_0 | Stage transition: HOLDING → FOSSILIZED
[21:17:29] env1_seed_0 | Fossilized (norm, Δacc +12.47%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +12.47%)
Batch 92: Episodes 184/200
  Env accuracies: ['76.6%', '76.4%']
  Avg acc: 76.5% (rolling: 75.1%)
  Avg reward: 185.2
  Actions: {'WAIT': 128, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_LIGHT': 0, 'GERMINATE_CONV_HEAVY': 0, 'FOSSILIZE': 2, 'CULL': 18}
  Successful: {'WAIT': 128, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_LIGHT': 0, 'GERMINATE_CONV_HEAVY': 0, 'FOSSILIZE': 2, 'CULL': 18}
  Policy loss: -0.0060, Value loss: 35.6822, Entropy: 0.3697, Entropy coef: 0.1000
[21:19:17] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[21:19:17] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[21:19:17] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[21:19:17] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[21:19:23] env0_seed_0 | Stage transition: TRAINING → BLENDING
[21:19:23] env1_seed_0 | Stage transition: TRAINING → BLENDING
[21:19:32] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[21:19:32] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[21:19:35] env0_seed_0 | Stage transition: SHADOWING → HOLDING
[21:19:35] env1_seed_0 | Stage transition: SHADOWING → HOLDING
[21:19:35] env0_seed_0 | Stage transition: HOLDING → FOSSILIZED
[21:19:35] env0_seed_0 | Fossilized (norm, Δacc +14.72%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +14.72%)
[21:19:35] env1_seed_0 | Stage transition: HOLDING → FOSSILIZED
[21:19:35] env1_seed_0 | Fossilized (norm, Δacc +15.26%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +15.26%)
Batch 93: Episodes 186/200
  Env accuracies: ['76.6%', '75.2%']
  Avg acc: 75.9% (rolling: 75.2%)
  Avg reward: 181.2
  Actions: {'WAIT': 125, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_LIGHT': 0, 'GERMINATE_CONV_HEAVY': 0, 'FOSSILIZE': 2, 'CULL': 21}
  Successful: {'WAIT': 125, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_LIGHT': 0, 'GERMINATE_CONV_HEAVY': 0, 'FOSSILIZE': 2, 'CULL': 21}
  Policy loss: -0.0020, Value loss: 35.2938, Entropy: 0.3986, Entropy coef: 0.1000
[21:21:22] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[21:21:22] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[21:21:22] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[21:21:22] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[21:21:29] env1_seed_0 | Stage transition: TRAINING → BLENDING
[21:21:31] env0_seed_0 | Stage transition: TRAINING → BLENDING
[21:21:37] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[21:21:39] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[21:21:41] env1_seed_0 | Stage transition: SHADOWING → HOLDING
[21:21:41] env1_seed_0 | Stage transition: HOLDING → FOSSILIZED
[21:21:41] env1_seed_0 | Fossilized (norm, Δacc +15.73%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +15.73%)
[21:21:42] env0_seed_0 | Stage transition: SHADOWING → HOLDING
[21:21:42] env0_seed_0 | Stage transition: HOLDING → FOSSILIZED
[21:21:42] env0_seed_0 | Fossilized (norm, Δacc +12.82%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +12.82%)
Batch 94: Episodes 188/200
  Env accuracies: ['75.0%', '75.8%']
  Avg acc: 75.4% (rolling: 75.4%)
  Avg reward: 176.8
  Actions: {'WAIT': 117, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_LIGHT': 0, 'GERMINATE_CONV_HEAVY': 0, 'FOSSILIZE': 2, 'CULL': 29}
  Successful: {'WAIT': 117, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_LIGHT': 0, 'GERMINATE_CONV_HEAVY': 0, 'FOSSILIZE': 2, 'CULL': 29}
  Policy loss: -0.0344, Value loss: 38.1488, Entropy: 0.4286, Entropy coef: 0.1000
[21:23:28] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[21:23:28] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[21:23:28] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[21:23:28] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[21:23:35] env0_seed_0 | Stage transition: TRAINING → BLENDING
[21:23:36] env1_seed_0 | Stage transition: TRAINING → BLENDING
[21:23:43] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[21:23:45] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[21:23:46] env0_seed_0 | Stage transition: SHADOWING → HOLDING
[21:23:46] env0_seed_0 | Stage transition: HOLDING → FOSSILIZED
[21:23:46] env0_seed_0 | Fossilized (norm, Δacc +17.18%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +17.18%)
[21:23:48] env1_seed_0 | Stage transition: SHADOWING → HOLDING
[21:23:48] env1_seed_0 | Stage transition: HOLDING → FOSSILIZED
[21:23:48] env1_seed_0 | Fossilized (norm, Δacc +13.67%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +13.67%)
Batch 95: Episodes 190/200
  Env accuracies: ['75.7%', '76.9%']
  Avg acc: 76.3% (rolling: 75.9%)
  Avg reward: 180.5
  Actions: {'WAIT': 120, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_LIGHT': 0, 'GERMINATE_CONV_HEAVY': 0, 'FOSSILIZE': 2, 'CULL': 26}
  Successful: {'WAIT': 120, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_LIGHT': 0, 'GERMINATE_CONV_HEAVY': 0, 'FOSSILIZE': 2, 'CULL': 26}
  Policy loss: -0.0261, Value loss: 36.4794, Entropy: 0.4876, Entropy coef: 0.1000

Blueprint Stats
  ---------------------------------------------------------------------------

Blueprint       Germ  Foss  Cull   Rate     ΔAcc    Churn
  ---------------------------------------------------------------------------

  attention        250     1   236   0.4%   +0.05%   +0.70%
  conv_heavy       171     2   159   1.2%   +3.45%   +0.78%
  conv_light       133     1   127   0.8%   +4.46%   +0.55%
  depthwise        165     1   153   0.6%   +7.56%   -0.37%
  norm             530   108   400  21.3%  +14.77%   +2.93%
Seed Scoreboard (env 0):
  Fossilized: 58 (+86.0K params, +90.8% of host)
  Pruned: 527
  Avg fossilize age: 11.2 epochs
  Avg cull age: 4.1 epochs
  Compute cost: 2.30x baseline
  Distribution: norm x56, conv_heavy x1, depthwise x1
Seed Scoreboard (env 1):
  Fossilized: 55 (+119.7K params, +126.4% of host)
  Pruned: 548
  Avg fossilize age: 11.3 epochs
  Avg cull age: 4.3 epochs
  Compute cost: 2.59x baseline
  Distribution: norm x52, conv_light x1, attention x1, conv_heavy x1

[21:25:34] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[21:25:34] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[21:25:34] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[21:25:34] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[21:25:40] env0_seed_0 | Stage transition: TRAINING → BLENDING
[21:25:40] env1_seed_0 | Stage transition: TRAINING → BLENDING
[21:25:49] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[21:25:49] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[21:25:52] env0_seed_0 | Stage transition: SHADOWING → HOLDING
[21:25:52] env1_seed_0 | Stage transition: SHADOWING → HOLDING
[21:25:52] env0_seed_0 | Stage transition: HOLDING → FOSSILIZED
[21:25:52] env0_seed_0 | Fossilized (norm, Δacc +18.79%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +18.79%)
[21:25:52] env1_seed_0 | Stage transition: HOLDING → FOSSILIZED
[21:25:52] env1_seed_0 | Fossilized (norm, Δacc +17.67%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +17.67%)
Batch 96: Episodes 192/200
  Env accuracies: ['76.0%', '76.5%']
  Avg acc: 76.3% (rolling: 76.0%)
  Avg reward: 176.5
  Actions: {'WAIT': 110, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_LIGHT': 0, 'GERMINATE_CONV_HEAVY': 0, 'FOSSILIZE': 2, 'CULL': 36}
  Successful: {'WAIT': 110, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_LIGHT': 0, 'GERMINATE_CONV_HEAVY': 0, 'FOSSILIZE': 2, 'CULL': 36}
  Policy loss: 0.0149, Value loss: 23.6685, Entropy: 0.4870, Entropy coef: 0.1000
[21:27:39] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[21:27:39] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[21:27:39] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[21:27:39] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[21:27:46] env0_seed_0 | Stage transition: TRAINING → BLENDING
[21:27:46] env1_seed_0 | Stage transition: TRAINING → BLENDING
[21:27:54] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[21:27:54] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[21:27:58] env0_seed_0 | Stage transition: SHADOWING → HOLDING
[21:27:58] env1_seed_0 | Stage transition: SHADOWING → HOLDING
[21:27:58] env0_seed_0 | Stage transition: HOLDING → FOSSILIZED
[21:27:58] env0_seed_0 | Fossilized (norm, Δacc +17.10%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +17.10%)
[21:27:58] env1_seed_0 | Stage transition: HOLDING → FOSSILIZED
[21:27:58] env1_seed_0 | Fossilized (norm, Δacc +8.68%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +8.68%)
Batch 97: Episodes 194/200
  Env accuracies: ['76.5%', '76.7%']
  Avg acc: 76.6% (rolling: 75.9%)
  Avg reward: 180.9
  Actions: {'WAIT': 120, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_LIGHT': 0, 'GERMINATE_CONV_HEAVY': 0, 'FOSSILIZE': 2, 'CULL': 26}
  Successful: {'WAIT': 120, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_LIGHT': 0, 'GERMINATE_CONV_HEAVY': 0, 'FOSSILIZE': 2, 'CULL': 26}
  Policy loss: -0.0017, Value loss: 30.5992, Entropy: 0.4687, Entropy coef: 0.1000
[21:29:45] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[21:29:45] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[21:29:45] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[21:29:45] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[21:29:52] env0_seed_0 | Stage transition: TRAINING → BLENDING
[21:29:52] env1_seed_0 | Stage transition: TRAINING → BLENDING
[21:30:00] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[21:30:00] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[21:30:03] env0_seed_0 | Stage transition: SHADOWING → HOLDING
[21:30:03] env1_seed_0 | Stage transition: SHADOWING → HOLDING
[21:30:03] env0_seed_0 | Stage transition: HOLDING → FOSSILIZED
[21:30:03] env0_seed_0 | Fossilized (norm, Δacc +16.46%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +16.46%)
[21:30:03] env1_seed_0 | Stage transition: HOLDING → FOSSILIZED
[21:30:03] env1_seed_0 | Fossilized (norm, Δacc +22.14%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +22.14%)
Batch 98: Episodes 196/200
  Env accuracies: ['76.1%', '76.9%']
  Avg acc: 76.5% (rolling: 76.0%)
  Avg reward: 177.1
  Actions: {'WAIT': 107, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_LIGHT': 0, 'GERMINATE_CONV_HEAVY': 0, 'FOSSILIZE': 2, 'CULL': 39}
  Successful: {'WAIT': 107, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_LIGHT': 0, 'GERMINATE_CONV_HEAVY': 0, 'FOSSILIZE': 2, 'CULL': 39}
  Policy loss: 0.0047, Value loss: 28.9270, Entropy: 0.4869, Entropy coef: 0.1000
[21:31:51] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[21:31:51] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[21:31:51] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[21:31:51] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[21:31:57] env0_seed_0 | Stage transition: TRAINING → BLENDING
[21:31:57] env1_seed_0 | Stage transition: TRAINING → BLENDING
[21:32:06] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[21:32:06] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[21:32:09] env0_seed_0 | Stage transition: SHADOWING → HOLDING
[21:32:09] env1_seed_0 | Stage transition: SHADOWING → HOLDING
[21:32:09] env0_seed_0 | Stage transition: HOLDING → FOSSILIZED
[21:32:09] env0_seed_0 | Fossilized (norm, Δacc +11.77%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +11.77%)
[21:32:09] env1_seed_0 | Stage transition: HOLDING → FOSSILIZED
[21:32:09] env1_seed_0 | Fossilized (norm, Δacc +11.94%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +11.94%)
Batch 99: Episodes 198/200
  Env accuracies: ['76.2%', '75.9%']
  Avg acc: 76.1% (rolling: 76.1%)
  Avg reward: 172.7
  Actions: {'WAIT': 107, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_LIGHT': 0, 'GERMINATE_CONV_HEAVY': 0, 'FOSSILIZE': 2, 'CULL': 39}
  Successful: {'WAIT': 107, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_LIGHT': 0, 'GERMINATE_CONV_HEAVY': 0, 'FOSSILIZE': 2, 'CULL': 39}
  Policy loss: -0.0205, Value loss: 33.2599, Entropy: 0.4634, Entropy coef: 0.1000
[21:33:56] env0_seed_0 | Germinated (norm, 0.1K params)
    [env0] Germinated 'env0_seed_0' (norm, 0.1K params)
[21:33:56] env1_seed_0 | Germinated (norm, 0.1K params)
    [env1] Germinated 'env1_seed_0' (norm, 0.1K params)
[21:33:56] env0_seed_0 | Stage transition: GERMINATED → TRAINING
[21:33:56] env1_seed_0 | Stage transition: GERMINATED → TRAINING
[21:34:03] env0_seed_0 | Stage transition: TRAINING → BLENDING
[21:34:03] env1_seed_0 | Stage transition: TRAINING → BLENDING
[21:34:11] env0_seed_0 | Stage transition: BLENDING → SHADOWING
[21:34:11] env1_seed_0 | Stage transition: BLENDING → SHADOWING
[21:34:15] env0_seed_0 | Stage transition: SHADOWING → HOLDING
[21:34:15] env1_seed_0 | Stage transition: SHADOWING → HOLDING
[21:34:15] env0_seed_0 | Stage transition: HOLDING → FOSSILIZED
[21:34:15] env0_seed_0 | Fossilized (norm, Δacc +22.01%)
    [env0] Fossilized 'env0_seed_0' (norm, Δacc +22.01%)
[21:34:15] env1_seed_0 | Stage transition: HOLDING → FOSSILIZED
[21:34:15] env1_seed_0 | Fossilized (norm, Δacc +13.11%)
    [env1] Fossilized 'env1_seed_0' (norm, Δacc +13.11%)
Batch 100: Episodes 200/200
  Env accuracies: ['75.3%', '76.6%']
  Avg acc: 76.0% (rolling: 76.1%)
  Avg reward: 177.1
  Actions: {'WAIT': 114, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_LIGHT': 0, 'GERMINATE_CONV_HEAVY': 0, 'FOSSILIZE': 2, 'CULL': 32}
  Successful: {'WAIT': 114, 'GERMINATE_NORM': 2, 'GERMINATE_ATTENTION': 0, 'GERMINATE_DEPTHWISE': 0, 'GERMINATE_CONV_LIGHT': 0, 'GERMINATE_CONV_HEAVY': 0, 'FOSSILIZE': 2, 'CULL': 32}
  Policy loss: 0.0157, Value loss: 22.1722, Entropy: 0.4080, Entropy coef: 0.1000

Blueprint Stats
  ---------------------------------------------------------------------------

Blueprint       Germ  Foss  Cull   Rate     ΔAcc    Churn
  ---------------------------------------------------------------------------

  attention        250     1   236   0.4%   +0.05%   +0.70%
  conv_heavy       171     2   159   1.2%   +3.45%   +0.78%
  conv_light       133     1   127   0.8%   +4.46%   +0.55%
  depthwise        165     1   153   0.6%   +7.56%   -0.37%
  norm             540   118   400  22.8%  +14.87%   +2.93%
Seed Scoreboard (env 0):
  Fossilized: 63 (+86.7K params, +91.4% of host)
  Pruned: 527
  Avg fossilize age: 11.2 epochs
  Avg cull age: 4.1 epochs
  Compute cost: 2.40x baseline
  Distribution: norm x61, conv_heavy x1, depthwise x1
Seed Scoreboard (env 1):
  Fossilized: 60 (+120.4K params, +127.0% of host)
  Pruned: 548
  Avg fossilize age: 11.2 epochs
  Avg cull age: 4.3 epochs
  Compute cost: 2.69x baseline
  Distribution: norm x57, conv_light x1, attention x1, conv_heavy x1

Loaded best weights (avg_acc=76.1%)
(.venv) john@nyx:~/esper-lite$
