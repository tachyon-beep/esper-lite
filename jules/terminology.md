# Esper Terminology

Because Esper applies Reinforcement Learning (RL) to control another Neural Network's training loop, there is significant overlap in common ML terminology. This glossary clarifies what terms mean in the context of Esper.

## Core Loop Definitions

| Term | Meaning in Esper |
| :--- | :--- |
| **Episode** | One complete host training run (e.g., 150 steps). This is equivalent to an RL trajectory. |
| **Step** | One policy decision point. Tamiyo observes the host state and chooses an action. Occurs synchronously with a Host Epoch. |
| **Batch** | A collection of parallel episodes used for one PPO update. (e.g., With `n_envs=10`, one batch = 10 episodes). |
| **Host Epoch** | One training iteration (forward/backward pass) of the host neural network. Occurs synchronously with an RL Step. |

*Example Context:* "At step 75 of episode 3, the host finished its 75th training epoch, and Tamiyo decided to GERMINATE."

## Morphogenetic Concepts

| Term | Meaning in Esper |
| :--- | :--- |
| **Host** | The base, primary neural network whose parameters and topology are being modified. |
| **Seed** | A new neural module (sub-network) that is introduced to the host. |
| **Slot (SeedSlot)** | A predefined location in the host's architecture where a seed can be germinated and attached. |
| **Fossilization** | The process of permanently committing a seed into the host's structure. |
| **Alpha Ramp** | The controlled scaling of a seed's influence (its output weight) as it transitions from isolated training to full blending. |
| **Drip Mechanism** | A reward mechanism where, after fossilization, rewards continue to "drip" to the policy based on the seed's ongoing contribution, preventing agents from gaming short-term gains. |
