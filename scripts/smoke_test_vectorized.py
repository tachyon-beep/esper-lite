import logging

import torch

from esper.simic.training.vectorized import train_ppo_vectorized

logging.basicConfig(level=logging.INFO)


def smoke_test_vectorized() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Run vectorized training for a very small number of episodes/epochs
    print("Starting smoke test for vectorized training...")
    try:
        train_ppo_vectorized(
            n_episodes=1,
            n_envs=1,
            max_epochs=2,  # Short run for smoke testing
            device=device,
            task="cifar_impaired",
            slots=["r0c0"],
            use_telemetry=True,
            gpu_preload=True,
            quiet_analytics=True
        )
        print("Smoke test completed successfully!")
    except Exception as e:
        print(f"Smoke test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    smoke_test_vectorized()
