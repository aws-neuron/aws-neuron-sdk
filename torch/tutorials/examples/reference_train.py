"""Run the common distributed entry point with CPU/Gloo defaults."""

from distributed_train import main

if __name__ == "__main__":
    main(default_device="cpu", default_backend="gloo")
