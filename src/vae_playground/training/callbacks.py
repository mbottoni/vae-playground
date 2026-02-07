"""Optional training callbacks (placeholder for future extensions)."""

from __future__ import annotations


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Parameters
    ----------
    patience : number of epochs with no improvement before stopping.
    min_delta : minimum change to qualify as an improvement.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best: float | None = None
        self.counter = 0

    def step(self, metric: float) -> bool:
        """Return ``True`` if training should stop."""
        if self.best is None or metric < self.best - self.min_delta:
            self.best = metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience
