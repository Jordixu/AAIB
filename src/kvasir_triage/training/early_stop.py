"""Early stopping on val PR-AUC."""


class EarlyStopping:
    def __init__(self, patience: int = 10) -> None:
        self.patience = patience
        self.best = -float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        """Call after each val epoch. Returns True if training should stop."""
        if metric > self.best:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    def state_dict(self) -> dict:
        return {
            "best": self.best,
            "counter": self.counter,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, state: dict) -> None:
        self.best = state["best"]
        self.counter = state["counter"]
        self.should_stop = state["should_stop"]
