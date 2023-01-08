class Accumulator:
    """For accumulating sums over `n` variables.
    Usage
    -------
        metric = Accumulator(3)
        metric.add(1., 0., 1.)
        acc = metric[2] / metric[0]
        metric.reset()
    """

    def __init__(self, n: int) -> None:
        self.data = [0.0] * n

    def add(self, *args) -> None:
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self) -> None:
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx: int) -> float:
        return self.data[idx]
