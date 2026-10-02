class LatencyModel:

    def __init__(
        self,
        latency_ms=10.0,
    ):

        self.latency_ms = (
            latency_ms
        )

    def apply(self):

        return self.latency_ms