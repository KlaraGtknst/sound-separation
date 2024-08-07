
class AudioLengthFilter:
    def __init__(self, min_len: int = 0, max_len: int = 10):
        self.min_len = min_len
        self.max_len = max_len

    def __call__(self, batch):
        return self.min_len <= batch["end_time"] - batch["start_time"] <= self.max_len
