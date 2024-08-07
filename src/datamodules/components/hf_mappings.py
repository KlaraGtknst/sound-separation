
class AudioSegmenting:
    def __init__(self, segment_length: int = 10, max_segments: int = 3):
        self.segment_length = segment_length
        self.max_segments = max_segments

    def __call__(self, batch):
        new_batch = {k :[] for k in batch.keys()}
        # iterate over all rows of batch
        for b_idx in range(len(batch["filepath"])):
            # skip audios with to long length
            if batch["length"][b_idx] > self.segment_length * self.max_segments:
                continue

            # add all keys to new_batch
            for key in batch.keys():
                # add duplicates if length is over 10, seconds
                for i in range((batch["length"][b_idx] // self.segment_length) + 1):
                    if key == "start_time":
                        new_batch[key] += [i * self.segment_length]
                    elif key == "end_time":
                        new_batch[key] += [min(( i +1) * self.segment_length, batch["length"][b_idx])]
                    else:
                        new_batch[key] += [batch[key][b_idx]]
        return new_batch
