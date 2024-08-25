
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
                for i in range((int(batch["length"][b_idx]) // self.segment_length) + 1):
                    if key == "start_time":
                        new_batch[key] += [i * self.segment_length]
                    elif key == "end_time":
                        new_batch[key] += [min(( i +1) * self.segment_length, batch["length"][b_idx])]
                    else:
                        new_batch[key] += [batch[key][b_idx]]
        return new_batch


import soundfile as sf

class LoadLengths:
    def __init__(self):
        pass

    def __call__(self, batch):
        for b_idx in range(len(batch["filepath"])):
            duration = sf.info(batch["filepath"][b_idx]).duration
            batch["length"][b_idx] = round(duration, 2)

        return batch


class FindSupervised:
    def __init__(self, length: float = 10, qualitys: tuple = ("A", "B"), secondarys_max: int = 0):
        self.length = length
        self.qualitys = qualitys
        self.secondarys_max = secondarys_max

    def __call__(self, batch):
        l = []
        for b_idx in range(len(batch["filepath"])):

            l.append(batch["length"][b_idx] <= self.length and batch["quality"][b_idx] in self.qualitys and len(batch["ebird_code_secondary"][b_idx]) <= self.secondarys_max)
        batch["is_supervised"] = l
        return batch

