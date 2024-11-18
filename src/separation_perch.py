import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from datasets import load_dataset_builder
from datasets import load_dataset
import soundfile as sf
import numpy as np
import torch
import librosa

from src.modules.models.tdcnn import TDConvNetpp, WaveformEncoder, WaveformDecoder
from src.datamodules.components.normalization import NormalizePeak
from src.modules.metrics.components.classification import TopKAccuracy, cmAP
from modules.models.perch import PerchModel

from torchmetrics import AUROC
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss

if os.path.exists("/home/mwirth/projects"):
    os.chdir("../")


def _classes_one_hot(labels, num_classes) -> torch.tensor:
    class_one_hot_matrix = torch.zeros(num_classes, dtype=torch.float32)

    for idx in labels:
        class_one_hot_matrix[idx] = 1
    #class_one_hot_matrix = torch.tensor(class_one_hot_matrix, dtype=torch.float32)
    return class_one_hot_matrix

def pad_to_length(arr, target_length=80000):
    current_length = len(arr)
    if current_length == target_length:
        return arr
    # Calculate padding
    padding_length = target_length - current_length
    padded_array = np.pad(arr, (0, padding_length), 'constant', constant_values=0)
    return padded_array


@torch.no_grad()
def main():
    if os.path.exists("/home/mwirth/projects"):
        #checkpoint = "/home/mwirth/projects/sound-seperation/checkpoints/new/epoch059-xcm_no_normal-loss_valid-10.4859-metric_valid10.4858.ckpt"
        #checkpoint = "/home/mwirth/projects/sound-seperation/checkpoints/new/epoch059-XCM-sisdr-loss_valid-11.4109-metric_valid11.4107.ckpt"
        checkpoint = "/home/mwirth/projects/sound-seperation/checkpoints/new/epoch013-loss_valid-11.2518-metric_valid11.2493.ckpt"
    else:
        #checkpoint = "/mnt/stud/work/deeplearninglab/ss2024/sound-separation/checkpoints/XCM-no_normal/epoch059-loss_valid-10.4859-metric_valid10.4858.ckpt"
        checkpoint = "/mnt/stud/work/deeplearninglab/ss2024/sound-separation/checkpoints/XCL/epoch013-loss_valid-11.2518-metric_valid11.2493.ckpt"

    print("using checkpoint:", checkpoint)

    datasets = ["NBP"]

    encoder = WaveformEncoder(256, 256)
    decoder = WaveformDecoder(256, 256)
    normalize = NormalizePeak(0.4)
    model = TDConvNetpp(in_chan=256, n_src=4, n_repeats=4, encoder=encoder, decoder=decoder)

    state_dict = torch.load(checkpoint, weights_only=False)
    state_dict = {k[6:]: v for k, v in state_dict["state_dict"].items()}
    model.load_state_dict(state_dict)

    #perch = hub.load('https://www.kaggle.com/models/google/bird-vocalization-classifier/TensorFlow2/bird-vocalization-classifier/4')
    #cs = pd.read_csv("data/perch/label_updated.csv")

    for name in datasets:
        print("-"*30)
        print("Dataset name:", name)
        print("-"*30)
        cache_dir = f"data/{name}_scape"

        #XCL = load_dataset_builder("DBD-research-group/BirdSet", "XCL")
        subset_builder = load_dataset_builder("DBD-research-group/BirdSet", f"{name}_scape", trust_remote_code=True)
        # idx = [builder.info.features["ebird_code"].str2int(name) for name in hsn.info.features["ebird_code"].names] # idx for
        #idx = [cs[cs["ebird2021"] == i].index.values[0] for i in subset_builder.info.features["ebird_code"].names]
        num_classes = len(subset_builder.info.features["ebird_code"].names)
        perch = PerchModel(num_classes=num_classes,
                           tfhub_version="4",
                           train_classifier=False,
                           restrict_logits=True,
                           label_path="data/perch/label_updated.csv",
                           pretrain_info={"hf_path":"DBD-research-group/BirdSet",
                                          "hf_name": name})

        ds = load_dataset("DBD-research-group/BirdSet", f"{name}_scape", cache_dir=cache_dir, trust_remote_code=True)

        t1 = TopKAccuracy()
        t5 = TopKAccuracy(topk=5)
        cmap = cmAP(num_labels=num_classes, thresholds=None)
        auroc = AUROC(task="multilabel", num_labels=num_classes, average='macro', thresholds=None)
        loss = BCEWithLogitsLoss()
        losses = []

        t1_org = TopKAccuracy()
        t5_org = TopKAccuracy(topk=5)
        cmap_org = cmAP(num_labels=num_classes, thresholds=None)
        auroc_org = AUROC(task="multilabel", num_labels=num_classes, average='macro', thresholds=None)
        loss_org = BCEWithLogitsLoss()
        losses_org = []

        for data in tqdm(ds["test_5s"]):
            #org_wave, org_sr = librosa.load(data["filepath"], sr=None)
            org_wave, org_sr = sf.read(data["filepath"])
            org_wave = librosa.resample(org_wave, orig_sr=org_sr, target_sr=32000)
            org_sr = 32000
            if org_wave.ndim != 1:
                org_wave = org_wave.swapaxes(1, 0)
                org_wave = librosa.to_mono(org_wave)

            assert org_wave.ndim == 1, f"{data['filepath']}, was not mono"
            assert org_sr == 32000, f"{data['filepath']}, was not 32kHz it was {org_sr}"

            wave = librosa.resample(org_wave, orig_sr=org_sr, target_sr=16_000)
            #wave = normalize(wave)
            wave = pad_to_length(wave, target_length=160_000) # pad to 10s audio
            wave = torch.tensor(wave).unsqueeze(0).float()
            _, est_wave, _ = model(wave)
            est_wave = est_wave.squeeze(0)[:,:80_000] # over-padded rest

            est_wave1 = est_wave
            waves = [pad_to_length(normalize(org_wave), 160_000)]
            for i in range(est_wave1.size(0)):
                w = est_wave1[i, :]
                w = librosa.resample(w.numpy(), orig_sr=16000, target_sr=32000)
                w = normalize(w)
                waves.append(w)
            est_wave1 = np.stack(waves, dtype=np.float64)

            #logits = torch.tensor(perch.infer_tf(est_wave1)["label"].numpy()[:, idx])
            logits = perch(torch.from_numpy(est_wave1).float())
            #logits = torch.from_numpy(logits.numpy()[:, idx])
            p = torch.sigmoid(logits)

            targets = _classes_one_hot(torch.tensor(data["ebird_code_multilabel"]), num_classes=num_classes).float()
            targets = targets.unsqueeze(0)

            # select channel with the highest probability as prediction
            #row_with_max = torch.argmax(p[1:,].max(dim=1)[0]) + 1
            #selected = p[row_with_max].unsqueeze(0)
            # gather the highest probabilities for each class
            selected = p[1:].max(dim=0)[0].unsqueeze(0)

            l = loss(logits[1:].max(dim=0)[0].unsqueeze(0), targets)
            #l = loss(logits[row_with_max].unsqueeze(0), targets)
            losses.append(l.item())
            t1(preds=selected, targets=targets.int())
            t5(preds=selected, targets=targets.int())
            cmap(logits=selected, labels=targets.int())
            auroc(preds=selected, target=targets.int())

            preds = p[0,].unsqueeze(0)

            losses_org.append((loss_org(logits[0,:].unsqueeze(0), targets)).item())
            t1_org(preds=preds, targets=targets.int())
            t5_org(preds=preds, targets=targets.int())
            cmap_org(logits=preds, labels=targets.int())
            auroc_org(preds=preds, target=targets.int())

        print("T1:", t1_org.compute(), "->", t1.compute())
        print("T5:", t5_org.compute(), "->", t5.compute())
        print("cmap:", cmap_org.compute(), "->", cmap.compute())
        print("auroc:", auroc_org.compute(), "->", auroc.compute())

        print("Loss:", np.mean(losses))
        print("Loss original:", np.mean(losses_org))

if __name__ == "__main__":
    main()










