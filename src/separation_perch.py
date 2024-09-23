import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from datasets import load_dataset_builder
from datasets import load_dataset
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import torch
import librosa
from torch.nn.functional import one_hot

from src.modules.models.tdcnn import TDConvNetpp, WaveformEncoder, WaveformDecoder
from src.datamodules.components.normalization import NormalizePeak
from src.modules.metrics.components.classification import TopKAccuracy, cmAP
from torchmetrics import AUROC
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss


def _classes_one_hot(labels, num_classes) -> torch.tensor:
    class_one_hot_matrix = torch.zeros(num_classes, dtype=torch.float32)

    for idx in labels:
        class_one_hot_matrix[idx] = 1
    #class_one_hot_matrix = torch.tensor(class_one_hot_matrix, dtype=torch.float32)
    return class_one_hot_matrix



@torch.no_grad()
def main():
    checkpoint = "/home/mwirth/projects/sound-seperation/checkpoints/new/epoch059-XCM-sisdr-loss_valid-11.4109-metric_valid11.4107.ckpt"
    datasets = ["HSN"]

    encoder = WaveformEncoder(256, 256)
    decoder = WaveformDecoder(256, 256)
    normalize = NormalizePeak(0.2)
    model = TDConvNetpp(in_chan=256, n_src=4, n_repeats=4, encoder=encoder, decoder=decoder)

    state_dict = torch.load(checkpoint, weights_only=False)
    state_dict = {k[6:]: v for k, v in state_dict["state_dict"].items()}
    model.load_state_dict(state_dict)

    perch = hub.load('https://www.kaggle.com/models/google/bird-vocalization-classifier/TensorFlow2/bird-vocalization-classifier/8')
    cs = pd.read_csv("../data/perch/label_updated.csv")

    for name in datasets:
        print("-"*30)
        print("Dataset name:", name)
        print("-"*30)
        cache_dir = f"../data/{name}"

        #XCL = load_dataset_builder("DBD-research-group/BirdSet", "XCL")
        subset_builder = load_dataset_builder("DBD-research-group/BirdSet", name)
        # idx = [builder.info.features["ebird_code"].str2int(name) for name in hsn.info.features["ebird_code"].names] # idx for
        idx = [cs[cs["ebird2021"] == i].index.values[0] for i in subset_builder.info.features["ebird_code"].names]

        ds = load_dataset("DBD-research-group/BirdSet", name, cache_dir=cache_dir)

        t1 = TopKAccuracy()
        t5 = TopKAccuracy(topk=5)
        cmap = cmAP(num_labels=len(idx), thresholds=None)
        auroc = AUROC(task="multilabel", num_labels=len(idx), average='macro', thresholds=None)
        loss = BCEWithLogitsLoss()
        losses = []

        t1_org = TopKAccuracy()
        t5_org = TopKAccuracy(topk=5)
        cmap_org = cmAP(num_labels=len(idx), thresholds=None)
        auroc_org = AUROC(task="multilabel", num_labels=len(idx), average='macro', thresholds=None)
        loss_org = BCEWithLogitsLoss()
        losses_org = []


        for data in tqdm(ds["test_5s"]):
            wave, sr = librosa.load(data["filepath"], sr=16000)
            wave = normalize(wave)
            wave = torch.tensor(wave).unsqueeze(0)
            masks, est_wave, weights = model(wave)
            est_wave = est_wave.squeeze(0)

            est_wave1 = torch.vstack([wave, est_wave])
            waves = []
            for i in range(est_wave1.size(0)):
                w = est_wave1[i, :]
                w = librosa.resample(w.numpy(), orig_sr=16000, target_sr=32000)
                w = normalize(w)
                waves.append(w)
            est_wave1 = np.stack(waves)

            logits = torch.tensor(perch.infer_tf(est_wave1)["label"].numpy()[:, idx])
            p = torch.sigmoid(logits)

            targets = _classes_one_hot(torch.tensor(data["ebird_code_multilabel"]), num_classes=len(idx)).float()
            targets = targets.unsqueeze(0)

            row_with_max = torch.argmax(p[1:,].max(dim=1)[0]) + 1
            selected = p[row_with_max].unsqueeze(0)

            l = loss(logits[row_with_max].unsqueeze(0), targets)
            losses.append(l.item())
            t1(preds=selected, targets=targets)
            t5(preds=selected, targets=targets)
            cmap(logits=selected, labels=targets.int())
            auroc(preds=selected, target=targets.int())


            preds = p[0,].unsqueeze(0)

            losses_org.append((loss_org(logits[0,:].unsqueeze(0), targets)).item())
            t1_org(preds=preds, targets=targets)
            t5_org(preds=preds, targets=targets)
            cmap_org(logits=preds, labels=targets.int())
            auroc_org(preds=preds, target=targets.int())

        print("T1:", t1.compute(), "->", t1_org.compute())
        print("T5:", t5.compute(), "->", t5_org.compute())
        print("cmap:", cmap.compute(), "->", cmap_org.compute())
        print("auroc:", auroc.compute(), "->", auroc_org.compute())

        print("Loss:", np.mean(losses))
        print("Loss original:", np.mean(losses_org))

if __name__ == "__main__":
    main()










