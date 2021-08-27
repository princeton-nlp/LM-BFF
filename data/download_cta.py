from primer_nlx.utils.dataregistry import download_data
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os


def main():
    filename = "cta_sentences-v1.csv"
    if os.path.exists(filename):
        os.remove(filename)
    download_data("cta_sentences", 1)
    df = pd.read_csv(filename)
    np.random.seed(42)
    df.columns = ["sentence", "label"]

    # This repo's code messes up with newlines
    # TODO: fix this
    df["sentence"] = df["sentence"].str.replace("\n", "  ")
    df["label"] = df["label"].astype(int)
    train, dev = train_test_split(df, test_size=0.3, stratify=df["label"])

    cta_path = "original/CTA"
    if not os.path.exists(cta_path):
        os.makedirs(cta_path)

    train.to_csv(os.path.join(cta_path, "train.tsv"), sep="\t", index=False)
    dev.to_csv(os.path.join(cta_path, "dev.tsv"), sep="\t", index=False)

if __name__ == "__main__":
    main()
