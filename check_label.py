import pandas as pd
from datasets import setup_dataset
from collections import Counter
from datasets.cocodots import CocoDots

def check_balance(path, label_column='label', threshold=0.1):
    # Carica dataset train da CSV
    data_root= "./data/coco"
    kwargs = {
    'base_size': 150,
    }
    dataset = setup_dataset(path, data_root, subset=1.0, shuffle=False,**kwargs)


    # Conta i valori della label "same"
    label_counter = Counter()
    for ann in dataset.data["serrelab_anns"]:
        label = ann["same"]
        label_counter[label] += 1

    # Stampa il risultato
    print("\nDistribuzione delle etichette (label 'same'):")
    for label, count in label_counter.items():
        print(f"Label {label}: {count} campioni")

    # Check bilanciamento
    total = sum(label_counter.values())
    for label, count in label_counter.items():
        print(f"Label {label}: {count} ({(count/total)*100:.2f}%)")



if __name__ == "__main__":
    # Percorso al tuo dataset train (modifica!)
    train_file = 'cocodots_train_flexmm'
    val_file = 'cocodots_val_flexmm'
    check_balance(train_file, label_column='same')
    check_balance(val_file, label_column='same')
