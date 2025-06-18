import json
import random
import os

def split_dataset(i,train_ratio=0.8, seed=42):
    """
    dataset_path: file json completo (tutto)
    output_dir: dove salvare i file cocodots_train.json e cocodots_val.json
    train_ratio: percentuale per il training set
    """
    dataset_path = f"/home/chiara/my_stuff/CRNN_code/rnn_rts/data/datasets/dot_dataset{i}.json" 
    output_dir = "/home/chiara/my_stuff/CRNN_code/rnn_rts/data/"
    train_ratio = 0.8

    with open(dataset_path, 'r') as f:
        data = json.load(f)

    images = data['images']
    dot_pairs = data['serrelab_anns']  # il tuo campo dot_pairs nel json
    categories = data['categories']

    # Shuffle immagini (usiamo seed per riproducibilità)
    random.seed(seed)
    random.shuffle(images)

    # Divido immagini in train e val
    train_size = int(len(images) * train_ratio)
    train_images = images[:train_size]
    val_images = images[train_size:]

    # Crea un set degli ID immagine per train e val
    train_img_ids = set(img['id'] for img in train_images)
    val_img_ids = set(img['id'] for img in val_images)

    # Divido le annotazioni in base a image_id
    train_dot_pairs = [ann for ann in dot_pairs if ann['image_id'] in train_img_ids]
    val_dot_pairs = [ann for ann in dot_pairs if ann['image_id'] in val_img_ids]

    # Prepara dict json per train e val
    train_data = {
        'images': train_images,
        'serrelab_anns': train_dot_pairs,
        'categories': categories
    }

    val_data = {
        'images': val_images,
        'serrelab_anns': val_dot_pairs,
        'categories': categories
    }

    # Salva su file
    os.makedirs(output_dir, exist_ok=True)

    mapping = {
    "_float3": "float_3seg",
    "_floatn": "float_varseg",
    "_binary": "binary_varseg",
    "_balanced_binary": "binary_bal_varseg"
    }


    train_path = os.path.join(output_dir, f'{mapping[i]}_train.json')
    val_path = os.path.join(output_dir,  f'{mapping[i]}_val.json')

    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=2)

    with open(val_path, 'w') as f:
        json.dump(val_data, f, indent=2)

    print(f"Train set: {len(train_images)} images, {len(train_dot_pairs)} annotations saved to {train_path}")
    print(f"Val set: {len(val_images)} images, {len(val_dot_pairs)} annotations saved to {val_path}")


if __name__ == "__main__":
    datasets = ["_balanced_binary","_binary", "_float3", "_floatn"] 
    for i in datasets:
        split_dataset(i)
