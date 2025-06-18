import os
from PIL import Image
import torch
from torchvision import models, transforms
from sklearn.decomposition import PCA
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# === CONFIG ===
IMAGE_FOLDER = '/home/chiara/my_stuff/CRNN_code/rnn_rts/data/coco/Overlapping_patches'

N_COMPONENTS = 6
PCA_OUTPUT_PATH = f'pca_vgg19_first_block_{N_COMPONENTS}_channels.joblib'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)
BATCH_SIZE = 32

# === Trasformazioni ===
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# === Carica VGG19 e prendi il primo blocco conv (2 conv + 2 ReLU + 1 MaxPool) ===
vgg19 = models.vgg19(pretrained=True).features.to(DEVICE).eval()
first_block = vgg19[:2].to(DEVICE).eval()  # Layers 0-4 inclusi

# === Carica immagini ===
image_paths = [os.path.join(IMAGE_FOLDER, f)
               for f in os.listdir(IMAGE_FOLDER)
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"Caricate {len(image_paths)} immagini da {IMAGE_FOLDER}")

all_features = []

with torch.no_grad():
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Estrazione Primo Blocco VGG19"):
        batch_paths = image_paths[i:i + BATCH_SIZE]
        images = []

        for path in batch_paths:
            try:
                img = Image.open(path).convert('RGB')
                img = preprocess(img)
                images.append(img)
            except Exception as e:
                print(f"Errore con {path}: {e}")

        if not images:
            continue

        batch = torch.stack(images).to(DEVICE)  # (B, 3, H, W)
        features = first_block(batch)             # (B, 64, H_out, W_out)

        B, C, H, W = features.shape
        flat = features.permute(0, 2, 3, 1).reshape(B * H * W, C)  # (B*H*W, 64)
        all_features.append(flat.cpu())

# Concatena tutti i vettori da tutte le immagini
X = torch.cat(all_features, dim=0).numpy()  # (Tot_pixel_totali, 64)
print(f"Totale vettori estratti per PCA: {X.shape}")

# Fit PCA con n_components=25
print("Fitting PCA...")
pca = PCA(n_components=N_COMPONENTS)
pca.fit(X)

# Salva PCA
joblib.dump(pca, PCA_OUTPUT_PATH)
print(f"PCA salvata in: {PCA_OUTPUT_PATH}")

# Plot varianza spiegata
explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)

plt.figure(figsize=(8, 5))
plt.plot(np.arange(1, len(explained) + 1), cumulative, marker='o', color='navy')
plt.axhline(0.90, color='red', linestyle='--', label='90% varianza')
plt.axhline(0.95, color='green', linestyle='--', label='95% varianza')
plt.title('PCA - Varianza Cumulativa')
plt.xlabel('Numero componenti')
plt.ylabel('Varianza spiegata cumulativa')
plt.grid(True)
plt.legend()
plt.tight_layout()

PLOT_PATH = 'pca_varianza_first_block.png'
os.makedirs(os.path.dirname(PLOT_PATH) or '.', exist_ok=True)
plt.savefig(PLOT_PATH)
plt.show()

print(f"Plot salvato in: {os.path.abspath(PLOT_PATH)}")


# === Salva solo la matrice di proiezione in torch ===
pca_matrix = torch.from_numpy(pca.components_.astype(np.float32))  # (n_components, original_dim)
torch.save(pca_matrix, f'pca_projection_matrix_{N_COMPONENTS}_channels.pt')
print("Matrice PCA salvata in: pca_projection_matrix.pt")