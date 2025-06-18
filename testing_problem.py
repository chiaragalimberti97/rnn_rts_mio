import numpy as np
import matplotlib.pyplot as plt
import os
import json
import torch
from collections import OrderedDict
from torch.utils.data import DataLoader
from utils.su_maps import get_su_values
from models import setup_model
from datasets import setup_dataset
import seaborn as sns



# LOSS PLOT (must be adjusted to also plot validation loss)
# ======================================================================================================================

def loss_plot(model_name, num_epochs):
    results_folder = f'results/{model_name}/'

    # Carica le loss salvate
    data_train = np.load(os.path.join(results_folder, 'train.npz'))
    data_val = np.load(os.path.join(results_folder, 'val.npz'))
    print("Chiavi nel dizionario val.npz:", list(data_val.keys()))


    key = "loss"
    print("Available keys in val.npz:", list(data_val.keys()))

    # Training loss: array di loss per step, calcola media per epoca
    train_losses = data_train[key]
    total_steps_train = len(train_losses)
    steps_per_epoch_train = total_steps_train // num_epochs
    print(f"📊 Totale step: {total_steps_train}, Epoche: {num_epochs}, Step/epoca: {steps_per_epoch_train}")

    train_loss_per_epoch = []
    for epoch in range(num_epochs):
        start = epoch * steps_per_epoch_train
        end = (epoch + 1) * steps_per_epoch_train
        epoch_loss_train = train_losses[start:end].mean()
        train_loss_per_epoch.append(epoch_loss_train)

    # Validation loss (già media per epoca)
    val_losses = data_val["val_loss"]
    print(len(val_losses))

    total_steps_val = len(val_losses)
    steps_per_epoch_val = total_steps_val // num_epochs
    print(f"📊 Totale step: {total_steps_val}, Epoche: {num_epochs}, Step/epoca: {steps_per_epoch_val}")

    val_loss_per_epoch = []
    for epoch in range(num_epochs):
        start = epoch * steps_per_epoch_val
        end = (epoch + 1) * steps_per_epoch_val
        epoch_loss_val = val_losses[start:end].mean()
        val_loss_per_epoch.append(epoch_loss_val)


    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_loss_per_epoch, label='Training Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_image = os.path.join(results_folder, "loss_curve.png")
    plt.savefig(output_image)
    print(f"✅ Grafico salvato in: {output_image}")
    plt.show()

# PLOTTING OF SEGMENTATION MAP FINALS AND TEMPORAL  (could be reduced to only a parametric dynamic with yes if only last step and no for each step and also adding the original seg map could be nice)
# ======================================================================================================================

def seg_map(model,dataset,x,index):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42)
    np.random.seed(42)
    img_index = index 
    sample = dataset[img_index]
    img = sample['image']

    fix_x = x[0]
    fix_y = x[1]

    step_size = 5
    c, h, w = img.shape

    # Costruisci griglia
    grid = np.zeros((h, w))
    grid[::step_size, ::step_size] = 1
    cue_y, cue_x = np.where(grid == 1)  

    # Prepara input ripetuto per ogni cue
    inputs = img[:3].unsqueeze(0).repeat(len(cue_x), 1, 1, 1).to(device)  # img senza canale dot
    dots = torch.zeros((len(cue_x), 1, h, w), device=device)
    dots[:, 0, fix_y, fix_x] = 1         # fissazione fissa
    dots[range(len(cue_x)), 0, cue_y, cue_x] = 1  # cue che varia

    inputs = torch.cat([inputs, dots], dim=1)  # aggiungi canale dot

    # Split in batch
    batch_size = 1
    batches = torch.split(inputs, batch_size)
    dots_batches = torch.split(dots, batch_size)

    model.eval()
    values_list = []
    time_list = []

    with torch.no_grad():
        for batch, dots_batch in zip(batches,dots_batches):
            output_dict = model(batch, 0, 0, testmode=True)
            scalars = create_dynamic(output_dict, model, index, fix_x, dots_batch)
            
            val = output_dict['output'][:,0]  # lista lunghezza T
            if val.dim() > 1:
                val = val.squeeze(-1)
            values_list.append(val.cpu())
            time_list.append(scalars)
            
    values = torch.cat(values_list, dim=0).numpy()  # shape: [n_cue_points]
    values = np.exp(values)  # trasformazione esponenziale

    vmin=0
    vmax=1
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    # Immagine originale
    axs[0].imshow(img[:3].permute(1, 2, 0).cpu())
    axs[0].set_title("Immagine originale")
    axs[0].axis("off")

    # Immagine con scatter dei valori
    scatter = axs[1].imshow(img[:3].permute(1, 2, 0).cpu())
    sc = axs[1].scatter(cue_x, cue_y, c=values, cmap="turbo", s=50, vmin = vmin, vmax = vmax)
    axs[1].scatter(fix_x, fix_y, color='white', s=100, marker='x')  # fissazione
    axs[1].set_title("Spatial uncertainty map")
    axs[1].axis("off")

    # Colorbar accanto
    fig.colorbar(sc, ax=axs[1], fraction=0.046, pad=0.04, label='Model output value')

    plt.tight_layout()


    # === Salvataggio immagine ===
    save_name = None
    save_dir = f"results/{model_name}/plots/image{index}"
    os.makedirs(save_dir, exist_ok=True)
    if save_name is None:
        save_name = f"segmap_index{index}_fix{fix_x}_{fix_y}.png"
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path)
    print(f"✅ Immagine salvata in: {save_path}")

    #plt.show()


    # Concatena tutte le serie temporali batch-wise:
    # time_series_list è lista di array shape (timesteps, batch_size=1)
    time_series_all = np.concatenate(time_list, axis=1)  # shape (timesteps, n_cue_points)

    n_timesteps = time_series_all.shape[0]

    # Plot per ogni timestep
    for t in range(n_timesteps):
        plt.figure(figsize=(8, 6))
        plt.imshow(img[:3].permute(1, 2, 0).cpu())
        
        # Prendi i valori al timestep t e applichiamo esponenziale (come per valori finali)
        vals_t = time_series_all[t, :]
        
        sc = plt.scatter(cue_x, cue_y, c=vals_t, cmap='turbo', s=50)
        plt.scatter(fix_x, fix_y, color='white', s=100, marker='x')
        plt.title(f"Spatial uncertainty map - timestep {t}")
        plt.axis('off')
        plt.colorbar(sc, label='Model output value')

        # Salva o mostra l'immagine
        save_dir = f"results/{model_name}/plots/image{index}/timesteps{fix_x}_{fix_y}"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"segmap_index{index}_fix{fix_x}_{fix_y}_t{t}.png")
        plt.savefig(save_path)
        print(f"✅ Salvata immagine timestep {t} in: {save_path}")
        plt.close()
def create_dynamic(output_dic, model,index,fix_x, x_dots):
    states = output_dic.get('states', None)
    if states is None:
        print("No states found in output_dic.")
        return None

    model.eval()
    values_over_time = []
    with torch.no_grad():
        for t, state in enumerate(states):
            if model_name == 'hgru' :
                readout_logits = model.readout(state)  # shape: (batch_size, n_classes)
            else : 
                readout_logits = model.readout(state, x_dots)  # shape: (batch_size, n_classes)
            readout_probs = torch.exp(readout_logits)
            val = readout_probs[:, 0]  # tensore (batch_size,)
            values_over_time.append(val.cpu().numpy())  # salva il vettore intero per timestep
    return np.array(values_over_time)  # shape (timesteps, batch_size)

# DATASET AND MODEL LOADING
# ======================================================================================================================

def set_data(model_name, checkpointstr, datastr, data_root):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setting up model
    with open(f'results/{model_name}/opts.json', 'rb') as f:
        train_opts = json.load(f)  # get hyperparameters the model was trained with

    model = setup_model(**train_opts)
    checkpoint = torch.load(f'results/{model_name}/saved_models/{checkpointstr}')
    state_dict = checkpoint['state_dict']

    if train_opts['parallel']:
        state_dict = OrderedDict([(k.replace('module.',''), v) for k, v in state_dict.items()])  

    model.load_state_dict(state_dict)
    model = model.eval()
    model = model.to(device)
    # Setting up dataset
    kwargs = {
    'base_size': 150,
    }

    dataset = setup_dataset(datastr, data_root, subset=1, shuffle=False, **kwargs)

    return model,dataset

# DATASET DISTRIBUTION CHECK ----> LABELS AND DOTS
# ======================================================================================================================

def check_labels(dataset):
    # Crea i bordi dei bin: [0.0, 0.1, ..., 1.0]
    bin_size = 0.01
    bins = np.arange(0, 1.0 + bin_size, bin_size)
    bin_labels = [f"{round(bins[i], 1)}–{round(bins[i+1], 1)}" for i in range(len(bins) - 1)]
    bin_counts = [0 for _ in range(len(bins) - 1)]

    for i in range(len(dataset)):
        sample = dataset[i]
        label = sample['label']

        # Converti a float se è tensor
        if isinstance(label, torch.Tensor):
            label = label.item()

        # Clipping per sicurezza
        label = max(0.0, min(1.0, float(label)))

        # Trova l'indice del bin (escludendo l'ultimo bordo)
        bin_index = min(int(label // bin_size), len(bin_counts) - 1)
        bin_counts[bin_index] += 1

    # Plot
    plt.figure(figsize=(10, 5))
    plt.bar(bin_labels, bin_counts, width=0.8, align='center')
    plt.xlabel("Intervallo label")
    plt.ylabel("Numero di occorrenze")
    plt.title("Distribuzione delle label continue (bin size = 0.10)")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
def check_dots(dataset,index):

    torch.manual_seed(42)
    np.random.seed(42)
    img_index = index 
    # Crea array vuoti
    fix_x, fix_y, cue_x, cue_y = [], [], [], []
    

    indices = [i for i in range(len(dataset)) if dataset[i]['id'] == img_index]

    for i in indices:        
        sample = dataset[i]
        fix_x.append(sample['fixation_x'])
        fix_y.append(sample['fixation_y'])
        cue_x.append(sample['cue_x'])
        cue_y.append(sample['cue_y'])
        x1 = sample['fixation_x']
        y1 = sample['fixation_y']
        x2 = sample['cue_x']
        y2 = sample['cue_y']
        img = sample['image'][:3].permute(1, 2, 0).cpu().numpy()  # Rimuove dot channel e lo porta a HWC per matplotlib

        if i ==0:
            plt.figure(figsize=(8, 8))
            plt.imshow(img)
            plt.grid(True, color='white', alpha=0.3, linestyle='--')
            plt.axis("off")

        plt.plot([x1, x2], [y1, y2], color='red', linewidth=0.8, alpha=0.7)
        plt.scatter([x1, x2], [y1, y2], color='blue', s=10)

    plt.title("Tutte le coppie di dot (fissazione → cue)")
    plt.show()   

       

    # Fixation heatmap
    plt.figure(figsize=(6, 5))
    sns.kdeplot(x=fix_x, y=fix_y, cmap="Reds", fill=True, thresh=0.05)
    plt.title("Fixation Point Heatmap")
    plt.xlim(0, 150)
    plt.ylim(0, 150)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()

    # Cue heatmap
    plt.figure(figsize=(6, 5))
    sns.kdeplot(x=cue_x, y=cue_y, cmap="Blues", fill=True, thresh=0.05)
    plt.title("Cue Point Heatmap")
    plt.xlim(0, 150)
    plt.ylim(0, 150)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()

# MAIN
# ======================================================================================================================


if __name__ == "__main__":

    index = 1000
    #model_name = "HGRU_VGG19"
    #model_name = "test_model"
    #model_name = "balanced_bynary_flexmm"
    model_name = "bynary_flexmm"
    checkpoint = "model_acc_6733_epoch_26_checkpoint.pth.tar"
    data_str ='cocodots_val_flexmm'
    data_root = "./data/coco"
    num_epochs = 26

    check = False
    loss_plot(model_name, num_epochs )
   



    if check == True:
        
        loss_plot(model_name, num_epochs )
        model, dataset = set_data(model_name, checkpoint, data_str,data_root)
        
        check_dots(dataset,index)
        
        fixation_points = [[10,10], [10,75] , [10, 130] , [75,10], [75,75], [75,130], [130,10], [130,75] ,[130,130]]
        #fixation_points =[[75,75]]
        for x in fixation_points:
            seg_map(model,dataset,x,index)
    

        check_labels(dataset)


   





   




