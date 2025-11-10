import torch
from tqdm import tqdm



def masked_mse_loss(pred, target, mask):
    """
    pred, target : (B, C, L)
    mask : (B, L)
    """
    mask = mask.unsqueeze(1).expand_as(pred)
    diff = (pred - target) ** 2 * mask
    return diff.sum() / mask.sum()



def train(model, diffusion, device, epochs, dataloader, lr=1e-5, max_len=210):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    result_loss = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}")

        for i, batch in progress_bar:
            # --- Batch brut : liste de séquences (2, L_i)
            seqs = [b[0].to(device) for b in batch]  # si dataset renvoie (x, label)
            B = len(seqs)

            # --- Déterminer la longueur max du batch
            batch_max_len = min(max(seq.shape[1] for seq in seqs), max_len)

            # --- Padding + mask
            x = torch.zeros(B, 2, batch_max_len, device=device)
            mask = torch.zeros(B, batch_max_len, dtype=torch.bool, device=device)
            for j, seq in enumerate(seqs):
                L = seq.shape[1]
                if L > batch_max_len:
                    x[j, :, :] = seq[:, :batch_max_len]
                    mask[j, :] = True
                else:
                    x[j, :, :L] = seq
                    mask[j, :L] = True

            # --- Étape de diffusion
            t = torch.randint(0, diffusion.noise_steps, (B,), device=device)
            noised_x, true_noise = diffusion.forward_diffusion(x, t)

            # --- Prédiction du bruit
            pred_noise = model(noised_x, t)

            # --- Calcul de la loss masquée
            loss = masked_mse_loss(pred_noise[:, :, :batch_max_len], true_noise[:, :, :batch_max_len], mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            result_loss.append(loss.item())
            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (i + 1))

    return model, result_loss
