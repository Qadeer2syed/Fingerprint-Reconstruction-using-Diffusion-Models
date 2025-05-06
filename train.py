# === train.py ===
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from dataset import InpaintingFromFilesDataset
from model import create_unet_inpainting, create_scheduler
import torch.nn.functional as F
from torchvision.models import vgg16
import torch.nn as nn
from pytorch_msssim import ssim
from torch_ema import ExponentialMovingAverage
from torch.optim.lr_scheduler import CosineAnnealingLR

# === Setup Device ===
def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        print(f"=== Using {torch.cuda.get_device_name(0)} ===")
        return device
    print("=== Using CPU ===")
    return torch.device("cpu")

DEVICE = setup_device()

# === Config ===
BATCH_SIZE = 16 if torch.cuda.is_available() else 4
EPOCHS = 100
LR = 1e-4

# === DataLoader ===
dataset = InpaintingFromFilesDataset(
    clean_dir="preprocessed/2013/clean",
    image_size=128,
    train=True
)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=2 if torch.cuda.is_available() else 0
)

# === Model, Scheduler, EMA ===
model = create_unet_inpainting().to(DEVICE)
if torch.cuda.is_available():
    model = torch.compile(model)
    model.enable_gradient_checkpointing()

scheduler = create_scheduler()
optimizer = AdamW(model.parameters(), lr=LR)

ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
scheduler_lr = CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-6
)

# === VGG Perceptual Loss Setup ===
vgg = vgg16(pretrained=True).features[:16].to(DEVICE)  # up to conv3_3
vgg.eval()
for p in vgg.parameters():
    p.requires_grad = False
loss_fn_vgg = nn.L1Loss()

def perceptual(x, y):
    return loss_fn_vgg(vgg(x.expand(-1,3,-1,-1)), vgg(y.expand(-1,3,-1,-1)))

# === Training Loop ===
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        masked = batch["masked"].to(DEVICE, non_blocking=True)
        mask = batch["mask"].to(DEVICE, non_blocking=True)
        target = batch["target"].to(DEVICE, non_blocking=True)

        noise = torch.randn_like(target)
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps,
            (target.size(0),), device=DEVICE
        )

        # Prepare noisy target and conditional input
        noisy_target = target * mask + noise * (1 - mask)
        x_cond = masked * mask + noisy_target * (1 - mask)
        model_input = torch.cat([x_cond, mask], dim=1)

        # Forward
        pred_noise = model(model_input, timesteps).sample

        # Predict clean image
        predicted_x0 = noisy_target - pred_noise

        # === Losses ===
        mse_loss = F.mse_loss(pred_noise * (1 - mask), noise * (1 - mask))

        ssim_loss = 1 - ssim(
            predicted_x0 * (1 - mask),
            target * (1 - mask),
            data_range=1.0,
            size_average=True
        )

        vgg_loss = perceptual(predicted_x0, target)

        # === Total Loss ===
        alpha = min(epoch / 20, 1.0)  # SSIM warmup
        loss = (
            0.9 * mse_loss +
            0.1 * ssim_loss
            # 0.1 * vgg_loss
        )

        # === Backward ===
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        ema.update()
        scheduler_lr.step()

        total_loss += loss.item()

    # === Save Checkpoints (normal + EMA versions) ===
    checkpoint = {
        'epoch': epoch + 1,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'loss': total_loss / len(loader),
        'config': model.config
    }
    torch.save(checkpoint, f"model_epoch{epoch+1}.pt")

    ema_checkpoint = {
        'epoch': epoch + 1,
        'model_state': ema.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'loss': total_loss / len(loader),
        'config': model.config
    }
    torch.save(ema_checkpoint, f"model_epoch{epoch+1}_ema.pt")

    print(f"Epoch {epoch+1} | Loss: {total_loss / len(loader):.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
