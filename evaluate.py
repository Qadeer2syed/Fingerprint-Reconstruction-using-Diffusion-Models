# === evaluate.py ===
import os
import torch
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from model import create_unet_inpainting, create_scheduler
from dataset import InpaintingFromFilesDataset
from collections import OrderedDict
from diffusers.configuration_utils import FrozenDict
import torch.serialization

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Helper Functions ===
def load_compiled_weights(checkpoint):
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model_state'].items():
        if k.startswith('_orig_mod.'):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def safe_load_checkpoint(model_path, device):
    try:
        with torch.serialization.safe_globals([FrozenDict]):
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        print("✅ Loaded weights with safe_globals (secure mode)")
        return checkpoint
    except Exception as e:
        print(f"⚠️ Safe load failed: {str(e)}")
        print("⚠️ Trying with weights_only=False (less secure)")
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            print("✅ Loaded weights with weights_only=False")
            return checkpoint
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load checkpoint: {str(e)}")

# === Main Evaluation Function ===
def evaluate(model_path, clean_dir, save_steps=False, use_ema=True):
    model = create_unet_inpainting().to(DEVICE)
    
    try:
        checkpoint = safe_load_checkpoint(model_path, DEVICE)
        
        if isinstance(checkpoint, dict):
            if 'model_state' in checkpoint:
                if any(k.startswith('_orig_mod.') for k in checkpoint['model_state']):
                    state_dict = load_compiled_weights(checkpoint)
                else:
                    state_dict = checkpoint['model_state']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Load model weights
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"⚠️ Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"⚠️ Unexpected keys: {unexpected_keys}")
        
        print(f"✅ Successfully loaded weights from {model_path}")
        
    except Exception as e:
        print(f"❌ Failed to load weights: {str(e)}")
        return

    model.eval()
    scheduler = create_scheduler()
    dataset = InpaintingFromFilesDataset(clean_dir, train=False)

    os.makedirs("results", exist_ok=True)
    if save_steps:
        os.makedirs("results/steps", exist_ok=True)

    total_psnr, total_ssim, count = 0, 0, 0

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataset, desc="Evaluating")):
            masked = batch["masked"].unsqueeze(0).to(DEVICE)
            mask = batch["mask"].unsqueeze(0).to(DEVICE)
            target = batch["target"].unsqueeze(0).to(DEVICE)

            x = torch.randn_like(masked) * (1 - mask) + masked * mask

            for t in reversed(range(scheduler.config.num_train_timesteps)):
                t_batch = torch.tensor([t], device=DEVICE)
                
                x_cond = masked * mask + x * (1 - mask)
                model_input = torch.cat([x_cond, mask], dim=1)
                pred_noise = model(model_input, t_batch).sample
                x = scheduler.step(pred_noise, t, x).prev_sample
                x = masked * mask + x * (1 - mask)

                if save_steps and t % 50 == 0:
                    step_img = x[0].clamp(0, 1).squeeze().cpu().numpy()
                    Image.fromarray((step_img * 255).astype(np.uint8)).save(
                        f"results/steps/step_{idx}_{t}.png"
                    )

                recon = x.clamp(0, 1).squeeze().cpu().numpy()
                target_np = target.squeeze().cpu().numpy()
                mask_np = mask.squeeze().cpu().numpy() < 0.5  # Missing regions
                corrupted_np = masked.squeeze().cpu().numpy()

            if np.sum(mask_np) > 10:
                total_psnr += psnr(target_np[mask_np], recon[mask_np], data_range=1.0)
                total_ssim += ssim(target_np[mask_np], recon[mask_np], data_range=1.0, win_size=3)
                count += 1

            Image.fromarray((recon * 255).astype(np.uint8)).save(f"results/{idx}_recon.png")
            Image.fromarray((mask_np * 255).astype(np.uint8)).save(f"results/{idx}_mask.png")
            Image.fromarray((corrupted_np * 255).astype(np.uint8)).save(f"results/{idx}_corrupted.png")

    if count > 0:
        print(f"\n🔍 Evaluation Results (on {count} images)")
        print(f"PSNR (inpainted regions): {total_psnr/count:.2f} dB")
        print(f"SSIM (inpainted regions): {total_ssim/count:.4f}")
    else:
        print("\n⚠️ No valid inpainted regions to evaluate!")

if __name__ == "__main__":
    # 📢 Tip: Use EMA checkpoint if available
    evaluate(
        model_path="model_epoch13_ema.pt",  # 🧠 Load EMA checkpoint
        clean_dir="Test_Images",
        save_steps=False
    )
