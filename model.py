from diffusers import UNet2DModel, DDPMScheduler

def create_unet_inpainting():
    return UNet2DModel(
        sample_size=128,
        in_channels=2,  # x_cond + mask
        out_channels=1,  # predicted noise
        layers_per_block=2,
        block_out_channels=(128, 256, 512),  # Increased capacity
        down_block_types=(
            "DownBlock2D",        # 128x128
            "AttnDownBlock2D",    # 64x64 with attention
            "AttnDownBlock2D"     # 32x32 with attention
        ),
        up_block_types=(
            "AttnUpBlock2D",      # 32x32
            "AttnUpBlock2D",      # 64x64
            "UpBlock2D"           # 128x128
        ),
        attention_head_dim=8,     # For attention blocks
        norm_num_groups=32,       # Better for small batch sizes
        add_attention=True,  # Ensure this is True (default)
    )

def create_scheduler():
    return DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",  # Better for images
        prediction_type="epsilon"           # Explicitly set
    )