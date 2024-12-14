import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from datasets import load_dataset
from accelerate import Accelerator
from tqdm.auto import tqdm
from torchvision import transforms
from PIL import Image
import wandb
from peft import LoraConfig, get_peft_model
from io import BytesIO
from glob import glob
import os

# Dataset class remains the same
class DynamicPreprocessingDataset(torch.utils.data.Dataset):
    def __init__(self, dir_A, dir_B, transform):
        self.images_A = sorted(glob(os.path.join(dir_A, "*")))  # Summer (A) images
        self.images_B = sorted(glob(os.path.join(dir_B, "*")))  # Winter (B) images
        self.transform = transform

    def __len__(self):
        return min(len(self.images_A), len(self.images_B))

    def __getitem__(self, idx):
        image_A = Image.open(self.images_A[idx]).convert("RGB")  # Summer image
        image_B = Image.open(self.images_B[idx]).convert("RGB")  # Winter image

        # Apply transforms
        image_A = self.transform(image_A)
        image_B = self.transform(image_B)

        return {"summer_images": image_A, "winter_images": image_B}

def create_dataloaders(data_dir, train_batch_size):
    # Define paths for train and test datasets
    train_dir_A = os.path.join(data_dir, "trainA")  # Summer images
    train_dir_B = os.path.join(data_dir, "trainB")  # Winter images
    test_dir_A = os.path.join(data_dir, "testA")    # Summer images
    test_dir_B = os.path.join(data_dir, "testB")    # Winter images

    # Transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # Create datasets
    train_dataset = DynamicPreprocessingDataset(train_dir_A, train_dir_B, transform)
    test_dataset = DynamicPreprocessingDataset(test_dir_A, test_dir_B, transform)

    # Split test dataset into validation and test sets
    val_size = int(0.5 * len(test_dataset))
    test_size = len(test_dataset) - val_size
    val_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [val_size, test_size])

    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=train_batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=train_batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader



def compute_loss(pipeline, dataloader, accelerator):
    pipeline.unet.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            summer_images = batch["summer_images"].to(accelerator.device)
            winter_images = batch["winter_images"].to(accelerator.device)
            summer_images = summer_images.to(dtype=torch.float16)
            winter_images = winter_images.to(dtype=torch.float16)

            latents = pipeline.vae.encode(summer_images).latent_dist.sample()
            latents = latents * pipeline.vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, pipeline.scheduler.num_train_timesteps,
                (latents.shape[0],), device=latents.device
            )
            noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)

            batch_size = summer_images.shape[0]
            # Get style conditioning from shoe images
            with torch.no_grad():
                style_latents = pipeline.vae.encode(winter_images).latent_dist.sample()
                encoder_hidden_states = pipeline.text_encoder(style_latents)[0]
                encoder_hidden_states = encoder_hidden_states.to(dtype=torch.float16)

            noise_pred = pipeline.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states
            ).sample
            loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())
            total_loss += loss.item()
            num_batches += 1

    pipeline.unet.train()
    return total_loss / num_batches


def train_lora(
    pretrained_model_path="runwayml/stable-diffusion-v1-5",
    output_dir="lora-trained-model",
    num_train_epochs=20,
    train_batch_size=16,
    learning_rate=1e-4,
    rank=8,
    project_name="edges2shoes-lora",
    max_grad_norm=1.0,
    data_dir=None, 
):
    wandb.init(project=project_name, config={
        "learning_rate": learning_rate,
        "epochs": num_train_epochs,
        "batch_size": train_batch_size,
        "rank": rank
    })
    
    # Initialize accelerator first
    accelerator = Accelerator(mixed_precision="fp16")
    
    # Load pipeline with float16
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_path,
        torch_dtype=torch.float16
    ).to(accelerator.device)
    
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

    # Configure LoRA
    supported_modules = [
        name for name, module in pipeline.unet.named_modules()
        if isinstance(module, torch.nn.Linear)
    ]

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=supported_modules,
        lora_dropout=0.0,
        bias="none",
    )

    pipeline.unet = get_peft_model(pipeline.unet, lora_config)
    
    # Create dataloaders
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(data_dir, train_batch_size)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(pipeline.unet.parameters(), lr=learning_rate)
    
    # Prepare for distributed training
    pipeline.unet, optimizer, train_dataloader = accelerator.prepare(
        pipeline.unet, optimizer, train_dataloader
    )

    for epoch in range(num_train_epochs):
        pipeline.unet.train()
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}")
        
        for batch in train_dataloader:
            summer_images = batch["summer_images"].to(accelerator.device)
            winter_images = batch["winter_images"].to(accelerator.device)
            summer_images = summer_images.to(dtype=torch.float16)
            winter_images = winter_images.to(dtype=torch.float16)

            with accelerator.accumulate(pipeline.unet):
                latents = pipeline.vae.encode(summer_images).latent_dist.sample()
                latents = latents * pipeline.vae.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, pipeline.scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=latents.device
                )
                noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)

                batch_size = summer_images.shape[0]

                # Get style conditioning from shoe images
                with torch.no_grad():
                    style_latents = pipeline.vae.encode(winter_images).latent_dist.sample()
                    encoder_hidden_states = pipeline.text_encoder(style_latents)[0]
                    encoder_hidden_states = encoder_hidden_states.to(dtype=torch.float16)

                noise_pred = pipeline.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states
                    ).sample

                # Convert to float32 for loss computation
                loss = torch.nn.functional.mse_loss(
                    noise_pred.float(),
                    noise.float()
                )

                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(pipeline.unet.parameters(), max_grad_norm)
                    
                optimizer.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item()})
            
        progress_bar.close()

        # Compute and log validation loss
        val_loss = compute_loss(pipeline, val_dataloader, accelerator)
        wandb.log({
            "train_loss": loss.item(),
            "val_loss": val_loss,
            "epoch": epoch
        })

        # Save checkpoint
        if (epoch + 1) % 2 == 0:
            pipeline.unet.save_pretrained(f"{output_dir}/checkpoint-{epoch+1}")

if __name__ == "__main__":
    train_lora(
        pretrained_model_path="runwayml/stable-diffusion-v1-5",
        output_dir="lora-summer2winter",
        num_train_epochs=10,
        train_batch_size=4,
        learning_rate=1e-4,
        rank=4,
        project_name="summer2winter-lora",
        data_dir="/home/ubuntu/summer2winter"
    )