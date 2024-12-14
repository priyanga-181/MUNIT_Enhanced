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

# Dataset class remains the same
class DynamicPreprocessingDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_pairs, transform):
        # self.dataset = dataset
        self.dataset_pairs = list(dataset_pairs)
        self.transform = transform

    def __len__(self):
        # return len(self.dataset)
        return len(self.dataset_pairs)

    def __getitem__(self, idx):
        cat_example, dog_example = self.dataset_pairs[idx]
        cat_image = self.transform(cat_example["image"].convert("RGB"))
        dog_image = self.transform(dog_example["image"].convert("RGB"))
        return {"cat_images": cat_image, "dog_images": dog_image}

def create_dataloaders(train_batch_size):
    # Load the AFHQ dataset
    dataset = load_dataset("huggan/AFHQ", split="train")
    dataset_splits = dataset.train_test_split(test_size=0.2, seed=42)
    
    # Split data
    train_dataset = dataset_splits["train"]
    temp_splits = dataset_splits["test"].train_test_split(test_size=0.5, seed=42)
    val_dataset = temp_splits["train"]
    test_dataset = temp_splits["test"]

    # Filter the dataset for cats and dogs
    def filter_by_label(dataset, label):
        return dataset.filter(lambda example: example["label"] == label)

    train_cats = filter_by_label(train_dataset, label=0)  # 0: Cats
    train_dogs = filter_by_label(train_dataset, label=1)  # 1: Dogs
    val_cats = filter_by_label(val_dataset, label=0)
    val_dogs = filter_by_label(val_dataset, label=1)
    test_cats = filter_by_label(test_dataset, label=0)
    test_dogs = filter_by_label(test_dataset, label=1)

    # Transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # Dataset combining cats as input and dogs as output
    train_dataset = DynamicPreprocessingDataset(zip(train_cats, train_dogs), transform)
    val_dataset = DynamicPreprocessingDataset(zip(val_cats, val_dogs), transform)
    test_dataset = DynamicPreprocessingDataset(zip(test_cats, test_dogs), transform)

    # Dataloaders
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
            cat_images = batch["cat_images"].to(accelerator.device)
            dog_images = batch["dog_images"].to(accelerator.device)
            # Move to float16 after transfer to device
            cat_images = cat_images.to(dtype=torch.float16)
            dog_images = dog_images.to(dtype=torch.float16)

            latents = pipeline.vae.encode(cat_images).latent_dist.sample()
            latents = latents * pipeline.vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, pipeline.scheduler.num_train_timesteps,
                (latents.shape[0],), device=latents.device
            )
            noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)

            batch_size = cat_images.shape[0]

            # Get style conditioning from shoe images
            with torch.no_grad():
                style_latents = pipeline.vae.encode(dog_images).latent_dist.sample()
                encoder_hidden_states = pipeline.text_encoder(style_latents)[0]
                encoder_hidden_states = encoder_hidden_states.to(dtype=torch.float16)

            noise_pred = pipeline.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states
            ).sample
            loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())  # Convert to float32 for loss
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
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(train_batch_size)
    
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
            cat_images = batch["cat_images"].to(accelerator.device)
            dog_images = batch["dog_images"].to(accelerator.device)
            cat_images = cat_images.to(dtype=torch.float16)
            dog_images = dog_images.to(dtype=torch.float16)

            with accelerator.accumulate(pipeline.unet):
                latents = pipeline.vae.encode(cat_images).latent_dist.sample()
                latents = latents * pipeline.vae.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, pipeline.scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=latents.device
                )
                noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)

                batch_size = cat_images.shape[0]

                # Get style conditioning from shoe images
                with torch.no_grad():
                    style_latents = pipeline.vae.encode(dog_images).latent_dist.sample()
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
        output_dir="lora-trained-model",
        num_train_epochs=10,
        train_batch_size=4,
        learning_rate=1e-4,
        rank=4,
        project_name="cats2dogs-lora",
    )