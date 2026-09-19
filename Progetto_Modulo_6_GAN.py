"""
PROGETTO FINALE: Conditional Wasserstein GAN with Gradient Penalty (cWGAN-GP)
Dataset: STL-10 | Framework: PyTorch

Obiettivo: generare immagini RGB 64x64 appartenenti a una classe specifica.
Loss Critic: E[C(fake)] - E[C(real)] + lambda_gp * GradientPenalty
Loss Generator: -E[C(fake)]
"""

# =========================================================================================
# 1. IMPORT LIBRERIE
# =========================================================================================
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# =========================================================================================
# 2. CONFIGURAZIONE E IPERPARAMETRI
# =========================================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo utilizzato: {device}")

# Model Iperparameters
img_size = 64
channels_img = 3
num_classes = 10
z_dim = 100
embedding_dim = 50

features_gen = 64
features_critic = 64

# Training Cycle
batch_size = 128
learning_rate = 0.0001
num_epochs = 100
n_critic = 5
lambda_gp = 10
samples_per_class = 5

# Saving pictures
save_dir = "cwgan_gp_results"
os.makedirs(save_dir, exist_ok=True)

# Classes names to train the model on
STL10_CLASSES = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]

# =========================================================================================
# 3. DATA PIPELINE
# =========================================================================================
# Resize to 64x64, conversion in Tensor and normalisation from [0,1] to [-1,1]. 

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Download Dataset and load with dataloader
train_dataset = datasets.STL10(root="./data", split="train", download=True, transform=transform)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
    pin_memory=torch.cuda.is_available(), drop_last=True
)

print(f"Numero immagini training: {len(train_dataset)}")
print(f"Numero batch: {len(train_loader)}")

# =========================================================================================
# 4. GENERATORE CONDIZIONATO
# =========================================================================================

class Generator(nn.Module):
    def __init__(self, z_dim, embedding_dim, num_classes, channels_img, features_gen):
        super().__init__()
        self.features_gen = features_gen
        self.label_embedding = nn.Embedding(num_classes, embedding_dim)

        self.project = nn.Sequential(
            nn.Linear(z_dim + embedding_dim, features_gen * 8 * 4 * 4),
            nn.BatchNorm1d(features_gen * 8 * 4 * 4),
            nn.ReLU(True)
        )

        self.main = nn.Sequential(
            nn.ConvTranspose2d(features_gen * 8, features_gen * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_gen * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_gen * 4, features_gen * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_gen * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_gen * 2, features_gen, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_gen),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_gen, channels_img, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Calculate the embedding vector from the label.
        class_embedding = self.label_embedding(labels)

        # Concatenation of the embedding to the input noise vector z.
        x = torch.cat((noise, class_embedding), dim=1)

        # Initial Linear part
        x = self.project(x)

        # Data Reshape in 2D domain
        x = x.view(x.size(0), self.features_gen * 8, 4, 4)

        return self.main(x)

# =========================================================================================
# 5. CRITICO CONDIZIONATO
# =========================================================================================
class Critic(nn.Module):
    def __init__(self, embedding_dim, num_classes, channels_img, features_critic, img_size):
        super().__init__()
        self.img_size = img_size
        self.label_embedding = nn.Embedding(num_classes, embedding_dim)
        self.label_to_image = nn.Linear(embedding_dim, img_size * img_size)
        input_channels = channels_img + 1

        self.main = nn.Sequential(
            nn.Conv2d(input_channels, features_critic, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features_critic, features_critic * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LayerNorm([features_critic * 2, 16, 16]),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features_critic * 2, features_critic * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LayerNorm([features_critic * 4, 8, 8]),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features_critic * 4, features_critic * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LayerNorm([features_critic * 8, 4, 4]),
            nn.LeakyReLU(0.2, inplace=True),

            # No final activaiton function to maintain real-numer score
            nn.Conv2d(features_critic * 8, 1, kernel_size=4, stride=1, padding=0)
        )

    def forward(self, images, labels):
        class_embedding = self.label_embedding(labels)
        label_map = self.label_to_image(class_embedding)
        label_map = label_map.view(labels.size(0), 1, self.img_size, self.img_size)
        x = torch.cat((images, label_map), dim=1)
        return self.main(x).view(-1)

# =========================================================================================
# 6. INIZIALIZZAZIONE PESI
# =========================================================================================
def weights_init(model):
    classname = model.__class__.__name__

    # Initialization of weights in accordance to best practices.
    if "Conv" in classname:
        if hasattr(model, "weight") and model.weight is not None:
            nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        if hasattr(model, "weight") and model.weight is not None:
            nn.init.normal_(model.weight.data, 1.0, 0.02)
        if hasattr(model, "bias") and model.bias is not None:
            nn.init.constant_(model.bias.data, 0)
    elif "Linear" in classname:
        nn.init.xavier_normal_(model.weight.data)
        if model.bias is not None:
            nn.init.constant_(model.bias.data, 0)

# =========================================================================================
# 7. GRADIENT PENALTY
# =========================================================================================


def compute_gradient_penalty(critic, real_images, fake_images, labels, device):
    batch_size = real_images.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = alpha * real_images + (1 - alpha) * fake_images
    interpolated.requires_grad_(True)

    mixed_scores = critic(interpolated, labels)
    gradients = torch.autograd.grad(
        outputs=mixed_scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    return torch.mean((gradient_norm - 1) ** 2)

# =========================================================================================
# 8. GALLERIA DELLE CLASSI
# =========================================================================================
def save_class_gallery(generator, fixed_noise, fixed_labels, epoch, class_names, samples_per_class, save_dir):
    generator.eval()
    with torch.no_grad():
        fake_images = generator(fixed_noise, fixed_labels)

    fake_images = ((fake_images + 1) / 2).clamp(0, 1).cpu()
    fig, axes = plt.subplots(
        len(class_names), samples_per_class,
        figsize=(samples_per_class * 2, len(class_names) * 2)
    )

    for class_idx, class_name in enumerate(class_names):
        for sample_idx in range(samples_per_class):
            image_idx = class_idx * samples_per_class + sample_idx
            image = fake_images[image_idx].permute(1, 2, 0).numpy()
            ax = axes[class_idx, sample_idx]
            ax.imshow(image)
            ax.axis("off")
            if sample_idx == 0:
                ax.set_ylabel(class_name, rotation=0, labelpad=35, va="center", fontsize=10)

    plt.suptitle(f"cWGAN-GP - Epoch {epoch}", fontsize=16)
    plt.tight_layout()
    path = os.path.join(save_dir, f"class_gallery_epoch_{epoch:03d}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    generator.train()
    print(f"Galleria salvata: {path}")

# =========================================================================================
# 9. GRAFICO LOSS DEL CRITICO
# =========================================================================================
def plot_critic_loss(critic_history, save_dir):
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, len(critic_history) + 1), critic_history, label="Critic Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Andamento della Loss del Critico - cWGAN-GP")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, "critic_loss.png")
    plt.savefig(path, dpi=150)
    plt.show()
    plt.close()
    print(f"Grafico loss salvato: {path}")

# =========================================================================================
# 10. CREAZIONE GENERATOR E CRITIC
# =========================================================================================
generator = Generator(z_dim, embedding_dim, num_classes, channels_img, features_gen).to(device)
critic = Critic(embedding_dim, num_classes, channels_img, features_critic, img_size).to(device)
generator.apply(weights_init)
critic.apply(weights_init)

# =========================================================================================
# 11. OPTIMIZER
# =========================================================================================
optimizer_gen = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.9))
optimizer_critic = optim.Adam(critic.parameters(), lr=learning_rate, betas=(0.5, 0.9))

# =========================================================================================
# 12. RUMORE FISSO PER LA VALIDAZIONE
# =========================================================================================
fixed_noise = torch.randn(num_classes * samples_per_class, z_dim, device=device)
fixed_labels = torch.arange(num_classes, device=device).repeat_interleave(samples_per_class)

# =========================================================================================
# 13. STORICO DELLE LOSS
# =========================================================================================
critic_history = []
generator_history = []

# =========================================================================================
# 14. IMMAGINI PRIMA DEL TRAINING
# =========================================================================================
save_class_gallery(generator, fixed_noise, fixed_labels, 0, STL10_CLASSES, samples_per_class, save_dir)

# =========================================================================================
# 15. CUSTOM TRAINING LOOP
# =========================================================================================
for epoch in range(1, num_epochs + 1):
    critic_losses_epoch = []
    generator_losses_epoch = []

    for batch_idx, (real_images, labels) in enumerate(train_loader):
        real_images = real_images.to(device)
        labels = labels.to(device).long()
        current_batch_size = real_images.size(0)

        # A. Training del Critico: n_critic aggiornamenti per ogni update del Generator.
        for _ in range(n_critic):
            noise = torch.randn(current_batch_size, z_dim, device=device)
            fake_images = generator(noise, labels)

            critic_real = critic(real_images, labels)
            critic_fake = critic(fake_images.detach(), labels)
            gp = compute_gradient_penalty(critic, real_images, fake_images.detach(), labels, device)
            loss_critic = torch.mean(critic_fake) - torch.mean(critic_real) + lambda_gp * gp

            optimizer_critic.zero_grad()
            loss_critic.backward()
            optimizer_critic.step()

        # B. Training del Generator: un aggiornamento.
        noise = torch.randn(current_batch_size, z_dim, device=device)
        fake_images = generator(noise, labels)
        output = critic(fake_images, labels)
        loss_generator = -torch.mean(output)

        optimizer_gen.zero_grad()
        loss_generator.backward()
        optimizer_gen.step()

        critic_losses_epoch.append(loss_critic.item())
        generator_losses_epoch.append(loss_generator.item())

        if batch_idx % 20 == 0:
            print(
                f"Epoch [{epoch:03d}/{num_epochs}] Batch [{batch_idx:03d}/{len(train_loader)}] | "
                f"Loss C: {loss_critic.item():.4f} | Loss G: {loss_generator.item():.4f} | GP: {gp.item():.4f}"
            )

    mean_critic_loss = np.mean(critic_losses_epoch)
    mean_generator_loss = np.mean(generator_losses_epoch)
    critic_history.append(mean_critic_loss)
    generator_history.append(mean_generator_loss)

    print(
        f"\n>>> Fine Epoch {epoch:03d}: Critic Loss media = {mean_critic_loss:.4f} | "
        f"Generator Loss media = {mean_generator_loss:.4f}\n"
    )

    if epoch % 10 == 0 or epoch == num_epochs // 2 or epoch == num_epochs:
        save_class_gallery(generator, fixed_noise, fixed_labels, epoch, STL10_CLASSES, samples_per_class, save_dir)

# =========================================================================================
# 16. GRAFICO FINALE DELLA LOSS DEL CRITICO
# =========================================================================================
plot_critic_loss(critic_history, save_dir)

# =========================================================================================
# 17. SALVATAGGIO MODELLI
# =========================================================================================
torch.save(generator.state_dict(), os.path.join(save_dir, "generator_final.pth"))
torch.save(critic.state_dict(), os.path.join(save_dir, "critic_final.pth"))
print("\nTraining completato.")
print(f"Risultati salvati nella cartella: {save_dir}")
