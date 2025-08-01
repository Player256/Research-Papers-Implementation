import tqdm, torch, random, numpy as np, os
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from datasets import load_dataset

from SRGAN.model.srresnet import SRResNet
from SRGAN.model.srgan import Discriminator
from SRGAN.dataset.srgan import SRGANDataset
from SRGAN.loss.srgan_loss import SRGANLoss

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G = SRResNet().to(device)
G.load_state_dict(torch.load("SRGAN/checkpoints/best_srresnet.pth"))
D = Discriminator().to(device)

g_opt = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.9, 0.999))
d_opt = torch.optim.Adam(D.parameters(), lr=1e-6, betas=(0.9, 0.999))

loss_fn = SRGANLoss().to(device)


train_set = load_dataset("ILSVRC/imagenet-1k", split="train[:1%]", streaming=False)
val_set = load_dataset("ILSVRC/imagenet-1k", split="validation[:1%]", streaming=False)

train_ds = SRGANDataset(train_set, hr_size=96, scale=4, train_mode=True)
val_ds = SRGANDataset(val_set, hr_size=96, scale=4, train_mode=False)

train_loader = DataLoader(
    train_ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
)


epochs = 100
warmup_steps = 10000  
global_step = 0

os.makedirs("SRGAN/checkpoints/GAN", exist_ok=True)

for epoch in range(epochs):
    G.train()
    D.train()
    pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")

    for batch in pbar:
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)

        with torch.no_grad():
            fake = G(lr)

        d_real = D(hr)
        d_fake = D(fake.detach())
        d_loss = loss_fn.adv.d_loss(d_real, d_fake)

        d_opt.zero_grad()
        d_loss.backward()
        clip_grad_norm_(D.parameters(), 5)
        d_opt.step()
        
        if global_step >= warmup_steps:
            fake = G(lr)
            d_fake = D(fake)
            g_tot, l_content, l_adv = loss_fn.g_total(fake, hr, d_fake)

            g_opt.zero_grad()
            g_tot.backward()
            clip_grad_norm_(G.parameters(), 1)
            g_opt.step()
        else:
            g_tot = torch.tensor(0.0)

        global_step += lr.size(0)

        pbar.set_postfix(
            g_loss=f"{g_tot.item():.4f}",
            d_loss=f"{d_loss.item():.4f}",
            step=global_step,
        )

    if (epoch + 1) % 5 == 0:
        G.eval()
        psnr_acc, ssim_acc, cnt = 0, 0, 0
        with torch.no_grad():
            for b in val_loader:
                sr = G(b["lr"].to(device))
                for i in range(sr.size(0)):
                    cnt += 1
                    mse = torch.nn.functional.mse_loss(sr[i], b["hr"][i].to(device))
                    psnr_acc += 20 * torch.log10(1.0 / torch.sqrt(mse)).item()
                    
        print(f"[Eval] Epoch {epoch+1}: PSNR {psnr_acc/cnt:.2f} dB")

    torch.save(G.state_dict(), f"SRGAN/checkpoints/GAN/G_epoch{epoch+1}.pth")
    torch.save(D.state_dict(), f"SRGAN/checkpoints/GAN/D_epoch{epoch+1}.pth")

print("Finished training.")
