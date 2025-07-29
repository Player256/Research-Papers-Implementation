import tqdm
import torch
from torch.utils.data import DataLoader
from ..model.srresnet import SRResNet
from .srresnet_test_dataset import SRResNetTestDataset

from skimage.metrics import structural_similarity as ssim

set5_path = "/home/ubuntu/oracle/Research-Papers-Implementation/SRGAN/test/1/set5/Set5"
set14_path = (
    "/home/ubuntu/oracle/Research-Papers-Implementation/SRGAN/test/1/set14/Set14"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SRResNet().to(device)
model.load_state_dict(torch.load("SRGAN/checkpoints/best_srresnet.pth"))
model.eval()

def load_dataset(path):
    test_dataset = SRResNetTestDataset(img_folder=path, hr_size=192, scale=4)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )
    
    return test_loader


def calculate_psnr(sr, hr):
    mse = torch.nn.functional.mse_loss(sr, hr)
    if mse == 0:
        return 100
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(sr, hr):
    sr_im = sr.permute(1, 2, 0).cpu().numpy()
    hr_im = hr.permute(1, 2, 0).cpu().numpy()
    return ssim(sr_im, hr_im, data_range=1.0, channel_axis=-1)


def test(model, test_loader, device):
    total_psnr = 0
    total_ssim = 0

    pbar = tqdm.tqdm(test_loader, desc="Testing")
    with torch.no_grad():
        for batch in pbar:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)

            sr = model(lr)

            psnr = calculate_psnr(sr[0], hr[0])
            ssim_value = calculate_ssim(sr[0], hr[0])

            total_psnr += psnr
            total_ssim += ssim_value

            pbar.set_postfix({"PSNR": psnr, "SSIM": ssim_value})
    avg_psnr = total_psnr / len(test_loader)
    avg_ssim = total_ssim / len(test_loader)
    print(f"Average PSNR: {avg_psnr:.2f}, Average SSIM: {avg_ssim:.4f}")
    return avg_psnr, avg_ssim


if __name__ == "__main__":
    test_loader = load_dataset(set5_path)
    print("Testing on Set5 dataset...")
    avg_psnr, avg_ssim = test(model, test_loader, device)
    print(f"Final Results - Average PSNR: {avg_psnr:.2f}, Average SSIM: {avg_ssim:.4f}")
    test_loader = load_dataset(set14_path)
    print("Testing on Set14 dataset...")
    avg_psnr, avg_ssim = test(model, test_loader, device)
    print(f"Final Results - Average PSNR: {avg_psnr:.2f}, Average SSIM: {avg_ssim:.4f}")
