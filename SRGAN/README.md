# Image Super-Resolution – SRResNet & SRGAN  
First iteration results: **SRResNet (x4) – Set5 29.69 dB / 0.900 SSIM, Set14 26.16 dB / 0.804 SSIM**  
Next step: finish training & debugging the adversarial counterpart **SRGAN**.

## Contents of this repository
```
SRGAN/
├── dataset/              # PyTorch `Dataset` classes for SRResNet & SRGAN
├── model/                # Network definitions (SRResNet, SRGAN-Generator, Discriminator)
├── loss/                 # MSE (SRResNet) and VGG-content + adversarial (SRGAN)
├── train/                # Training scripts
├── test/                 # Inference / evaluation scripts
└── checkpoints/          # Pre-trained weights drop here after training
```

## Quick start

1.  Install Python packages (tested with PyTorch 2.2 + torchvision 0.17):

    ```bash
    conda create -n srgan python=3.10
    conda activate srgan
    pip install torch torchvision tqdm scikit-image kagglehub datasets wandb
    ```

2.  Download evaluation datasets (Set5 & Set14):

    ```python
    import kagglehub, shutil, os
    root = kagglehub.dataset_download("ll01dm/set-5-14-super-resolution-dataset")
    shutil.copytree(os.path.join(root, "Set5"),  "test/1/set5/Set5",  dirs_exist_ok=True)
    shutil.copytree(os.path.join(root, "Set14"), "test/1/set14/Set14", dirs_exist_ok=True)
    ```

3.  (Optional) train SRResNet from scratch:

    ```bash
    python -m SRGAN.train.srresnet       # ~20 epochs on 35 % of ImageNet-1K
    ```

4.  Evaluate the saved model:
    Model weights:- https://drive.google.com/file/d/1B7u-QYPKqrM1HSJx5Cw0fa-nZGFEDfpr/view?usp=sharing

    ```bash
    python -m SRGAN.test.srresnet_test   # uses checkpoints/best_srresnet.pth
    ```

   On the very first run the script produced the metrics shown in the screenshot:  

   | Dataset | PSNR (dB) | SSIM |
   |---------|-----------|------|
   | Set5    | **29.69** | **0.9001** |
   | Set14   | **26.16** | **0.8035** |

## What’s next?
-  Move on to the adversarial stage: `python -m SRGAN.train.srgan`  
  – still being debugged; expect higher perceptual quality once discriminator & VGG-loss are fully stable.


