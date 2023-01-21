import torch
import torchvision

def save_some_examples(gen, val_loader, epoch, device):
    x, y = next(iter(val_loader))
    x, y = x.to(device).squeeze(1), y.to(device).squeeze(1)
    gen.eval()
    with torch.no_grad():
        _, y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5 # remove normalization
        torchvision.utils.save_image(y_fake, f"/y_gen_{epoch}.png")
        torchvision.utils.save_image(x * 0.5 + 0.5, f"/input_{epoch}.png")
        if epoch == 1:
            torchvision.utils.save_image(y, f"/label_{epoch}.jpg")
    gen.train()

def save_checkpoint(model, optimizer, filename="my_ckpt.pth"):
    print("=> saving checkpoint...")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr, device):
    print("=> loading checkpoint...")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

