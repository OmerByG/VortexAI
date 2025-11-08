import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import os
import re
import time
import sys

from model import VortexModel, TextDataset


def load_config():
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ config.json bulunamadı!")
        return None


def save_checkpoint(model, optimizer, epoch, loss, best_loss, config):
    checkpoint_dir = config["files"]["checkpoint_dir"]
    model_name = config["files"]["model_name"]

    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "best_loss": best_loss,
    }

    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_last.pth")

    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
        except Exception as e:
            print(f"⚠️ Checkpoint silinemedi: {e}")

    torch.save(checkpoint, checkpoint_path)

    if loss <= best_loss:
        best_path = os.path.join(checkpoint_dir, f"{model_name}_best.pth")
        torch.save(checkpoint, best_path)
        return True

    return False


def load_checkpoint(model, optimizer, config):
    checkpoint_dir = config["files"]["checkpoint_dir"]
    model_name = config["files"]["model_name"]
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_last.pth")

    if os.path.exists(checkpoint_path):
        print(f"📂 Checkpoint bulundu: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]

        print(f"✅ Eğitim Epoch {start_epoch}'den devam edecek")
        print(f"📊 Son Loss: {checkpoint['loss']:.4f}")
        print(f"🏆 En İyi Loss: {best_loss:.4f}")

        return start_epoch, best_loss
    else:
        print("⚠️ Checkpoint bulunamadı, sıfırdan başlıyor...")
        return 0, float("inf")


def train():
    print("=" * 60)
    print("🚀 VortexAI Eğitim Başlatıldı")
    print("=" * 60)

    config = load_config()
    if not config:
        return

    data_file = config["files"]["data_file"]
    vocab_file = config["files"]["vocab_file"]

    try:
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        text = " ".join([item["sentence"] for item in data])
        text = text.lower()
        text = re.sub(r"[^a-zA-ZçğıöşüÇĞİÖŞÜ0-9\s]", "", text)
        text = text.split()
        print(f"✅ {len(text)} kelime yüklendi")
    except FileNotFoundError:
        print(f"❌ {data_file} bulunamadı! (python dategen.py çalıştır)")
        return

    try:
        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab = json.load(f)["word2idx"]
        vocab_size = len(vocab)
        print(f"✅ Kelime dağarcığı: {vocab_size} kelime")
    except FileNotFoundError:
        print(f"❌ {vocab_file} bulunamadı! (python tokenizer.py çalıştır)")
        return

    batch_size = config["training"]["batch_size"]
    dataset = TextDataset(text, vocab)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"✅ Dataset hazır: {len(dataset)} örnek")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Cihaz: {device}")

    model = VortexModel(
        vocab_size=vocab_size,
        embed_dim=config["model"]["embed_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    start_epoch, best_loss = load_checkpoint(model, optimizer, config)

    print("\n" + "=" * 60)
    print("📚 Eğitim Başlıyor...")
    print("💡 Ctrl+C ile durdurup devam edebilirsin")
    print("=" * 60 + "\n")

    epochs = config["training"]["epochs"]
    save_every = config["training"]["save_every"]

    bar_length = 20

    try:
        for epoch in range(start_epoch, epochs):
            start_time = time.time()
            model.train()
            total_loss = 0

            for x, y in loader:
                x, y = x.to(device), y.to(device)

                pred, _ = model(x)
                loss = criterion(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            progress = int(((epoch + 1) / epochs) * 20)

            bar = "█" * progress + "-" * (20 - progress)
            elapsed_time = time.time() - start_time

            sys.stdout.write(
                f"\r📘 Epoch {epoch+1:3d}/{epochs} |{bar}| "
                f"Loss: {avg_loss:.4f} | Best: {best_loss:.4f} | ⏱️ {elapsed_time:.2f}s"
            )
            sys.stdout.flush()

            if epoch % 5 == 0 or epoch == epochs - 1:
                sys.stdout.write("\n")

            if epoch % save_every == 0 or epoch == epochs - 1:
                is_best = save_checkpoint(model, optimizer, epoch, avg_loss, best_loss, config)
                if is_best:
                    best_loss = avg_loss

    except KeyboardInterrupt:
        print("\n⚠️ Eğitim durduruldu, son durum kaydediliyor...")
        save_checkpoint(model, optimizer, epoch, avg_loss, best_loss, config)
        print("✅ Checkpoint kaydedildi (devam edebilirsin)")

    final_path = f"{config['files']['model_name']}{config['files']['model_version']}.pth"
    torch.save(model.state_dict(), final_path)

    print("\n" + "=" * 60)
    print("✅ Eğitim tamamlandı!")
    print(f"📁 Final model: {final_path}")
    print(f"📁 Checkpoints: {config['files']['checkpoint_dir']}/")
    print("💡 Metin üretmek için: python generate.py")
    print("=" * 60)


if __name__ == "__main__":
    train()