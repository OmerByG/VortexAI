import torch
import json

from model import VortexModel


def load_config():
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ config.json bulunamadı!")
        return None


def generate_text(model, start_word, word2idx, idx2word, device, max_length=20, temperature=1.0):
    model.eval()
    
    if start_word not in word2idx:
        print(f"❌ '{start_word}' kelimesi bilinmiyor!")
        print(f"💡 Bilinen kelimeler: {', '.join(list(word2idx.keys())[:10])}...")
        return None
    
    generated = [start_word]
    x = torch.tensor([[word2idx[start_word]]]).to(device)
    hidden = None
    
    with torch.no_grad():
        for _ in range(max_length):
            out, hidden = model(x, hidden)
            
            if len(out.shape) == 3:
                out = out.squeeze(1)
            
            logits = out[0] / temperature
            probs = torch.softmax(logits, dim=0)
            
            idx = torch.multinomial(probs, 1).item()
            word = idx2word[idx]
            
            generated.append(word)
            x = torch.tensor([[idx]]).to(device)
    
    return " ".join(generated)


def main():
    print("=" * 60)
    print("🧠 VortexAI Metin Üretici")
    print("=" * 60)

    config = load_config()
    if not config:
        return
    
    vocab_file = config["files"]["vocab_file"]
    model_version = config["files"]["model_version"]
    model_name = config["files"]["model_name"]
    
    try:
        with open(vocab_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        word2idx = data["word2idx"]
        idx2word = {int(k): v for k, v in data["idx2word"].items()}
        vocab_size = len(word2idx)
        print(f"✅ Kelime dağarcığı: {vocab_size} kelime")
    except FileNotFoundError:
        print(f"❌ {vocab_file} bulunamadı!")
        print("💡 Önce çalıştırın: python tokenizer.py")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Cihaz: {device}")
    
    model = VortexModel(
        vocab_size,
        config["model"]["embed_dim"],
        config["model"]["hidden_dim"],
        config["model"]["num_layers"],
        config["model"]["dropout"]
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(f"{model_name}{model_version}.pth", map_location=device))
        print(f"✅ Model yüklendi: {model_name}{model_version}.pth")
    except FileNotFoundError:
        print(f"❌ {model_name}{model_version}.pth bulunamadı!")
        print("💡 Önce eğitim yapın: python train.py")
        return
    
    print("\n" + "=" * 60)
    print("Komutlar:")
    print("  - Kelime girin → Metin üretir")
    print("  - 'temp X' → Sıcaklık (0.5=güvenli, 1.5=yaratıcı)")
    print("  - 'len X' → Uzunluk ayarı")
    print("  - 'quit' → Çıkış")
    print("=" * 60 + "\n")
    
    temperature = 1.0
    max_length = 20
    
    while True:
        try:
            user_input = input("🔤 Kelime: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == "quit":
                print("👋 Görüşmek üzere!")
                break
            
            if user_input.lower().startswith("temp "):
                try:
                    temperature = float(user_input.split()[1])
                    print(f"🌡️  Sıcaklık: {temperature}")
                except:
                    print("❌ Geçersiz! Örnek: temp 1.5")
                continue
            
            if user_input.lower().startswith("len "):
                try:
                    max_length = int(user_input.split()[1])
                    print(f"📏 Uzunluk: {max_length}")
                except:
                    print("❌ Geçersiz! Örnek: len 30")
                continue
            
            result = generate_text(
                model, 
                user_input, 
                word2idx, 
                idx2word, 
                device,
                max_length,
                temperature
            )
            
            if result:
                print(f"\n🧠 VortexAI: {result}\n")
        
        except KeyboardInterrupt:
            print("\n👋 Görüşmek üzere!")
            break
        except Exception as e:
            print(f"❌ Hata: {e}")


if __name__ == "__main__":
    main()