import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json


def load_config():
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("[*] config.json bulunamadı!")
        return None


def generate_text(model, tokenizer, prompt, max_length=50, temperature=1.0, top_k=50, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def main():
    print("=" * 60)
    print("[*] VortexAI - GPT-2 Metin Üretici")
    print("=" * 60)
    
    config = load_config()
    if not config:
        return
    
    model_name = config["files"]["model_name"]
    
    print(f"\n📥 Model yükleniyor: {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        print(f"[*] Model yüklendi")
        print(f"[*] Cihaz: {device}")
    except Exception as e:
        print(f"[*] Model yüklenemedi: {e}")
        print("[*] Önce eğitim yapın: python train_gpt2.py")
        return
    
    print("\n" + "=" * 60)
    print("Komutlar:")
    print("  - Metin girin → Devamını üretir")
    print("  - 'temp X' → Sıcaklık (0.5=güvenli, 1.5=yaratıcı)")
    print("  - 'len X' → Uzunluk ayarı")
    print("  - 'quit' → Çıkış")
    print("=" * 60 + "\n")
    
    temperature = 1.0
    max_length = 50
    
    while True:
        try:
            user_input = input("[*] Prompt: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == "quit":
                print("[*] Görüşmek üzere!")
                break
            
            if user_input.lower().startswith("temp "):
                try:
                    temperature = float(user_input.split()[1])
                    print(f"[*]🌡️  Sıcaklık: {temperature}")
                except:
                    print("[*] Geçersiz! Örnek: temp 1.5")
                continue
            
            if user_input.lower().startswith("len "):
                try:
                    max_length = int(user_input.split()[1])
                    print(f"[*] Uzunluk: {max_length}")
                except:
                    print("[*] Geçersiz! Örnek: len 100")
                continue
            
            print("\n🤖 VortexAI düşünüyor...")
            result = generate_text(
                model,
                tokenizer,
                user_input,
                max_length=max_length,
                temperature=temperature
            )
            
            print(f"\n[*] VortexAI:\n{result}\n")
            print("-" * 60 + "\n")
        
        except KeyboardInterrupt:
            print("\n[*] Görüşmek üzere!")
            break
        except Exception as e:
            print(f"[*] Hata: {e}")


if __name__ == "__main__":
    main()