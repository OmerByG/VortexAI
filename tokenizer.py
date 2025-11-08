import json
import re

def load_config():
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ config.json bulunamadı!")
        return None

def create_tokenizer():
    print("=" * 60)
    print("📝 VortexAI Tokenizer")
    print("=" * 60)
    
    config = load_config()
    if not config:
        return
    
    data_file = config["files"]["data_file"]
    vocab_file = config["files"]["vocab_file"]
    
    try:
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"✅ {data_file} yüklendi: {len(data)} cümle")
    except FileNotFoundError:
        print(f"❌ {data_file} bulunamadı!")
        return

    text = " ".join([item["sentence"] for item in data])
    text = text.lower()
    text = re.sub(r"[^a-zA-ZçğıöşüÇĞİÖŞÜ0-9\s]", "", text)
    words = text.split()
    
    special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    unique_words = sorted(set(words))
    vocab = special_tokens + unique_words
    
    print(f"📊 Toplam kelime: {len(words)}")
    print(f"📚 Benzersiz kelime: {len(vocab)}")
    
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    vocab_data = {
        "word2idx": word2idx,
        "idx2word": idx2word,
        "vocab_size": len(vocab),
        "total_words": len(words)
    }
    
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Tokenizer kaydedildi: {vocab_file}")
    print(f"📝 İlk 10 kelime: {', '.join(vocab[:10])}")
    print("\n✅ Hazır! Şimdi eğitime başlayın:")
    print("   python train.py")
    print("=" * 60)

if __name__ == "__main__":
    create_tokenizer()