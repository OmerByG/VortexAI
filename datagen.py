import json
import random

subjects = [
    {"word": "ben", "pos": "pronoun", "role": "özne", "meaning": "konuşan kişi"},
    {"word": "sen", "pos": "pronoun", "role": "özne", "meaning": "konuşulan kişi"},
    {"word": "o", "pos": "pronoun", "role": "özne", "meaning": "üçüncü kişi"},
]

verbs = [
    {"word": "yapıyorum", "pos": "verb", "role": "yüklem", "meaning": "bir işi gerçekleştirmek"},
    {"word": "gidiyorum", "pos": "verb", "role": "yüklem", "meaning": "bir yere doğru hareket etmek"},
    {"word": "konuşuyorum", "pos": "verb", "role": "yüklem", "meaning": "sözlü iletişim kurmak"},
    {"word": "uyuyorum", "pos": "verb", "role": "yüklem", "meaning": "dinlenmek için gözleri kapatmak"},
]

adjectives = [
    {"word": "mutlu", "pos": "adjective", "role": "sıfat", "meaning": "sevinçli, neşeli"},
    {"word": "üzgün", "pos": "adjective", "role": "sıfat", "meaning": "kederli, mutsuz"},
    {"word": "yorgun", "pos": "adjective", "role": "sıfat", "meaning": "enerjisi azalmış"},
]

questions = [
    {"word": "ne", "pos": "question", "role": "soru eki", "meaning": "bilinmeyeni sormak için kullanılır"},
    {"word": "nerede", "pos": "question", "role": "soru eki", "meaning": "yer sormak için kullanılır"},
    {"word": "nasılsın", "pos": "question", "role": "yüklem", "meaning": "hal-hatır sormak"},
]

punct = [
    {"word": ".", "pos": "punct", "role": "cümle sonu", "meaning": "ifadeyi bitirir"},
    {"word": "?", "pos": "punct", "role": "cümle sonu", "meaning": "soru cümlesi olduğunu belirtir"},
]

patterns = [
    ["subject", "verb", "punct"],
    ["subject", "adjective", "verb", "punct"],
    ["subject", "question", "punct"],
    ["question", "subject", "verb", "punct"]
]

def generate_sentence():
    pattern = random.choice(patterns)
    tokens = []

    for part in pattern:
        if part == "subject":
            tokens.append(random.choice(subjects))
        elif part == "verb":
            tokens.append(random.choice(verbs))
        elif part == "adjective":
            tokens.append(random.choice(adjectives))
        elif part == "question":
            tokens.append(random.choice(questions))
        elif part == "punct":
            tokens.append(random.choice(punct))

    sentence = " ".join([t["word"] for t in tokens if t["pos"] != "punct"]) + tokens[-1]["word"]
    return {"sentence": sentence, "tokens": tokens}


def main():
    data = [generate_sentence() for _ in range(200)]

    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("✅ Örnek cümleler üretildi ve 'data.json' dosyasına kaydedildi.")


if __name__ == "__main__":
    main()
