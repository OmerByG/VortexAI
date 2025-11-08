import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from torch.utils.data import Dataset
import json

def load_config():
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("[*] config.json bulunamadı!")
        return None

class VortexDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.sentences = []
        for item in data:
            self.sentences.append(item["text"])
        
        print(f"[*] {len(self.sentences)} cümle yüklendi")
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        
        encoding = self.tokenizer(
            sentence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()
        }

def train():
    print("=" * 60)
    print("[*] VortexAI - GPT-2 Türkçe Fine-tuning")
    print("=" * 60)
    
    config = load_config()
    if not config:
        return
    
    data_file = config["files"]["data_file"]
    model_name = config["files"]["model_name"]
    checkpoint_dir = config["files"]["checkpoint_dir"]
    
    print("\n📥 Türkçe GPT-2 modeli indiriliyor...")
    base_model = config["model"]["base_model"]
    
    model = AutoModelForCausalLM.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"[*] Model yüklendi: {base_model}")
    print(f"[*] Model parametreleri: {model.num_parameters():,}")
    
    print("\n[*] Dataset hazırlanıyor...")
    try:
        dataset = VortexDataset(data_file, tokenizer)
    except FileNotFoundError:
        print(f"[*] {data_file} bulunamadı!")
        return
    
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=config["training"]["epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        learning_rate=config["training"]["learning_rate"],
        save_steps=config["training"]["save_every"] * len(dataset) // config["training"]["batch_size"],
        save_total_limit=3,
        logging_steps=10,
        logging_dir=f"{checkpoint_dir}/logs",
        report_to="none",
        fp16=torch.cuda.is_available(),
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    print("\n" + "=" * 60)
    print("[*] Eğitim başlıyor...")
    print("[*] Ctrl+C ile durdurup devam edebilirsiniz")
    print("=" * 60 + "\n")
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n[*] Eğitim durduruldu!")
        trainer.save_model(f"{checkpoint_dir}/interrupted")
        print("[*] Model kaydedildi")
        return
    
    final_path = f"{model_name}"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    
    print("\n" + "=" * 60)
    print("[*] Eğitim tamamlandı!")
    print(f"[*] Final model: {final_path}/")
    print(f"[*] Checkpoints: {checkpoint_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    train()