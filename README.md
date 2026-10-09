<div align="center">

# 🦙 Llama-2 Fine-Tuning with LoRA & QLoRA

**Fine-tune Llama-2-7B-chat on a 4-bit quantised base with QLoRA to build a payment-dispute (chargeback) assistant.**

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗_Transformers-FFD21E)
![PEFT](https://img.shields.io/badge/PEFT-LoRA%20%2F%20QLoRA-informational)

</div>

---

## ✨ Overview

Prepared for the **Chase *AI digital agent* challenge**, `LLMByArashOmranpour_DATASCIENTIST.ipynb` shows the full LLM fine-tuning workflow:

1. 🧊 Load **`NousResearch/Llama-2-7b-chat-hf`** in **4-bit** precision with `bitsandbytes` (QLoRA).
2. 📝 Build a training set from custom conversations (`formatted_conversations.txt`, loaded with `datasets`) in the Llama-2 `[INST] ... [/INST]` chat format.
3. 🔧 Attach **LoRA** adapters with PEFT and train with `SFTTrainer` / `TrainingArguments`.
4. 💬 Test the model with a `text-generation` pipeline using chargeback questions (for example *"I want a chargeback for transactions with a valid 3D secure"*).
5. 🔗 Reload the base model in FP16 and **merge** the LoRA weights into a standalone model.

## 🚀 Getting Started

Needs a CUDA GPU - Google Colab (T4) is enough for QLoRA.

```bash
git clone https://github.com/Arashomranpour/LLMFineTuning_Lora_and_Qlora.git
cd LLMFineTuning_Lora_and_Qlora
pip install torch transformers peft trl bitsandbytes accelerate datasets jupyter
jupyter notebook LLMByArashOmranpour_DATASCIENTIST.ipynb
```

## 📁 Project Structure

```
.
└── LLMByArashOmranpour_DATASCIENTIST.ipynb   # QLoRA fine-tuning, inference and weight merge
```

## 🛠️ Tech Stack

`PyTorch` · `Hugging Face Transformers` · `PEFT (LoRA/QLoRA)` · `TRL` · `bitsandbytes`
