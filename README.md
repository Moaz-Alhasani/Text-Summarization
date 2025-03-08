# Pegasus-SAMSum Summarization Fine-Tuning

This repository contains code to fine-tune Google's PEGASUS model on the SAMSum dataset for dialogue summarization. The SAMSum dataset consists of human-annotated summaries of chat dialogues, making it ideal for training abstractive summarization models.

## 📌 Features
- Fine-tunes `google/pegasus-cnn_dailymail` on the SAMSum dataset.
- Uses Hugging Face's `datasets` and `transformers` libraries.
- Supports GPU acceleration with PyTorch.
- Includes a Trainer API for easy fine-tuning.
- Saves the fine-tuned model and tokenizer for future inference.

## 🚀 Installation
Make sure you have Python installed, then run the following commands:

```bash
pip install --upgrade datasets transformers torch py7zr tqdm nltk matplotlib pandas
```

## 📂 Dataset
The dataset used for training is SAMSum, which can be loaded directly using Hugging Face's `datasets` library:

```python
from datasets import load_dataset
dataset_samsum = load_dataset("samsum")
```

## 🔥 Training the Model
Run the script to fine-tune the PEGASUS model on the SAMSum dataset:

```python
python train.py
```

## 🛠 Model and Tokenizer Saving
After training, the fine-tuned model and tokenizer are saved as:
```bash
pegasus-samsum-model/
tokenizer/
```
These can be loaded for inference as follows:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("pegasus-samsum-model")
model = AutoModelForSeq2SeqLM.from_pretrained("pegasus-samsum-model")
```

## 📈 Evaluation
The script includes evaluation on the validation set. You can modify `train.py` to customize hyperparameters, batch sizes, or evaluation metrics.

## 💡 Usage
To summarize new dialogues using the fine-tuned model:

```python
def summarize_text(dialogue):
    inputs = tokenizer(dialogue, return_tensors="pt", truncation=True, max_length=1024)
    summary_ids = model.generate(inputs["input_ids"], max_length=128, num_beams=5, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

sample_dialogue = "Hey, how's your project going?"  
summary = summarize_text(sample_dialogue)
print(summary)
```

## 📜 License
This project is released under the MIT License.

## 🤝 Contributing
Feel free to open issues or submit pull requests if you find any improvements!

## ⭐ Acknowledgments
- [Hugging Face](https://huggingface.co/) for providing the `transformers` and `datasets` libraries.
- Google Research for the PEGASUS model.
- The creators of the SAMSum dataset.

