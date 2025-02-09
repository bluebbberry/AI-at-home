import logging
import torch
import torch.nn.utils.prune as prune
import torch.jit
import base64
import io
import gzip
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast, TrainingArguments, Trainer, \
    DefaultDataCollator
from datasets import Dataset

# Initialize logging
logging.basicConfig(level=logging.INFO)


class QuestionAnsweringService:
    def __init__(self, model_name='distilbert-base-uncased', num_epochs=3, batch_size=2, lr=3e-5):
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        self.model = DistilBertForQuestionAnswering.from_pretrained(model_name)

    def preprocess_data(self):
        dataset = {
            'train': [
                {'context': "The capital of France is Paris.", 'question': "What is the capital of France?",
                 'answers': {'answer_start': [24], 'text': ['Paris']}}
            ],
            'validation': [
                {'context': "The Great Wall of China is in China.", 'question': "Where is the Great Wall of China?",
                 'answers': {'answer_start': [27], 'text': ['China']}}
            ]
        }

        def preprocess_function(example):
            inputs = self.tokenizer(example['question'], example['context'], truncation=True, padding='max_length',
                                    max_length=512, return_offsets_mapping=True)

            # Find the start and end character positions in the tokenized text
            offset_mapping = inputs.pop("offset_mapping")

            answer_start_char = example['answers']['answer_start'][0]
            answer_end_char = answer_start_char + len(example['answers']['text'][0])

            # Convert character positions to token indices
            start_positions = end_positions = None
            for idx, (start, end) in enumerate(offset_mapping):
                if start_positions is None and start <= answer_start_char < end:
                    start_positions = idx
                if end_positions is None and start < answer_end_char <= end:
                    end_positions = idx

            if start_positions is None or end_positions is None:
                start_positions, end_positions = 0, 0  # Default to first token if misalignment occurs

            inputs["start_positions"] = start_positions
            inputs["end_positions"] = end_positions

            return inputs

        train_dataset = Dataset.from_list([preprocess_function(example) for example in dataset['train']])
        val_dataset = Dataset.from_list([preprocess_function(example) for example in dataset['validation']])

        return train_dataset, val_dataset

    def train_model(self):
        train_dataset, val_dataset = self.preprocess_data()
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            logging_dir="./logs",
            save_strategy="no"  # Disable auto-saving to avoid quantization issues
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=DefaultDataCollator()
        )
        trainer.train()

    def answer_question(self, question, context):
        inputs = self.tokenizer(question, context, return_tensors="pt", truncation=True, padding=True, max_length=512)

        with torch.no_grad():
            outputs = self.model(**inputs)

        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        start_index = torch.argmax(start_scores, dim=1).item()
        end_index = torch.argmax(end_scores, dim=1).item()

        if start_index >= end_index or start_index < 0 or end_index >= inputs["input_ids"].size(1):
            return "Unable to determine answer."

        # Convert token indices to actual words
        answer_tokens = inputs["input_ids"][0][start_index:end_index + 1]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)

        # Print debug information
        print(f"Question: {question}")
        print(f"Context: {context}")
        print(f"Predicted Start Index: {start_index}, End Index: {end_index}")
        print(f"Predicted Answer: {answer}")

        return answer

    def encode_model(self, model):
        # Apply quantization before saving
        quantized_model = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)

        buffer = io.BytesIO()
        example_input = torch.randint(0, 30522, (1, 512))  # Example input tensor
        traced_model = torch.jit.trace(quantized_model, (example_input,), strict=False)
        torch.jit.save(traced_model, buffer)
        buffer.seek(0)
        compressed_model = gzip.compress(buffer.getvalue())
        b64_model = base64.b64encode(compressed_model).decode('utf-8')
        print("Model saved as compressed Base64.")
        return b64_model


if __name__ == "__main__":
    qa_service = QuestionAnsweringService()
    qa_service.train_model()
    question = "What is the capital of France?"
    context = "The capital of France is Paris."
    answer = qa_service.answer_question(question, context)
    print(f"Answer: {answer}")
