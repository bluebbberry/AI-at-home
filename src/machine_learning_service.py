import logging
from transformers import BertForQuestionAnswering, BertTokenizerFast, Trainer, TrainingArguments
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader


class QuestionAnsweringService:
    def __init__(self, model_name='bert-base-uncased', num_epochs=3, batch_size=8, lr=3e-5):
        """
        Initialize the QuestionAnsweringService for training a BERT model.

        Args:
            model_name (str): Name of the pre-trained BERT model to use.
            num_epochs (int): Number of epochs for fine-tuning.
            batch_size (int): Batch size for training and evaluation.
            lr (float): Learning rate for the optimizer.
        """
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr

        # Load the pre-trained BERT model and tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model = BertForQuestionAnswering.from_pretrained(model_name)

    def preprocess_data(self, dataset_name='squad'):
        """
        Preprocess the dataset for training the BERT model.

        Args:
            dataset_name (str): Name of the Hugging Face dataset to use (default: 'squad').

        Returns:
            train_dataset, val_dataset: Tokenized and preprocessed training and validation datasets.
        """
        logging.info("Loading and preprocessing dataset...")

        # Load the dataset
        dataset = load_dataset(dataset_name)

        def tokenize_function(example):
            return self.tokenizer(
                example['question'],
                example['context'],
                truncation=True,
                padding='max_length',
                max_length=384
            )

        def preprocess_training_data(examples):
            tokenized_examples = tokenize_function(examples)
            start_positions = []
            end_positions = []

            for i, offsets in enumerate(tokenized_examples['offset_mapping']):
                input_ids = tokenized_examples['input_ids'][i]
                cls_index = input_ids.index(self.tokenizer.cls_token_id)

                # Find start and end token positions
                start_char = examples['answers'][i]['answer_start'][0]
                end_char = start_char + len(examples['answers'][i]['text'][0])

                token_start_index = 0
                while offsets[token_start_index] is None or offsets[token_start_index][0] <= start_char:
                    token_start_index += 1

                token_end_index = len(offsets) - 1
                while offsets[token_end_index] is None or offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1

                start_positions.append(token_start_index)
                end_positions.append(token_end_index)

            tokenized_examples['start_positions'] = start_positions
            tokenized_examples['end_positions'] = end_positions

            return tokenized_examples

        # Process the dataset
        train_dataset = dataset['train'].map(preprocess_training_data, batched=True)
        val_dataset = dataset['validation'].map(preprocess_training_data, batched=True)

        train_dataset.set_format(type='torch',
                                 columns=['input_ids', 'attention_mask', 'start_positions', 'end_positions'])
        val_dataset.set_format(type='torch',
                               columns=['input_ids', 'attention_mask', 'start_positions', 'end_positions'])

        return train_dataset, val_dataset

    def train_model(self):
        """
        Train the BERT model for question answering.
        """
        train_dataset, val_dataset = self.preprocess_data()

        logging.info("Setting up training arguments...")
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=self.lr,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            weight_decay=0.01,
            save_total_limit=2,
            save_strategy="epoch",
            logging_dir="./logs",
            logging_steps=10
        )

        logging.info("Initializing Trainer...")
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer
        )

        logging.info("Starting training...")
        trainer.train()

    def answer_question(self, question, context):
        """
        Use the fine-tuned BERT model to answer a question based on the given context.

        Args:
            question (str): The question to answer.
            context (str): The context providing the answer.

        Returns:
            answer (str): The predicted answer from the model.
        """
        inputs = self.tokenizer(question, context, return_tensors="pt", truncation=True, padding=True, max_length=384)

        with torch.no_grad():
            outputs = self.model(**inputs)

        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        start_index = torch.argmax(start_logits)
        end_index = torch.argmax(end_logits)

        input_ids = inputs['input_ids'][0]
        answer_tokens = input_ids[start_index: end_index + 1]

        return self.tokenizer.decode(answer_tokens, skip_special_tokens=True)


# Example Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    qa_service = QuestionAnsweringService()

    # Train the model
    qa_service.train_model()

    # Test the model with a simple question
    question = "What is the capital of France?"
    context = "France is a country in Europe. Its capital is Paris."
    answer = qa_service.answer_question(question, context)
    print(f"Answer: {answer}")
