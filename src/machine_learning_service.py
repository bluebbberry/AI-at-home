import logging
from transformers import BertForQuestionAnswering, BertTokenizerFast, Trainer, TrainingArguments
import torch

# Initialize logging
logging.basicConfig(level=logging.INFO)


class QuestionAnsweringService:
    def __init__(self, model_name='bert-base-uncased', num_epochs=3, batch_size=2, lr=3e-5):
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr

        # Load model and tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model = BertForQuestionAnswering.from_pretrained(model_name)

    def preprocess_data(self):
        # Hard-coded simple dataset
        dataset = {
            'train': [
                {
                    'context': "The capital of France is Paris. It is located in Europe.",
                    'question': "What is the capital of France?",
                    'answers': {'answer_start': [30], 'text': ['Paris']}
                },
                {
                    'context': "The Eiffel Tower is located in Paris, France.",
                    'question': "Where is the Eiffel Tower?",
                    'answers': {'answer_start': [28], 'text': ['Paris']}
                }
            ],
            'validation': [
                {
                    'context': "The Great Wall of China is located in China. It is a historic monument.",
                    'question': "Where is the Great Wall of China?",
                    'answers': {'answer_start': [30], 'text': ['China']}
                }
            ]
        }

        # Function to preprocess and tokenize data
        def preprocess_function(examples):
            tokenized_examples = self.tokenizer(
                examples['question'], examples['context'], truncation=True, padding=True, max_length=512
            )

            # Add start and end positions for each answer
            start_positions = []
            end_positions = []
            for i in range(len(examples['answers']['answer_start'])):
                start_positions.append(examples['answers']['answer_start'][i])
                end_positions.append(examples['answers']['answer_start'][i] + len(examples['answers']['text'][i]) - 1)

            tokenized_examples['start_positions'] = start_positions
            tokenized_examples['end_positions'] = end_positions
            return tokenized_examples

        # Tokenize and preprocess the training and validation datasets
        train_dataset = self.tokenize_data(dataset['train'], preprocess_function)
        val_dataset = self.tokenize_data(dataset['validation'], preprocess_function)

        return train_dataset, val_dataset

    def tokenize_data(self, data, preprocess_function):
        tokenized_data = [preprocess_function(example) for example in data]
        return tokenized_data

    def train_model(self):
        train_dataset, val_dataset = self.preprocess_data()

        # Set up training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=1e-5,  # Lower learning rate
            per_device_train_batch_size=8,  # Lower batch size if needed
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            save_total_limit=2,
            save_strategy="epoch",
            logging_dir="./logs",
            logging_steps=10,
            max_grad_norm=1.0,  # Gradient clipping to avoid explosion
        )

        # Initialize the Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer
        )

        # Start training
        trainer.train()

    def answer_question(self, question, context):
        # Tokenize question and context
        inputs = self.tokenizer(question, context, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract answer span
        start_index = torch.argmax(outputs.start_logits)
        end_index = torch.argmax(outputs.end_logits)

        # Get the answer tokens
        answer_tokens = inputs['input_ids'][0][start_index:end_index + 1]

        # Decode and return the answer
        return self.tokenizer.decode(answer_tokens, skip_special_tokens=True)


# Example Usage
if __name__ == "__main__":
    qa_service = SimpleQAService()

    # Train the model
    qa_service.train_model()

    # Test with a sample question and context
    question = "What is the capital of France?"
    context = "The capital of France is Paris. It is located in Europe."
    answer = qa_service.answer_question(question, context)
    print(f"Answer: {answer}")
