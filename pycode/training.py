from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from config import MODEL_OUTPUT_DIR, CONTROL_TOKENS, MODE
from dataset_creation import creat_dataset_dict,preprocess_function
from tokenization import load_tokenizer_and_model, update_tokenizer

def train_model():
    tokenizer, model = load_tokenizer_and_model()
    update_tokenizer(tokenizer, model, CONTROL_TOKENS, MODE)

    dataset = creat_dataset_dict()
    tokenized_dataset = dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_strategy="epoch",
        num_train_epochs=5,
        fp16=False,
        load_best_model_at_end=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['valid'],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.evaluate(tokenized_dataset['valid'])
    model.save_pretrained(MODEL_OUTPUT_DIR)