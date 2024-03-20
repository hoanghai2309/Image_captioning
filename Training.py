import os
import datasets
from transformers import VisionEncoderDecoderModel, ViTImageProcessor,AutoTokenizer
os.environ["WANDB_DISABLED"] = "true"
import nltk
try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    nltk.download("punkt", quiet=True)

image_encoder_model = "google/vit-base-patch16-224-in21k"
text_decode_model = "gpt2"
#directory of pretrained image_encoder_model (VIT), text_decode_model (GPT2)
output_dir = "vit-gpt-model"
model = VisionEncoderDecoderModel.from_pretrained(output_dir)
# image feature extractor
feature_extractor = ViTImageProcessor.from_pretrained(output_dir)
# text tokenizer
tokenizer = AutoTokenizer.from_pretrained(output_dir)
# GPT2 only has bos/eos tokens but not decoder_start/pad tokens
tokenizer.pad_token = tokenizer.eos_token
# update the model config
model.config.eos_token_id = tokenizer.eos_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

ds = datasets.load_dataset("coco_dataset_script.py", "2017", data_dir="dataset")

import random
from datasets import DatasetDict
# create sample dataset
minids = DatasetDict()
minids['train'] = ds['train'].select(indices=random.sample(range(len(ds['train'])), 80))
minids['validation'] = ds['validation'].select(indices=random.sample(range(len(ds['validation'])), 80))
minids['test'] = ds['test'].select(indices=random.sample(range(len(ds['test'])), 16))
print(type(minids))


from PIL import Image
# text preprocessing step
def tokenization_fn(captions, max_target_length):
    """Run tokenization on captions."""
    labels = tokenizer(captions,
                       padding="max_length",
                       max_length=max_target_length).input_ids

    return labels
# image preprocessing step
def feature_extraction_fn(image_paths, check_image=True):
    """
    Run feature extraction on images
    If `check_image` is `True`, the examples that fails during `Image.open()` will be caught and discarded.
    Otherwise, an exception will be thrown.
    """
    model_inputs = {}
    if check_image:
        images = []
        to_keep = []
        for image_file in image_paths:
            try:
                img = Image.open(image_file)
                if img.mode != "RGB":
                    img = img.convert(mode="RGB")
                images.append(img)
                to_keep.append(True)
            except Exception:
                to_keep.append(False)
    else:
        images = [Image.open(image_file) for image_file in image_paths]

    encoder_inputs = feature_extractor(images=images, return_tensors="np")

    return encoder_inputs.pixel_values


def preprocess_fn(examples, max_target_length, check_image=True):
    """Run tokenization + image feature extraction"""
    image_paths = examples['image_path']
    captions = examples['caption']

    model_inputs = {}
    # This contains image path column
    model_inputs['labels'] = tokenization_fn(captions, max_target_length)
    model_inputs['pixel_values'] = feature_extraction_fn(image_paths, check_image=check_image)

    return model_inputs

processed_dataset = minids.map(
    function=preprocess_fn,
    batched=True,
    fn_kwargs={"max_target_length": 128},
    remove_columns=minids['train'].column_names
)


from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels
import evaluate
metric = evaluate.load("rouge")
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds,
                                                     decoded_labels)

    result = metric.compute(predictions=decoded_preds,
                            references=decoded_labels,
                            use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    return result


from transformers import default_data_collator
import numpy as np
training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    output_dir="./image-captioning-output",
)
ignore_pad_token_for_loss = True
# instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=feature_extractor,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=processed_dataset['train'],
    eval_dataset=processed_dataset['validation'],
    data_collator=default_data_collator,
)
trainer.train()
trainer.save_model("demo")
tokenizer.save_pretrained("demo")
