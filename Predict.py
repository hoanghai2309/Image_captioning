from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import os

saved_model="model"
model = VisionEncoderDecoderModel.from_pretrained(saved_model, local_files_only=True)
feature_extractor = ViTImageProcessor.from_pretrained(saved_model, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(saved_model, local_files_only=True)


max_length = 16
num_beams = 4
no_repeat_ngram_size=2
gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "no_repeat_ngram_size": no_repeat_ngram_size}


def predict_step(directory_path):
  image_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path)]
  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds


print(predict_step("image"))
import evaluate
metric = evaluate.load("rouge")