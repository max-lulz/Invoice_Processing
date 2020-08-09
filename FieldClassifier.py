import os

import cv2
import nltk
import numpy as np
import pandas as pd
import pdf2image
from fastai.text import DatasetType
from fastai.text import load_learner
from nltk.corpus import stopwords

from TextRecog import ImageTextRecog

nltk.data.path.append('./')
if not os.path.isdir("corpora"):
    nltk.download("stopwords", download_dir="./")
stop_words = stopwords.words("english")

field_model_path = "./models/field_classifier/"
field_model_name = "export.pkl"


def preprocess_text(text_df):
    text_df["text"] = text_df["text"].fillna(" ")
    text_df["text"] = text_df["text"].str.replace(r"[^a-zA-Z\d]", " ")

    tokenized_df = text_df["text"].apply(lambda x: x.split())
    tokenized_df = tokenized_df.apply(lambda x: [item for item in x if item not in stop_words])

    detokenized_df = []

    for tokens in tokenized_df:
        detokenized_df.append(' '.join(tokens))

    text_df["text"] = detokenized_df

    return text_df


def get_classifier(model_path, model_name):
    classifier = load_learner(path=model_path, file=model_name)

    return classifier


def classify_text(bounding_boxes, model_path=field_model_path, model_name=field_model_name):
    text_list = [box[1] for line in bounding_boxes for box in line]

    text_df = pd.DataFrame({"text": text_list})
    text_df = preprocess_text(text_df)

    text_classifier = get_classifier(model_path=model_path, model_name=model_name)
    text_classifier.data.add_test(text_df["text"])

    probs, labels = text_classifier.get_preds(ds_type=DatasetType.Test)
    predictions = probs.argmax(axis=1).numpy()

    pred_boxes = []

    num_elems = 0
    for line in bounding_boxes:
        line_length = len(line)
        pred_line = []

        for j, box in enumerate(line):
            new_box = box.copy()
            new_box.append((predictions[num_elems + j] == 0))
            pred_line.append(new_box)

        pred_boxes.append(pred_line)
        num_elems += line_length

    return pred_boxes


if __name__ == "__main__":
    pages = pdf2image.convert_from_path("GRiD_Sample Invoices II/Sample1.pdf")
    for i, page in enumerate(pages):
        image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

        print(classify_text(ImageTextRecog(image, "").get_image_text()))
