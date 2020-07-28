import imutils
import pytesseract
import Hierarchy
import cv2
import random
import time
import tesserocr
from PIL import Image
import numpy as np


def detect_text(image_region):
    pytesseract.image_to_data(image_region, output_type="dict")


def get_tesseract_bbox(image_path):
    random.seed()
    im = cv2.imread(image_path)
    out = pytesseract.image_to_data(im, output_type="dict")

    level_col = {}

    for i, text in enumerate(out['text']):
        if int(out["conf"][i]) >= 30 and text and not text.isspace():
            print(out["line_num"][i])
            if out["line_num"][i] not in level_col:
                level_col[out["line_num"][i]] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            cv2.rectangle(im, (out["left"][i], out["top"][i]),
                          (out["left"][i] + out["width"][i], out["top"][i] + out["height"][i]),
                          color=level_col[out["line_num"][i]], thickness=2)
            # print(text)

    cv2.imwrite("outTess3.jpg", im)


def get_merged_hierarchical_bbox(image_path):
    lev = Hierarchy.get_hierarchy_levels(image_path)

    for key in lev.keys():
        for val in lev[key]:
            val[2] -= val[0] - 15
            val[3] -= val[1]
            val[0] -= 15

    new_levels = []

    for val in lev.values():
        new_levels.append(
            list(filter(lambda x: x[3] + x[1] < image.shape[0] and x[2] + x[0] < image.shape[0], sorted(val))))

    new_levels = sorted(new_levels, key=lambda x: x[0][1])

    merged_levels = []

    for val in new_levels:
        new_val = []
        merged_bbox = val[0]
        for i in range(1, len(val)):
            if val[i][0] < val[i - 1][0] + val[i - 1][2]:
                merged_bbox[1] = min(merged_bbox[1], val[i - 1][1], val[i][1])
                merged_bbox[2] = max(merged_bbox[2], val[i][0] + val[i][2] - merged_bbox[0])
                merged_bbox[3] = max(merged_bbox[3], val[i - 1][1] + val[i - 1][3] - merged_bbox[1],
                                     val[i][1] + val[i][3] - merged_bbox[1])

            else:
                new_val.append(merged_bbox)
                merged_bbox = val[i]

        if new_val[-1:] != merged_bbox:
            new_val.append(merged_bbox)

        merged_levels.append(new_val)

    for stuff in new_levels:

        for box in stuff:
            cv2.rectangle(image, (box[0], box[1]), (box[2] + box[0], box[1] + box[3]), (0, 0, 255), 2)


if __name__ == "__main__":
    im_path = "Image_Out/image00.jpg"
    image = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    for i, val in enumerate(new_levels):
        new_levels[i] += val
        # print()
        recs, weights = cv2.groupRectangles(new_levels[i] + val, 2, eps=0.8)

        for box in recs:
            # print(box)
            cv2.rectangle(image, (box[0] - 5, box[1]), (box[2] + box[0] + 5, box[1] + box[3]), (0, 0, 255), 2)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)

    with tesserocr.PyTessBaseAPI() as api:
        api.SetPageSegMode(tesserocr.PSM.AUTO_OSD)

        api.SetImage(pil_image)

        boxes = api.GetComponentImages(tesserocr.RIL.WORD, True)

        for _, box, _, _ in boxes:
            x, y, w, h = box['x'], box['y'], box['w'], box['h']
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # for val in new_lev:
        #     row = []
        #     for rec in val:
        #         if rec[3] < image.shape[0]:
        #             api.SetImage(Image.fromarray(image[rec[1]-5:rec[3]+5, rec[0]-5:rec[2]+5]))
        #
        #             boxes = api.GetComponentImages(tesserocr.RIL.WORD, True)
        #
        #             text = api.GetUTF8Text().strip()
        #             row.append(text)
        #
        #             # Hierarchy.debug_image(image[rec[1]-5:rec[3]+5, rec[0]-5:rec[2]+5])
        #     print(*row)
        #     print("\n")

    Hierarchy.debug_image(imutils.resize(image, width=800))
