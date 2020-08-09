import os
import time
import cv2
from PIL import Image
import pdf2image
import numpy as np
import Hierarchy
import tesserocr
import random
import glob
import concurrent.futures
import os


def func_time(func):
    def timer(*args, **kwargs):
        init_time = time.time()
        func_return_val = func(*args, **kwargs)
        print("Time taken for " + func.__name__ + ": {}".format(time.time() - init_time))

        return func_return_val

    return timer


def is_level(box1, box2):
    return True if ((box2[1] - 2 <= centre(box1)[1] <= box2[3] + box2[1] + 2) or (
            box1[1] - 2 <= centre(box2)[1] <= box1[3] + box1[1] + 2)) else False


def centre(box):
    x = box[0] + (box[2]) // 2
    y = box[1] + (box[3]) // 2

    return x, y


def merge_box(box1, box2):
    min_x = min(box1[0], box2[0])
    min_y = min(box1[1], box2[1])
    max_w = max(box1[0] + box1[2], box2[0] + box2[2]) - min_x
    max_h = max(box1[1] + box1[3], box2[1] + box2[3]) - min_y

    merged_box = [min_x, min_y, max_w, max_h]

    return merged_box


def get_line(boxes):
    line_dict = {}

    is_done = [-1] * len(boxes)
    last_num = 0

    for i in range(len(boxes)):

        if is_done[i] == -1:
            last_num += 1
            curr_num = last_num
            is_done[i] = curr_num
            line_dict[curr_num] = [boxes[i]]

        else:
            curr_num = is_done[i]

        for j in range(i + 1, len(boxes)):
            if is_done[j] == -1:
                if is_level(boxes[i], boxes[j]):
                    is_done[j] = is_done[i]
                    line_dict[curr_num].append(boxes[j])

                else:
                    break

    return line_dict


def get_merged_boundingboxes(boxes):
    merged_boxes = []

    image_lines = get_line(boxes)

    for line in image_lines.values():

        line.sort()
        new_line = []

        is_merged = [False] * len(line)

        for i in range(len(line)):
            merged_boundingbox = line[i]

            if not is_merged[i]:
                is_merged[i] = True

                for j in range(i + 1, len(line)):
                    if line[j][0] < merged_boundingbox[0] + merged_boundingbox[2] and is_level(merged_boundingbox,
                                                                                               line[j]) and not \
                            is_merged[j]:
                        is_merged[j] = True
                        merged_boundingbox = merge_box(merged_boundingbox, line[j])

                    elif not line[j][0] < merged_boundingbox[0] + merged_boundingbox[2]:
                        break

                new_line.append(merged_boundingbox)

        merged_boxes.append(new_line)

    merged_boxes.sort(key=lambda x: x[0][1])

    return merged_boxes


def filter_boxes(box, api):
    api.SetRectangle(box[0], box[1], box[2], box[3])
    text = api.GetUTF8Text()

    return (api.MeanTextConf() >= 30 and text and not text.isspace()), text


class ImageTextRecog:

    def __init__(self, image, pdf_name):

        self.original_image = image
        self.pdf_name = pdf_name.split(".")[0]

    @staticmethod
    def split_text_box(bounding_box, text):

        lines = list(filter(lambda x: x != "" and not x.isspace(), text.split("\n")))
        num_lines = len(lines)
        print(lines, text)

        line_height = bounding_box[3] // num_lines

        line_sep_boxes = [
            [[bounding_box[0], bounding_box[1] + (i * line_height), bounding_box[2], line_height], text_line] for
            i, text_line in enumerate(lines)]

        return line_sep_boxes

    def detect_sparse_words(self):

        image = self.original_image.copy()
        image = Hierarchy.mask_lines(image, 80)

        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        sparse_boundingboxes = []

        with tesserocr.PyTessBaseAPI() as api:
            api.SetPageSegMode(tesserocr.PSM.SPARSE_TEXT)

            api.SetImage(pil_image)
            box_data = api.GetComponentImages(tesserocr.RIL.WORD, True)

            for (_, box, _, _) in box_data:
                sparse_boundingboxes.append([box['x'], box['y'], box['w'], box['h']])

        return sparse_boundingboxes

    def get_text_boundingboxes(self):

        image = self.original_image.copy()
        pil_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        sparse_word_boxes = self.detect_sparse_words()
        text_boxes = []

        with tesserocr.PyTessBaseAPI() as api:
            api.SetPageSegMode(tesserocr.PSM.SINGLE_BLOCK)

            api.SetImage(Image.fromarray(pil_image))

            for box in sparse_word_boxes:
                is_text, text = filter_boxes([box[0] - 5, box[1] - 5, box[2] + 5, box[3] + 5], api)

                if is_text:
                    text_boxes.append(box)

        return text_boxes

    @func_time
    def get_image_text(self):

        image = self.original_image.copy()
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        _, boxes = Hierarchy.get_contour_bounding_boxes(image, get_vertices=False)

        boxes.extend(self.get_text_boundingboxes())

        boxes = list(
            filter(lambda x: x[3] + x[1] < image.shape[0] and x[2] + x[0] < image.shape[1] and x[3] <= 100, boxes))
        boxes.sort(key=lambda x: x[1])

        merged_text_boxes = get_merged_boundingboxes(boxes)

        with tesserocr.PyTessBaseAPI() as api:
            api.SetPageSegMode(tesserocr.PSM.SINGLE_BLOCK)

            api.SetImage(pil_image)

            for i, line in enumerate(merged_text_boxes):

                filtered_line = []
                for box in line:
                    is_text, text = filter_boxes(box, api)

                    if is_text:
                        splitted_lines = self.split_text_box(box, text.strip())

                        for split_line in splitted_lines:
                            filtered_line.append(split_line)

                merged_text_boxes[i] = filtered_line

        for val in merged_text_boxes:
            col = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for box in val:
                box = box[0]
                cv2.rectangle(image, (box[0], box[1]), (box[2] + box[0], box[1] + box[3]), col, 2)

        out_dir = "Output/{}".format(self.pdf_name)
        cv2.imwrite(out_dir + "/1.jpg", image)

        return merged_text_boxes


def load_image(pdf_path):
    pages = pdf2image.convert_from_path(pdf_path)
    out_dir = "Output/{}".format(pdf_path.split("/")[1].split(".")[0])
    os.mkdir(out_dir)

    for i, page in enumerate(pages):
        image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

        print(ImageTextRecog(image, pdf_path.split("/")[1]).get_image_text())


if __name__ == "__main__":

    random.seed()
    # os.environ["OMP_THREAD_LIMIT"] = '1'

    pdf_dir = "GRiD_Sample Invoices II/"
    pdf_list = glob.glob("GRiD_Sample Invoices II/*.pdf")

    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        init_time = time.time()
        for pdf_file, _ in zip(pdf_list, executor.map(load_image, pdf_list)):
            print(pdf_file)

        print(time.time() - init_time)
    # init_time = time.time()
    # for pdf in pdf_list:
    #     print(pdf)
    #     load_image(pdf)
    # print(time.time() - init_time)
