import os
import random
import time
import cv2
import imutils
import pytesseract
from PIL import Image
import Hierarchy
import tesserocr
import numpy as np


def merge_bboxes(boxes):
    merged_boxes = []
    level_dict = {}

    is_done = [-1] * len(boxes)
    last_num = 0

    for i in range(len(boxes)):

        if is_done[i] == -1:
            last_num += 1
            curr_num = last_num
            is_done[i] = curr_num
            level_dict[curr_num] = [boxes[i]]

        else:
            curr_num = is_done[i]

        for j in range(i + 1, len(boxes)):
            if is_done[j] == -1:
                if is_level(boxes[i], boxes[j]):
                    is_done[j] = is_done[i]
                    level_dict[curr_num].append(boxes[j])

                else:
                    break

    for val in level_dict.values():
        new_val = []
        val.sort()

        merged_bbox = val[0]
        for i in range(1, len(val)):
            if val[i][0] < merged_bbox[0] + merged_bbox[2]:
                # merged_bbox[0] = min(merged_bbox[0], val[i - 1][0], val[i][0])
                merged_bbox[1] = min(merged_bbox[1], val[i - 1][1], val[i][1])
                merged_bbox[2] = max(merged_bbox[2], val[i - 1][0] + val[i - 1][2] - merged_bbox[0],
                                     val[i][0] + val[i][2] - merged_bbox[0])
                merged_bbox[3] = max(merged_bbox[3], val[i - 1][1] + val[i - 1][3] - merged_bbox[1],
                                     val[i][1] + val[i][3] - merged_bbox[1])

            else:
                new_val.append(merged_bbox)
                merged_bbox = val[i]

        if new_val[-1:] != merged_bbox:
            new_val.append(merged_bbox)

        merged_boxes.append(new_val)

    merged_boxes.sort(key=lambda x: x[0][1])

    return merged_boxes


def tess_process(image):
    orig_image = image.copy()
    pil_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hor_kernel = np.zeros((1, 80), dtype=np.uint8)  # square kernel with middle row set to 1
    hor_kernel[:, :] = 1  # can be changed to a single row kernel

    ver_kernel = np.zeros((80, 1), dtype=np.uint8)
    ver_kernel[:, :] = 1

    hor_mask = cv2.morphologyEx(~image, cv2.MORPH_OPEN, hor_kernel, iterations=1)  # make masks that contain
    ver_mask = cv2.morphologyEx(~image, cv2.MORPH_OPEN, ver_kernel, iterations=1)  # horizontal and vertical lines

    _, hor_mask = cv2.threshold(hor_mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, ver_mask = cv2.threshold(ver_mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    image += hor_mask  # removes horizontal and vertical lines
    image += ver_mask  # improves text blob detection

    boxes = get_tesseract_bbox(image)

    new_boxes = []

    with tesserocr.PyTessBaseAPI() as api:
        api.SetPageSegMode(tesserocr.PSM.SINGLE_BLOCK)

        api.SetImage(Image.fromarray(pil_image))

        total = len(boxes)
        found = 0

        for box in boxes:
            is_text, text = filter_boxes([box[0] - 5, box[1] - 5, box[2] + 5, box[3] + 5], api)

            if is_text:
                # found += 1
                new_boxes.append(box)
                # cv2.rectangle(orig_image, (box[0], box[1]), (box[2] + box[0], box[1] + box[3]), (0, 255, 0), 1)

    # print("Image{}, Found: {}, Total: {}".format(count, found, total))
    # cv2.imwrite("Output/Image{}.jpg".format(count), orig_image)

    # return image
    return new_boxes


def func_time(func):
    def timer(*args, **kwargs):
        init_time = time.time()
        func(*args, **kwargs)
        print("Time taken for " + func.__name__ + ": {}".format(time.time() - init_time))

    return timer


def filter_boxes(box, api):
    api.SetRectangle(box[0], box[1], box[2], box[3])
    text = api.GetUTF8Text()

    return (api.MeanTextConf() >= 30 and text and not text.isspace()), text


def is_level(box1, box2):
    return True if ((box2[1] - 2 <= centre(box1)[1] <= box2[3] + box2[1] + 2) or (
            box1[1] - 2 <= centre(box2)[1] <= box1[3] + box1[1] + 2)) else False


def centre(box):
    x = box[0] + (box[2]) // 2
    y = box[1] + (box[3]) // 2

    return x, y


def detect_text(image_region):
    pytesseract.image_to_data(image_region, output_type="dict")


def get_tesseract_bbox(image):
    im = image.copy()
    image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)

    tess_bbox = []

    with tesserocr.PyTessBaseAPI() as api:
        api.SetPageSegMode(tesserocr.PSM.SPARSE_TEXT)

        api.SetImage(pil_image)
        box_data = api.GetComponentImages(tesserocr.RIL.WORD, True)

        for (_, box, _, _) in box_data:
            tess_bbox.append([box['x'], box['y'], box['w'], box['h']])

    return tess_bbox


@func_time
def test_bbox(image_path):
    image = cv2.imread(image_path)
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pre_image = Hierarchy.preprocess(image_path)
    _, boxes = Hierarchy.get_bounding_boxes(pre_image, False)

    boxes += tess_process(image)

    boxes = list(filter(lambda x: x[3] + x[1] < image.shape[0] and x[2] + x[0] < image.shape[1] and x[3] <= 100, boxes))

    # for i,_ in enumerate(boxes):
    #     boxes[i][0] -= 5
    #     boxes[i][2] += 5

    boxes.sort(key=lambda x: x[1])

    image_lines = {}
    line_num = 0
    word_line = boxes[0:1]

    for i in range(1, len(boxes)):
        if is_level(boxes[i], boxes[i - 1]):
            word_line.append(boxes[i])
        else:
            image_lines[line_num] = word_line
            line_num += 1
            word_line = [boxes[i]]

    # del boxes

    image_lines[line_num] = word_line

    for key in image_lines.keys():
        image_lines[key].sort()

    # for key, val in image_lines.items():
        # val = val2
        # for _ in range(100):
        #     new_val = []
        #     merged_bbox = val[0]
        #     for i in range(1, len(val)):
        #         if val[i][0] <= val[i - 1][0] + val[i - 1][2] and is_level(val[i], merged_bbox):
        #             merged_bbox[0] = min(merged_bbox[0], val[i - 1][0], val[i][0])
        #             # merged_bbox[1] = min(merged_bbox[1], val[i - 1][1], val[i][1])
        #             merged_bbox[2] = max(merged_bbox[2], val[i - 1][0] + val[i - 1][2] - merged_bbox[0],
        #                                  val[i][0] + val[i][2] - merged_bbox[0])
        #             # merged_bbox[3] = max(merged_bbox[3], val[i - 1][1] + val[i - 1][3] - merged_bbox[1],
        #             #                      val[i][1] + val[i][3] - merged_bbox[1])
        #
        #         else:
        #             new_val.append(merged_bbox)
        #             merged_bbox = val[i]
        #
        #     if new_val[-1:] != merged_bbox:
        #         new_val.append(merged_bbox)
        #
        #     val = new_val
        # new_val = []
        # merged_bbox = val[0]
        # for i in range(1, len(val)):
        #     if val[i][0] < merged_bbox[0] + merged_bbox[2]:
        #         merged_bbox[0] = min(merged_bbox[0], val[i - 1][0], val[i][0])
        #         # merged_bbox[1] = min(merged_bbox[1], val[i - 1][1], val[i][1])
        #         merged_bbox[2] = max(merged_bbox[2], val[i - 1][0] + val[i - 1][2] - merged_bbox[0],
        #                              val[i][0] + val[i][2] - merged_bbox[0])
        #         # merged_bbox[3] = max(merged_bbox[3], val[i - 1][1] + val[i - 1][3] - merged_bbox[1],
        #         #                      val[i][1] + val[i][3] - merged_bbox[1])
        #
        #     else:
        #         new_val.append(merged_bbox)
        #         merged_bbox = val[i]
        #
        # if new_val[-1:] != merged_bbox:
        #     new_val.append(merged_bbox)

        # image_merged_boxes[key] = val

    image_merged_boxes = merge_bboxes(boxes)

    del image_lines

    with tesserocr.PyTessBaseAPI() as api:  # can be added to merge loop above
        api.SetPageSegMode(tesserocr.PSM.SINGLE_BLOCK)

        api.SetImage(pil_image)

        total = 0
        found = 0

        for i, line in enumerate(image_merged_boxes):
            filtered_line = []
            for box in line:
                is_text, text = filter_boxes(box, api)
                total += 1

                if is_text:
                    found += 1
                    filtered_line.append([box, text.strip()])

            image_merged_boxes[key] = filtered_line

    for val in image_merged_boxes.values():
        col = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for box in val:
            box = box[0]
            cv2.rectangle(image, (box[0], box[1]), (box[2] + box[0], box[1] + box[3]), col, 2)

    cv2.imwrite("Output/image{}.jpg".format(count), image)


def get_merged_hierarchical_bbox(image_path):
    lev = Hierarchy.get_hierarchy_levels(image_path)
    image = cv2.imread(image_path)

    for key in lev.keys():
        for val in lev[key]:
            val[2] -= val[0] - 5
            val[3] -= val[1]
            val[0] -= 5

    new_levels = []

    for val in lev.values():
        new_val = list(filter(lambda x: x[3] + x[1] < image.shape[0] and x[2] + x[0] < image.shape[1], sorted(val)))

        if len(new_val) != 0:
            new_levels.append(new_val)

    print(new_levels)

    new_levels = sorted(new_levels, key=lambda x: x[0][1])

    merged_levels = []

    for val in new_levels:
        new_val = []
        merged_bbox = val[0]
        for i in range(1, len(val)):
            if val[i][0] < val[i - 1][0] + val[i - 1][2]:
                merged_bbox[1] = min(merged_bbox[1], val[i - 1][1], val[i][1])
                merged_bbox[2] = max(merged_bbox[2], val[i - 1][0] + val[i - 1][2] - merged_bbox[0],
                                     val[i][0] + val[i][2] - merged_bbox[0])
                merged_bbox[3] = max(merged_bbox[3], val[i - 1][1] + val[i - 1][3] - merged_bbox[1],
                                     val[i][1] + val[i][3] - merged_bbox[1])

            else:
                new_val.append(merged_bbox)
                merged_bbox = val[i]

        if new_val[-1:] != merged_bbox:
            new_val.append(merged_bbox)

        merged_levels.append(new_val)

    for lev in merged_levels:
        col = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for box in lev:
            cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), col, 2)

    Hierarchy.debug_image(imutils.resize(image, width=800))
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_dir = "Image_Out/"
    count = 0

    for file in os.listdir(image_dir):
        if file.endswith(".jpg"):
            test_bbox(image_dir + file)
            # tess_process(cv2.imread(image_dir + file))
            count += 1
