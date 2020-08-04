import os
import time
import cv2
from PIL import Image
import Hierarchy
import tesserocr


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


def merge_box(box1, box2):
    min_x = min(box1[0], box2[0])
    min_y = min(box1[1], box2[1])
    max_w = max(box1[0] + box1[2], box2[0] + box2[2]) - min_x
    max_h = max(box1[1] + box1[3], box2[1] + box2[3]) - min_y

    merged_box = [min_x, min_y, max_w, max_h]

    return merged_box


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

                for j in range(i+1, len(line)):
                    if line[j][0] < merged_boundingbox[0] + merged_boundingbox[2] and is_level(merged_boundingbox, line[j]) and not is_merged[j]:
                        is_merged[j] = True
                        merged_boundingbox = merge_box(merged_boundingbox, line[j])

                    elif not line[j][0] < merged_boundingbox[0] + merged_boundingbox[2]:
                        break

                new_line.append(merged_boundingbox)

        merged_boxes.append(new_line)

    merged_boxes.sort(key=lambda x: x[0][1])

    return merged_boxes


def get_text_boundingboxes(image):

    pil_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    image = Hierarchy.mask_lines(image.copy(), 80)

    sparse_word_boxes = detect_sparse_words(image)
    text_boxes = []

    with tesserocr.PyTessBaseAPI() as api:
        api.SetPageSegMode(tesserocr.PSM.SINGLE_BLOCK)

        api.SetImage(Image.fromarray(pil_image))

        for box in sparse_word_boxes:
            is_text, text = filter_boxes([box[0] - 5, box[1] - 5, box[2] + 5, box[3] + 5], api)

            if is_text:
                text_boxes.append(box)

    return text_boxes


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


def detect_sparse_words(image):

    pil_image = Image.fromarray(cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB))

    sparse_boundingboxes = []

    with tesserocr.PyTessBaseAPI() as api:
        api.SetPageSegMode(tesserocr.PSM.SPARSE_TEXT)

        api.SetImage(pil_image)
        box_data = api.GetComponentImages(tesserocr.RIL.WORD, True)

        for (_, box, _, _) in box_data:
            sparse_boundingboxes.append([box['x'], box['y'], box['w'], box['h']])

    return sparse_boundingboxes


@func_time
def get_image_text(image_path):

    image = cv2.imread(image_path)
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    _, boxes = Hierarchy.get_contour_bounding_boxes(image, get_vertices=False)

    boxes.extend(get_text_boundingboxes(image))

    boxes = list(filter(lambda x: x[3] + x[1] < image.shape[0] and x[2] + x[0] < image.shape[1] and x[3] <= 100, boxes))
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
                    filtered_line.append([box, text.strip()])
                    # print(text.strip())

            merged_text_boxes[i] = filtered_line

    # for val in merged_text_boxes:
    #     col = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    #     for box in val:
    #         box = box[0]
    #         cv2.rectangle(image, (box[0], box[1]), (box[2] + box[0], box[1] + box[3]), col, 2)

    # cv2.imwrite("Output/image{}.jpg".format(count), image)


if __name__ == "__main__":
    image_dir = "Image_Out/"
    count = 0

    for file in os.listdir(image_dir):
        if file.endswith(".jpg"):
            get_image_text(image_dir + file)
            count += 1
