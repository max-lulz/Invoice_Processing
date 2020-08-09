import cv2
import pdf2image
import imutils
import numpy as np
import random
import os


def debug_image(img, name="Debug"):
    cv2.imshow(name, img)
    cv2.waitKey(0)


def load_image(pdf_path, output_path=""):
    pages = pdf2image.convert_from_path(pdf_path)

    i = 0
    for page in pages:
        page.save(output_path + "{}{}.jpg".format(pdf_path.split("/")[-1], i), "JPEG")
        i += 1


def get_hierarchy_levels(image_path):
    original_image = cv2.imread(image_path)
    filtered_image = preprocess(image_path)

    contour_hierarchy, bounding_boxes = get_contour_bounding_boxes(filtered_image)

    hier_image = original_image.copy()
    vis = {}
    levels = {}

    def is_level(box1, box2):

        return 0 if box2[1] - 5 <= centre(box1)[1] <= box2[3] + 5 or box1[1] - 5 <= centre(box2)[1] <= box1[
            3] + 5 else 1

    def centre(box):
        x = (box[0] + box[2]) // 2
        y = (box[1] + box[3]) // 2

        return x, y

    def dfs(node, curr_level):
        prev_node = contour_hierarchy[node][0]
        next_node = contour_hierarchy[node][1]
        child_node = contour_hierarchy[node][2]

        vis[node] = True

        if curr_level in levels:
            levels[curr_level].append(bounding_boxes[node])
        else:
            levels[curr_level] = [bounding_boxes[node]]

        if prev_node == -1 and next_node == -1:  # No adjacent node -> no contour at same level
            if child_node == -1:  # No child or adjacent nodes -> end of hierarchy
                return

            elif child_node not in vis:
                cv2.line(hier_image, (centre(bounding_boxes[node])),
                         (centre(bounding_boxes[child_node])),
                         color=(0, 0, 255))
                dfs(child_node, curr_level + is_level(bounding_boxes[node], bounding_boxes[child_node]))

        else:
            if prev_node != -1 and prev_node not in vis:
                cv2.line(hier_image, (centre(bounding_boxes[node])),
                         (centre(bounding_boxes[prev_node])),
                         color=(0, 0, 255))
                dfs(prev_node, curr_level + is_level(bounding_boxes[node], bounding_boxes[prev_node]))

            if next_node != -1 and next_node not in vis:
                cv2.line(hier_image, (centre(bounding_boxes[node])),
                         (centre(bounding_boxes[next_node])),
                         color=(0, 0, 255))
                dfs(next_node, curr_level + is_level(bounding_boxes[node], bounding_boxes[next_node]))

    dfs(0, 0)

    # debug_image(hier_image)
    # print(levels)
    return levels


def mask_lines(image, line_length):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    horizontal_kernel = np.zeros((1, line_length), dtype=np.uint8)  # square kernel with middle row set to 1
    horizontal_kernel[:, :] = 1  # can be changed to a single row kernel

    vertical_kernel = np.zeros((line_length, 1), dtype=np.uint8)
    vertical_kernel[:, :] = 1

    horizontal_mask = cv2.morphologyEx(~image, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)  # make masks that contain
    vertical_mask = cv2.morphologyEx(~image, cv2.MORPH_OPEN, vertical_kernel, iterations=1)  # horizontal and vertical lines

    _, horizontal_mask = cv2.threshold(horizontal_mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, vertical_mask = cv2.threshold(vertical_mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    image += horizontal_mask  # removes horizontal and vertical lines
    image += vertical_mask  # improves text blob detection

    return image


def dilate_text(image):

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(5, 5))
    image = cv2.morphologyEx(~image, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    image = cv2.erode(image, kernel=close_kernel, iterations=1)
    image = cv2.dilate(image, kernel=close_kernel, iterations=3)
    image = ~image

    return image


def preprocess(image, line_length=31):

    image = mask_lines(image, line_length)
    image = dilate_text(image)

    return image


def get_contour_bounding_boxes(image, get_vertices=True):

    image = image.copy()

    image = preprocess(image)

    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        x2 = x + w if get_vertices else w
        y2 = y + h if get_vertices else h

        bounding_boxes.append([x, y, x2, y2])

    return hierarchy[0], bounding_boxes


if __name__ == "__main__":
    random.seed()

    path = "GRiD_Sample Invoices II/"
    out_path = "Image_Out/"

    os.mkdir(out_path, mode=0o777, dir_fd=None)

    for file in os.listdir(path):
        if file.endswith(".pdf"):
            print("Loading " + file)
            load_image(path + file, out_path)

    for file in os.listdir(out_path):
        if file.endswith(".jpg"):
            print("Processing " + file)

            lev = get_hierarchy_levels(out_path + file)
            im = cv2.imread(out_path + file)

            for val in lev.values():
                col = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                for rec in val:
                    cv2.rectangle(im, (rec[0], rec[1]), (rec[2], rec[3]), color=col, thickness=2)

            debug_image(imutils.resize(im, width=1000))
