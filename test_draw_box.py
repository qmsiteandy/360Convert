import cv2
import json
import numpy as np


if __name__ == "__main__":

    frame = cv2.imread('./test/origin.jpg')
    frame_copy = frame.copy()

    # optput result from person_tracker
    # outputs = [{'class': 'person', 'name': 'person', 'bbox': [1049, 1188,  315,  225], 'confidence': 0.89111328125}, {'class': 'person', 'name': 'person', 'bbox': [2925, 1125,   99,  234], 'confidence': 0.80615234375}, {'class': 'person', 'name': 'person', 'bbox': [ 738, 1335,  185,   81], 'confidence': 0.74609375}, {'class': 'person', 'name': 'person', 'bbox': [1918,  998,   60,  147], 'confidence': 0.5458984375}, {'class': 'person', 'name': 'person', 'bbox': [1854, 1028,   75,  168], 'confidence': 0.48681640625}, {'class': 'person', 'name': 'person', 'bbox': [1922, 1070,  111,  189], 'confidence': 0.482421875}, {'class': 'person', 'name': 'person', 'bbox': [3798, 1083,   42,   81], 'confidence': 0.45703125}, {'class': 'person', 'name': 'person', 'bbox': [3765, 1082,   66,   84], 'confidence': 0.415771484375}]
    outputs = None
    # 打開 JSON 文件
    with open("test/predict_output.json", "r") as file:
        # 使用 json.load 讀取並解析 JSON 數據
        outputs = json.load(file)

    # 建立透明的 frame
    box_frame = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8)

    # 在透明的 frame 繪製框線
    for data in outputs:
        # print(data)

        bbox = data['bbox']
        conf = data['confidence']

        if conf > 0.5:
            # draw bounding box and text
            cv2.rectangle(box_frame, tuple(bbox[:2]), tuple([bbox[0]+bbox[2], bbox[1]+bbox[3]]), (0,0,255), 3, cv2.LINE_AA)
            top_center = (int(bbox[0] + (bbox[2])/2), max(bbox[1], 10))
            cv2.putText(box_frame, "ID: {}".format(data['name']), top_center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA) 

    # blended_frame = cv2.add(frame_copy, box_frame)
    # cv2.addWeighted(frame_copy, 1, box_frame, 1, 0)

    output_path = './output/'
    cv2.imwrite(f'{output_path}draw_box.jpg', box_frame)


def equirectangular_projection(image):
    # 獲取圖片尺寸
    height, width = image.shape[:2]

    # 定義 equirectangular 投影映射函數
    map_x = np.zeros((height, width), np.float32)
    map_y = np.zeros((height, width), np.float32)

    # 設定 equirectangular 投影映射
    for i in range(height):
        for j in range(width):
            theta = (2 * j / width - 1) * np.pi  # 經度映射
            phi = (i / height - 0.5) * np.pi      # 緯度映射

            x = int((theta / np.pi + 1) * (width - 1) / 2)
            y = int((phi / np.pi + 0.5) * (height - 1))

            map_x[i, j] = x
            map_y[i, j] = y

    # 進行 equirectangular 投影
    equirectangular_image = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    return equirectangular_image