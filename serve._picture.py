from marshal import dumps
from xml.etree.ElementTree import tostringlist
import cv2
import time
import random
import numpy as np
from rknn.api import RKNN
import json
from flask import Flask, request
import base64
import os

"""
yolov5 预测脚本 for rknn
"""


def get_max_scale(img, max_w, max_h):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1)
    return scale


def get_new_size(img, scale):
    return tuple(map(int, np.array(img.shape[:2][::-1]) * scale))


class AutoScale:
    def __init__(self, img, max_w, max_h):
        self._src_img = img
        self.scale = get_max_scale(img, max_w, max_h)
        self._new_size = get_new_size(img, self.scale)
        self.__new_img = None

    @property
    def size(self):
        return self._new_size

    @property
    def new_img(self):
        if self.__new_img is None:
            self.__new_img = cv2.resize(self._src_img, self._new_size)
        return self.__new_img


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def filter_boxes(boxes, box_confidences, box_class_probs, conf_thres):
    box_scores = box_confidences * box_class_probs  # 条件概率， 在该cell存在物体的概率的基础上是某个类别的概率
    box_classes = np.argmax(box_scores, axis=-1)  # 找出概率最大的类别索引

    print("box_classes:", box_classes)
    box_class_scores = np.max(box_scores, axis=-1)  # 最大类别对应的概率值
    pos = np.where(box_class_scores >= conf_thres)  # 找出概率大于阈值的item
    # pos = box_class_scores >= conf_thres  # 找出概率大于阈值的item
    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]
    return boxes, classes, scores


def nms_boxes(boxes, scores, iou_thres):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def letterbox(img, new_wh=(416, 416), color=(114, 114, 114)):
    a = AutoScale(img, *new_wh)
    new_img = a.new_img
    h, w = new_img.shape[:2]
    new_img = cv2.copyMakeBorder(
        new_img, 0, new_wh[1] - h, 0, new_wh[0] - w, cv2.BORDER_CONSTANT, value=color)
    return new_img, (new_wh[0] / a.scale, new_wh[1] / a.scale)


def load_model0(model_path, npu_id):
    rknn = RKNN()
    devs = rknn.list_devices()
    device_id_dict = {}
    for index, dev_id in enumerate(devs[-1]):
        if dev_id[:2] != 'TS':
            device_id_dict[0] = dev_id
        if dev_id[:2] == 'TS':
            device_id_dict[1] = dev_id
    print('-->loading model : ' + model_path)
    rknn.load_rknn(model_path)
    print('--> Init runtime environment on: ' + device_id_dict[npu_id])
    ret = rknn.init_runtime(device_id=device_id_dict[npu_id])
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')
    return rknn


def load_rknn_model(PATH):
    rknn = RKNN()
    print('--> Loading model')
    ret = rknn.load_rknn(PATH)
    if ret != 0:
        print('load rknn model failed')
        exit(ret)
    print('done')
    ret = rknn.init_runtime(target='RK1808')
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')
    return rknn


class RKNNDetector:
    def __init__(self, model, wh, masks, anchors, names):
        self.wh = wh
        self._masks = masks
        self._anchors = anchors
        self.names = names
        if isinstance(model, str):
            model = load_rknn_model(model)
        self._rknn = model
        self.draw_box = True

    def _predict(self, img_src, _img, gain, conf_thres=0.2, iou_thres=0.45):
        src_h, src_w = img_src.shape[:2]
        # _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        t0 = time.time()
        pred_onx = self._rknn.inference(inputs=[_img])
        print("inference time:\t", time.time() - t0)
        boxes, classes, scores = [], [], []
        for t in range(3):
            input0_data = sigmoid(pred_onx[t][0])
            c, h, w = input0_data.shape
            # (1, 21, 80, 80)

            input0_data = np.transpose(input0_data, (1, 2, 0))
            input0_data = input0_data.reshape((h, w, 3, -1))
            grid_h, grid_w, _, _ = input0_data.shape

            anchors = [self._anchors[i] for i in self._masks[t]]

            box_confidence = input0_data[..., 4]
            box_confidence = np.expand_dims(box_confidence, axis=-1)
            box_class_probs = input0_data[..., 5:]

            box_xy = input0_data[..., :2]
            box_wh = input0_data[..., 2:4]

            col = np.tile(np.arange(0, grid_w), grid_h).reshape(-1, grid_w)
            row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_w)
            col = col.reshape((grid_h, grid_w, 1, 1)).repeat(3, axis=-2)
            row = row.reshape((grid_h, grid_w, 1, 1)).repeat(3, axis=-2)
            grid = np.concatenate((col, row), axis=-1)
            box_xy = box_xy * 2 - 0.5 + grid
            box_wh = (box_wh * 2) ** 2 * anchors
            box_xy /= (grid_w, grid_h)  # 计算原尺寸的中心
            box_wh /= self.wh  # 计算原尺寸的宽高
            box_xy -= (box_wh / 2.)  # 计算原尺寸的中心
            box = np.concatenate((box_xy, box_wh), axis=-1)
            res = filter_boxes(box, box_confidence,
                               box_class_probs, conf_thres)
            boxes.append(res[0])
            classes.append(res[1])
            scores.append(res[2])
        boxes, classes, scores = np.concatenate(
            boxes), np.concatenate(classes), np.concatenate(scores)
        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]
            keep = nms_boxes(b, s, iou_thres)
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])
        if len(nboxes) < 1:
            return img_src, [''], ['']
        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)
        label_list = []
        box_list = []
        score_list = []

        for (x, y, w, h), score, cl in zip(boxes, scores, classes):
            x *= gain[0]
            y *= gain[1]
            w *= gain[0]
            h *= gain[1]
            x1 = max(0, np.floor(x).astype(int))
            y1 = max(0, np.floor(y).astype(int))
            x2 = min(src_w, np.floor(x + w + 0.5).astype(int))
            y2 = min(src_h, np.floor(y + h + 0.5).astype(int))

            label_list.append(self.names[int(cl)])
            box_list.append((x1, y1, x2, y2))
            score_list
            if self.draw_box:
                plot_one_box((x1, y1, x2, y2), img_src,
                             label=self.names[int(cl)])

        return img_src, label_list, box_list

    def predict_resize(self, img_src, conf_thres=0.4, iou_thres=0.45):
        """
        预测一张图片，预处理使用resize
        return: labels,boxes
        """
        _img = cv2.resize(img_src, self.wh)
        gain = img_src.shape[:2][::-1]
        return self._predict(img_src, _img, gain, conf_thres, iou_thres, )

    def predict(self, img_src, conf_thres=0.4, iou_thres=0.45):
        """
        预测一张图片，预处理保持宽高比
        return: labels,boxes
        """
        _img, gain = letterbox(img_src, self.wh)
        return self._predict(img_src, _img, gain, conf_thres, iou_thres)

    def close(self):
        self._rknn.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close



app = Flask(__name__)


@app.route('/detect', methods=["POST"])
def play_1():

    opt = {
        "code": 0,
        "id": "string",
        "data": [{
            "alog": {
                "images": [],
                'labels':[],
                'position':{
                    "x_min": {},
                    "y_min": {},
                    "x_max": {},
                    "y_max": {},
                }
            }
        }]
    }

    data = request.get_data().decode('utf-8')  # 捕捉客户端传来的数据
    data = json.loads(data)

    if data["algo_type"] == "fire" : 
        RKNN_MODEL_PATH = '/home/su/Downloads/yolov5_rknn/weights/fire_smoke.rknn'
        SIZE = (640, 640)
        MASKS = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        ANCHORS = [[10, 13], [16, 30], [33, 23], [30, 61], [
            62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
        CLASSES = ("smoke", "fire", "person")
        model = load_rknn_model(RKNN_MODEL_PATH)
        detector = RKNNDetector(model, SIZE, MASKS, ANCHORS, CLASSES)

    if data["algo_type"] == "hat" : 
        RKNN_MODEL_PATH = '/home/su/Downloads/yolov5_rknn/weights/hat_640.rknn'
        SIZE = (640, 640)
        MASKS = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        ANCHORS = [[10, 13], [16, 30], [33, 23], [30, 61], [
            62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
        CLASSES = ("person","hat")
        model = load_rknn_model(RKNN_MODEL_PATH)
        detector = RKNNDetector(model, SIZE, MASKS, ANCHORS, CLASSES)

    if data["data_type"] == "0":

        image_b = data["img"]
        n = 0

        for image_b64 in image_b:
            n = n+1  # 获取dict中'img'标签的数据
            image_decode = base64.b64decode(
                image_b64)  # 进行base64解码工作 base64->数组
            # fromstring实现了字符串到Ascii码的转换
            nparr = np.fromstring(image_decode, np.uint8)
            # 将 nparr 数据转换(解码)成图像格式
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            im, lable_pre, box_pre = detector.predict(img_np)
            
            opt['data'][0]["alog"]['position']['x_min']['这是第{}张图片:'.format(n)] = [
            ]
            opt['data'][0]["alog"]['position']['y_min']['这是第{}张图片:'.format(n)] = [
            ]
            opt['data'][0]["alog"]['position']['x_max']['这是第{}张图片:'.format(n)] = [
            ]
            opt['data'][0]["alog"]['position']['y_max']['这是第{}张图片:'.format(n)] = [
            ]
            if lable_pre != [""]:
                opt["code"] = 1
                img_src_1 = cv2.imencode('.jpg', im)[1]
                base64_data = base64.b64encode(img_src_1)
                base64_data = base64_data.decode()
                opt['data'][0]["alog"]["images"].append(base64_data)
                opt['data'][0]["alog"]['labels'].append(lable_pre)
                for i in box_pre:
                    opt['data'][0]["alog"]['position']['x_min']['这是第{}张图片:'.format(
                        n)].append(str(i[0]))
                    opt['data'][0]["alog"]['position']['y_min']['这是第{}张图片:'.format(
                        n)].append(str(i[1]))
                    opt['data'][0]["alog"]['position']['x_max']['这是第{}张图片:'.format(
                        n)].append(str(i[2]))
                    opt['data'][0]["alog"]['position']['y_max']['这是第{}张图片:'.format(
                        n)].append(str(i[3]))
            else:
                opt['data'][0]["alog"]["images"].append(image_b64)
                opt['data'][0]["alog"]['labels'].append([])
                opt['data'][0]["alog"]['position']['x_min']['这是第{}张图片:'.format(
                    n)].append("0")
                opt['data'][0]["alog"]['position']['y_min']['这是第{}张图片:'.format(
                    n)].append("0")
                opt['data'][0]["alog"]['position']['x_max']['这是第{}张图片:'.format(
                    n)].append("0")
                opt['data'][0]["alog"]['position']['y_max']['这是第{}张图片:'.format(
                    n)].append("0")

        json_data = json.dumps(opt).encode('utf8')
        model.release()
        return json_data

    if data["data_type"] == "1":
        image_b = data["img"]
        n = 1
    # rtmp://rtmp01open.ys7.com:1935/v3/openlive/C44410380_13_1?expire=1669887658&id=388022017657929728&t=4a28b61516a33b7e3887a1bfb828672af29365b31fb1b999cce071032d6c8dd4&ev=100
        cap = cv2.VideoCapture(data["img"])
        # _, img_copy = cap.read()
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # fps=24

        # videoWriter = cv2.VideoWriter(
        #     '/home/su/Downloads/123.mp4', fourcc, fps, (1920, 1080))  # (1280, 720)为视频大小
        while cap.isOpened:
            time.sleep(1)
            n = n+1
            if n % 1 == 0:
                ret, Img = cap.read()
                if not ret:
                    return "读取错误，请检查视频流路径是否正确"
                im, lable_pre, box_pre = detector.predict(Img)
                
                opt['data'][0]["alog"]['position']['x_min']['{}'.format(n)] = [
                ]
                opt['data'][0]["alog"]['position']['y_min']['{}'.format(n)] = [
                ]
                opt['data'][0]["alog"]['position']['x_max']['{}'.format(n)] = [
                ]
                opt['data'][0]["alog"]['position']['y_max']['{}'.format(n)] = [
                ]
                if lable_pre != [""]:
                    opt["code"] = 1
                    img_src_1 = cv2.imencode('.jpg', im)[1]
                    base64_data_2 = base64.b64encode(img_src_1)
                    base64_data_2 = base64_data_2.decode()
                    opt['data'][0]["alog"]["images"].append(base64_data_2)
                    opt['data'][0]["alog"]['labels'].append(lable_pre)
                    for i in box_pre:
                        opt['data'][0]["alog"]['position']['x_min']['{}'.format(n)].append(str(i[0]))
                        opt['data'][0]["alog"]['position']['y_min']['{}'.format(n)].append(str(i[1]))
                        opt['data'][0]["alog"]['position']['x_max']['{}'.format(n)].append(str(i[2]))
                        opt['data'][0]["alog"]['position']['y_max']['{}'.format(n)].append(str(i[3]))
                else:
                    img_src_1 = cv2.imencode('.jpg', Img)[1]
                    base64_data_2 = base64.b64encode(img_src_1)
                    base64_data_2 = base64_data_2.decode()
                    opt['data'][0]["alog"]["images"].append(base64_data_2)
                    opt['data'][0]["alog"]['labels'].append([])
                    opt['data'][0]["alog"]['position']['x_min']['{}'.format(n)].append("0")
                    opt['data'][0]["alog"]['position']['y_min']['{}'.format(n)].append("0")
                    opt['data'][0]["alog"]['position']['x_max']['{}'.format(n)].append("0")
                    opt['data'][0]["alog"]['position']['y_max']['{}'.format(n)].append("0")

            json_data = json.dumps(opt).encode('utf8')
            model.release()
            return json_data
                # fps = 24  # 视频帧率
            #    # w=img.shape[1]

            # videoWriter.write(img)

        #     cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        #     cv2.imshow('img', img)

        #     if cv2.waitKey(40) & 0xFF == ord('q'):
        #         break
        # else:
        #     ret, img = cap.read()
        #     if not ret:
        #         continue
        #     cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        #     cv2.imshow('img', img)
        #     if cv2.waitKey(40) & 0xFF == ord('q'):
        #         break
    # videoWriter.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    app.run(host='192.168.2.29', port=5000)
