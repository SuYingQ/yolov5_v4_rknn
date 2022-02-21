import time
import requests
import cv2
import json
import base64
import os
import numpy as np
path = '/home/su/Downloads/yolov5_rknn/imgs/'
img_path = os.listdir(path)
l = []

for img_p in img_path:
    img_path_ = os.path.join(path, img_p)
    f = open(img_path_, 'rb')
    base64_data = base64.b64encode(f.read())
    f.close()
    base64_data = base64_data.decode()
    l.append(base64_data)
# data_1 = {"algo_type":"fire","data_type":'0','img':l}	# 传输的数据格式
# r_1 = requests.post('http://192.168.2.14:5000/detect',data =json.dumps(data_1))
# r_json_1 = r_1.json()
    # if r_json_1["code"]==1 :
    #     for i in range(len(r_json_1['data'][0]["alog"]["images"])):
    #         image_decode = base64.b64decode(r_json_1['data'][0]["alog"]["images"][i])	# 进行base64解码工作 base64->数组
    #         nparr = np.fromstring(image_decode, np.uint8)	# fromstring实现了字符串到Ascii码的转换
    #         img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #         cv2.imwrite('/home/su/Downloads/yolov5_rknn/out/{}'.format(img_path[i]),img_np)
    #     cv2.imshow('{}'.format(img_path[0]),img_np)
    #     key=cv2.waitKey(0)
    #     while True:
    #         if key== ord('q'):
    #             break
    # else:
    #     print("未检测到目标")
    # cv2.destroyAllWindows()
n = 0
while True:
    time.sleep(1)
    n = n+1
    if n % 2 == 0:
        data_2 = {"algo_type": "hat", "data_type": '1',
                  'img': 'rtmp://rtmp01open.ys7.com:1935/v3/openlive/C44410380_13_1?expire=1669887658&id=388022017657929728&t=4a28b61516a33b7e3887a1bfb828672af29365b31fb1b999cce071032d6c8dd4&ev=100'}
        r_2 = requests.post('http://192.168.2.29:5000/detect',
                            data=json.dumps(data_2))

        r_json_2 = r_2.json()
        if r_json_2["code"] == 1:
            for i in range(len(r_json_2['data'][0]["alog"]["images"])):
                image_decode = base64.b64decode(
                    r_json_2['data'][0]["alog"]["images"][i])  # 进行base64解码工作 base64->数组
                # fromstring实现了字符串到Ascii码的转换
                nparr = np.fromstring(image_decode, np.uint8)
                img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                cv2.imwrite(
                    '/home/su/Downloads/yolov5_rknn/out/{}'.format(img_path[i]), img_np)
            cv2.imshow('{}'.format(img_path[0]), img_np)
            key = cv2.waitKey(0)
            while True:
                if key == ord('q'):
                    break
        else:
            print("未检测到目标")
        cv2.destroyAllWindows()
