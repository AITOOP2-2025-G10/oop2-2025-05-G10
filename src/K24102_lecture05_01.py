import numpy as np
import cv2
from my_module.K21999.lecture05_camera_image_capture import MyVideoCapture

def lecture05_01():

    # カメラキャプチャ実行
    app = MyVideoCapture()
    capture_img = app.run()  # run() が cv2.Mat を返す想定


    # 画像をローカル変数に保存
    google_img : cv2.Mat = cv2.imread('images/google.png')
    # capture_img : cv2.Mat = "implement me"

    g_hight, g_width, g_channel = google_img.shape
    c_hight, c_width, c_channel = capture_img.shape
    print(google_img.shape)
    print(capture_img.shape)
    for y in range(g_hight):
        for x in range(g_width):
            b, g, r = google_img[y, x]
            # もし白色(255,255,255)だったら置き換える
            if (b, g, r) == (255, 255, 255):
                tile_x = x % c_width
                tile_y = y % c_hight
                google_img[y, x] = capture_img[tile_y, tile_x]
                #implement me

    # 書き込み処理
    # implement me
    cv2.imwrite('images/google_with_camera.png', google_img)

