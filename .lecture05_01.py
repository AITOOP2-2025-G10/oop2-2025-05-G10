import cv2
import numpy as np
import os
# スクリーンショットの構成に基づき、src/my_module/camera_capture.py からMyVideoCaptureをインポート
# 修正点: 相対パス (from .) を 'src' から始まる絶対パスに変更
from src.my_module.camera_capture import MyVideoCapture

# --- 定数定義 ---
GOOGLE_IMG_PATH = './images/google.png'
OUTPUT_DIR = './output_images'
OUTPUT_IMG_PATH = os.path.join(OUTPUT_DIR, 'google_replaced_by_camera.png')

# 白色の閾値 (BGR)
# (250, 250, 250) から (255, 255, 255) までを「白」とみなす
LOWER_WHITE = np.array([250, 250, 250])
UPPER_WHITE = np.array([255, 255, 255])

def lecture05_01():
    """
    課題5-1のメイン処理:
    1. カメラを起動し、'q'キーで画像を1枚キャプチャする。
    2. Google画像を読み込む。
    3. キャプチャ画像をGoogle画像と同じサイズにリサイズする。
    4. Google画像の「白色」部分を特定するマスクを作成する。
    5. マスクを使い、Google画像の非白色部分（ロゴ等）とカメラ画像の背景部分を合成する。
    6. 結果をファイルに保存する。
    """
    
    # 1. カメラを起動し、画像をキャプチャ
    print("カメラを起動します。'q'キーを押してキャプチャしてください。")
    cam_app = MyVideoCapture()
    cam_app.run() # 'q'が押されるまでループ
    
    camera_img = cam_app.get_img()
    if camera_img is None:
        print("画像がキャプチャされませんでした。処理を終了します。")
        return

    # OpenCVはBGRで画像を扱うため、(640x480)のBGR画像が取得される
    print(f"カメラ画像(640x480)を取得しました: {camera_img.shape}")

    # 2. Google画像を読み込む
    google_img = cv2.imread(GOOGLE_IMG_PATH)
    if google_img is None:
        print(f"エラー: Google画像が見つかりません: {GOOGLE_IMG_PATH}")
        return
    
    # (1280x640)のBGR画像が読み込まれる
    print(f"Google画像(1280x640)を読み込みました: {google_img.shape}")
    g_height, g_width, _ = google_img.shape

    # 3. キャプチャ画像をGoogle画像と同じサイズ(1280x640)にリサイズ
    resized_camera_img = cv2.resize(camera_img, (g_width, g_height))

    # 4. Google画像の白色部分を特定するマスクを作成
    # white_maskは、白色の部分が255、それ以外が0になるグレースケール画像
    white_mask = cv2.inRange(google_img, LOWER_WHITE, UPPER_WHITE)
    
    # マスクを反転（非白色部分(ロゴなど)が255になるマスク）
    white_mask_inv = cv2.bitwise_not(white_mask)

    # 5. 画像を合成
    # 5a. Google画像から非白色部分（ロゴ、ボタンなど）を抽出
    google_foreground = cv2.bitwise_and(google_img, google_img, mask=white_mask_inv)

    # 5b. リサイズしたカメラ画像から、Google画像の白色部分に相当する領域を抽出
    camera_background = cv2.bitwise_and(resized_camera_img, resized_camera_img, mask=white_mask)

    # 5c. 前景（ロゴ）と背景（カメラ画像）を足し合わせる
    result_img = cv2.add(google_foreground, camera_background)

    # 6. 結果をファイルに保存
    # 出力先ディレクトリがなければ作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cv2.imwrite(OUTPUT_IMG_PATH, result_img)
    
    print(f"処理が完了しました。画像を {OUTPUT_IMG_PATH} に保存しました。")

    # 結果を表示
    cv2.imshow('Result Image - Press any key to exit', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # このファイルが直接実行された場合もlecture05_01を実行
    lecture05_01()

