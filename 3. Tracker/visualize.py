import os
import cv2
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

def visualize_tracking_results(txt_path, img_dir, output_path, fps=30):
    """
    トラッキング結果のテキストファイルと画像シーケンスから、
    結果を可視化した動画を生成します。

    Args:
        txt_path (str): トラッキング結果のテキストファイルのパス。
        img_dir (str): 連番JPG画像が格納されているディレクトリのパス。
        output_path (str): 出力するMP4動画の保存パス。
        fps (int): 出力動画のフレームレート。
    """
    # --- 1. 検出結果の読み込み ---
    print(f"トラッキング結果を読み込んでいます: {txt_path}")
    try:
        # MOT Challengeのフォーマットに合わせてカラム名を定義
        columns = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z']
        df = pd.read_csv(txt_path, header=None, names=columns)
    except pd.errors.EmptyDataError:
        print(f"エラー: {txt_path} が空か、読み込めません。")
        return
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません: {txt_path}")
        return

    # --- 2. IDごとにランダムな色を割り当て ---
    # 各追跡IDに一貫した色を割り当てるための辞書を作成
    unique_ids = df['id'].unique()
    colors = {int(tid): (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for tid in unique_ids}

    # --- 3. 画像ファイルと動画出力の準備 ---
    print(f"画像ファイルを読み込んでいます: {img_dir}")
    try:
        # 画像ファイル名を取得し、数値順に正しくソート
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')], 
                           key=lambda x: int(os.path.splitext(x)[0]))
        if not img_files:
            print(f"エラー: {img_dir} にJPG画像が見つかりません。")
            return
    except FileNotFoundError:
        print(f"エラー: ディレクトリが見つかりません: {img_dir}")
        return

    # 最初の画像から動画のサイズを取得
    first_image_path = os.path.join(img_dir, img_files[0])
    first_frame = cv2.imread(first_image_path)
    if first_frame is None:
        print(f"エラー: 最初の画像 {first_image_path} を読み込めません。")
        return
    height, width, _ = first_frame.shape

    # 動画書き出し用のVideoWriterを準備
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4用のコーデック
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"動画の書き出し準備が完了しました: {output_path}")

    # --- 4. 各フレームを処理して動画を生成 ---
    # tqdmを使ってプログレスバーを表示
    for frame_idx, img_file in enumerate(tqdm(img_files, desc="動画を生成中"), 1):
        # 画像を読み込む
        img_path = os.path.join(img_dir, img_file)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"警告: フレーム {img_path} が読み込めませんでした。スキップします。")
            continue

        # 現在のフレームに対応する検出結果を抽出
        detections_in_frame = df[df['frame'] == frame_idx]

        # 抽出した検出結果を描画
        for _, row in detections_in_frame.iterrows():
            track_id = int(row['id'])
            x = int(row['bb_left'])
            y = int(row['bb_top'])
            w = int(row['bb_width'])
            h = int(row['bb_height'])
            
            # バウンディングボックスを描画
            color = colors.get(track_id, (0, 255, 0))  # IDに色があれば使い、なければ緑
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # ID番号をバウンディングボックスの上に描画
            label = f"ID: {track_id}"
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 描画したフレームを動画に書き込む
        writer.write(frame)

    # --- 5. 終了処理 ---
    writer.release()
    print(f"\n動画の生成が完了しました。ファイルを確認します...")

    # --- 6. デバッグ用のファイル存在確認 ---
    # ファイルが実際にディスクに存在するか確認
    if os.path.exists(output_path):
        # ファイルサイズを確認（0バイト以上であれば成功とみなす）
        file_size = os.path.getsize(output_path)
        if file_size > 0:
            print(f"成功: '{output_path}' が作成されました (サイズ: {file_size / 1024:.2f} KB)。")
            print("ファイルブラウザに表示されない場合は、ブラウザの更新ボタンを押してみてください。")
        else:
            print(f"エラー: '{output_path}' は作成されましたが、ファイルサイズが0バイトです。動画の書き込みに失敗した可能性があります。")
    else:
        print(f"エラー: '{output_path}' が見つかりません。ファイルの作成に失敗しました。")


if __name__ == '__main__':
    # コマンドラインから引数を受け取るためのパーサーを作成
    parser = argparse.ArgumentParser(description="MOTトラッキング結果を動画として可視化します。")
    parser.add_argument('--txt_path', type=str, required=True, 
                        help='トラッキング結果のテキストファイルへのパス。例: MOT17-04-FRCNN.txt')
    parser.add_argument('--img_dir', type=str, required=True, 
                        help='連番JPG画像が格納されているディレクトリへのパス。例: MOT17-04-FRCNN/img1/')
    parser.add_argument('--output_path', type=str, default='./output_video.mp4', 
                        help='出力するMP4動画の保存パス。')
    parser.add_argument('--fps', type=int, default=30, 
                        help='出力動画のフレームレート。')

    args = parser.parse_args()

    visualize_tracking_results(args.txt_path, args.img_dir, args.output_path, args.fps)
