
# 瞬き検出システム


## 環境セットアップ
```bash
pip install -r requirements.txt
```

## 基本的な使い方

### config.yaml を使った実行

```bash
# 基本実行（CPU、動画入力）
python -m app

# GPU実行
python -m app runtime=gpu

# カメラ入力 + GPU + デバッグモード
python -m app stream=camera runtime=gpu
```


### 設定ファイル構造

```
app/config/
├── config.yaml    # メイン設定（分割版）
├── config_simple.yaml     # 統合設定
├── stream/                # 入力設定
│   ├── video.yaml
│   ├── camera.yaml
│   └── socket.yaml
├── pipeline/              # 処理パイプライン
│   ├── face_detector/
│   ├── face_recognizer/
│   └── blink_detector/
├── debug/                 # デバッグ設定
└── runtime/               # 実行環境
    ├── cpu.yaml
    └── gpu.yaml
```

### パラメータ調整例

```bash
# YOLO検出閾値調整
python -m app runtime=gpu \
  pipeline.face_detector.yolo.object_score_threshold=0.5

# 顔認識精度調整
python -m app pipeline/face_recognizer=insight_face \
  pipeline.face_recognizer.insightface.similarity_threshold=0.7

# 瞬き検出閾値調整
python -m app runtime=gpu \
  pipeline.blink_detector.lstm_base_01_blink_detector.lstm_base_01.blink_threshold=0.6
```
