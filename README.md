
# SSBUFrameAnalyzer

## Usage

```
python3 SSBUFrameAnalyzer.py DIGIT_DICTIONARY_FILE SCREENSHOT_FILE
```

1280x720のカラー画像のみ受け付けます（mainの場合内部で自動リサイズします）。

`DIGIT_DICTIONALY_FILE`はダメージ値の数値画像から計算したHOG特徴量を格納したjsonです（Releasesに添付）。

いまのところ、2人対戦のみです（領域の切り出しを手打ちでやってるので面倒）。トレーニングも4人対戦の配置なので動かないです（エラーチェックもしてない）。

## Requirements

- Tesseract
    - `apt install tesseract-ocr tesseract-ocr-jpn`
- pip install -r requirements.txt
    - Pillow
    - pyocr
    - numpy
    - scikit-image

