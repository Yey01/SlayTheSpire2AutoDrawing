import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def convert_color_to_onnx_lineart(image_path, model_path, max_side):
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        resized = img

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = np.expand_dims(rgb.transpose(2, 0, 1), axis=0)

    session = ort.InferenceSession(str(model_path))
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: tensor})[0]
    lineart = (output[0, 0] * 255).clip(0, 255).astype(np.uint8)

    if lineart.shape[:2] != (h, w):
        lineart = cv2.resize(lineart, (w, h), interpolation=cv2.INTER_LINEAR)

    return lineart, img.shape[:2], resized.shape[:2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="models/line_drawing/line-drawings.onnx")
    parser.add_argument("--max-side", type=int, default=900)
    parser.add_argument("--output-max-side", type=int, default=0)
    args = parser.parse_args()

    image_path = Path(args.input).resolve()
    model_path = Path(args.model).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    lineart, original_shape, infer_shape = convert_color_to_onnx_lineart(
        image_path, model_path, args.max_side
    )
    output_shape = lineart.shape[:2]
    if args.output_max_side and max(output_shape) > args.output_max_side:
        scale = args.output_max_side / max(output_shape)
        lineart = cv2.resize(lineart, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        output_shape = lineart.shape[:2]

    line_raw = output_dir / "line_raw.png"
    cv2.imwrite(str(line_raw), lineart)

    dark_ratio = float(np.count_nonzero(lineart < 220) / lineart.size)
    summary = {
        "input": str(image_path),
        "model": str(model_path),
        "line_raw": str(line_raw),
        "max_side": args.max_side,
        "output_max_side": args.output_max_side,
        "original_shape": list(original_shape),
        "inference_shape": list(infer_shape),
        "output_shape": list(output_shape),
        "dark_ratio_lt220": round(dark_ratio, 4),
    }
    (output_dir / "line_raw_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
