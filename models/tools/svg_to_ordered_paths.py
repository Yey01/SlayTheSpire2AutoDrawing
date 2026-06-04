import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path


def _strip_namespace(tag):
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _parse_viewbox(root):
    view_box = root.attrib.get("viewBox") or root.attrib.get("viewbox")
    if view_box:
        vals = [float(v) for v in re.split(r"[\s,]+", view_box.strip()) if v]
        if len(vals) == 4:
            return vals
    width = _parse_svg_length(root.attrib.get("width", "0"))
    height = _parse_svg_length(root.attrib.get("height", "0"))
    return [0.0, 0.0, width, height]


def _parse_svg_length(value):
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else 0.0


def _sample_with_svgpathtools(svg_path, step):
    try:
        from svgpathtools import svg2paths2
    except ImportError:
        return None

    paths, attrs, svg_attrs = svg2paths2(str(svg_path))
    sampled = []
    metas = []
    for idx, path in enumerate(paths):
        length = float(path.length(error=1e-4)) if path else 0.0
        if length <= 0:
            continue
        count = max(2, int(math.ceil(length / step)) + 1)
        points = []
        for i in range(count):
            t = i / (count - 1)
            z = path.point(t)
            points.append([int(round(z.real)), int(round(z.imag))])
        points = _dedupe(points)
        if len(points) >= 2:
            sampled.append(points)
            metas.append({
                "kind": "line",
                "level": "detail",
                "source": "svg_path",
                "svg_index": idx,
                "svg_fill": attrs[idx].get("fill", ""),
                "svg_stroke": attrs[idx].get("stroke", ""),
            })
    return sampled, metas, svg_attrs


def _dedupe(points):
    result = []
    last = None
    for p in points:
        cur = (p[0], p[1])
        if cur != last:
            result.append(p)
            last = cur
    return result


def _tokenize_path_data(d):
    return re.findall(r"[MmLlHhVvZz]|-?\d+(?:\.\d+)?(?:e[-+]?\d+)?", d)


def _parse_simple_path_data(d):
    tokens = _tokenize_path_data(d)
    points = []
    idx = 0
    cmd = None
    x = 0.0
    y = 0.0
    start = None

    def is_cmd(tok):
        return len(tok) == 1 and tok.isalpha()

    while idx < len(tokens):
        if is_cmd(tokens[idx]):
            cmd = tokens[idx]
            idx += 1
        if cmd is None:
            break

        if cmd in ("M", "m"):
            first = True
            while idx + 1 < len(tokens) and not is_cmd(tokens[idx]):
                nx = float(tokens[idx])
                ny = float(tokens[idx + 1])
                idx += 2
                if cmd == "m":
                    x += nx
                    y += ny
                else:
                    x, y = nx, ny
                points.append([int(round(x)), int(round(y))])
                if first:
                    start = (x, y)
                    first = False
                    cmd = "l" if cmd == "m" else "L"
        elif cmd in ("L", "l"):
            while idx + 1 < len(tokens) and not is_cmd(tokens[idx]):
                nx = float(tokens[idx])
                ny = float(tokens[idx + 1])
                idx += 2
                if cmd == "l":
                    x += nx
                    y += ny
                else:
                    x, y = nx, ny
                points.append([int(round(x)), int(round(y))])
        elif cmd in ("H", "h"):
            while idx < len(tokens) and not is_cmd(tokens[idx]):
                nx = float(tokens[idx])
                idx += 1
                x = x + nx if cmd == "h" else nx
                points.append([int(round(x)), int(round(y))])
        elif cmd in ("V", "v"):
            while idx < len(tokens) and not is_cmd(tokens[idx]):
                ny = float(tokens[idx])
                idx += 1
                y = y + ny if cmd == "v" else ny
                points.append([int(round(x)), int(round(y))])
        elif cmd in ("Z", "z"):
            if start is not None:
                x, y = start
                points.append([int(round(x)), int(round(y))])
            cmd = None
        else:
            # Curves require svgpathtools. Stop this path instead of producing wrong points.
            return []

    return _dedupe(points)


def _fallback_parse(svg_path):
    tree = ET.parse(svg_path)
    root = tree.getroot()
    paths = []
    metas = []
    for idx, el in enumerate(root.iter()):
        if _strip_namespace(el.tag) != "path":
            continue
        d = el.attrib.get("d", "")
        pts = _parse_simple_path_data(d)
        if len(pts) >= 2:
            paths.append(pts)
            metas.append({
                "kind": "line",
                "level": "detail",
                "source": "svg_path_simple",
                "svg_index": idx,
                "svg_fill": el.attrib.get("fill", ""),
                "svg_stroke": el.attrib.get("stroke", ""),
            })
    return paths, metas, {"viewBox": " ".join(str(v) for v in _parse_viewbox(root))}


def convert(svg_path, output_json, sample_step):
    svg_path = Path(svg_path)
    sampled = _sample_with_svgpathtools(svg_path, sample_step)
    if sampled is None:
        paths, metas, svg_attrs = _fallback_parse(svg_path)
        parser = "fallback_simple_path_parser"
    else:
        paths, metas, svg_attrs = sampled
        parser = "svgpathtools"

    payload = {
        "source_svg": str(svg_path),
        "parser": parser,
        "sample_step": sample_step,
        "path_count": len(paths),
        "point_count": sum(len(p) for p in paths),
        "svg_attrs": svg_attrs,
        "ordered_paths": paths,
        "ordered_metas": metas,
    }
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Convert stroke-like SVG paths to ordered_paths JSON.")
    parser.add_argument("--svg", required=True, help="Input SVG file.")
    parser.add_argument("--output", required=True, help="Output JSON file.")
    parser.add_argument("--sample-step", type=float, default=2.0, help="Curve sampling step in SVG units.")
    args = parser.parse_args()
    payload = convert(args.svg, args.output, args.sample_step)
    print(json.dumps({
        "source_svg": payload["source_svg"],
        "parser": payload["parser"],
        "path_count": payload["path_count"],
        "point_count": payload["point_count"],
        "output": args.output,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
