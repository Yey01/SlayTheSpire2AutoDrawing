# -*- coding: utf-8 -*-

import hashlib
import json
import math
import os
import shutil
import sys
import threading

import cv2
import numpy as np

from models.device_runtime import resolve_torch_device


class DeepSketchVectorizerMixin:
    def _set_status(self, text, color="blue"):
        self.root.after(0, lambda: self.status_label.config(text=text, foreground=color))

    def _clear_cuda_cache(self):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _get_deepsketch_cache_key(self, line_raw_bytes):
        import hashlib
        payload = line_raw_bytes
        payload += self.deepsketch_model_size.encode()
        payload += str(self.deepsketch_max_side).encode()
        payload += str(getattr(self, "deepsketch_oom_retry_sides", [])).encode()
        payload += resolve_torch_device(getattr(self, "deepsketch_device", "auto")).encode()
        payload += str(getattr(self, "deepsketch_sample_step", 2.0)).encode()
        payload += str(getattr(self, "deepsketch_use_bezier", True)).encode()
        payload += b"v15"
        return hashlib.sha256(payload).hexdigest()[:16]

    def _get_deepsketch_cache_dir(self, cache_key):
        cache_root = os.path.join(os.path.dirname(self.image_path) if self.image_path else ".",
                                   ".cache", "deepsketch", cache_key)
        os.makedirs(cache_root, exist_ok=True)
        return cache_root

    def get_image_output_dir(self, mode=None, onnx=False):
        if self.image_path:
            image_name = os.path.splitext(os.path.basename(self.image_path))[0]
        else:
            image_name = "current_image"
        safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in image_name).strip("_")
        if not safe_name:
            safe_name = "current_image"
        if mode:
            suffix = f"onnx_{mode}" if onnx else mode
            safe_name = f"{safe_name}_{suffix}"
        if getattr(sys, "frozen", False):
            base_dir = os.path.dirname(sys.executable)
        else:
            base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        root_dir = os.path.join(base_dir, "output")
        output_dir = os.path.join(root_dir, safe_name)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def clear_image_output_dir(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for name in os.listdir(output_dir):
            path = os.path.join(output_dir, name)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                try:
                    os.remove(path)
                except OSError:
                    pass

    def get_debug_output_dir(self, mode, onnx=False):
        debug_dir = os.path.join(self.get_image_output_dir(mode, onnx=onnx), "debug")
        os.makedirs(debug_dir, exist_ok=True)
        return debug_dir

    def cleanup_legacy_debug_outputs(self):
        if not self.image_path:
            return
        image_dir = os.path.dirname(self.image_path)
        if not os.path.isdir(image_dir):
            return
        for name in os.listdir(image_dir):
            if not name.startswith("debug_"):
                continue
            if not name.lower().endswith((".png", ".jpg", ".jpeg", ".json")):
                continue
            try:
                os.remove(os.path.join(image_dir, name))
            except OSError:
                pass

    def _load_cached_deepsketch(self, cache_dir):
        cached_svg = os.path.join(cache_dir, "sample_final.svg")
        cached_json = os.path.join(cache_dir, "ordered_optimized.json")
        cached_stats = os.path.join(cache_dir, "stats.json")
        if os.path.exists(cached_svg) and os.path.exists(cached_json):
            import json as _json
            with open(cached_json, "r", encoding="utf-8") as f:
                data = _json.load(f)
            paths = [item["path"] for item in data]
            metas = [item["meta"] for item in data]
            stats = {}
            if os.path.exists(cached_stats):
                with open(cached_stats, "r", encoding="utf-8") as f:
                    stats = _json.load(f)
            stats["cache_hit"] = True
            return paths, metas, stats, cached_svg
        return None, None, None, None

    def _save_cached_deepsketch(self, cache_dir, svg_path, optimized_paths, optimized_metas, stats):
        import json as _json
        import shutil as _shutil
        cached_svg = os.path.join(cache_dir, "sample_final.svg")
        try:
            if os.path.exists(svg_path):
                _shutil.copy2(svg_path, cached_svg)
        except Exception:
            pass
        cached_json = os.path.join(cache_dir, "ordered_optimized.json")
        export = []
        for path, meta in zip(optimized_paths, optimized_metas):
            export.append({"path": path, "meta": meta, "length": self._path_length(path)})
        with open(cached_json, "w", encoding="utf-8") as f:
            _json.dump(export, f, indent=2, ensure_ascii=False)
        cached_stats = os.path.join(cache_dir, "stats.json")
        stats["cache_hit"] = False
        with open(cached_stats, "w", encoding="utf-8") as f:
            _json.dump(stats, f, indent=2, ensure_ascii=False)

    def _get_deepsketch_resize_attempts(self):
        min_viable = getattr(self, "deepsketch_gpu_min_viable_side", 384)
        base_side = int(getattr(self, "deepsketch_max_side", 700))
        attempts = [base_side]
        if getattr(self, "deepsketch_auto_reduce_on_oom", True):
            for side in getattr(self, "deepsketch_oom_retry_sides", []):
                try:
                    side = int(side)
                except (TypeError, ValueError):
                    continue
                if min_viable <= side < base_side and side not in attempts:
                    attempts.append(side)
        return attempts

    def _try_load_paths_from_existing_output(self, output_dir):
        high_quality_json = os.path.join(output_dir, "ordered_paths_high_quality.json")
        if not os.path.exists(high_quality_json):
            return None, None, None
        try:
            with open(high_quality_json, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return None, None, None

        if data.get("version", 0) < 14:
            return None, None, None

        paths = []
        metas = []
        for item in data.get("paths", []):
            pts = item.get("points", [])
            if len(pts) < 2:
                continue
            paths.append(pts)
            metas.append({
                "kind": "line",
                "level": item.get("level", "detail"),
                "source": item.get("source", ""),
                "length": item.get("length", 0),
                "svg_index": item.get("svg_index", 0),
            })

        stats = {}
        quality_stats_path = os.path.join(output_dir, "path_quality_stats.json")
        if os.path.exists(quality_stats_path):
            try:
                with open(quality_stats_path, "r", encoding="utf-8") as f:
                    stats = json.load(f)
            except Exception:
                pass

        stats["output_reused"] = True
        stats["optimized_paths"] = len(paths)
        stats["optimized_points"] = sum(len(p) for p in paths)
        return paths, metas, stats

    def process_color_image_paths_deepsketch(self, img, deepsketch_input, onnx_image=None):
        import time as _time
        t_start = _time.time()
        output_dir = self.get_image_output_dir("quality", onnx=onnx_image is not None)

        reused_paths, reused_metas, reused_stats = self._try_load_paths_from_existing_output(output_dir)
        if reused_paths is not None:
            self._set_status("已复用已有输出文件，跳过推理...", "green")
            self.debug_deepsketch_stats = reused_stats
            self._preview_image = self.preview_image_from_deepsketch_input(deepsketch_input)
            return reused_paths, reused_metas

        self.clear_image_output_dir(output_dir)
        work_dir = os.path.join(output_dir, "_work")
        work_input_dir = os.path.join(work_dir, "input")
        work_vector_dir = os.path.join(work_dir, "vectorize")
        os.makedirs(work_input_dir, exist_ok=True)
        os.makedirs(work_vector_dir, exist_ok=True)

        line_raw_path, scale_info = self.prepare_deepsketch_input(
            deepsketch_input, work_input_dir, output_dir, onnx_image=onnx_image
        )

        if self.deepsketch_cache_enabled:
            with open(line_raw_path, "rb") as f:
                line_raw_bytes = f.read()
            cache_key = self._get_deepsketch_cache_key(line_raw_bytes)
            cache_dir = self._get_deepsketch_cache_dir(cache_key)
            cached_paths, cached_metas, cached_stats, cached_svg = self._load_cached_deepsketch(cache_dir)
            if cached_paths is not None:
                self._set_status("已命中 DeepSketch 缓存，跳过推理...", "green")
                self.debug_deepsketch_stats = cached_stats
                self.write_deepsketch_outputs(output_dir, line_raw_path, cached_svg,
                                              cached_paths, cached_metas, img.shape[:2])
                if getattr(self, "deepsketch_save_ordered_paths_json", False):
                    self._save_ordered_paths_json(output_dir, cached_paths, cached_metas,
                                                  {"_parser": cached_stats.get("parser", "unknown"),
                                                   "_raw_path_count": cached_stats.get("raw_path_count", len(cached_paths)),
                                                   "_raw_point_count": cached_stats.get("raw_point_count", 0),
                                                   "width": cached_stats.get("svg_size", [0, 0])[1] if cached_stats.get("svg_size") else 0,
                                                   "height": cached_stats.get("svg_size", [0, 0])[0] if cached_stats.get("svg_size") else 0},
                                                  cached_stats)
                self.save_inference_config(output_dir, "quality", onnx_image is not None)
                self.cleanup_legacy_debug_outputs()
                shutil.rmtree(work_dir, ignore_errors=True)
                self._preview_image = self.preview_image_from_deepsketch_input(deepsketch_input)
                return cached_paths, cached_metas
        else:
            cache_key = None
            cache_dir = None

        self._set_status("正在运行 DeepSketch full，可能需要 1-4 分钟...", "darkorange")
        try:
            self._show_deepsketch_progress()
        except Exception:
            pass
        self._clear_cuda_cache()
        t_ds_start = _time.time()
        try:
            svg_path = self.run_deepsketch_vectorize(line_raw_path, work_vector_dir)
        finally:
            self._clear_cuda_cache()
            try:
                self._hide_deepsketch_progress()
            except Exception:
                pass
        t_ds_end = _time.time()

        self._set_status("正在解析 SVG...")
        raw_paths, raw_metas, svg_info = self.load_svg_as_ordered_paths(svg_path)
        mapped_paths = self.map_deepsketch_paths_to_image(raw_paths, svg_info, scale_info, img.shape[:2])

        self._set_status("正在压缩路径...")
        t_opt_start = _time.time()
        optimized_paths, optimized_metas, stats = self.optimize_deepsketch_paths(
            mapped_paths, raw_metas, img.shape[:2])
        t_opt_end = _time.time()

        stats["elapsed_deepsketch_sec"] = round(t_ds_end - t_ds_start, 2)
        stats["elapsed_optimize_sec"] = round(t_opt_end - t_opt_start, 2)
        stats["elapsed_total_sec"] = round(_time.time() - t_start, 2)
        stats["enabled"] = True
        stats["model_size"] = self.deepsketch_model_size
        stats["device"] = getattr(self, "deepsketch_runtime_device", self.deepsketch_device)
        stats["runtime_resize_to"] = getattr(self, "deepsketch_runtime_resize_to", self.deepsketch_max_side)
        stats["line_raw_shape"] = list(deepsketch_input.shape[:2])
        stats["svg_size"] = [svg_info.get("height", 0), svg_info.get("width", 0)]
        stats["sample_step"] = getattr(self, "deepsketch_sample_step", 2.0)
        stats["use_bezier"] = getattr(self, "deepsketch_use_bezier", True)
        stats["parser"] = svg_info.get("_parser", "unknown")
        stats["parser_error"] = svg_info.get("_parser_error", None)
        stats["raw_path_count"] = svg_info.get("_raw_path_count", 0)
        stats["raw_point_count"] = svg_info.get("_raw_point_count", 0)
        self.debug_deepsketch_stats = stats

        if self.deepsketch_cache_enabled and cache_dir is not None:
            try:
                self._save_cached_deepsketch(cache_dir, svg_path, optimized_paths, optimized_metas, stats)
            except Exception:
                pass

        self.write_deepsketch_outputs(output_dir, line_raw_path, svg_path,
                                      optimized_paths, optimized_metas, img.shape[:2])
        if getattr(self, "deepsketch_save_ordered_paths_json", False):
            self._save_ordered_paths_json(output_dir, optimized_paths, optimized_metas, svg_info, stats)
        self._save_path_quality_stats(output_dir, svg_info, stats)
        self.save_inference_config(output_dir, "quality", onnx_image is not None)
        self.cleanup_legacy_debug_outputs()
        shutil.rmtree(work_dir, ignore_errors=True)
        self._preview_image = self.preview_image_from_deepsketch_input(deepsketch_input)
        return optimized_paths, optimized_metas

    def preview_image_from_deepsketch_input(self, deepsketch_input):
        if len(deepsketch_input.shape) == 3:
            return cv2.cvtColor(deepsketch_input, cv2.COLOR_BGR2RGB)
        return deepsketch_input

    def prepare_deepsketch_input(self, deepsketch_input, work_input_dir, final_output_dir, onnx_image=None):
        h, w = deepsketch_input.shape[:2]
        max_side = max(h, w)
        scale = 1.0
        if max_side > self.deepsketch_max_side:
            scale = self.deepsketch_max_side / max_side
            resized = cv2.resize(deepsketch_input, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            resized = deepsketch_input

        if onnx_image is not None:
            cv2.imwrite(os.path.join(final_output_dir, "onnx.png"), onnx_image)

        line_raw_path = os.path.join(work_input_dir, "input.png")
        cv2.imwrite(line_raw_path, resized)

        scale_info = {
            "original_shape": [h, w],
            "input_shape": [resized.shape[0], resized.shape[1]],
            "scale": float(scale),
        }
        return line_raw_path, scale_info

    def run_deepsketch_vectorize(self, line_raw_path, output_dir):
        import subprocess
        import time
        repo = self.deepsketch_repo_path
        model_dir = self.deepsketch_model_dir

        udf_name = "udf_" + self.deepsketch_model_size + ".pth"
        ndc_name = "ndc_" + self.deepsketch_model_size + ".pth"
        udf_path = os.path.join(model_dir, udf_name)
        ndc_path = os.path.join(model_dir, ndc_name)

        if not os.path.exists(udf_path) or not os.path.exists(ndc_path):
            raise FileNotFoundError(
                f"DeepSketch model weights not found: {udf_path}, {ndc_path}")

        input_dir = os.path.dirname(line_raw_path)
        vector_dir = os.path.join(output_dir, "vector")
        os.makedirs(vector_dir, exist_ok=True)

        requested_device = getattr(self, "deepsketch_device", "auto")
        device = resolve_torch_device(requested_device)
        if str(requested_device).lower() in ("cuda", "gpu") and device != "cuda":
            self._set_status("当前 PyTorch 不支持 CUDA，DeepSketch 自动改用 CPU...", "darkorange")

        def build_cmd(run_device, resize_to):
            script_args = [
                "--input", input_dir,
                "--output", output_dir,
                "--model_udf", udf_path,
                "--model_ndc", ndc_path,
                "--device", run_device,
                "--resize_to", str(resize_to),
                "--refine",
                "--skip_vis",
            ]
            if getattr(self, "deepsketch_use_bezier", True):
                script_args.append("--bezier")
            if getattr(sys, "frozen", False):
                return [sys.executable, "--deepsketch-worker"] + script_args
            return [sys.executable, os.path.join(repo, "predict_s1.py")] + script_args

        def find_final_svg():
            svg_dir = os.path.join(output_dir, "svg_full")
            candidates = []
            if os.path.isdir(svg_dir):
                for fname in os.listdir(svg_dir):
                    if fname.endswith("_final.svg"):
                        candidates.append(os.path.join(svg_dir, fname))
            if candidates:
                candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                return candidates[0]
            return None

        def log_has_total():
            log_path = os.path.join(output_dir, "sketchvg_log.txt")
            if not os.path.exists(log_path):
                return False
            try:
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    return "total in" in f.read()
            except OSError:
                return False

        def parse_stdout_progress(line):
            """Match actual stdout keywords from predict_s1.py CLI mode."""
            line_lower = line.lower()
            if "opening" in line_lower:
                return 5
            if "resize" in line_lower or "image size" in line_lower:
                return 10
            if "work mode" in line_lower or "working mode" in line_lower:
                return 12
            # Model inference (UDF+NDC) produces no stdout — log file covers this gap
            if "surgery" in line_lower:
                return 65
            if "bfd time" in line_lower:
                return 78
            if "rdp simplify" in line_lower:
                return 84
            if "bezier" in line_lower:
                return 92
            return None

        def run_cmd_until_svg_complete(cmd, env):
            proc = subprocess.Popen(
                cmd, cwd=repo, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, errors="ignore", encoding="utf-8"
            )
            progress = 0
            log_path = os.path.join(output_dir, "sketchvg_log.txt")

            def read_stdout():
                nonlocal progress
                try:
                    for line in proc.stdout:
                        stripped = line.strip()
                        if stripped:
                            p = parse_stdout_progress(stripped)
                            if p is not None and p > progress:
                                progress = p
                                try:
                                    self._deepsketch_progress = progress
                                except Exception:
                                    pass
                except Exception:
                    pass

            def read_log_progress():
                """Poll sketchvg_log.txt — the CLI writes progress markers here
                including the model-inference phase which has no stdout output."""
                nonlocal progress
                while proc.poll() is None:
                    try:
                        if os.path.exists(log_path):
                            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                                content = f.read().lower()
                            if "vectorized" in content and progress < 55:
                                progress = 55
                                self._deepsketch_progress = 55
                            elif "usm surgery" in content and progress < 65:
                                progress = 65
                                self._deepsketch_progress = 65
                            elif "stroke grouping" in content and progress < 82:
                                progress = 82
                                self._deepsketch_progress = 82
                            elif "total in" in content and progress < 98:
                                progress = 98
                                self._deepsketch_progress = 98
                    except Exception:
                        pass
                    time.sleep(2)

            reader = threading.Thread(target=read_stdout, daemon=True)
            reader.start()
            log_reader = threading.Thread(target=read_log_progress, daemon=True)
            log_reader.start()

            try:
                while True:
                    rc = proc.poll()
                    completed_svg = find_final_svg()
                    if completed_svg and log_has_total():
                        if rc is None:
                            try:
                                proc.terminate()
                                proc.wait(timeout=3)
                            except Exception:
                                try:
                                    proc.kill()
                                except Exception:
                                    pass
                        self._deepsketch_progress = 100
                        return completed_svg
                    if rc is not None:
                        if rc != 0:
                            raise subprocess.CalledProcessError(rc, cmd)
                        self._deepsketch_progress = 100
                        return completed_svg
                    try:
                        self.root.update()
                    except Exception:
                        pass
                    time.sleep(0.5)
            finally:
                try:
                    self._deepsketch_progress = 100
                except Exception:
                    pass

        resize_attempts = self._get_deepsketch_resize_attempts()
        attempts = []
        if device == "cuda":
            for resize_to in resize_attempts:
                attempts.append(("cuda", resize_to, False))
                attempts.append(("cuda", resize_to, True))
            cpu_side = resize_attempts[-1]
            plan = getattr(self, "_precision_plan", None)
            if plan is not None and plan.cpu_max_side > cpu_side:
                cpu_side = plan.cpu_max_side
            attempts.append(("cpu", cpu_side, False))
        else:
            attempts.append((device, resize_attempts[0], False))

        last_error = None
        for run_device, resize_to, disable_cudnn in attempts:
            if run_device == "cuda":
                self._clear_cuda_cache()
            env = os.environ.copy()
            if run_device == "cuda":
                env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
            if disable_cudnn:
                env["DEEPSKETCH_DISABLE_CUDNN"] = "1"
                self._set_status("CUDA/cuDNN 失败，正在禁用 cuDNN 重试...", "darkorange")
            elif run_device == "cpu" and device == "cuda":
                self._set_status("CUDA 重试失败，正在降级 CPU 运行 DeepSketch...", "darkorange")
            if run_device == "cuda" and resize_to != resize_attempts[0] and not disable_cudnn:
                self._set_status(f"DeepSketch CUDA retry with max side {resize_to}...", "darkorange")
            try:
                svg_path = run_cmd_until_svg_complete(build_cmd(run_device, resize_to), env)
                self.deepsketch_runtime_device = run_device + ("_no_cudnn" if disable_cudnn else "")
                self.deepsketch_runtime_resize_to = resize_to
                break
            except subprocess.CalledProcessError as exc:
                last_error = exc
        else:
            raise last_error

        if svg_path:
            return svg_path

        raw_svg = os.path.join(output_dir, "svg_full", "sample_final.svg")
        if not os.path.exists(raw_svg):
            raw_fname = os.path.splitext(os.path.basename(line_raw_path))[0]
            alt_raw = os.path.join(output_dir, "svg_full", raw_fname + "_final.svg")
            if os.path.exists(alt_raw):
                return alt_raw
            alt_raw2 = os.path.join(output_dir, "svg_full", raw_fname + "_refine.svg")
            if os.path.exists(alt_raw2):
                return alt_raw2
        return raw_svg

    def load_svg_as_ordered_paths(self, svg_path):
        import xml.etree.ElementTree as ET
        paths = []
        metas = []
        svg_info = {"width": 512, "height": 512, "_parser": "none"}

        if not os.path.exists(svg_path):
            svg_info["_parser_error"] = "SVG file not found"
            return paths, metas, svg_info

        try:
            tree = ET.parse(svg_path)
            root = tree.getroot()
        except Exception as e:
            svg_info["_parser_error"] = f"XML parse failed: {e}"
            return paths, metas, svg_info

        w_str = root.get("width", "")
        h_str = root.get("height", "")
        for val, key in [(w_str, "width"), (h_str, "height")]:
            if val:
                try:
                    num = val.replace("px", "").replace("pt", "").strip()
                    svg_info[key] = float(num)
                except ValueError:
                    pass

        viewbox = root.get("viewBox", "")
        if viewbox:
            parts = viewbox.strip().split()
            if len(parts) >= 4:
                try:
                    svg_info["width"] = float(parts[2])
                    svg_info["height"] = float(parts[3])
                except ValueError:
                    pass

        sampled = self._sample_svg_paths_with_svgpathtools(svg_path, svg_info)
        if sampled is not None:
            return sampled

        # Fallback: hand-written parser — NOT high-quality, only for diagnostics
        ns = {"svg": "http://www.w3.org/2000/svg"}
        path_elems = root.findall(".//svg:path", ns)
        if not path_elems:
            path_elems = root.findall(".//{http://www.w3.org/2000/svg}path")
        if not path_elems:
            path_elems = root.findall(".//path")

        for elem in path_elems:
            d = elem.get("d", "")
            if not d:
                continue
            points = self._parse_svg_path_d(d)
            if len(points) < 2:
                continue
            paths.append(points)
            metas.append({
                "kind": "line",
                "level": "detail",
                "source": "deepsketch_svg_fallback",
            })

        svg_info["_parser"] = "regex_fallback"
        svg_info["_raw_path_count"] = len(paths)
        svg_info["_raw_point_count"] = sum(len(p) for p in paths)
        return paths, metas, svg_info

    def _sample_svg_paths_with_svgpathtools(self, svg_path, svg_info):
        try:
            from svgpathtools import svg2paths2
        except ImportError:
            svg_info["_parser_error"] = "svgpathtools not installed"
            return None

        try:
            svg_paths, attrs, svg_attrs = svg2paths2(svg_path)
        except Exception as e:
            svg_info["_parser_error"] = f"svgpathtools svg2paths2 failed: {e}"
            return None

        self._merge_svg_attrs_into_info(svg_attrs, svg_info)
        sample_step = max(0.25, float(getattr(self, "deepsketch_sample_step", 2.0)))
        paths = []
        metas = []
        raw_total_points = 0

        for idx, svg_path_obj in enumerate(svg_paths):
            try:
                length = float(svg_path_obj.length(error=1e-4))
            except Exception:
                length = 0.0
            if length <= 0:
                continue

            count = max(2, int(math.ceil(length / sample_step)) + 1)
            points = []
            for i in range(count):
                t = i / (count - 1)
                try:
                    point = svg_path_obj.point(t)
                except Exception:
                    continue
                pt = [float(point.real), float(point.imag)]
                if not points or math.hypot(pt[0] - points[-1][0], pt[1] - points[-1][1]) > 1e-6:
                    points.append(pt)

            if len(points) >= 2:
                raw_total_points += len(points)
                attr = attrs[idx] if idx < len(attrs) else {}
                paths.append(points)
                metas.append({
                    "kind": "line",
                    "level": "detail",
                    "source": "deepsketch_svgpathtools",
                    "svg_index": idx,
                    "svg_fill": attr.get("fill", ""),
                    "svg_stroke": attr.get("stroke", ""),
                    "length": length,
                })

        svg_info["_parser"] = "svgpathtools"
        svg_info["_raw_path_count"] = len(paths)
        svg_info["_raw_point_count"] = raw_total_points
        return paths, metas, svg_info

    def _merge_svg_attrs_into_info(self, svg_attrs, svg_info):
        def parse_len(value):
            import re
            match = re.search(r"-?\d+(?:\.\d+)?", str(value))
            return float(match.group(0)) if match else None

        viewbox = svg_attrs.get("viewBox") or svg_attrs.get("viewbox") if svg_attrs else None
        if viewbox:
            parts = str(viewbox).replace(",", " ").split()
            if len(parts) >= 4:
                try:
                    svg_info["width"] = float(parts[2])
                    svg_info["height"] = float(parts[3])
                    return
                except ValueError:
                    pass

        if svg_attrs:
            width = parse_len(svg_attrs.get("width", ""))
            height = parse_len(svg_attrs.get("height", ""))
            if width and width > 0:
                svg_info["width"] = width
            if height and height > 0:
                svg_info["height"] = height

    def _parse_svg_path_d(self, d_str):
        import re
        points = []
        tokens = re.findall(r"[a-zA-Z]+|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", d_str)
        i = 0
        current_cmd = None
        x, y = 0.0, 0.0
        start_x, start_y = 0.0, 0.0

        def take_float():
            nonlocal i
            if i < len(tokens):
                try:
                    v = float(tokens[i])
                    i += 1
                    return v
                except ValueError:
                    pass
            return None

        while i < len(tokens):
            tok = tokens[i]
            if tok.isalpha():
                current_cmd = tok.upper()
                i += 1
                continue
            if current_cmd is None:
                i += 1
                continue

            if current_cmd == "M":
                vx = take_float()
                vy = take_float()
                if vx is not None and vy is not None:
                    x, y = vx, vy
                    start_x, start_y = x, y
                    points.append([round(x), round(y)])
                current_cmd = "L"
            elif current_cmd == "m":
                vx = take_float()
                vy = take_float()
                if vx is not None and vy is not None:
                    x += vx
                    y += vy
                    start_x, start_y = x, y
                    points.append([round(x), round(y)])
                current_cmd = "l"
            elif current_cmd == "L":
                vx = take_float()
                vy = take_float()
                if vx is not None and vy is not None:
                    x, y = vx, vy
                    points.append([round(x), round(y)])
            elif current_cmd == "l":
                vx = take_float()
                vy = take_float()
                if vx is not None and vy is not None:
                    x += vx
                    y += vy
                    points.append([round(x), round(y)])
            elif current_cmd == "C":
                for _ in range(3):
                    take_float()
                    take_float()
                vx = take_float()
                vy = take_float()
                if vx is not None and vy is not None:
                    x, y = vx, vy
                    points.append([round(x), round(y)])
            elif current_cmd == "c":
                for _ in range(3):
                    take_float()
                    take_float()
                vx = take_float()
                vy = take_float()
                if vx is not None and vy is not None:
                    x += vx
                    y += vy
                    points.append([round(x), round(y)])
            elif current_cmd in ("Z", "z"):
                x, y = start_x, start_y
                if points and (points[-1][0] != round(x) or points[-1][1] != round(y)):
                    points.append([round(x), round(y)])
                i += 1
                continue
            else:
                i += 1
                continue

        return points

    def map_deepsketch_paths_to_image(self, paths, svg_info, scale_info, image_shape):
        svg_w = svg_info.get("width", 512)
        svg_h = svg_info.get("height", 512)
        input_h, input_w = scale_info["input_shape"]
        orig_h, orig_w = scale_info["original_shape"]

        if svg_w <= 0 or svg_h <= 0:
            sx = sy = 1.0
        else:
            sx = input_w / svg_w
            sy = input_h / svg_h

        scale = scale_info["scale"]
        if scale < 1.0:
            sx_svg_to_orig = sx / scale
            sy_svg_to_orig = sy / scale
        else:
            sx_svg_to_orig = sx
            sy_svg_to_orig = sy

        mapped = []
        for path in paths:
            mp = [[max(0.0, min(float(orig_w - 1), x * sx_svg_to_orig)),
                   max(0.0, min(float(orig_h - 1), y * sy_svg_to_orig))]
                  for x, y in path]
            mapped.append(mp)
        return mapped

    def optimize_deepsketch_paths(self, paths, metas, image_shape):
        stats = {
            "parser": "unknown",
            "raw_paths": len(paths),
            "raw_points": sum(len(p) for p in paths),
            "target_points": self.deepsketch_target_points,
            "max_points": self.deepsketch_max_points,
            "max_paths": self.deepsketch_max_paths,
            "removed_short_paths": 0,
            "removed_by_path_limit": 0,
            "out_of_bounds_points": 0,
            "rdp_epsilon_final": self.deepsketch_simplify_epsilon,
        }

        h, w = image_shape
        filtered = []
        filtered_metas = []
        removed_short = 0

        for path, meta in zip(paths, metas):
            # clamp float coords to image bounds
            clamped = []
            oob = 0
            for px, py in path:
                cx = max(0.0, min(float(w - 1), float(px)))
                cy = max(0.0, min(float(h - 1), float(py)))
                if abs(cx - px) > 1e-4 or abs(cy - py) > 1e-4:
                    oob += 1
                clamped.append([cx, cy])
            stats["out_of_bounds_points"] += oob

            # remove consecutive duplicates (float-safe)
            dedup = [clamped[0]]
            for pt in clamped[1:]:
                if math.hypot(pt[0] - dedup[-1][0], pt[1] - dedup[-1][1]) > 1e-4:
                    dedup.append(pt)

            if len(dedup) < 2:
                removed_short += 1
                continue

            plen = self._path_length(dedup)
            if plen < self.deepsketch_min_path_length:
                removed_short += 1
                continue

            # point spacing filter
            if self.deepsketch_min_point_spacing > 0:
                spaced = [dedup[0]]
                for pt in dedup[1:-1]:
                    if math.hypot(pt[0] - spaced[-1][0], pt[1] - spaced[-1][1]) >= self.deepsketch_min_point_spacing:
                        spaced.append(pt)
                spaced.append(dedup[-1])
            else:
                spaced = dedup

            if len(spaced) < 2:
                removed_short += 1
                continue

            filtered.append(spaced)
            filtered_metas.append(meta)

        stats["removed_short_paths"] = removed_short

        # RDP simplify — cv2.approxPolyDP needs int32, convert internally
        base_epsilon = self.deepsketch_simplify_epsilon
        paths_rdp = [self._rdp_simplify_path(p, base_epsilon) for p in filtered]
        total_pts = sum(len(p) for p in paths_rdp)

        epsilon = base_epsilon
        for _round in range(5):
            if total_pts <= self.deepsketch_max_points:
                break
            epsilon *= 1.35
            paths_rdp = [self._rdp_simplify_path(p, epsilon) for p in filtered]
            total_pts = sum(len(p) for p in paths_rdp)

        stats["rdp_epsilon_final"] = round(epsilon, 2)

        # path scoring and sorting
        scored = []
        for i, (path, meta) in enumerate(zip(paths_rdp, filtered_metas)):
            plen = self._path_length(path)
            curvature = self._estimate_curvature(path)
            score = plen + curvature * 8.0
            scored.append((path, meta, score, plen))

        scored.sort(key=lambda x: x[2], reverse=True)

        # classify by length percentile
        all_lengths = sorted([s[3] for s in scored], reverse=True)
        p15_idx = max(0, int(len(all_lengths) * 0.15) - 1)
        p15_threshold = all_lengths[p15_idx] if all_lengths else 0

        optimized_paths = []
        optimized_metas = []
        for path, meta, score, plen in scored:
            if len(optimized_paths) >= self.deepsketch_max_paths:
                stats["removed_by_path_limit"] += 1
                continue

            if plen >= p15_threshold:
                level = "outline"
            elif plen < 15 and self._estimate_curvature(path) > 0.4:
                level = "detail"
            else:
                level = "normal"

            optimized_paths.append(path)
            optimized_metas.append({
                "kind": "line",
                "level": level,
                "source": meta.get("source", "deepsketch_svg"),
                "length": plen,
                "curvature": self._estimate_curvature(path),
            })

        stats["optimized_paths"] = len(optimized_paths)
        stats["optimized_points"] = sum(len(p) for p in optimized_paths)

        return optimized_paths, optimized_metas, stats

    def _rdp_simplify_path(self, path, epsilon):
        if len(path) < 3:
            return path
        contour = np.array([[[p[0], p[1]]] for p in path], dtype=np.float32)
        simplified = cv2.approxPolyDP(contour, epsilon, False)
        return [[float(p[0][0]), float(p[0][1])] for p in simplified]

    def write_deepsketch_outputs(self, output_dir, line_raw_path, svg_path,
                                 optimized_paths, optimized_metas, image_shape):
        import shutil as _shutil

        svg_out = os.path.join(output_dir, "deepsketch.svg")
        if os.path.exists(svg_path):
            try:
                _shutil.copy2(svg_path, svg_out)
            except Exception:
                pass

        h, w = image_shape
        final_img = np.zeros((h, w, 3), dtype=np.uint8)
        colors = {"outline": (0, 255, 0), "normal": (255, 255, 0), "detail": (0, 0, 255)}
        for path, meta in zip(optimized_paths, optimized_metas):
            level = meta.get("level", "normal")
            color = colors.get(level, (255, 255, 255))
            for i in range(1, len(path)):
                pt1 = (int(round(path[i-1][0])), int(round(path[i-1][1])))
                pt2 = (int(round(path[i][0])), int(round(path[i][1])))
                cv2.line(final_img, pt1, pt2, color, 1)
        cv2.imwrite(os.path.join(output_dir, "final_paths.png"), final_img)

    def _save_ordered_paths_json(self, output_dir, paths, metas, svg_info, stats):
        export = {
            "version": 14,
            "source_svg": "deepsketch.svg",
            "parser": svg_info.get("_parser", "unknown"),
            "sample_step": getattr(self, "deepsketch_sample_step", 2.0),
            "svg_info": {
                "width": svg_info.get("width", 0),
                "height": svg_info.get("height", 0),
                "viewBox": svg_info.get("viewBox", []),
            },
            "raw_path_count": svg_info.get("_raw_path_count", stats.get("raw_paths", len(paths))),
            "raw_point_count": svg_info.get("_raw_point_count", stats.get("raw_points", sum(len(p) for p in paths))),
            "optimized_path_count": stats.get("optimized_paths", len(paths)),
            "optimized_point_count": stats.get("optimized_points", sum(len(p) for p in paths)),
            "paths": [],
        }
        for path, meta in zip(paths, metas):
            export["paths"].append({
                "points": path,
                "length": meta.get("length", 0),
                "level": meta.get("level", "detail"),
                "source": meta.get("source", ""),
                "svg_index": meta.get("svg_index", 0),
            })
        json_path = os.path.join(output_dir, "ordered_paths_high_quality.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(export, f, indent=2, ensure_ascii=False)

    def _save_path_quality_stats(self, output_dir, svg_info, stats):
        delay_var = getattr(self, "delay_var", None)
        if hasattr(delay_var, "get"):
            delay = delay_var.get()
        else:
            delay = delay_var
        quality_stats = {
            "version": 14,
            "parser": svg_info.get("_parser", "unknown"),
            "parser_error": svg_info.get("_parser_error", None),
            "use_bezier": getattr(self, "deepsketch_use_bezier", True),
            "sample_step": getattr(self, "deepsketch_sample_step", 2.0),
            "raw_path_count": svg_info.get("_raw_path_count", stats.get("raw_paths", 0)),
            "raw_point_count": svg_info.get("_raw_point_count", stats.get("raw_points", 0)),
            "optimized_path_count": stats.get("optimized_paths", 0),
            "optimized_point_count": stats.get("optimized_points", 0),
            "rdp_epsilon_final": stats.get("rdp_epsilon_final", 0),
            "recommended_drag_step": getattr(self, "deepsketch_quality_draw_step", 1),
            "delay": delay,
        }
        stats_path = os.path.join(output_dir, "path_quality_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(quality_stats, f, indent=2, ensure_ascii=False)

    def save_inference_config(self, output_dir, mode, onnx_enabled, extra_stats=None):
        import datetime
        config = {
            "version": 14,
            "image_name": os.path.splitext(os.path.basename(self.image_path))[0] if self.image_path else "",
            "image_path": self.image_path or "",
            "mode": mode,
            "onnx_enabled": onnx_enabled,
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "onnx_lineart": {
                "max_side": self.onnx_lineart_max_side,
                "enhance": self.onnx_lineart_enhance,
                "clean_mode": getattr(self, "onnx_clean_mode", "balanced"),
            },
            "deepsketch": {
                "precision": getattr(self, "deepsketch_precision", "medium"),
                "model_size": self.deepsketch_model_size,
                "device": self.deepsketch_device,
                "max_side": self.deepsketch_max_side,
                "auto_reduce_on_oom": self.deepsketch_auto_reduce_on_oom,
                "oom_retry_sides": self.deepsketch_oom_retry_sides,
                "target_points": self.deepsketch_target_points,
                "max_points": self.deepsketch_max_points,
                "max_paths": self.deepsketch_max_paths,
                "min_path_length": self.deepsketch_min_path_length,
                "min_point_spacing": self.deepsketch_min_point_spacing,
                "simplify_epsilon": self.deepsketch_simplify_epsilon,
                "sample_step": getattr(self, "deepsketch_sample_step", 2.0),
                "use_bezier": getattr(self, "deepsketch_use_bezier", True),
                "save_ordered_paths_json": getattr(self, "deepsketch_save_ordered_paths_json", True),
                "quality_draw_step": getattr(self, "deepsketch_quality_draw_step", 1),
                "cache_enabled": self.deepsketch_cache_enabled,
                "runtime_resize_to": getattr(self, "deepsketch_runtime_resize_to", self.deepsketch_max_side),
                "runtime_device": getattr(self, "deepsketch_runtime_device", self.deepsketch_device),
            },
            "outputs": {
                "onnx": "onnx.png" if onnx_enabled else None,
                "svg": "deepsketch.svg" if mode == "quality" else None,
                "final_paths": "final_paths.png",
            },
        }
        config_path = os.path.join(output_dir, "inference_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def load_inference_config_for_current_image(self, mode, onnx_enabled):
        if not self.image_path:
            return None
        image_name = os.path.splitext(os.path.basename(self.image_path))[0]
        safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in image_name).strip("_")
        if not safe_name:
            return None
        if onnx_enabled:
            suffix = f"onnx_{mode}"
        else:
            suffix = mode
        dir_name = f"{safe_name}_{suffix}"
        if getattr(sys, "frozen", False):
            base_dir = os.path.dirname(sys.executable)
        else:
            base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        config_path = os.path.join(base_dir, "output", dir_name, "inference_config.json")
        if not os.path.exists(config_path):
            return None
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def apply_inference_config(self, config):
        if not config or config.get("version", 0) < 13:
            return False
        onnx_cfg = config.get("onnx_lineart", {})
        if onnx_cfg:
            if "max_side" in onnx_cfg:
                self.onnx_lineart_max_side = onnx_cfg["max_side"]
            if "enhance" in onnx_cfg:
                self.onnx_lineart_enhance = onnx_cfg["enhance"]
            if "clean_mode" in onnx_cfg:
                self.onnx_clean_mode = onnx_cfg["clean_mode"]
        ds_cfg = config.get("deepsketch", {})
        if ds_cfg:
            if "precision" in ds_cfg:
                self.deepsketch_precision = ds_cfg["precision"]
                if hasattr(self, "deepsketch_precision_var"):
                    self.deepsketch_precision_var.set(ds_cfg["precision"])
            if "model_size" in ds_cfg:
                self.deepsketch_model_size = ds_cfg["model_size"]
            if "device" in ds_cfg:
                self.deepsketch_device = ds_cfg["device"]
            if "max_side" in ds_cfg:
                self.deepsketch_max_side = ds_cfg["max_side"]
            if "auto_reduce_on_oom" in ds_cfg:
                self.deepsketch_auto_reduce_on_oom = ds_cfg["auto_reduce_on_oom"]
            if "oom_retry_sides" in ds_cfg:
                self.deepsketch_oom_retry_sides = ds_cfg["oom_retry_sides"]
            if "target_points" in ds_cfg:
                self.deepsketch_target_points = ds_cfg["target_points"]
            if "max_points" in ds_cfg:
                self.deepsketch_max_points = ds_cfg["max_points"]
            if "max_paths" in ds_cfg:
                self.deepsketch_max_paths = ds_cfg["max_paths"]
            if "min_path_length" in ds_cfg:
                self.deepsketch_min_path_length = ds_cfg["min_path_length"]
            if "min_point_spacing" in ds_cfg:
                self.deepsketch_min_point_spacing = ds_cfg["min_point_spacing"]
            if "simplify_epsilon" in ds_cfg:
                self.deepsketch_simplify_epsilon = ds_cfg["simplify_epsilon"]
            if "sample_step" in ds_cfg:
                self.deepsketch_sample_step = ds_cfg["sample_step"]
            if "use_bezier" in ds_cfg:
                self.deepsketch_use_bezier = ds_cfg["use_bezier"]
            if "cache_enabled" in ds_cfg:
                self.deepsketch_cache_enabled = ds_cfg["cache_enabled"]
            if "save_ordered_paths_json" in ds_cfg:
                self.deepsketch_save_ordered_paths_json = ds_cfg["save_ordered_paths_json"]
            if "quality_draw_step" in ds_cfg:
                self.deepsketch_quality_draw_step = ds_cfg["quality_draw_step"]
        return True
