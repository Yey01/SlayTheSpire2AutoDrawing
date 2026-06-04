# -*- coding: utf-8 -*-

import json
import os

import cv2
import numpy as np


class LineartPipelineMixin:
    def load_reusable_onnx_lineart(self, image_shape, preferred_mode):
        """Reuse an existing ONNX line-art image for the current source image."""
        if not self.image_path:
            return None, None

        modes = []
        if preferred_mode:
            modes.append(preferred_mode)
        for mode in ("quality", "fast"):
            if mode not in modes:
                modes.append(mode)

        target_h, target_w = image_shape[:2]
        for mode in modes:
            output_dir = self.get_image_output_dir(mode, onnx=True)
            onnx_path = os.path.join(output_dir, "onnx.png")
            if not os.path.exists(onnx_path):
                continue
            lineart = cv2.imread(onnx_path, cv2.IMREAD_GRAYSCALE)
            if lineart is None:
                continue
            if lineart.shape[:2] != (target_h, target_w):
                lineart = cv2.resize(lineart, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            return lineart, onnx_path

        return None, None

    def get_or_create_onnx_lineart(self, img, preferred_mode):
        lineart_gray, onnx_path = self.load_reusable_onnx_lineart(img.shape[:2], preferred_mode)
        if lineart_gray is not None:
            source_dir = os.path.basename(os.path.dirname(onnx_path))
            self._set_status(f"Reusing ONNX line_raw: {source_dir}", "green")
            return lineart_gray

        self._set_status("Generating ONNX line_raw...")
        return self.convert_color_to_onnx_lineart(img)

    def _process_lineart_image_paths(self, img):
        """Existing line-art pipeline."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self._process_lineart_gray_paths(gray, img.shape[:2], source="skeleton")

    def write_fast_outputs(self, onnx_image, skeleton, ordered_paths, ordered_metas, image_shape):
        output_dir = self.get_image_output_dir("fast", onnx=onnx_image is not None)
        self.clear_image_output_dir(output_dir)

        if onnx_image is not None:
            cv2.imwrite(os.path.join(output_dir, "onnx.png"), onnx_image)
        cv2.imwrite(os.path.join(output_dir, "skeleton.png"), 255 - skeleton)

        h, w = image_shape
        final_img = np.zeros((h, w, 3), dtype=np.uint8)
        colors = {"outline": (0, 255, 0), "normal": (255, 255, 0),
                  "fill": (255, 0, 0), "detail": (0, 0, 255),
                  "cover": (0, 255, 255)}
        for path, meta in zip(ordered_paths, ordered_metas):
            level = meta.get("level", "normal") if isinstance(meta, dict) else meta
            kind = meta.get("kind", "line") if isinstance(meta, dict) else "line"
            color = colors["cover"] if kind == "cover" else colors.get(level, (255, 255, 255))
            for i in range(1, len(path)):
                cv2.line(final_img, path[i-1], path[i], color, 1)
        cv2.imwrite(os.path.join(output_dir, "final_paths.png"), final_img)
        return output_dir

    def _process_lineart_gray_paths(self, gray, image_shape, source="skeleton",
                                     preset_ink_mask=None, line_raw=None,
                                     onnx_clean_strategy=None):
        """Shared gray-lineart to paths pipeline."""
        line_art_gray = gray
        is_onnx = (source == "onnx_lineart")

        if preset_ink_mask is not None:
            ink_mask = preset_ink_mask
        else:
            ink_mask = self.build_ink_mask(line_art_gray)

        line_mask, fill_mask = self.split_line_and_fill_masks(gray, ink_mask)

        clean_mask = self.preprocess_line_mask(line_mask)
        skeleton = self.skeletonize_mask(clean_mask)
        self._preview_image = skeleton
        raw_paths = self.trace_skeleton_paths(skeleton)

        if is_onnx and len(raw_paths) > 1000:
            raw_paths = [p for p in raw_paths if self._path_length(p) >= 8]

        bridged_paths = self.bridge_small_gaps(raw_paths)

        if is_onnx:
            density_map = self.build_path_density_map(bridged_paths, image_shape)
            merged_paths = self.merge_onnx_paths_grid(bridged_paths, image_shape, density_map)
        else:
            pre_density_map = self.build_path_density_map(bridged_paths, image_shape)
            merged_paths = self.merge_nearby_paths(bridged_paths, pre_density_map)

        simplified_paths = [self.simplify_path_adaptive(p) for p in merged_paths]

        density_map = self.build_path_density_map(simplified_paths, image_shape)

        min_len = self.min_len_var.get()
        line_paths = []
        line_metas = []
        for p in simplified_paths:
            if len(p) < max(2, min_len // 5):
                continue
            level = self.classify_path_v2(p, density_map)
            line_paths.append(p)
            line_metas.append({
                "kind": "line", "level": level,
                "length": self._path_length(p),
                "curvature": self._estimate_curvature(p),
                "source": source
            })

        thin_mask = None
        thick_mask = None
        dist_map = None
        cover_paths = []
        cover_metas = []

        if is_onnx and self.should_enable_onnx_cover(onnx_clean_strategy, line_raw):
            thin_mask, thick_mask, dist_map = self.split_thin_thick_onnx_mask(ink_mask)
            cover_paths, cover_metas = self.generate_onnx_stroke_cover_paths(thick_mask, line_raw)
            line_paths += cover_paths
            line_metas += cover_metas

        if is_onnx:
            fill_paths, fill_metas = [], []
            fill_mask = np.zeros_like(ink_mask)
        else:
            fill_regions = self.detect_fill_regions(gray, fill_mask)
            fill_paths, fill_metas = self.generate_fill_paths(fill_regions, gray)

        if not line_paths and not fill_paths:
            return [], []

        ordered_paths, ordered_metas = self.order_draw_tasks(
            line_paths, line_metas, fill_paths, fill_metas
        )
        output_dir = self.write_fast_outputs(line_raw if is_onnx else None, skeleton,
                                             ordered_paths, ordered_metas, image_shape)
        self.save_inference_config(output_dir, "fast", is_onnx)
        self.cleanup_legacy_debug_outputs()

        if self.debug_output:
            debug_dir = self.get_debug_output_dir("fast", onnx=is_onnx)
            cv2.imwrite(os.path.join(debug_dir, "debug_mask.png"), ink_mask)
            cv2.imwrite(os.path.join(debug_dir, "debug_line_mask.png"), line_mask)
            cv2.imwrite(os.path.join(debug_dir, "debug_fill_mask.png"), fill_mask)
            cv2.imwrite(os.path.join(debug_dir, "debug_clean.png"), clean_mask)
            cv2.imwrite(os.path.join(debug_dir, "debug_skeleton.png"), skeleton)

            debug_merged = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
            for path in merged_paths:
                for i in range(1, len(path)):
                    cv2.line(debug_merged, path[i-1], path[i], (255, 255, 255), 1)
            cv2.imwrite(os.path.join(debug_dir, "debug_merged_paths.png"), debug_merged)

            if hasattr(self, "debug_bridge_segments") and self.debug_bridge_segments:
                debug_bridges = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
                for seg in self.debug_bridge_segments:
                    for i in range(1, len(seg)):
                        cv2.line(debug_bridges, seg[i-1], seg[i], (255, 0, 255), 1)
                cv2.imwrite(os.path.join(debug_dir, "debug_bridges.png"), debug_bridges)

            if is_onnx and self.debug_onnx_merge_stats is not None:
                import json as _json
                debug_onnx_merge_img = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
                for path in merged_paths:
                    for i in range(1, len(path)):
                        cv2.line(debug_onnx_merge_img, path[i-1], path[i], (255, 255, 255), 1)
                cv2.imwrite(os.path.join(debug_dir, "debug_onnx_merge.png"), debug_onnx_merge_img)
                self.debug_onnx_merge_stats["output_paths"] = len(merged_paths)
                import time as _time
                self.debug_onnx_merge_stats["timestamp"] = _time.strftime("%Y-%m-%d %H:%M:%S")
                with open(os.path.join(debug_dir, "debug_onnx_merge_stats.json"), "w") as f:
                    _json.dump(self.debug_onnx_merge_stats, f, indent=2, ensure_ascii=False)

            if is_onnx and thin_mask is not None:
                cv2.imwrite(os.path.join(debug_dir, "debug_onnx_thin_mask.png"), thin_mask)
                cv2.imwrite(os.path.join(debug_dir, "debug_onnx_thick_mask.png"), thick_mask)
                dist_viz = np.zeros_like(dist_map)
                if dist_map.max() > 0:
                    dist_viz = (dist_map / dist_map.max() * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(debug_dir, "debug_onnx_distance.png"), dist_viz)

                debug_cover = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
                for path in cover_paths:
                    for i in range(1, len(path)):
                        cv2.line(debug_cover, path[i-1], path[i], (0, 255, 0), 1)
                cv2.imwrite(os.path.join(debug_dir, "debug_onnx_cover_paths.png"), debug_cover)

                if self.debug_onnx_cover_stats is not None:
                    with open(os.path.join(debug_dir, "debug_onnx_cover_stats.json"), "w") as f:
                        _json.dump(self.debug_onnx_cover_stats, f, indent=2, ensure_ascii=False)

            if is_onnx and self.debug_onnx_cover_decision is not None:
                import json as _json2
                with open(os.path.join(debug_dir, "debug_onnx_cover_decision.json"), "w") as f:
                    _json2.dump(self.debug_onnx_cover_decision, f, indent=2, ensure_ascii=False)

            debug_fill_regions_img = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
            if not is_onnx:
                for region in fill_regions:
                    mask = region["mask"]
                    ys, xs = np.where(mask > 0)
                    for x, y in zip(xs, ys):
                        if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
                            debug_fill_regions_img[y, x] = (255, 0, 0)
            cv2.imwrite(os.path.join(debug_dir, "debug_fill_regions.png"), debug_fill_regions_img)

            debug_final = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
            colors = {"outline": (0, 255, 0), "normal": (255, 255, 0),
                      "fill": (255, 0, 0), "detail": (0, 0, 255),
                      "cover": (0, 255, 255)}
            for path, meta in zip(ordered_paths, ordered_metas):
                level = meta.get("level", "normal") if isinstance(meta, dict) else meta
                kind = meta.get("kind", "line") if isinstance(meta, dict) else "line"
                if kind == "cover":
                    color = colors["cover"]
                else:
                    color = colors.get(level, (255, 255, 255))
                for i in range(1, len(path)):
                    cv2.line(debug_final, path[i-1], path[i], color, 1)
            cv2.imwrite(os.path.join(debug_dir, "debug_final_paths_v2.png"), debug_final)

        return ordered_paths, ordered_metas

    def process_color_image_paths(self, img):
        """ONNX-based color image pipeline: ONNX line-art -> skeletonize path tracing."""
        self._set_status("正在生成 ONNX line_raw...")
        lineart_gray = self.get_or_create_onnx_lineart(img, "fast")

        if self.color_path_backend == "deepsketch_full" or self.deepsketch_enabled:
            return self.process_color_image_paths_deepsketch(img, lineart_gray, onnx_image=lineart_gray)

        self._set_status("正在处理线稿 clean + skeleton...")
        clean_gray, clean_mask, strategy = self.prepare_onnx_lineart_mask(lineart_gray)

        if self.debug_output:
            debug_dir = self.get_debug_output_dir("fast", onnx=True)
            cv2.imwrite(os.path.join(debug_dir, "debug_onnx_lineart_raw.png"), lineart_gray)
            cv2.imwrite(os.path.join(debug_dir, "debug_onnx_lineart_clean.png"), clean_gray)
            cv2.imwrite(os.path.join(debug_dir, "debug_onnx_clean_mask.png"), clean_mask)
            import json
            clean_params = self.get_onnx_clean_params(lineart_gray, strategy)
            stats = {
                "source": "onnx_lineart",
                "max_side": self.onnx_lineart_max_side,
                "enhance": self.onnx_lineart_enhance,
                "image_shape": list(img.shape[:2]),
                "lineart_shape": list(lineart_gray.shape[:2]),
                "clean_mode": self.onnx_clean_mode,
                "selected_strategy": strategy,
                "min_area": clean_params["min_area"],
                "close_kernel": clean_params["close_kernel"],
            }
            if "adaptive_c" in clean_params:
                stats["adaptive_c"] = clean_params["adaptive_c"]
            if self.debug_onnx_clean_strategy is not None:
                stats["clean_strategy"] = self.debug_onnx_clean_strategy
                with open(os.path.join(debug_dir, "debug_onnx_clean_strategy.json"), "w") as f:
                    json.dump(self.debug_onnx_clean_strategy, f, indent=2, ensure_ascii=False)
            with open(os.path.join(debug_dir, "debug_onnx_stats.json"), "w") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)

        self._preview_image = clean_gray
        return self._process_lineart_gray_paths(
            clean_gray, img.shape[:2], source="onnx_lineart",
            preset_ink_mask=clean_mask, line_raw=lineart_gray,
            onnx_clean_strategy=strategy,
        )

    def get_onnx_lineart_session(self):
        if self.onnx_lineart_session is not None:
            return self.onnx_lineart_session
        import onnxruntime as ort
        model_path = self.onnx_lineart_model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        self.onnx_lineart_session = ort.InferenceSession(model_path)
        return self.onnx_lineart_session

    def convert_color_to_onnx_lineart(self, img):
        session = self.get_onnx_lineart_session()
        h, w = img.shape[:2]
        max_side = max(h, w)
        scale = 1.0
        if max_side > self.onnx_lineart_max_side:
            scale = self.onnx_lineart_max_side / max_side
            resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            resized = img

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.expand_dims(rgb.transpose(2, 0, 1), axis=0)

        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: tensor})[0]
        lineart = (output[0, 0] * 255).clip(0, 255).astype(np.uint8)

        if lineart.shape[:2] != (h, w):
            lineart = cv2.resize(lineart, (w, h), interpolation=cv2.INTER_LINEAR)

        return lineart

    def get_onnx_clean_params(self, gray, strategy=None):
        if strategy is None:
            strategy = getattr(self, "onnx_clean_mode", "balanced")
        base_params = {
            "fine": {"adaptive_c": 7, "min_area": 10, "close_kernel": 1},
            "balanced": {"adaptive_c": 5, "min_area": 8, "close_kernel": 2},
            "strong": {"adaptive_c": 3, "min_area": 6, "close_kernel": 2},
        }
        if strategy in base_params:
            return base_params[strategy]
        return {"min_area": 8, "close_kernel": 2}

    def _select_onnx_clean_strategy(self, lineart_gray):
        dark_ratio = np.count_nonzero(lineart_gray < 220) / lineart_gray.size
        std_gray = float(np.std(lineart_gray))
        mode = self.onnx_clean_mode
        if mode == "auto":
            strategy = "balanced"
        elif mode in ("balanced", "fine", "strong", "otsu", "pct35", "pct40"):
            strategy = mode
        else:
            strategy = "balanced"
        return strategy, dark_ratio, std_gray

    def build_onnx_clean_mask(self, gray, strategy, params):
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        if strategy == "balanced":
            block_size = max(3, min(gray.shape) // 40 * 2 + 1)
            if block_size % 2 == 0:
                block_size += 1
            adaptive = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, block_size, params.get("adaptive_c", 5)
            )
            _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            mask = cv2.bitwise_and(adaptive, otsu)
        elif strategy == "otsu":
            _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        elif strategy.startswith("pct"):
            pct_val = 35 if "35" in strategy else 40
            threshold = np.percentile(blur, pct_val)
            _, mask = cv2.threshold(blur, int(threshold), 255, cv2.THRESH_BINARY_INV)
        else:
            _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        min_area = params.get("min_area", 8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cleaned = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cleaned[labels == i] = 255
        mask = cleaned

        ck = params.get("close_kernel", 2)
        if ck >= 2:
            kernel = np.ones((ck, ck), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return mask

    def prepare_onnx_lineart_mask(self, lineart_gray):
        strategy, dark_ratio, std_gray = self._select_onnx_clean_strategy(lineart_gray)
        params = self.get_onnx_clean_params(lineart_gray, strategy)
        clean_mask = self.build_onnx_clean_mask(lineart_gray, strategy, params)
        clean_gray = 255 - clean_mask
        self.debug_onnx_clean_strategy = {
            "mode": self.onnx_clean_mode,
            "selected_strategy": strategy,
            "dark_ratio_lt220": round(dark_ratio, 4),
            "std_gray": round(std_gray, 2),
            "min_area": params["min_area"],
            "close_kernel": params["close_kernel"],
        }
        if "adaptive_c" in params:
            self.debug_onnx_clean_strategy["adaptive_c"] = params["adaptive_c"]
        return clean_gray, clean_mask, strategy

    def prepare_onnx_lineart_for_vectorize(self, lineart_gray):
        clean_gray, _, _ = self.prepare_onnx_lineart_mask(lineart_gray)
        return clean_gray

