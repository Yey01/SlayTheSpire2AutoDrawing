# -*- coding: utf-8 -*-

import json
import os
import sys


class RuntimeConfigMixin:
    def resource_path(self, relative_path):
        if hasattr(sys, "_MEIPASS"):
            base_dir = sys._MEIPASS
        else:
            base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        return os.path.join(base_dir, relative_path)

    def load_runtime_config(self):
        defaults = {
            "backend": {"default_color_path_backend": "onnx_skeleton", "enable_deepsketch": False, "quality_backend": "deepsketch_full"},
            "onnx_lineart": {"model_path": "models/line_drawing/line-drawings.onnx", "max_side": 900, "enhance": False},
            "deepsketch": {
                "repo_path": "models/deepsketch",
                "model_dir": "models/deepsketch/model",
                "model_size": "full", "device": "auto", "max_side": 700,
                "auto_reduce_on_oom": True, "oom_retry_sides": [600, 512, 448],
                "target_points": 12000, "max_points": 18000, "max_paths": 1800,
                "min_path_length": 3.0, "min_point_spacing": 1.0, "simplify_epsilon": 0.8,
                "sample_step": 1.5, "use_bezier": True,
                "save_ordered_paths_json": True, "quality_draw_step": 1,
                "cache_enabled": True,
                "precision_mode": "auto",
                "precision": "medium",
                "precision_presets": {"low": 448, "medium": 512, "high": 600},
                "gpu_min_viable_side": 448,
                "cpu_max_time_seconds": 120.0,
            },
            "debug": {"enabled": False, "keep_last_debug": False},
        }

        try:
            import json as _json
            if os.path.exists(self.runtime_config_file):
                with open(self.runtime_config_file, "r", encoding="utf-8") as f:
                    cfg = _json.load(f)
            else:
                cfg = {}
        except Exception:
            cfg = {}

        backend_cfg = cfg.get("backend", {})
        self.color_path_backend = backend_cfg.get("default_color_path_backend", defaults["backend"]["default_color_path_backend"])
        self.deepsketch_enabled = backend_cfg.get("enable_deepsketch", defaults["backend"]["enable_deepsketch"])

        onnx_cfg = cfg.get("onnx_lineart", {})
        model_path_rel = onnx_cfg.get("model_path", defaults["onnx_lineart"]["model_path"])
        self.onnx_lineart_model_path = self.resource_path(model_path_rel)
        self.onnx_lineart_max_side = onnx_cfg.get("max_side", defaults["onnx_lineart"]["max_side"])
        self.onnx_lineart_enhance = onnx_cfg.get("enhance", defaults["onnx_lineart"]["enhance"])

        ds_cfg = cfg.get("deepsketch", {})
        self.deepsketch_repo_path = self.resource_path(ds_cfg.get("repo_path", defaults["deepsketch"]["repo_path"]))
        self.deepsketch_model_dir = self.resource_path(ds_cfg.get("model_dir", defaults["deepsketch"]["model_dir"]))
        self.deepsketch_model_size = ds_cfg.get("model_size", defaults["deepsketch"]["model_size"])
        self.deepsketch_device = ds_cfg.get("device", defaults["deepsketch"]["device"])
        self.deepsketch_max_side = ds_cfg.get("max_side", defaults["deepsketch"]["max_side"])
        self.deepsketch_auto_reduce_on_oom = ds_cfg.get("auto_reduce_on_oom", defaults["deepsketch"]["auto_reduce_on_oom"])
        self.deepsketch_oom_retry_sides = ds_cfg.get("oom_retry_sides", defaults["deepsketch"]["oom_retry_sides"])
        self.deepsketch_target_points = ds_cfg.get("target_points", defaults["deepsketch"]["target_points"])
        self.deepsketch_max_points = ds_cfg.get("max_points", defaults["deepsketch"]["max_points"])
        self.deepsketch_max_paths = ds_cfg.get("max_paths", defaults["deepsketch"]["max_paths"])
        self.deepsketch_min_path_length = ds_cfg.get("min_path_length", defaults["deepsketch"]["min_path_length"])
        self.deepsketch_min_point_spacing = ds_cfg.get("min_point_spacing", defaults["deepsketch"]["min_point_spacing"])
        self.deepsketch_simplify_epsilon = ds_cfg.get("simplify_epsilon", defaults["deepsketch"]["simplify_epsilon"])
        self.deepsketch_sample_step = ds_cfg.get("sample_step", defaults["deepsketch"]["sample_step"])
        self.deepsketch_use_bezier = ds_cfg.get("use_bezier", defaults["deepsketch"]["use_bezier"])
        self.deepsketch_save_ordered_paths_json = ds_cfg.get("save_ordered_paths_json", defaults["deepsketch"]["save_ordered_paths_json"])
        self.deepsketch_quality_draw_step = ds_cfg.get("quality_draw_step", defaults["deepsketch"]["quality_draw_step"])
        self.deepsketch_cache_enabled = ds_cfg.get("cache_enabled", defaults["deepsketch"]["cache_enabled"])
        self.deepsketch_precision = ds_cfg.get("precision", "medium")
        self.deepsketch_precision_presets = ds_cfg.get("precision_presets", {"low": 448, "medium": 512, "high": 600})
        self.deepsketch_precision_mode = ds_cfg.get("precision_mode", "auto")
        self.deepsketch_gpu_min_viable_side = ds_cfg.get("gpu_min_viable_side", 384)
        self.deepsketch_cpu_max_time_seconds = ds_cfg.get("cpu_max_time_seconds", 120.0)

        debug_cfg = cfg.get("debug", {})
        self.debug_output = debug_cfg.get("enabled", defaults["debug"]["enabled"])

