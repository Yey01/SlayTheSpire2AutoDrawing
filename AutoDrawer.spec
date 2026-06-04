# -*- mode: python ; coding: utf-8 -*-

import os


def assert_cpu_only_torch_bundle():
    if os.environ.get("ALLOW_CUDA_TORCH_BUNDLE") == "1":
        return
    try:
        import torch
    except Exception as exc:
        raise SystemExit(f"PyInstaller build requires torch to be importable: {exc}") from exc

    cuda_version = getattr(torch.version, "cuda", None)
    if cuda_version:
        raise SystemExit(
            "Refusing to build the CPU-only exe with a CUDA PyTorch install "
            f"(torch {torch.__version__}, CUDA {cuda_version}).\n"
            "Create a clean CPU build environment and install requirements-cpu.txt, "
            "then run PyInstaller again. Set ALLOW_CUDA_TORCH_BUNDLE=1 only when "
            "intentionally building a large CUDA/GPU exe."
        )


assert_cpu_only_torch_bundle()


datas = [
    ("config.txt", "."),
    ("runtime_config.json", "."),
    ("models/line_drawing/line-drawings.onnx", "models/line_drawing"),
    ("models/deepsketch/predict_s1.py", "models/deepsketch"),
    ("models/deepsketch/edge_distance_aabb.py", "models/deepsketch"),
    ("models/deepsketch/model/udf_full.pth", "models/deepsketch/model"),
    ("models/deepsketch/model/ndc_full.pth", "models/deepsketch/model"),
    ("models/deepsketch/dataset", "models/deepsketch/dataset"),
    ("models/deepsketch/network", "models/deepsketch/network"),
    ("models/deepsketch/utils", "models/deepsketch/utils"),
    ("models/deepsketch/web", "models/deepsketch/web"),
]

hiddenimports = [
    # DeepSketch worker local modules. pathex below makes these importable at build time.
    "edge_distance_aabb",
    "network.udc",
    "network.keypoint",
    "network.thin",
    "network.anime2sketch",
    "dataset.preprocess",
    "dataset.augmentation",
    "utils.ndc_tools",
    "utils.keypt_tools",
    "utils.svg_tools",
    "utils.dual_contouring",
    "utils.fitCurves",
    "fitCurves",
    "bezier",
    # Third-party runtime dependencies used by the GUI, ONNX pipeline, and DeepSketch.
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torchvision",
    "torchvision.transforms",
    "torchvision.transforms.functional",
    "torchvision.utils",
    "cv2",
    "onnxruntime",
    "skimage",
    "skimage.morphology",
    "skimage.feature",
    "skimage.util",
    "scipy",
    "scipy.interpolate",
    "scipy.signal",
    "scipy.spatial",
    "scipy.sparse.csgraph",
    "scipy.ndimage",
    "sklearn",
    "sklearn.neighbors",
    "svgpathtools",
    "svgpathtools.document",
    "aabbtree",
    "rdp",
    "colorama",
    "ndjson",
    "tqdm",
    "matplotlib",
    "matplotlib.pyplot",
]

excludes = [
    "IPython",
    "jupyter",
    "notebook",
    "pytest",
    "tests",
    "tensorflow",
    "tensorboard",
    "dask",
    "pydiffvg",
    "ttools",
    "playwright",
    "langchain",
    "googleapiclient",
    "qianfan",
    "modelscope",
    "psycopg2",
    "sqlalchemy",
]


a = Analysis(
    ["AutoDrawer.py"],
    pathex=[
        ".",
        "models/deepsketch",
        "models/deepsketch/utils",
        "models/deepsketch/dataset",
        "models/deepsketch/network",
    ],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={
        "matplotlib": {
            "backends": ["TkAgg", "Agg"],
        },
    },
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="AutoDrawer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=["icon.ico"],
)
