"""Единое описание локальных моделей для CLI, планировщика и панели."""
from pathlib import Path

from ..config import project_paths
from ..progress import emit
from .runtime import RuntimeManager


def manager_for(config: dict, config_path, kind: str = "text") -> RuntimeManager:
    paths = project_paths(config, config_path)
    llm = config["llm"]
    if kind == "text":
        return RuntimeManager(paths["data"] / "runtime", runtime_root=Path(llm["runtime_root"]),
                              endpoint=llm["endpoint"], model=llm["model"])
    if kind != "vision":
        raise ValueError("неизвестная модель")
    root = Path(__file__).resolve().parents[2] / "llm_runtime_vision"
    models = sorted(p for p in root.glob("*.gguf") if "mmproj" not in p.name.lower())
    projectors = sorted(root.glob("*mmproj*.gguf"))
    return RuntimeManager(paths["data"] / "runtime", runtime_dir=root,
                          weights=models[0] if models else root / "missing-model.gguf",
                          endpoint=llm["vision_endpoint"], model=llm["vision_model"],
                          extra_args=["--mmproj", str(projectors[0] if projectors else root / "missing-mmproj.gguf")])


def ensure_models(config: dict, config_path, *, vision: bool = False) -> list[str]:
    warnings = []
    for kind in (["text", "vision"] if vision else ["text"]):
        emit("models", f"Подготовка {'Bonsai' if kind == 'text' else 'Qwen-VL'}")
        try:
            manager_for(config, config_path, kind).ensure_ready(timeout=180)
        except Exception as error:
            warnings.append(f"{kind}: {error}")
    return warnings
