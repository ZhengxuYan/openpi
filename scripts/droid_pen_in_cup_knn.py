#!/usr/bin/env python3
"""Find DROID pen-in-cup frames and run action-independent KNN over openpi features.

This script is intentionally read-only with respect to the DROID dataset roots. It writes
all generated artifacts under --output-dir.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import html
import json
import logging
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
import openpi.models.pi0 as _pi0
import openpi.models.pi0_fast as _pi0_fast
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


DEFAULT_CHECKPOINTS = {
    "pi0": "gs://openpi-assets/checkpoints/pi0_droid",
    "pi0_fast": "gs://openpi-assets/checkpoints/pi0_fast_droid",
    "pi05": "gs://openpi-assets/checkpoints/pi05_droid",
}

MODEL_CONFIGS = {
    "pi0": "pi0_droid",
    "pi0_fast": "pi0_fast_droid",
    "pi05": "pi05_droid",
}


@dataclasses.dataclass(frozen=True)
class FrameRecord:
    frame_id: int
    episode_id: str
    step_id: str
    frame_index: int
    prompt: str
    base_image: str
    wrist_image: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlds-data-dir", default="/iliad/group/datasets/droid")
    parser.add_argument("--raw-data-dir", default="/iliad/group/datasets/droid_raw")
    parser.add_argument(
        "--raw-annotation-json",
        default=None,
        help="Optional path to an aggregated DROID raw annotation JSON. If omitted, common locations are tried.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--work-dir",
        default=None,
        help=(
            "Directory for intermediate frame caches. Defaults to --output-dir. On SLURM, prefer $SLURM_TMPDIR "
            "to reduce writes to /iliad."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--version", default="1.0.1")
    parser.add_argument("--pi0-checkpoint", default=DEFAULT_CHECKPOINTS["pi0"])
    parser.add_argument("--pi0-fast-checkpoint", default=DEFAULT_CHECKPOINTS["pi0_fast"])
    parser.add_argument("--pi05-checkpoint", default=DEFAULT_CHECKPOINTS["pi05"])
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Only build the pen_in_cup manifest and image artifacts; skip model loading and KNN.",
    )
    parser.add_argument(
        "--include-self",
        action="store_true",
        help="Include each frame as its own nearest neighbor. By default self-neighbors are excluded.",
    )
    parser.add_argument(
        "--keep-raw-frame-cache",
        action="store_true",
        help="Copy intermediate raw .npz frame caches to --output-dir. Off by default to reduce storage use.",
    )
    return parser.parse_args()


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _is_pen_in_cup(text: str) -> bool:
    normalized = _normalize_text(text)
    compact = normalized.replace(" ", "_")
    return "pen_in_cup" in compact or ("pen" in normalized.split() and "cup" in normalized.split())


def _decode(value: Any) -> str:
    value = _first_scalar(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _first_scalar(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    if arr.size == 0:
        return ""
    return arr.reshape(-1)[0].item()


def _as_image(image: Any) -> np.ndarray:
    """Converts TFDS image tensors or encoded bytes to uint8 HWC arrays."""
    image = _first_scalar(image) if isinstance(image, bytes | np.bytes_) else image
    if isinstance(image, bytes | np.bytes_):
        import tensorflow as tf

        image = tf.io.decode_image(image, expand_animations=False, dtype=tf.uint8).numpy()

    image = np.asarray(image)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if np.issubdtype(image.dtype, np.floating):
        max_value = 1.0 if float(np.nanmax(image)) <= 1.0 else 255.0
        image = np.clip(image, 0.0, max_value) / max_value * 255.0
    return image.astype(np.uint8)


def _save_image(path: Path, image: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_as_image(image)).save(path, quality=88)


def _resolve_tfds_data_dir(rlds_data_dir: str) -> Path:
    """Accept either the TFDS data_dir parent or the droid dataset directory itself."""
    path = Path(rlds_data_dir)
    if (path / "droid").exists():
        return path
    if path.name == "droid" and any((path / version).exists() for version in ("1.0.1", "1.0.0")):
        return path.parent
    return path


def _extract_prompts(traj: dict[str, Any]) -> list[str]:
    prompts = []
    for key in ("language_instruction", "language_instruction_2", "language_instruction_3"):
        if key in traj:
            prompt = _decode(traj[key]).strip()
            if prompt and prompt not in prompts:
                prompts.append(prompt)
    return prompts


def _metadata_string(traj: dict[str, Any], *path: str) -> str:
    value: Any = traj
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return ""
        value = value[key]
    return _decode(value)


def _scan_raw_annotations(raw_data_dir: str, raw_annotation_json: str | None, output_dir: Path) -> None:
    """Best-effort raw annotation scan for auditability; RLDS remains source of frame truth."""
    root = Path(raw_data_dir)
    candidate_paths = [Path(raw_annotation_json)] if raw_annotation_json else []
    candidate_paths.extend(
        [
            root / "1.0.1" / "aggregated-annotations-030724.json",
            root / "aggregated-annotations-030724.json",
        ]
    )
    candidate_paths = [path for path in candidate_paths if path.exists()]
    if not candidate_paths:
        logging.warning(
            "No raw annotation JSON found in common locations under %s; skipping raw annotation audit scan.",
            root,
        )
        return

    matches: list[dict[str, str]] = []
    for json_path in candidate_paths:
        try:
            data = json.loads(json_path.read_text())
        except Exception as exc:  # noqa: BLE001
            logging.debug("Skipping unreadable JSON %s: %s", json_path, exc)
            continue
        for pointer, text in _walk_strings(data):
            if _is_pen_in_cup(text):
                matches.append({"json_path": str(json_path), "json_pointer": pointer, "text": text})

    if matches:
        path = output_dir / "raw_annotation_matches.json"
        path.write_text(json.dumps(matches, indent=2))
        logging.info("Wrote %d raw annotation matches to %s", len(matches), path)
    else:
        logging.warning("No raw annotation matches found in %s", ", ".join(map(str, candidate_paths)))


def _walk_strings(value: Any, pointer: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, str):
        yield pointer or "/", value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _walk_strings(child, f"{pointer}/{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_strings(child, f"{pointer}/{index}")


def _build_manifest(args: argparse.Namespace, output_dir: Path, work_dir: Path) -> list[FrameRecord]:
    import tensorflow as tf
    import tensorflow_datasets as tfds

    tf.config.set_visible_devices([], "GPU")

    data_dir = _resolve_tfds_data_dir(args.rlds_data_dir)
    logging.info("Opening TFDS DROID from data_dir=%s version=%s", data_dir, args.version)
    builder = tfds.builder("droid", data_dir=str(data_dir), version=args.version)
    dataset = builder.as_dataset(split="train", shuffle_files=False)

    raw_npz_dir = work_dir / "raw_frames"
    image_dir = work_dir / "images"
    raw_npz_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    records: list[FrameRecord] = []
    for episode in tfds.as_numpy(dataset):
        prompts = _extract_prompts(episode)
        matching_prompts = [prompt for prompt in prompts if _is_pen_in_cup(prompt)]
        if not matching_prompts:
            continue

        prompt = matching_prompts[0]
        recording_folder = _metadata_string(
            episode, "traj_metadata", "episode_metadata", "recording_folderpath"
        )
        file_path = _metadata_string(episode, "traj_metadata", "episode_metadata", "file_path")
        if file_path and not re.search("success", file_path):
            continue

        episode_id = f"{recording_folder}--{file_path}".strip("-")
        obs = episode["observation"]
        num_steps = int(np.asarray(obs["joint_position"]).shape[0])

        for step in range(num_steps):
            if args.max_frames is not None and len(records) >= args.max_frames:
                _write_manifest(output_dir / "pen_in_cup_manifest.csv", records)
                logging.info("Reached --max-frames=%d", args.max_frames)
                return records

            frame_id = len(records)
            step_id = f"{episode_id}--{step}"
            base = _as_image(obs["exterior_image_1_left"][step])
            wrist = _as_image(obs["wrist_image_left"][step])
            state = np.concatenate(
                [
                    np.asarray(obs["joint_position"][step], dtype=np.float32),
                    np.asarray(obs["gripper_position"][step], dtype=np.float32).reshape(-1)[:1],
                ],
                axis=0,
            )

            base_rel = Path("images") / f"frame_{frame_id:06d}_base.jpg"
            wrist_rel = Path("images") / f"frame_{frame_id:06d}_wrist.jpg"
            _save_image(output_dir / base_rel, base)
            _save_image(output_dir / wrist_rel, wrist)
            np.savez_compressed(
                raw_npz_dir / f"frame_{frame_id:06d}.npz",
                base=base,
                wrist=wrist,
                state=state,
                prompt=np.asarray(prompt),
            )

            records.append(
                FrameRecord(
                    frame_id=frame_id,
                    episode_id=episode_id,
                    step_id=step_id,
                    frame_index=step,
                    prompt=prompt,
                    base_image=str(base_rel),
                    wrist_image=str(wrist_rel),
                )
            )

    _write_manifest(output_dir / "pen_in_cup_manifest.csv", records)
    return records


def _write_manifest(path: Path, records: list[FrameRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dataclasses.fields(FrameRecord)])
        writer.writeheader()
        for record in records:
            writer.writerow(dataclasses.asdict(record))
    logging.info("Wrote %d manifest rows to %s", len(records), path)


def _raw_observation(work_dir: Path, frame_id: int) -> dict[str, Any]:
    data = np.load(work_dir / "raw_frames" / f"frame_{frame_id:06d}.npz")
    return {
        "observation/exterior_image_1_left": data["base"],
        "observation/wrist_image_left": data["wrist"],
        "observation/joint_position": data["state"][:7],
        "observation/gripper_position": data["state"][7:8],
        "prompt": str(data["prompt"]),
    }


def _collate(dicts: list[dict[str, Any]]) -> dict[str, Any]:
    return jax.tree.map(lambda *xs: np.stack(xs, axis=0), *dicts)


def _last_valid_token(tokens: jax.Array, mask: jax.Array) -> jax.Array:
    positions = jnp.broadcast_to(jnp.arange(mask.shape[1])[None, :], mask.shape)
    indices = jnp.max(jnp.where(mask, positions, 0), axis=1)
    return jnp.take_along_axis(tokens, indices[:, None, None], axis=1)[:, 0, :]


def _make_pi0_prefix_extractor(model: Any):
    graphdef, state = nnx.split(model)

    def extract(state: nnx.State, observation: _model.Observation) -> jax.Array:
        module = nnx.merge(graphdef, state)
        observation = _model.preprocess_observation(None, observation, train=False)
        prefix_tokens, prefix_mask, prefix_ar_mask = module.embed_prefix(observation)
        prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = module.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )
        assert suffix_out is None
        return _last_valid_token(prefix_out, prefix_mask)

    jitted = jax.jit(extract)
    return lambda observation: jitted(state, observation)


def _make_fast_prefix_extractor(model: Any):
    graphdef, state = nnx.split(model)

    def extract(state: nnx.State, observation: _model.Observation) -> jax.Array:
        module = nnx.merge(graphdef, state)
        observation = _model.preprocess_observation(
            None, observation, train=False, image_keys=list(observation.images.keys())
        )
        token_embeddings, input_mask, ar_mask = module.embed_inputs(observation)
        attn_mask = _pi0_fast.make_attn_mask(input_mask, ar_mask)
        pre_logits, _, _ = module.PaliGemma.llm(
            embedded_prefix=token_embeddings,
            mask=attn_mask,
            return_prelogits=True,
        )
        return _last_valid_token(pre_logits, input_mask)

    jitted = jax.jit(extract)
    return lambda observation: jitted(state, observation)


def _extract_features_for_model(
    *,
    model_key: str,
    checkpoint: str,
    records: list[FrameRecord],
    output_dir: Path,
    work_dir: Path,
    batch_size: int,
) -> np.ndarray:
    logging.info("Loading %s from %s", model_key, checkpoint)
    policy = _policy_config.create_trained_policy(_config.get_config(MODEL_CONFIGS[model_key]), checkpoint)
    if getattr(policy, "_is_pytorch_model", False):
        raise ValueError("This KNN script currently supports JAX checkpoints only, not PyTorch safetensors.")

    model = policy._model  # noqa: SLF001
    transform = policy._input_transform  # noqa: SLF001
    extract = _make_fast_prefix_extractor(model) if model_key == "pi0_fast" else _make_pi0_prefix_extractor(model)

    features = []
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        transformed = []
        for record in batch_records:
            raw = _raw_observation(work_dir, record.frame_id)
            if "actions" in raw:
                raise AssertionError("Action field unexpectedly present before model transform.")
            item = transform(raw)
            if "actions" in item:
                raise AssertionError("Action field unexpectedly present after model transform.")
            transformed.append(item)

        observation = _model.Observation.from_dict(_collate(transformed))
        batch_features = np.asarray(extract(observation))
        features.append(batch_features.astype(np.float32))
        logging.info("%s: extracted %d/%d frames", model_key, start + len(batch_records), len(records))

    feature_array = np.concatenate(features, axis=0)
    norms = np.linalg.norm(feature_array, axis=1, keepdims=True)
    feature_array = feature_array / np.clip(norms, 1e-12, None)

    np.save(output_dir / f"features_{model_key}.npy", feature_array)
    metadata_path = output_dir / f"features_{model_key}.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model": model_key,
                "config": MODEL_CONFIGS[model_key],
                "checkpoint": checkpoint,
                "shape": list(feature_array.shape),
                "pooling": "last_valid_prefix_token",
                "uses_actions": False,
            },
            indent=2,
        )
    )
    logging.info("Wrote %s features with shape %s", model_key, feature_array.shape)
    return feature_array


def _compute_knn(features: np.ndarray, k: int, *, include_self: bool) -> tuple[np.ndarray, np.ndarray]:
    n = features.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=np.int32), np.empty((0, 0), dtype=np.float32)
    effective_k = min(k, n if include_self else max(n - 1, 0))
    if effective_k == 0:
        return np.empty((n, 0), dtype=np.int32), np.empty((n, 0), dtype=np.float32)

    all_indices = np.empty((n, effective_k), dtype=np.int32)
    all_distances = np.empty((n, effective_k), dtype=np.float32)
    chunk_size = max(1, min(1024, 1_000_000 // max(n, 1)))

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        sims = features[start:stop] @ features.T
        if not include_self:
            rows = np.arange(stop - start)
            cols = np.arange(start, stop)
            sims[rows, cols] = -np.inf
        candidate_count = min(effective_k, sims.shape[1])
        part = np.argpartition(-sims, kth=candidate_count - 1, axis=1)[:, :candidate_count]
        part_sims = np.take_along_axis(sims, part, axis=1)
        order = np.argsort(-part_sims, axis=1)
        sorted_indices = np.take_along_axis(part, order, axis=1)
        sorted_sims = np.take_along_axis(part_sims, order, axis=1)
        all_indices[start:stop] = sorted_indices
        all_distances[start:stop] = np.sqrt(np.maximum(2.0 - 2.0 * sorted_sims, 0.0))

    return all_indices, all_distances


def _write_knn_csv(path: Path, records: list[FrameRecord], indices: np.ndarray, distances: np.ndarray) -> None:
    with path.open("w", newline="") as f:
        fieldnames = [
            "query_frame_id",
            "neighbor_rank",
            "neighbor_frame_id",
            "distance",
            "query_step_id",
            "neighbor_step_id",
            "query_prompt",
            "neighbor_prompt",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for query_index, record in enumerate(records):
            for rank, neighbor_index in enumerate(indices[query_index]):
                neighbor = records[int(neighbor_index)]
                writer.writerow(
                    {
                        "query_frame_id": record.frame_id,
                        "neighbor_rank": rank + 1,
                        "neighbor_frame_id": neighbor.frame_id,
                        "distance": float(distances[query_index, rank]),
                        "query_step_id": record.step_id,
                        "neighbor_step_id": neighbor.step_id,
                        "query_prompt": record.prompt,
                        "neighbor_prompt": neighbor.prompt,
                    }
                )
    logging.info("Wrote KNN CSV to %s", path)


def _write_html(output_dir: Path, records: list[FrameRecord], model_knn: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    model_sections = []
    for model_key, (indices, distances) in model_knn.items():
        rows = []
        for query_idx, record in enumerate(records):
            neighbor_cells = []
            for rank, neighbor_idx in enumerate(indices[query_idx]):
                neighbor = records[int(neighbor_idx)]
                neighbor_cells.append(
                    f"""
                    <figure>
                      <img src="{html.escape(neighbor.base_image)}" alt="neighbor {neighbor.frame_id}">
                      <figcaption>#{neighbor.frame_id} r{rank + 1}<br>d={distances[query_idx, rank]:.4f}</figcaption>
                    </figure>
                    """
                )
            rows.append(
                f"""
                <section class="query">
                  <div class="query-main">
                    <img src="{html.escape(record.base_image)}" alt="query {record.frame_id}">
                    <div>
                      <h3>Frame #{record.frame_id}</h3>
                      <p>{html.escape(record.prompt)}</p>
                      <code>{html.escape(record.step_id)}</code>
                    </div>
                  </div>
                  <div class="neighbors">{''.join(neighbor_cells)}</div>
                </section>
                """
            )
        model_sections.append(
            f"""
            <section class="model">
              <h2>{html.escape(model_key)}</h2>
              {''.join(rows)}
            </section>
            """
        )

    page = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>DROID pen_in_cup KNN</title>
      <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; color: #1f2933; }}
        h1 {{ margin-bottom: 4px; }}
        .summary {{ color: #52606d; margin-bottom: 24px; }}
        .model {{ border-top: 1px solid #d9e2ec; padding-top: 18px; margin-top: 28px; }}
        .query {{ display: grid; grid-template-columns: minmax(320px, 420px) 1fr; gap: 18px; padding: 14px 0; border-top: 1px solid #edf2f7; }}
        .query-main {{ display: grid; grid-template-columns: 144px 1fr; gap: 12px; align-items: start; }}
        img {{ width: 144px; height: 144px; object-fit: cover; border: 1px solid #d9e2ec; }}
        .neighbors {{ display: flex; flex-wrap: wrap; gap: 10px; }}
        figure {{ margin: 0; width: 144px; }}
        figcaption {{ font-size: 12px; color: #52606d; line-height: 1.35; }}
        code {{ display: block; margin-top: 8px; overflow-wrap: anywhere; font-size: 12px; color: #334e68; }}
        p {{ margin: 0; }}
      </style>
    </head>
    <body>
      <h1>DROID pen_in_cup KNN</h1>
      <p class="summary">{len(records)} frames. Embeddings are action-independent last valid prefix-token features.</p>
      {''.join(model_sections)}
    </body>
    </html>
    """
    path = output_dir / "index.html"
    path.write_text(page)
    logging.info("Wrote HTML report to %s", path)


def _sync_frame_artifacts(work_dir: Path, output_dir: Path, *, keep_raw_frame_cache: bool) -> None:
    if work_dir.resolve() == output_dir.resolve():
        return
    dirnames = ["images"]
    if keep_raw_frame_cache:
        dirnames.append("raw_frames")
    for dirname in dirnames:
        src = work_dir / dirname
        dst = output_dir / dirname
        if not src.exists():
            continue
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        logging.info("Copied %s to %s", src, dst)


def _cleanup_raw_frame_cache(work_dir: Path, *, keep_raw_frame_cache: bool) -> None:
    if keep_raw_frame_cache:
        return
    raw_dir = work_dir / "raw_frames"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
        logging.info("Removed intermediate raw frame cache at %s", raw_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    output_dir = Path(args.output_dir)
    work_dir = Path(args.work_dir) if args.work_dir else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    _scan_raw_annotations(args.raw_data_dir, args.raw_annotation_json, output_dir)
    records = _build_manifest(args, output_dir, work_dir)
    if not records:
        raise RuntimeError("No pen_in_cup frames found. Check dataset paths/version and matching prompts.")
    if args.scan_only:
        _sync_frame_artifacts(work_dir, output_dir, keep_raw_frame_cache=args.keep_raw_frame_cache)
        _cleanup_raw_frame_cache(work_dir, keep_raw_frame_cache=args.keep_raw_frame_cache)
        return

    checkpoints = {
        "pi0": args.pi0_checkpoint,
        "pi0_fast": args.pi0_fast_checkpoint,
        "pi05": args.pi05_checkpoint,
    }
    model_knn: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model_key, checkpoint in checkpoints.items():
        features = _extract_features_for_model(
            model_key=model_key,
            checkpoint=checkpoint,
            records=records,
            output_dir=output_dir,
            work_dir=work_dir,
            batch_size=args.batch_size,
        )
        indices, distances = _compute_knn(features, args.k, include_self=args.include_self)
        _write_knn_csv(output_dir / f"knn_{model_key}.csv", records, indices, distances)
        model_knn[model_key] = (indices, distances)

    _sync_frame_artifacts(work_dir, output_dir, keep_raw_frame_cache=args.keep_raw_frame_cache)
    _cleanup_raw_frame_cache(work_dir, keep_raw_frame_cache=args.keep_raw_frame_cache)
    _write_html(output_dir, records, model_knn)


if __name__ == "__main__":
    main()
