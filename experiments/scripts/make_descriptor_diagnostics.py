"""Build descriptor-level diagnostic checks for the revised manuscript.

The checks are deliberately descriptor-level rather than trained-model
ablations. They ask whether the fixed component candidate-edge descriptor
contains residual-consistent information beyond simple chainage, global anomaly,
distance-only exposure, uniform edge averaging, and component-label disruption.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial import KDTree

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.diagnostics.context import TARGET_NAMES, build_descriptor_context
from src.diagnostics.spatial_interaction import (
    COMPONENT_NAMES,
    component_descriptors_for_snapshot,
    persistence_residuals,
    safe_spearman,
    vp_anomaly_from_standardized_attrs,
)


CASE_CONFIGS = [
    "config/bsll_dyk1017_205.yaml",
    "config/bsll_dyk1017_205_h3.yaml",
    "config/sjls_dyk1252_411.yaml",
]

CASE_LABELS = {
    "bsll_dyk1017_205": "BSLL h=1",
    "bsll_dyk1017_205_h3": "BSLL h=3",
    "sjls_dyk1252_411": "SJLS h=3",
}

DIAGNOSTIC_PAIRS = {
    "bsll_dyk1017_205": ("front_shield", "AdvanceRate"),
    "bsll_dyk1017_205_h3": ("front_shield", "ShieldPressure"),
    "sjls_dyk1252_411": ("cutterhead", "ShieldPressure"),
}

ATTR_INDEX = {
    "Vp": 0,
    "Vs": 1,
    "E": 2,
    "Vp/Vs": 3,
    "nu": 4,
    "rho": 5,
}


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(data: dict[str, Any], path: Path) -> None:
    def convert(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=convert)


def low_attribute_score(rock_attrs: np.ndarray, col: int) -> np.ndarray:
    """Bounded low-attribute anomaly score from standardized attributes."""
    values = rock_attrs[:, col].astype(np.float32)
    return np.clip((1.645 - values) / (2.0 * 1.645), 0.0, 1.0).astype(np.float32)


def descriptor_series_for_component(context: dict[str, Any], component: str) -> dict[str, np.ndarray]:
    """Return proposed and null descriptor series for one component.

    The per-step all-component $I_c$ matrix is returned so that the caller can
    run the component-label disruption diagnostic against the residual.
    """
    proposed = []
    chainage = []
    global_anomaly = []
    distance_only = []
    uniform_edge = []
    per_step_all_ic: list[list[float]] = []

    for graph_seq, ch in zip(context["test_graph_seqs"], context["test_chainages"]):
        snapshot = graph_seq[-1]
        rows = component_descriptors_for_snapshot(
            snapshot,
            anomaly_reference=context.get("vp_anomaly_reference"),
        )
        by_component = {row.component: row for row in rows}
        row = by_component[component]

        rock_attrs = snapshot.rock_attrs.detach().cpu().numpy()
        q = vp_anomaly_from_standardized_attrs(rock_attrs, context.get("vp_anomaly_reference"))
        proposed.append(row.interaction_intensity)
        chainage.append(float(ch))
        global_anomaly.append(float(np.mean(q)) if len(q) else 0.0)
        distance_only.append(row.geometric_exposure)
        uniform_edge.append(row.mean_anomaly)
        per_step_all_ic.append([by_component[name].interaction_intensity for name in COMPONENT_NAMES.values()])

    return {
        "proposed": np.asarray(proposed, dtype=np.float64),
        "chainage_only": np.asarray(chainage, dtype=np.float64),
        "global_vp_anomaly": np.asarray(global_anomaly, dtype=np.float64),
        "distance_only_exposure": np.asarray(distance_only, dtype=np.float64),
        "uniform_edge_anomaly": np.asarray(uniform_edge, dtype=np.float64),
        "all_component_ic": np.asarray(per_step_all_ic, dtype=np.float64),
    }


def component_intensity_from_edges(
    src: np.ndarray,
    dst: np.ndarray,
    weights: np.ndarray,
    anomaly_scores: np.ndarray,
    tbm_components: np.ndarray,
    component: str | None,
) -> float:
    """Compute a weighted anomaly intensity for a component or pooled surface."""
    if len(src) == 0:
        return 0.0
    if component is None:
        mask = np.ones(len(src), dtype=bool)
    else:
        component_ids = {name: cid for cid, name in COMPONENT_NAMES.items()}
        mask = tbm_components[dst] == component_ids[component]
    if not np.any(mask):
        return 0.0
    w = np.asarray(weights[mask], dtype=np.float64)
    q = np.asarray(anomaly_scores[src[mask]], dtype=np.float64)
    denom = float(np.sum(w))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(w * q) / denom)


def distance_only_candidate_edges(rock_coords: np.ndarray, tbm_positions: np.ndarray, tau_edge: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return all active-rock to TBM pairs within tau_edge with distance-only weights."""
    tree = KDTree(tbm_positions)
    pairs = tree.query_ball_point(rock_coords, r=tau_edge, workers=-1)
    src, dst, distances = [], [], []
    for i, neighbours in enumerate(pairs):
        for j in neighbours:
            delta = rock_coords[i] - tbm_positions[j]
            d = float(np.linalg.norm(delta))
            if d <= 1e-8:
                continue
            src.append(i)
            dst.append(j)
            distances.append(d)
    if not src:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0, dtype=np.float64)
    distances_arr = np.asarray(distances, dtype=np.float64)
    return (
        np.asarray(src, dtype=int),
        np.asarray(dst, dtype=int),
        np.exp(-distances_arr / tau_edge),
    )


def relation_support_variant_series(context: dict[str, Any], component: str) -> dict[str, np.ndarray]:
    """Ablate relation-support choices while keeping the fixed component/readout."""
    variants: dict[str, list[float]] = {
        "proposed": [],
        "no_normal_screen": [],
        "distance_only_edges": [],
        "pooled_tbm_surface": [],
    }
    tau_edge = float(context["tau_edge"])
    for graph_seq in context["test_graph_seqs"]:
        snapshot = graph_seq[-1]
        rock_attrs = snapshot.rock_attrs.detach().cpu().numpy()
        q = vp_anomaly_from_standardized_attrs(rock_attrs, context.get("vp_anomaly_reference"))
        tbm_components = snapshot.tbm_components.argmax(dim=1).detach().cpu().numpy()
        rock_coords = snapshot.hetero_data["rock"].x.detach().cpu().numpy()[:, :3]
        tbm_x = snapshot.hetero_data["tbm"].x.detach().cpu().numpy()
        tbm_positions = tbm_x[:, :3]
        tbm_normals = tbm_x[:, 3:6]

        edge_store = snapshot.hetero_data["rock", "interact", "tbm"]
        edge_index = edge_store.edge_index.detach().cpu().numpy()
        src_existing = edge_index[0]
        dst_existing = edge_index[1]
        weights_existing = np.asarray(edge_store["edge_attrs"]["geometry_prior"], dtype=np.float64).reshape(-1)
        variants["proposed"].append(
            component_intensity_from_edges(src_existing, dst_existing, weights_existing, q, tbm_components, component)
        )
        variants["pooled_tbm_surface"].append(
            component_intensity_from_edges(src_existing, dst_existing, weights_existing, q, tbm_components, None)
        )

        src_no_norm, dst_no_norm, dist_weights = distance_only_candidate_edges(rock_coords, tbm_positions, tau_edge)
        if len(src_no_norm):
            delta = rock_coords[src_no_norm] - tbm_positions[dst_no_norm]
            distances = np.linalg.norm(delta, axis=1) + 1e-8
            kappa = np.maximum(0.0, np.sum(tbm_normals[dst_no_norm] * delta, axis=1) / distances)
            normal_weight = dist_weights * kappa
        else:
            normal_weight = dist_weights
        variants["no_normal_screen"].append(
            component_intensity_from_edges(src_no_norm, dst_no_norm, normal_weight, q, tbm_components, component)
        )
        variants["distance_only_edges"].append(
            component_intensity_from_edges(src_no_norm, dst_no_norm, dist_weights, q, tbm_components, component)
        )
    return {key: np.asarray(value, dtype=np.float64) for key, value in variants.items()}


def delta_field_rows(case_id: str, component: str, response: str, series: dict[str, np.ndarray], residual: np.ndarray) -> list[dict[str, Any]]:
    """Compare static and step-change field readouts against residual changes."""
    chainage = series["chainage_only"]
    ic = series["proposed"]
    rows = []
    comparisons = [
        ("Ic_vs_residual", ic, residual, chainage),
        ("DeltaIc_vs_residual", np.diff(ic), residual[1:], chainage[1:]),
        ("DeltaIc_vs_DeltaResidual", np.diff(ic), np.diff(residual), chainage[1:]),
    ]
    for comparison, x, y, ch in comparisons:
        rho, _ = safe_spearman(x, y)
        detrended_rho, _ = safe_spearman(linear_detrend(x, ch), linear_detrend(y, ch))
        rows.append(
            {
                "case_id": case_id,
                "case_label": CASE_LABELS[case_id],
                "component": component,
                "response": response,
                "comparison": comparison,
                "n": len(x),
                "spearman_r": rho,
                "detrended_spearman_r": detrended_rho,
            }
        )
    return rows


def relation_support_ablation_rows(case_id: str, component: str, response: str, context: dict[str, Any], residual: np.ndarray, chainage: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for variant, values in relation_support_variant_series(context, component).items():
        rho, _ = safe_spearman(values, residual)
        detrended_rho, _ = safe_spearman(linear_detrend(values, chainage), linear_detrend(residual, chainage))
        rows.append(
            {
                "case_id": case_id,
                "case_label": CASE_LABELS[case_id],
                "component": component,
                "response": response,
                "variant": variant,
                "n": len(values),
                "spearman_r": rho,
                "detrended_spearman_r": detrended_rho,
            }
        )
    return rows


def edge_concentration_rows(case_id: str, component: str, response: str, context: dict[str, Any]) -> list[dict[str, Any]]:
    """Quantify how strongly weighted edge contributions concentrate."""
    rows = []
    component_ids = {name: cid for cid, name in COMPONENT_NAMES.items()}
    for sample_idx, (chainage, graph_seq) in enumerate(zip(context["test_chainages"], context["test_graph_seqs"])):
        snapshot = graph_seq[-1]
        rock_attrs = snapshot.rock_attrs.detach().cpu().numpy()
        q = vp_anomaly_from_standardized_attrs(rock_attrs, context.get("vp_anomaly_reference"))
        edge_store = snapshot.hetero_data["rock", "interact", "tbm"]
        edge_index = edge_store.edge_index.detach().cpu().numpy()
        src = edge_index[0]
        dst = edge_index[1]
        weights = np.asarray(edge_store["edge_attrs"]["geometry_prior"], dtype=np.float64).reshape(-1)
        tbm_components = snapshot.tbm_components.argmax(dim=1).detach().cpu().numpy()
        mask = tbm_components[dst] == component_ids[component]
        contrib = weights[mask] * q[src[mask]]
        uniform = q[src[mask]]
        total = float(np.sum(contrib))
        if len(contrib) == 0 or total <= 1e-12:
            top5_share = 0.0
            top10_share = 0.0
            effective_edges = 0.0
            entropy = 0.0
            rank_overlap_top10 = 0.0
        else:
            order = np.argsort(contrib)[::-1]
            p = contrib / total
            top5_share = float(np.sum(contrib[order[:5]]) / total)
            top10_share = float(np.sum(contrib[order[:10]]) / total)
            effective_edges = float(1.0 / np.sum(p * p))
            entropy = float(-np.sum(p * np.log(p + 1e-12)) / np.log(len(p))) if len(p) > 1 else 0.0
            weighted_top = set(order[: min(10, len(order))].tolist())
            uniform_order = np.argsort(uniform)[::-1]
            uniform_top = set(uniform_order[: min(10, len(uniform_order))].tolist())
            denom = max(1, len(weighted_top | uniform_top))
            rank_overlap_top10 = float(len(weighted_top & uniform_top) / denom)
        rows.append(
            {
                "case_id": case_id,
                "case_label": CASE_LABELS[case_id],
                "component": component,
                "response": response,
                "sample_idx": sample_idx,
                "chainage": float(chainage),
                "edge_count": int(len(contrib)),
                "top5_contribution_share": top5_share,
                "top10_contribution_share": top10_share,
                "effective_edge_number": effective_edges,
                "contribution_entropy": entropy,
                "weighted_unweighted_top10_jaccard": rank_overlap_top10,
            }
        )
    return rows


def component_label_permutation_test(
    all_ic: np.ndarray,
    target_component: str,
    residual: np.ndarray,
    chainage: np.ndarray,
    n_perm: int = 1000,
) -> tuple[float, float, int]:
    """Component-label permutation diagnostic on detrended series.

    At each chainage step, component labels are randomly permuted and the value
    assigned to the target component is used to build a disrupted descriptor
    series.
    Both the observed and permuted series are linearly detrended against
    chainage before computing the Spearman correlation with the detrended
    residual, consistent with Table~``tab:null_comparison``. This is repeated
    ``n_perm`` times. The returned proportion counts permutations whose
    absolute correlation is at least as large as the observed absolute
    correlation.
    """
    rng = np.random.default_rng(20260625)
    n_steps, n_comp = all_ic.shape
    target_idx = list(COMPONENT_NAMES.values()).index(target_component)

    obs_series = all_ic[:, target_idx]
    obs_detrended = linear_detrend(obs_series, chainage)
    res_detrended = linear_detrend(residual, chainage)
    obs_r, _ = safe_spearman(obs_detrended, res_detrended)

    perm_rs = np.empty(n_perm, dtype=np.float64)
    for p in range(n_perm):
        permuted = np.empty(n_steps, dtype=np.float64)
        for s in range(n_steps):
            perm = rng.permutation(n_comp)
            permuted[s] = all_ic[s, perm[target_idx]]
        perm_detrended = linear_detrend(permuted, chainage)
        r, _ = safe_spearman(perm_detrended, res_detrended)
        perm_rs[p] = r

    p_value = float(np.mean(np.abs(perm_rs) >= np.abs(obs_r)))
    return p_value, obs_r, n_perm


def component_series_reassignment_rows(
    case_id: str,
    all_ic: np.ndarray,
    target_component: str,
    response: str,
    residual: np.ndarray,
    chainage: np.ndarray,
) -> list[dict[str, Any]]:
    """Compare complete component series without step-wise relabelling."""
    res_detrended = linear_detrend(residual, chainage)
    rows = []
    for idx, component in enumerate(COMPONENT_NAMES.values()):
        comp_detrended = linear_detrend(all_ic[:, idx], chainage)
        rho, _ = safe_spearman(comp_detrended, res_detrended)
        rows.append(
            {
                "case_id": case_id,
                "case_label": CASE_LABELS[case_id],
                "target_component": target_component,
                "component": component,
                "response": response,
                "n": len(residual),
                "detrended_spearman_r": rho,
                "is_target": component == target_component,
            }
        )
    return rows


def alignment_offset_rows(case_id: str, component: str, response: str, series: dict[str, np.ndarray], residual: np.ndarray) -> list[dict[str, Any]]:
    """Approximate +/-1 m longitudinal alignment uncertainty by shifting descriptor chainage."""
    chainage = series["chainage_only"]
    values = series["proposed"]
    rows = []
    for offset in [-1.0, 0.0, 1.0]:
        shifted_values = np.interp(chainage + offset, chainage, values, left=np.nan, right=np.nan)
        mask = ~np.isnan(shifted_values)
        r, _ = safe_spearman(shifted_values[mask], residual[mask])
        detrended_r, _ = safe_spearman(
            linear_detrend(shifted_values[mask], chainage[mask]),
            linear_detrend(residual[mask], chainage[mask]),
        )
        rows.append(
            {
                "case_id": case_id,
                "case_label": CASE_LABELS[case_id],
                "component": component,
                "response": response,
                "offset_m": offset,
                "n": int(mask.sum()),
                "spearman_r": r,
                "detrended_spearman_r": detrended_r,
            }
        )
    return rows


def response_residual_series(context: dict[str, Any], response: str) -> np.ndarray:
    residuals = persistence_residuals(context["y_test"], context["X_test"])
    idx = TARGET_NAMES.index(response)
    return residuals[:, idx].astype(np.float64)


def linear_detrend(values: np.ndarray, chainage: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    chainage = np.asarray(chainage, dtype=np.float64)
    if len(values) < 3 or np.std(values) == 0 or np.std(chainage) == 0:
        return values - np.mean(values)
    coef = np.polyfit(chainage, values, deg=1)
    return values - np.polyval(coef, chainage)


def circular_shift_pvalue(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    observed, _ = safe_spearman(x, y)
    if len(x) < 4:
        return 1.0, 0
    shifted = []
    for shift in range(1, len(x)):
        r, _ = safe_spearman(x, np.roll(y, shift))
        shifted.append(r)
    shifted = np.asarray(shifted, dtype=np.float64)
    p = (1.0 + float(np.sum(np.abs(shifted) >= abs(observed)))) / (1.0 + len(shifted))
    return float(p), int(len(shifted))


def anomaly_variant_series(context: dict[str, Any], component: str) -> dict[str, np.ndarray]:
    variants: dict[str, list[float]] = {
        "Vp_main": [],
        "Vs_low": [],
        "VpVs_low": [],
        "Vp_Vs_mean": [],
    }
    for graph_seq in context["test_graph_seqs"]:
        snapshot = graph_seq[-1]
        rock_attrs = snapshot.rock_attrs.detach().cpu().numpy()
        q_vp = vp_anomaly_from_standardized_attrs(rock_attrs, context.get("vp_anomaly_reference"))
        q_vs = low_attribute_score(rock_attrs, ATTR_INDEX["Vs"])
        q_ratio = low_attribute_score(rock_attrs, ATTR_INDEX["Vp/Vs"])
        q_multi = np.mean(np.vstack([q_vp, q_vs]), axis=0).astype(np.float32)
        for name, q in [
            ("Vp_main", q_vp),
            ("Vs_low", q_vs),
            ("VpVs_low", q_ratio),
            ("Vp_Vs_mean", q_multi),
        ]:
            rows = component_descriptors_for_snapshot(snapshot, anomaly_scores=q)
            variants[name].append(next(row.interaction_intensity for row in rows if row.component == component))
    return {key: np.asarray(value, dtype=np.float64) for key, value in variants.items()}


def top_contributing_edges(context: dict[str, Any], component: str, response: str, top_n: int = 8) -> list[dict[str, Any]]:
    residual = response_residual_series(context, response)
    sample_idx = int(np.argmax(np.abs(residual)))
    snapshot = context["test_graph_seqs"][sample_idx][-1]
    chainage = float(context["test_chainages"][sample_idx])
    rock_attrs = snapshot.rock_attrs.detach().cpu().numpy()
    q = vp_anomaly_from_standardized_attrs(rock_attrs, context.get("vp_anomaly_reference"))

    data = snapshot.hetero_data
    edge_store = data["rock", "interact", "tbm"]
    edge_index = edge_store.edge_index.detach().cpu().numpy()
    src = edge_index[0]
    dst = edge_index[1]
    weights = np.asarray(edge_store["edge_attrs"]["geometry_prior"], dtype=np.float64).reshape(-1)
    distances = np.asarray(edge_store["edge_attrs"]["distance"], dtype=np.float64).reshape(-1)
    kappas = np.asarray(edge_store["edge_attrs"]["kappa"], dtype=np.float64).reshape(-1)

    tbm_components = snapshot.tbm_components.argmax(dim=1).detach().cpu().numpy()
    component_ids = {name: cid for cid, name in COMPONENT_NAMES.items()}
    mask = tbm_components[dst] == component_ids[component]
    contribution = weights * q[src]
    order = np.argsort(contribution[mask])[::-1][:top_n]
    masked_indices = np.where(mask)[0][order]

    rock_x = snapshot.hetero_data["rock"].x.detach().cpu().numpy()
    tbm_x = snapshot.hetero_data["tbm"].x.detach().cpu().numpy()
    rock_node_ids = snapshot.hetero_data["rock"].node_ids.detach().cpu().numpy()
    tbm_node_ids = snapshot.hetero_data["tbm"].node_ids.detach().cpu().numpy()

    rows = []
    for rank, edge_pos in enumerate(masked_indices, start=1):
        rock_local = int(src[edge_pos])
        tbm_local = int(dst[edge_pos])
        rows.append(
            {
                "sample_idx": sample_idx,
                "chainage": chainage,
                "component": component,
                "response": response,
                "residual": float(residual[sample_idx]),
                "rank": rank,
                "rock_node_id": int(rock_node_ids[rock_local]),
                "tbm_node_id": int(tbm_node_ids[tbm_local]),
                "rock_x": float(rock_x[rock_local, 0]),
                "rock_y": float(rock_x[rock_local, 1]),
                "rock_z": float(rock_x[rock_local, 2]),
                "tbm_x": float(tbm_x[tbm_local, 0]),
                "tbm_y": float(tbm_x[tbm_local, 1]),
                "tbm_z": float(tbm_x[tbm_local, 2]),
                "distance": float(distances[edge_pos]),
                "kappa": float(kappas[edge_pos]),
                "weight": float(weights[edge_pos]),
                "anomaly_score": float(q[rock_local]),
                "weighted_contribution": float(contribution[edge_pos]),
            }
        )
    return rows


def process_case(config_path: Path, output_dir: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    case_id = cfg.get("case", {}).get("id", config_path.stem)
    context = build_descriptor_context(cfg, "cpu")
    component, response = DIAGNOSTIC_PAIRS[case_id]
    descriptor_series = descriptor_series_for_component(context, component)
    residual = response_residual_series(context, response)
    chainage = descriptor_series["chainage_only"]

    primary_rows = []
    null_rows = []
    # all_component_ic is a matrix for the permutation test, not a descriptor variant
    all_ic_matrix = descriptor_series.pop("all_component_ic")
    for variant, values in descriptor_series.items():
        r, _ = safe_spearman(values, residual)
        detrended_r, _ = safe_spearman(
            linear_detrend(values, chainage),
            linear_detrend(residual, chainage),
        )
        perm_p, n_perm = circular_shift_pvalue(values, residual)
        row = {
            "case_id": case_id,
            "case_label": CASE_LABELS[case_id],
            "component": component,
            "response": response,
            "variant": variant,
            "n": len(values),
            "spearman_r": r,
            "detrended_spearman_r": detrended_r,
            "circular_shift_p": perm_p,
            "n_circular_shifts": n_perm,
        }
        null_rows.append(row)
        if variant == "proposed":
            primary_rows.append(row)

    # Formal component-label permutation test (detrended, consistent with
    # Table null_comparison)
    perm_p_value, perm_obs_r, n_label_perm = component_label_permutation_test(
        all_ic_matrix, component, residual, chainage, n_perm=1000
    )
    permutation_row = {
        "case_id": case_id,
        "case_label": CASE_LABELS[case_id],
        "component": component,
        "response": response,
        "variant": "component_label_permutation",
        "n": len(residual),
        "observed_detrended_r": perm_obs_r,
        "permutation_p": perm_p_value,
        "n_permutations": n_label_perm,
    }
    null_rows.append(permutation_row)
    component_series_rows = component_series_reassignment_rows(
        case_id, all_ic_matrix, component, response, residual, chainage
    )

    anomaly_rows = []
    for variant, values in anomaly_variant_series(context, component).items():
        r, _ = safe_spearman(values, residual)
        detrended_r, _ = safe_spearman(
            linear_detrend(values, chainage),
            linear_detrend(residual, chainage),
        )
        anomaly_rows.append(
            {
                "case_id": case_id,
                "case_label": CASE_LABELS[case_id],
                "component": component,
                "response": response,
                "anomaly_variant": variant,
                "n": len(values),
                "spearman_r": r,
                "detrended_spearman_r": detrended_r,
            }
        )

    trace_rows = top_contributing_edges(context, component, response)
    offset_rows = alignment_offset_rows(case_id, component, response, descriptor_series, residual)
    delta_rows = delta_field_rows(case_id, component, response, descriptor_series, residual)
    relation_rows = relation_support_ablation_rows(case_id, component, response, context, residual, chainage)
    edge_concentration = edge_concentration_rows(case_id, component, response, context)

    case_dir = output_dir / case_id
    # Separate permutation result from the null comparison table to keep CSV
    # fieldnames consistent across rows.
    permutation_rows = [permutation_row]
    null_rows_without_perm = [r for r in null_rows if r.get("variant") != "component_label_permutation"]
    write_csv(null_rows_without_perm, case_dir / "primary_null_comparison.csv")
    write_csv(permutation_rows, case_dir / "component_label_permutation.csv")
    write_csv(component_series_rows, case_dir / "component_series_reassignment.csv")
    write_csv(anomaly_rows, case_dir / "anomaly_sensitivity.csv")
    write_csv(offset_rows, case_dir / "alignment_offset_sensitivity.csv")
    write_csv(trace_rows, case_dir / "top_contributing_edges.csv")
    write_csv(delta_rows, case_dir / "delta_field_association.csv")
    write_csv(relation_rows, case_dir / "relation_support_ablation.csv")
    write_csv(edge_concentration, case_dir / "edge_contribution_concentration.csv")
    save_json(
        {
            "case_id": case_id,
            "case_label": CASE_LABELS[case_id],
            "config": str(config_path),
            "primary_pair": {"component": component, "response": response},
            "split_counts": context["split_counts"],
            "null_comparison": null_rows_without_perm,
            "component_label_permutation": permutation_row,
            "component_series_reassignment": component_series_rows,
            "anomaly_sensitivity": anomaly_rows,
            "alignment_offset_sensitivity": offset_rows,
            "delta_field_association": delta_rows,
            "relation_support_ablation": relation_rows,
            "edge_contribution_concentration": edge_concentration,
            "traceability_top_edges": trace_rows,
        },
        case_dir / "descriptor_diagnostics_summary.json",
    )
    return {
        "case_id": case_id,
        "primary_rows": primary_rows,
        "null_rows": null_rows,
        "anomaly_rows": anomaly_rows,
        "offset_rows": offset_rows,
        "trace_rows": trace_rows,
        "delta_rows": delta_rows,
        "relation_rows": relation_rows,
        "edge_concentration_rows": edge_concentration,
        "permutation_row": permutation_row,
        "component_series_rows": component_series_rows,
    }


def main() -> None:
    exp_dir = Path(__file__).resolve().parent.parent
    os.chdir(exp_dir)
    output_dir = exp_dir / "outputs" / "descriptor_diagnostics"
    all_primary = []
    all_null = []
    all_anomaly = []
    all_offset = []
    all_trace = []
    all_delta = []
    all_relation = []
    all_edge_concentration = []
    all_permutation = []
    all_component_series = []
    for rel_config in CASE_CONFIGS:
        result = process_case(exp_dir / rel_config, output_dir)
        all_primary.extend(result["primary_rows"])
        all_null.extend(result["null_rows"])
        all_anomaly.extend(result["anomaly_rows"])
        all_offset.extend(result["offset_rows"])
        all_trace.extend(result["trace_rows"])
        all_delta.extend(result["delta_rows"])
        all_relation.extend(result["relation_rows"])
        all_edge_concentration.extend(result["edge_concentration_rows"])
        all_permutation.append(result["permutation_row"])
        all_component_series.extend(result["component_series_rows"])

    # Exclude permutation rows from null comparison table (different schema)
    all_null_without_perm = [r for r in all_null if r.get("variant") != "component_label_permutation"]
    write_csv(all_primary, output_dir / "primary_pair_diagnostics.csv")
    write_csv(all_null_without_perm, output_dir / "null_model_comparison.csv")
    write_csv(all_permutation, output_dir / "component_label_permutation.csv")
    write_csv(all_component_series, output_dir / "component_series_reassignment.csv")
    write_csv(all_anomaly, output_dir / "anomaly_definition_sensitivity.csv")
    write_csv(all_offset, output_dir / "alignment_offset_sensitivity.csv")
    write_csv(all_trace, output_dir / "top_contributing_edges_all.csv")
    write_csv(all_delta, output_dir / "delta_field_association.csv")
    write_csv(all_relation, output_dir / "relation_support_ablation.csv")
    write_csv(all_edge_concentration, output_dir / "edge_contribution_concentration.csv")
    print(f"Saved descriptor diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
