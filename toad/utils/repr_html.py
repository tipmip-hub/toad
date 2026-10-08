"""
Helpers for HTML representation of the TOAD object.

Used by TOAD._repr_html_() to build the collapsible variable hierarchy.
"""

import base64
import os
import uuid
from typing import TYPE_CHECKING

from . import _attrs

if TYPE_CHECKING:
    import xarray as xr


def build_variable_hierarchy(
    base_vars: list[str],
    shift_vars: list[str],
    cluster_vars: list[str],
    data: "xr.Dataset",
) -> dict:
    """Build a hierarchy dict mapping base vars to their shifts and cluster variables.

    Args:
        base_vars: List of base variable names.
        shift_vars: List of shift variable names.
        cluster_vars: List of cluster variable names.
        data: xarray Dataset containing the variables and their attributes.

    Returns:
        Dict of shape {base_var: {"shifts": [...], "clusters": [...]}}.
    """
    hierarchy: dict = {}

    for base_var in base_vars:
        hierarchy[base_var] = {"shifts": [], "clusters": []}

    for shift_var in shift_vars:
        shift_data = data[shift_var]
        base_var = shift_data.attrs.get(
            _attrs.BASE_VARIABLE, shift_var.split("_dts")[0]
        )
        if base_var not in hierarchy:
            hierarchy[base_var] = {"shifts": [], "clusters": []}
        hierarchy[base_var]["shifts"].append({"name": shift_var, "clusters": []})

    for cluster_var in cluster_vars:
        cluster_data = data[cluster_var]
        base_var = cluster_data.attrs.get(_attrs.BASE_VARIABLE)
        shifts_var = cluster_data.attrs.get(_attrs.SHIFTS_VARIABLE)
        if base_var and base_var in hierarchy:
            if shifts_var:
                for shift_info in hierarchy[base_var]["shifts"]:
                    if shift_info["name"] == shifts_var:
                        shift_info["clusters"].append(cluster_var)
                        break
            else:
                hierarchy[base_var]["clusters"].append(cluster_var)

    return hierarchy


def _cluster_count_for_var(data: "xr.Dataset", cluster_var: str) -> int:
    """Get number of clusters for a cluster variable (excluding noise).

    NetCDF round-trips can store a single id (e.g. only noise ``-1``) as a
    scalar attr; coerce to 1-d like ``TOAD.get_cluster_ids``.
    """
    import numpy as np

    raw = data[cluster_var].attrs.get(_attrs.CLUSTER_IDS)
    if raw is None:
        return 0
    ids = np.atleast_1d(np.asarray(raw, dtype=np.int64))
    return int(np.sum(ids != -1))


def render_hierarchy_html(
    hierarchy: dict,
    data: "xr.Dataset",
) -> str:
    """Render the variable hierarchy as HTML with collapsible sections.

    Args:
        hierarchy: Dict from build_variable_hierarchy().
        data: xarray Dataset (for cluster counts).

    Returns:
        HTML string for the variable table, or empty string if hierarchy is empty.
    """
    if not hierarchy:
        return ""

    instance_id = str(uuid.uuid4()).replace("-", "")
    hierarchy_html: list[str] = []

    for base_var in sorted(hierarchy.keys()):
        info = hierarchy[base_var]
        shift_count = len(info["shifts"])
        cluster_count = sum(len(s["clusters"]) for s in info["shifts"]) + len(
            info["clusters"]
        )
        base_id = f"{instance_id}_base_{base_var.replace('.', '_')}"

        if shift_count == 0 and cluster_count == 0:
            hierarchy_html.append(_render_base_no_children(base_var))
            continue

        hierarchy_html.append(
            _render_base_expandable(
                base_var, base_id, shift_count, cluster_count, instance_id
            )
        )
        hierarchy_html.append(
            _render_shifts(
                info["shifts"],
                data,
                instance_id,
            )
        )
        hierarchy_html.append("</div></div>")

    return _wrap_hierarchy_table(
        "".join(hierarchy_html),
        instance_id,
    )


def render_consensus_variables_html(
    data: "xr.Dataset",
    consensus_label_vars: list[str],
) -> str:
    """Render a compact section listing consensus label variables and their rate fields.

    Consensus outputs are not folded into the base/shift/cluster tree; they appear in a
    separate block below.

    Returns:
        Empty string when ``consensus_label_vars`` is empty, so the section is omitted entirely.
    """
    if not consensus_label_vars:
        return ""

    rows: list[str] = []
    for name in sorted(consensus_label_vars):
        n_clusters = _cluster_count_for_var(data, name)
        from toad import TOAD

        rate_name = TOAD.consensus_rate_var_name(name)
        has_rate = rate_name in data.data_vars
        rate_note = (
            f'<span style="opacity: 0.55; font-size: 0.85em;"> + {rate_name}</span>'
            if has_rate
            else ""
        )
        rows.append(
            f"""
                <div style="margin: 6px 0;">
                    <span style="color: black; background-color: #D4C4F5; padding: 2px 4px; border-radius: 4px;">consensus</span>
                    <span style="font-family: monospace;">{name}</span>{rate_note}
                    <span style="opacity: 0.5; font-size: 0.85em;"> ({n_clusters} clusters)</span>
                </div>
            """
        )

    return f"""
            <div style='margin: 12px 0px 10px 0px;'>
                <h4 style="margin: 5px 0; font-size: 1.1em;">Consensus variables</h4>
                <div style="font-family: monospace; font-size: 1.0em; border: 1px solid #ddd; padding: 10px; line-height: 1.45;">
                    {"".join(rows)}
                </div>
            </div>
            """


def _render_base_no_children(base_var: str) -> str:
    """Render a base variable row with no derived variables."""
    return f"""
                <div style="margin: 2px 0;">
                    <span style="font-family: monospace; font-weight: bold; opacity: 0;">▶</span>
                    <span style="color: black; background-color: #A8D5FF; padding: 2px 4px; border-radius: 4px;">base var</span> {base_var}
                    <span style="opacity: 0.5; font-size: 0.85em;">
                        (no shifts or clusterings)
                    </span>
                </div>
                """


def _render_base_expandable(
    base_var: str,
    base_id: str,
    shift_count: int,
    cluster_count: int,
    instance_id: str,
) -> str:
    """Render an expandable base variable row."""
    return f"""
                <div style="margin: 2px 0;">
                    <span onclick="toggleSection_{instance_id}('{base_id}')" style="cursor: pointer; user-select: none;">
                        <span id="{base_id}_arrow" style="font-family: monospace; font-weight: bold;">▶</span>
                        <span style="color: black; background-color: #A8D5FF; padding: 2px 4px; border-radius: 4px;">base var</span> {base_var}
                        <span style="opacity: 0.5; font-size: 0.85em;">
                            ({shift_count} shifts, {cluster_count} clusterings)
                        </span>
                    </span>
                    <div id="{base_id}_content" style="display: none; margin-left: 20px; margin-top: 5px;">
                """


def _render_shifts(
    shifts_info: list[dict],
    data: "xr.Dataset",
    instance_id: str,
) -> str:
    """Render shift variables and their cluster variables."""
    parts: list[str] = []
    for shift_info in shifts_info:
        shift_var = shift_info["name"]
        shift_clusters = shift_info["clusters"]
        shift_id = f"{instance_id}_shift_{shift_var.replace('.', '_')}"

        if shift_clusters:
            parts.append(
                f"""
                        <div style="margin: 4px 0;">
                            <span onclick="toggleSection_{instance_id}('{shift_id}')" style="cursor: pointer; user-select: none;">
                                <span id="{shift_id}_arrow" style="font-family: monospace; font-weight: bold;">▶</span>
                                <span style="color: black; background-color: #FFE0A3; padding: 2px 4px; border-radius: 4px;">shifts var</span> {shift_var} <span style="opacity: 0.5; font-size: 0.85em;">({len(shift_clusters)} clusterings)</span>
                            </span>
                            <div id="{shift_id}_content" style="display: none; margin-left: 20px; margin-top: 3px;">
                        """
            )
            for cluster_var in shift_clusters:
                n_clusters = _cluster_count_for_var(data, cluster_var)
                parts.append(
                    f"""
                            <div style="margin-left: 12px; padding: 2px 0px;">
                                <span style="color: black; background-color: #B8E6C1; padding: 2px 4px; border-radius: 4px;">cluster var</span> {cluster_var} <span style="opacity: 0.5; font-size: 0.85em;">({n_clusters} clusters)</span>
                            </div>
                            """
                )
            parts.append("</div></div>")
        else:
            parts.append(
                f"""
                        <div style="margin: 2px 0;">
                            <span style="font-family: monospace; font-weight: bold; opacity: 0;">▶</span>
                            <span style="color: black; background-color: #FFE0A3; padding: 2px 4px; border-radius: 4px;">shifts var</span> {shift_var}  <span style="opacity: 0.5; font-size: 0.85em;">({len(shift_clusters)} clusterings)</span>
                        </div>
                        """
            )
    return "".join(parts)


def _wrap_hierarchy_table(hierarchy_html: str, instance_id: str) -> str:
    """Wrap hierarchy HTML in the table div and toggle script."""
    return f"""
            <div style='margin: 10px 0px;'>
                <h4 style="margin: 5px 0; font-size: 1.1em;">Variable Hierarchy:</h4>
                <div style="font-family: monospace; font-size: 1.0em; border: 1px solid #ddd; padding: 10px; line-height: 1.4;">
                    {hierarchy_html}
                </div>
            </div>
            
            <script>
            function toggleSection_{instance_id}(sectionId) {{
                const content = document.getElementById(sectionId + '_content');
                const arrow = document.getElementById(sectionId + '_arrow');
                
                if (content.style.display === 'none') {{
                    content.style.display = 'block';
                    arrow.textContent = '▼';
                }} else {{
                    content.style.display = 'none';
                    arrow.textContent = '▶';
                }}
            }}

            // Auto-expand logic
            function autoExpand_{instance_id}() {{
                let visibleClusterings = 0;
                const maxVisible = 10;
                
                const baseSections = document.querySelectorAll('[id^="{instance_id}_base_"][id$="_arrow"]');
                const sectionCounts = [];
                
                baseSections.forEach(baseArrow => {{
                    const baseId = baseArrow.id.replace('_arrow', '');
                    const baseContent = document.getElementById(baseId + '_content');
                    const clusterCount = baseContent.querySelectorAll('[style*="background-color: lightgreen"]').length;
                    sectionCounts.push({{baseId, clusterCount}});
                }});
                
                sectionCounts.sort((a, b) => a.clusterCount - b.clusterCount);
                
                for (const {{baseId, clusterCount}} of sectionCounts) {{
                    if (visibleClusterings + clusterCount <= maxVisible) {{
                        const baseContent = document.getElementById(baseId + '_content');
                        const baseArrow = document.getElementById(baseId + '_arrow');
                        
                        baseContent.style.display = 'block';
                        baseArrow.textContent = '▼';
                        
                        const shiftSections = baseContent.querySelectorAll('[id^="{instance_id}_shift_"][id$="_arrow"]');
                        shiftSections.forEach(shiftArrow => {{
                            const shiftId = shiftArrow.id.replace('_arrow', '');
                            const shiftContent = document.getElementById(shiftId + '_content');
                            shiftArrow.textContent = '▼';
                            shiftContent.style.display = 'block';
                        }});
                        
                        visibleClusterings += clusterCount;
                    }}
                }}
            }}

            autoExpand_{instance_id}();
            </script>
            """


def load_toad_logo_html() -> str:
    """Load the TOAD logo and return as base64-encoded HTML img tag, or empty string on failure."""
    try:
        # repr_html.py is in toad/utils/; repo root is two levels up
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        logo_path = os.path.abspath(
            os.path.join(repo_root, "docs", "source", "resources", "toad.png")
        )
        if os.path.exists(logo_path):
            with open(logo_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode()
            return f'<img src="data:image/png;base64,{img_data}" style="height: 40px; margin-right: 10px; vertical-align: middle;">'
    except Exception:
        pass
    return ""
