from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from submissions.force_directed_sa.types import Benchmark, Net, Node, PinRef

_FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_NODE_ROW_RE = re.compile(
    r"^\s*(\d+)\s+([-+eE0-9.]+)\s+([-+eE0-9.]+)\s+(\S+)\s+(\d+)\s*$"
)


@dataclass
class _RawNode:
    name: str
    inputs: list[str]
    attrs: dict[str, object]


def _parse_plc_header(plc_path: Path) -> tuple[int, int, float, float, float, float, int, float]:
    grid_cols = 0
    grid_rows = 0
    width = 0.0
    height = 0.0
    routes_h = 0.0
    routes_v = 0.0
    smooth_range = 0
    overlap_threshold = 0.0
    for line in plc_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("#"):
            break
        if "Columns" in line and "Rows" in line:
            nums = _FLOAT_RE.findall(line)
            if len(nums) >= 2:
                grid_cols = int(float(nums[0]))
                grid_rows = int(float(nums[1]))
        elif line.startswith("# Width") and "Height" in line:
            nums = _FLOAT_RE.findall(line)
            if len(nums) >= 2:
                width = float(nums[0])
                height = float(nums[1])
        elif line.startswith("# Width"):
            nums = _FLOAT_RE.findall(line)
            if nums:
                width = float(nums[0])
        elif line.startswith("# Routes per micron"):
            nums = _FLOAT_RE.findall(line)
            if len(nums) >= 2:
                routes_h = float(nums[0])
                routes_v = float(nums[1])
        elif line.startswith("# Smoothing factor"):
            nums = _FLOAT_RE.findall(line)
            if nums:
                smooth_range = int(float(nums[0]))
        elif line.startswith("# Overlap threshold"):
            nums = _FLOAT_RE.findall(line)
            if nums:
                overlap_threshold = float(nums[0])
        elif line.startswith("# Height"):
            nums = _FLOAT_RE.findall(line)
            if nums:
                height = float(nums[0])
    if width <= 0.0 or height <= 0.0:
        for line in plc_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# Width") and "Height" in line:
                nums = _FLOAT_RE.findall(line)
                if len(nums) >= 2:
                    width = float(nums[0])
                    height = float(nums[1])
                    break
    if grid_cols <= 0 or grid_rows <= 0:
        raise ValueError(f"Could not parse placement grid from {plc_path}")
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"Could not parse canvas dimensions from {plc_path}")
    return grid_cols, grid_rows, width, height, routes_h, routes_v, smooth_range, overlap_threshold


def _parse_plc_rows(plc_path: Path) -> dict[int, tuple[float, float, str, bool]]:
    result: dict[int, tuple[float, float, str, bool]] = {}
    for line in plc_path.read_text(encoding="utf-8").splitlines():
        m = _NODE_ROW_RE.match(line)
        if not m:
            continue
        idx = int(m.group(1))
        x = float(m.group(2))
        y = float(m.group(3))
        orient = m.group(4)
        fixed = m.group(5) == "1"
        result[idx] = (x, y, orient, fixed)
    if not result:
        raise ValueError(f"Could not parse node rows from {plc_path}")
    return result


def _parse_attr(lines: list[str], start_idx: int) -> tuple[tuple[str, object] | None, int]:
    key: str | None = None
    val: object | None = None
    depth = 1
    i = start_idx + 1
    while i < len(lines):
        stripped = lines[i].strip()
        depth += lines[i].count("{")
        depth -= lines[i].count("}")
        if stripped.startswith("key:"):
            key = stripped.split('"')[1]
        elif stripped.startswith("f:"):
            nums = _FLOAT_RE.findall(stripped)
            if nums:
                val = float(nums[0])
        elif stripped.startswith("placeholder:"):
            parts = stripped.split('"')
            if len(parts) >= 2:
                val = parts[1]
        if depth == 0:
            break
        i += 1
    if key is None:
        return None, i
    return (key, val), i


def _iter_node_blocks(pb_text: str) -> list[list[str]]:
    lines = pb_text.splitlines()
    blocks: list[list[str]] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() != "node {":
            i += 1
            continue
        depth = 1
        block = [lines[i]]
        i += 1
        while i < len(lines) and depth > 0:
            block.append(lines[i])
            depth += lines[i].count("{")
            depth -= lines[i].count("}")
            i += 1
        blocks.append(block)
    return blocks


def _parse_raw_nodes(pb_path: Path) -> list[_RawNode]:
    raw_nodes: list[_RawNode] = []
    text = pb_path.read_text(encoding="utf-8")
    for block in _iter_node_blocks(text):
        name = ""
        inputs: list[str] = []
        attrs: dict[str, object] = {}
        i = 0
        while i < len(block):
            stripped = block[i].strip()
            if stripped.startswith("name:"):
                parts = stripped.split('"')
                if len(parts) >= 2:
                    name = parts[1]
            elif stripped.startswith("input:"):
                parts = stripped.split('"')
                if len(parts) >= 2:
                    inputs.append(parts[1])
            elif stripped.startswith("attr {"):
                parsed, i = _parse_attr(block, i)
                if parsed is not None:
                    k, v = parsed
                    attrs[k] = v
            i += 1
        raw_nodes.append(_RawNode(name=name, inputs=inputs, attrs=attrs))
    return raw_nodes


def _norm_type(raw: object) -> str:
    if not isinstance(raw, str):
        return ""
    return raw.strip().upper()


def load_benchmark(benchmark_dir: Path) -> Benchmark:
    benchmark_id = benchmark_dir.name
    pb_path = benchmark_dir / "netlist.pb.txt"
    plc_path = benchmark_dir / "initial.plc"
    if not pb_path.exists() or not plc_path.exists():
        raise FileNotFoundError(f"Missing benchmark files in {benchmark_dir}")

    grid_cols, grid_rows, canvas_w, canvas_h, routes_h, routes_v, smooth, overlap = _parse_plc_header(
        plc_path
    )
    plc_rows = _parse_plc_rows(plc_path)
    raw_nodes = _parse_raw_nodes(pb_path)

    nodes: list[Node] = []
    physical_name_to_idx: dict[str, int] = {}
    pin_name_to_ref: dict[str, PinRef] = {}
    pin_name_to_weight: dict[str, float] = {}
    pin_name_to_macro: dict[str, str] = {}

    physical_raw_indices: list[int] = []
    for raw_idx, rn in enumerate(raw_nodes):
        ntype = _norm_type(rn.attrs.get("type"))
        if rn.name == "__metadata__":
            continue
        if ntype == "MACRO_PIN":
            macro_name = str(rn.attrs.get("macro_name", ""))
            x_off = float(rn.attrs.get("x_offset", 0.0) or 0.0)
            y_off = float(rn.attrs.get("y_offset", 0.0) or 0.0)
            pin_name_to_weight[rn.name] = float(rn.attrs.get("weight", 1.0) or 1.0)
            if macro_name:
                pin_name_to_macro[rn.name] = macro_name
                pin_name_to_ref[rn.name] = PinRef(node_idx=-1, x_offset=x_off, y_offset=y_off)
            continue
        if ntype == "":
            continue
        width = float(rn.attrs.get("width", 0.0) or 0.0)
        height = float(rn.attrs.get("height", 0.0) or 0.0)
        x = float(rn.attrs.get("x", 0.0) or 0.0)
        y = float(rn.attrs.get("y", 0.0) or 0.0)
        node = Node(
            name=rn.name,
            node_type=ntype,
            width=width,
            height=height,
            x=x,
            y=y,
            orientation="N",
            is_fixed=False,
        )
        physical_name_to_idx[rn.name] = len(nodes)
        nodes.append(node)
        physical_raw_indices.append(raw_idx)

    for pin_name, pref in list(pin_name_to_ref.items()):
        macro_name = pin_name_to_macro.get(pin_name, "")
        node_idx = physical_name_to_idx.get(macro_name)
        if node_idx is None:
            pin_name_to_ref.pop(pin_name, None)
            continue
        pin_name_to_ref[pin_name] = PinRef(node_idx=node_idx, x_offset=pref.x_offset, y_offset=pref.y_offset)

    for plc_idx, node in enumerate(nodes):
        row = plc_rows.get(plc_idx)
        if row is None:
            continue
        x, y, orient, fixed = row
        node.x = x
        node.y = y
        node.orientation = "N" if orient == "-" else orient
        node.is_fixed = fixed

    nets: list[Net] = []
    for rn in raw_nodes:
        if not rn.inputs:
            continue
        endpoint_names = [rn.name, *rn.inputs]
        pin_refs: list[PinRef] = []
        seen: set[tuple[int, float, float]] = set()
        for ep in endpoint_names:
            pref = pin_name_to_ref.get(ep)
            if pref is not None and pref.node_idx >= 0:
                key = (pref.node_idx, pref.x_offset, pref.y_offset)
                if key not in seen:
                    seen.add(key)
                    pin_refs.append(pref)
                continue
            node_idx = physical_name_to_idx.get(ep)
            if node_idx is None:
                continue
            key = (node_idx, 0.0, 0.0)
            if key not in seen:
                seen.add(key)
                pin_refs.append(PinRef(node_idx=node_idx, x_offset=0.0, y_offset=0.0))
        if len(pin_refs) < 2:
            continue
        weight = float(rn.attrs.get("weight", pin_name_to_weight.get(rn.name, 1.0)) or 1.0)
        nets.append(Net(pin_refs=pin_refs, weight=weight, name=rn.name))

    movable_indices = [
        i
        for i, n in enumerate(nodes)
        if n.node_type == "MACRO"
        and not n.name.startswith("Grp_")
        and not n.is_fixed
        and n.width > 0.0
        and n.height > 0.0
    ]
    node_to_nets: list[list[int]] = [[] for _ in nodes]
    for net_idx, net in enumerate(nets):
        touched = {pr.node_idx for pr in net.pin_refs}
        for node_idx in touched:
            node_to_nets[node_idx].append(net_idx)

    return Benchmark(
        benchmark_id=benchmark_id,
        canvas_width=canvas_w,
        canvas_height=canvas_h,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        routes_h=routes_h,
        routes_v=routes_v,
        smooth_range=smooth,
        overlap_threshold=overlap,
        nodes=nodes,
        nets=nets,
        movable_indices=movable_indices,
        node_to_nets=node_to_nets,
    )

