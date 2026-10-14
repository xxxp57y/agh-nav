#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""navlive_map.py —— 楼层图渲染的**定稿实现**（服务器侧与本地离线都用这一份）。

两个坑（都在服务器上实测踩过，写在这里免得后人再踩）：

1. `top_down_map["map"]` **不是灰度图**，是 habitat 的"语义索引图"：每个像素是
   `habitat.utils.visualizations.maps` 里的常量（0 无效 / 1 可行走 / 2 墙 / 4 起点 /
   6 目标 / 7 最短路径 / 9 目标框）。直接 `Image.fromarray` 存出来几乎是纯黑
   （像素值只有 0~17），必须先过 `colorize_topdown_map()` 才有人能看的图。
2. `fog_of_war.draw=True` 时未探索区域会被压暗成 0.5 倍；要一张"整层楼底图"就得关掉它
   （`FogOfWarConfig(draw=False, visibility_dist=100, fov=180)`）。

本模块只做渲染，不碰仿真：输入 map / fog / agent_map_coord / 轨迹点，输出一张 PNG。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def colorize(map_info: Dict[str, Any], *, use_fog: bool = False):
    """语义图 → RGB。`use_fog=False` 时忽略 fog_of_war_mask（拿到完整楼层图）。"""
    import numpy as np
    from habitat.utils.visualizations import maps as habitat_maps

    raw = np.asarray(map_info["map"])
    fog = np.asarray(map_info["fog_of_war_mask"]) if use_fog and map_info.get("fog_of_war_mask") is not None else None
    return habitat_maps.colorize_topdown_map(raw, fog)


def crop_to_content(image, pad: int = 16):
    """把楼层图裁到有内容的外接矩形（工作台里做自适应缩放用）。

    注意：必须按"非白"来找内容 —— 语义图里 0（无效）是白色 [255,255,255]。
    """
    import numpy as np

    arr = np.asarray(image)
    # 白底 = 无效区域；取"明显不是白"的像素
    non_white = (arr.astype(np.int16).sum(axis=2) < 720) if arr.ndim == 3 else (arr < 240)
    rows = np.where(non_white.any(axis=1))[0]
    cols = np.where(non_white.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return image, (0, 0)
    r0, r1 = max(int(rows[0]) - pad, 0), min(int(rows[-1]) + pad + 1, arr.shape[0])
    c0, c1 = max(int(cols[0]) - pad, 0), min(int(cols[-1]) + pad + 1, arr.shape[1])
    return image.crop((c0, r0, c1, r1)), (c0, r0)


def render_with_origin(
    map_info: Dict[str, Any],
    path_rc: Sequence[Tuple[float, float]] = (),
    *,
    goal_rc: Optional[Tuple[float, float]] = None,
    use_fog: bool = False,
    crop: bool = True,
    scale: float = 1.0,
    pad: int = 16,
    wall_color: Tuple[int, int, int] = (250, 250, 248),
    floor_color: Tuple[int, int, int] = (206, 208, 205),
    path_color: Tuple[int, int, int] = (232, 90, 20),
    landmarks: Sequence[Dict[str, Any]] = (),
    landmark_color: Tuple[int, int, int] = (44, 110, 200),
):
    """与 `render` 相同，但**同时返回裁剪原点**，方便调用方继续在同一坐标系上画图。

    坐标系与 habitat 一致：(row, col)。`agent_map_coord` 就是 (row, col)。
    `landmarks` 是 [{'name': str, 'rc': (row, col)}] —— 画成小方点 + 名字，
    这样"要去的客厅"在图上看得见，而不是只有一个绿点。
    返回值：(PIL.Image, (origin_col, origin_row))
    """
    import numpy as np
    from PIL import Image, ImageDraw

    raw = np.asarray(map_info["map"])
    arr = np.zeros(raw.shape + (3,), dtype=np.uint8)
    valid = raw != 0
    arr[valid] = floor_color
    border = (raw == 2) | (raw == 9)
    arr[border] = wall_color
    if use_fog and map_info.get("fog_of_war_mask") is not None:
        fog = np.asarray(map_info["fog_of_war_mask"]).astype(bool)
        hidden = (~fog) & valid
        arr[hidden] = (np.asarray(floor_color) * 0.6).astype(np.uint8)

    image = Image.fromarray(arr)
    if crop:
        image, origin = crop_to_content(image, pad=pad)
    else:
        origin = (0, 0)

    draw = ImageDraw.Draw(image)
    ox, oy = origin
    if len(path_rc) > 1:
        pts = [(float(c) - ox, float(r) - oy) for r, c in path_rc]
        draw.line(pts, fill=path_color, width=3, joint="curve")
    for item in landmarks:
        rc = item.get("rc")
        if not rc:
            continue
        r, c = float(rc[0]) - oy, float(rc[1]) - ox
        if not (0 <= c <= image.width and 0 <= r <= image.height):
            continue
        size = 5
        draw.rectangle([c - size, r - size, c + size, r + size], fill=landmark_color, outline=(255, 255, 255), width=1)
        name = str(item.get("name") or "")
        if name:
            # 服务器上这个 Pillow(9.5) 的默认字体渲染会抛
            # `'ImageFont' object has no attribute 'getmask2'`，所以文字是"尽力而为"：
            # 画不上就算了——前端会用自己的字体在 SVG 上标名字，地图本身的方块照画。
            try:
                draw.text((c + 7, r - 6), name, fill=landmark_color)
            except Exception:  # noqa: BLE001
                pass
    if goal_rc is not None:
        gr, gc = float(goal_rc[0]) - oy, float(goal_rc[1]) - ox
        radius = 6
        draw.ellipse([gc - radius, gr - radius, gc + radius, gr + radius], fill=(0, 160, 60))

    if scale != 1.0 and scale > 0:
        image = image.resize(
            (max(int(image.width * scale), 1), max(int(image.height * scale), 1)),
            Image.LANCZOS,
        )
    return image, origin


def render(map_info: Dict[str, Any], path_rc: Sequence[Tuple[float, float]] = (), **kwargs):
    """只返回图片（大多数调用方不需要原点）。"""
    image, _origin = render_with_origin(map_info, path_rc, **kwargs)
    return image


def draw_robot(image, coord_rc: Tuple[float, float], angle_rad: float, origin=(0, 0), radius: int = 9):
    """在楼层图上画机器人：橙色圆 + 朝向线（对应参考图里"橙色 = 机器人"）。"""
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    r, c = float(coord_rc[0]) - origin[1], float(coord_rc[1]) - origin[0]
    draw.ellipse([c - radius, r - radius, c + radius, r + radius], fill=(240, 120, 10), outline=(120, 50, 0), width=2)
    # habitat 的 agent_angle 是地图坐标系角度；朝向线给个直观指向
    length = radius * 2.2
    ex = c + length * math.cos(float(angle_rad))
    ey = r - length * math.sin(float(angle_rad))
    draw.line([(c, r), (ex, ey)], fill=(120, 50, 0), width=3)
    return image


def render_bytes(map_info: Dict[str, Any], **kwargs) -> bytes:
    """给 HTTP 层用：直接返回 PNG 字节。"""
    import io

    image = render(map_info, **kwargs)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()
