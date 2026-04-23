"""从视频提取的步态特征与临床平衡量表评分的相关性分析。

临床评分数据来源 VFile_parsed.csv:
    - FAC: 功能性步行分类 (1-5/6, 数值越高 = 步行能力越强)
    - BBS: Berg 平衡量表 (0-56, 数值越高 = 平衡能力越好)
    - TIS: 躯干功能评定量表 (0-23, 数值越高 = 躯干控制越好)
    - FMA_LE: Fugl-Meyer 下肢运动功能 (0-34, 数值越高 = 运动功能越好)

视频提取的步态特征:
    - BOS: 支撑底面积（双侧足跟间距，经肩宽归一化）
    - 变异性: 各归一化宽度的标准差/CV（步态稳定性指标）
    - 步频: 每秒足跟碰撞事件数
    - 垂直摆动: 足跟 y 坐标的标准差（步行效率代理指标）
    - 事件对称性: 左右足跟碰撞次数之差
"""

import csv
import math
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------
# 路径初始化：通过向上查找 CLAUDE.md 确定项目根目录
# ---------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
for _ in range(10):
    if (PROJECT_ROOT / "CLAUDE.md").exists():
        break
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 数据文件路径
DEFAULT_PARSED_CSV = PROJECT_ROOT / "data/Accessment/VFile_parsed.csv"   # 临床评分元数据
DEFAULT_ANGLES_DIR = PROJECT_ROOT / "output/export"                       # 视频提取结果（_angles_*.csv）

# ================================================================
# 临床评分配置：量表名称 → 满分值（用于归一化到百分比）
# ================================================================

SCORE_MAX: dict[str, float] = {
    "fac":    5.0,   # Functional Ambulation Classification（满分 5）
    "bbs":   56.0,   # Berg Balance Scale（满分 56）
    "tis":   23.0,   # Trunk Impairment Scale（满分 23）
    "fma_le": 34.0,  # Fugl-Meyer Lower Extremity（满分 34）
}

# 正则：从 assessment_text 中匹配各量表数值
_SCORE_PATTERNS: dict[str, re.Pattern] = {
    "fac":    re.compile(r"FAC\s*(\d+(?:\.\d+)?)\s*级"),
    "bbs":    re.compile(r"BBS\s*(\d+(?:\.\d+)?)\s*分"),
    "tis":    re.compile(r"TIS\s*(\d+(?:\.\d+)?)\s*分"),
    "fma_le": re.compile(r"FMA-?LE\s*(\d+(?:\.\d+)?)\s*分"),
}


def _parse_assessment_text(text: str) -> dict[str, float]:
    """从评估文本中解析各量表评分，返回 {量表名: 原始分数}。

    优先取"基线："阶段的数值（若存在）。
    """
    if not text:
        return {}

    result: dict[str, float] = {}
    has_baseline = "基线" in text

    for score_name, pattern in _SCORE_PATTERNS.items():
        if has_baseline:
            # 取"基线："到"末期："之间的内容进行匹配
            bi = text.find("基线：")
            ei = text.find("末期：")
            search_range = text[bi:ei] if bi >= 0 and ei > bi else text
        else:
            search_range = text

        m = pattern.search(search_range)
        if m:
            result[score_name] = float(m.group(1))

    return result


def _normalize_scores(raw_scores: dict[str, float]) -> dict[str, float]:
    """将原始评分归一化到百分比（0~100），无效值跳过。

    Args:
        raw_scores: {量表名: 原始分数}，如 {"bbs": 37, "tis": 11}

    Returns:
        {量表名: 百分比}，如 {"bbs_pct": 66.07, "tis_pct": 47.83}
    """
    normalized = {}
    for name, raw in raw_scores.items():
        max_val = SCORE_MAX.get(name)
        if max_val and raw > 0:
            normalized[f"{name}_pct"] = (raw / max_val) * 100.0
    return normalized


def _composite_score(pct_dict: dict[str, float]) -> float | None:
    """计算综合评分：所有可用量表百分比的算术平均。

    Args:
        pct_dict: {量表名_pct: 百分比}，如 {"bbs_pct": 66.07, "tis_pct": 47.83}

    Returns:
        综合百分比（0~100），若没有任何有效评分则返回 None
    """
    vals = [v for k, v in pct_dict.items() if k.endswith("_pct") and math.isfinite(v)]
    return sum(vals) / len(vals) if vals else None


# ================================================================
# 工具函数
# ================================================================

def parse_float_or_nan(v: str) -> float:
    """将 CSV 单元格字符串转为 float，空白/无效值返回 NaN。"""
    if not v:
        return float("nan")
    try:
        return float(v)
    except ValueError:
        return float("nan")


def load_parsed_csv(path: Path) -> list[dict[str, str]]:
    """读取 VFile_parsed.csv，返回所有行（字典列表）。"""
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_angles_csv(path: Path) -> list[dict[str, str]]:
    """读取 *_angles_*.csv，返回所有帧数据行。"""
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_events_csv(path: Path) -> tuple[list[float], list[float]]:
    """读取 *_events_*.csv，返回（左足跟碰撞时刻列表, 右足跟碰撞时刻列表）。"""
    left_times, right_times = [], []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        for row in reader:
            if len(row) < 2:
                continue
            side, t = row[0].strip(), row[1].strip()
            try:
                if side == "left":
                    left_times.append(float(t))
                elif side == "right":
                    right_times.append(float(t))
            except ValueError:
                continue
    return left_times, right_times


# ================================================================
# 统计辅助函数（处理含 NaN 的数据）
# ================================================================

def valid(values: list[float]) -> list[float]:
    """返回序列中所有有限值（非 NaN / Inf）。"""
    return [v for v in values if math.isfinite(v)]


def nanmean(values: list[float]) -> float:
    """序列的算术平均值，忽略 NaN；空序列返回 NaN。"""
    v = valid(values)
    return sum(v) / len(v) if v else float("nan")


def nanstd(values: list[float]) -> float:
    """序列的标准差（总体标准差，分母为 n），忽略 NaN；少于 2 个值返回 NaN。"""
    v = valid(values)
    if len(v) < 2:
        return float("nan")
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))


# ================================================================
# 特征提取
# ================================================================

def extract_row_features(
    angles_rows: list[dict[str, str]],
    left_ev: list[float],
    right_ev: list[float],
    duration: float,
) -> dict[str, float]:
    """从一个视频片段中提取步态/平衡特征。

    Args:
        angles_rows: *_angles_*.csv 的所有帧数据行
        left_ev:     左侧足跟碰撞时刻列表
        right_ev:    右侧足跟碰撞时刻列表
        duration:    视频片段时长（秒），用于计算步频

    Returns:
        特征字典，键为特征名，值为浮点数
    """
    f: dict[str, float] = {}

    # ---- 从 CSV 读取各列，转换为 float ----
    knee_w   = [parse_float_or_nan(r.get("knee_width_norm", ""))      for r in angles_rows]
    ankle_w  = [parse_float_or_nan(r.get("ankle_width_norm", ""))    for r in angles_rows]
    bos      = [parse_float_or_nan(r.get("base_of_support_norm", "")) for r in angles_rows]
    hip_w    = [parse_float_or_nan(r.get("hip_width_norm", ""))      for r in angles_rows]
    shoulder = [parse_float_or_nan(r.get("shoulder_width", ""))      for r in angles_rows]
    l_heel_y = [parse_float_or_nan(r.get("left_heel_y", ""))        for r in angles_rows]
    r_heel_y = [parse_float_or_nan(r.get("right_heel_y", ""))       for r in angles_rows]

    # ---- 支撑底面积（BOS = Base of Support）统计 ----
    # BOS = 双侧足跟中心距离（经肩宽归一化）
    # 数值小 = 站立基线窄（平衡差），数值大 = 步态中正常摆动
    f["bos_mean"]  = nanmean(bos)                                  # 平均值
    f["bos_std"]   = nanstd(bos)                                   # 标准差（步态稳定性）
    # 变异系数：标准差/均值，消除量纲差异，跨被试可比
    f["bos_cv"]    = abs(f["bos_std"] / f["bos_mean"]) if abs(f["bos_mean"]) > 1e-8 else float("nan")
    f["bos_range"] = (max(valid(bos)) - min(valid(bos))) if len(valid(bos)) > 1 else float("nan")  # 极差

    # ---- 下肢各关节宽度的变异性（步幅一致性） ----
    # 标准差大 = 步态不规律，波动大
    f["knee_width_std"]   = nanstd(knee_w)
    f["ankle_width_std"]  = nanstd(ankle_w)
    f["hip_width_std"]    = nanstd(hip_w)
    f["shoulder_width_std"] = nanstd(shoulder)  # 肩宽稳定性（镜头运动敏感度参考）

    # ---- BOS 时序稳定性 ----
    # 每帧 BOS 与均值的 min/max 比值的均值；1.0 = 全程稳定
    bv = valid(bos)
    f["bos_stability"] = (
        sum(min(b, f["bos_mean"]) / max(b, f["bos_mean"]) for b in bv) / len(bv)
        if bv else float("nan")
    )

    # ---- 步态事件统计 ----
    f["left_heel_strikes"]  = len(left_ev)                              # 左脚着地次数
    f["right_heel_strikes"] = len(right_ev)                             # 右脚着地次数
    f["event_count_diff"]   = abs(len(left_ev) - len(right_ev))        # 左右事件数差（0=对称）
    f["cadence"]            = (len(left_ev) + len(right_ev)) / duration if duration > 0 else float("nan")  # 步频（步/秒）

    # ---- 垂直摆动（足跟 y 坐标波动） ----
    # 步行效率低时身体上下起伏大，y 标准差增大
    f["left_heel_y_std"]   = nanstd(l_heel_y)
    f["right_heel_y_std"]  = nanstd(r_heel_y)
    f["vertical_oscillation_mean"] = nanmean([f["left_heel_y_std"], f["right_heel_y_std"]])

    # ---- 变异系数（CV = std/mean） ----
    # 与 bos_cv 同理，消除均值差异，可跨视频比较
    f["knee_width_cv"]  = abs(f["knee_width_std"]  / nanmean(knee_w))  if abs(nanmean(knee_w))  > 1e-8 else float("nan")
    f["ankle_width_cv"] = abs(f["ankle_width_std"] / nanmean(ankle_w)) if abs(nanmean(ankle_w)) > 1e-8 else float("nan")

    return f


# ================================================================
# 主流程
# ================================================================

def main() -> None:
    # ---- 1. 加载临床评分元数据 ----
    parsed_rows = load_parsed_csv(DEFAULT_PARSED_CSV)
    print(f"Loaded {len(parsed_rows)} rows from VFile_parsed.csv\n")

    results: list[dict] = []
    skipped = 0

    # ---- 2. 遍历每个患者/视频行，提取特征 ----
    for idx, meta in enumerate(parsed_rows):
        # ---- 读取并合并评分：CSV 列优先，再从 assessment_text 补充 ----
        subject    = meta.get("subject_name", "").strip()
        video      = meta.get("video_name", "").strip()
        affected   = meta.get("Affected Side", "").strip()
        start_time = meta.get("start_time", "").strip()

        raw_scores: dict[str, float] = {}
        # CSV 直接列
        for name in SCORE_MAX:
            val_str = meta.get(name, "").strip()
            if val_str:
                try:
                    raw_scores[name] = float(val_str)
                except ValueError:
                    pass
        # assessment_text 补充（CSV 列已有时不覆盖）
        text_scores = _parse_assessment_text(meta.get("assessment_text", "").strip())
        for name, val in text_scores.items():
            raw_scores.setdefault(name, val)

        # 归一化到百分比
        pct_scores = _normalize_scores(raw_scores)
        # 综合百分比
        composite = _composite_score(pct_scores)
        pct_scores["composite_pct"] = composite if composite is not None else float("nan")

        # 根据 start_time 构建片段标签（如 "4_40to50s"）
        if start_time:
            parts = start_time.replace(":", "_").split("_")
            try:
                start_s = float(parts[0]) * 60.0 + float(parts[1]) if len(parts) >= 2 else float(parts[0])
            except ValueError:
                start_s = 0.0
        else:
            start_s = 0.0
        start_tag = str(int(start_s)) if float(start_s).is_integer() else str(start_s).replace(".", "p")
        segment_tag = f"{start_tag}to{int(start_s + 10)}s"

        # 查找对应的 *_angles_*.csv 文件
        stem = Path(video).stem
        candidates = list(DEFAULT_ANGLES_DIR.glob(f"{stem}_angles_{segment_tag}.csv"))
        if not candidates:
            # 兜底：模糊匹配
            candidates = [f for f in DEFAULT_ANGLES_DIR.glob(f"{stem}*.csv") if "_angles_" in f.name]

        if not candidates:
            skipped += 1
            print(f"  [{idx+1}] {subject}/{video}: angles CSV not found (skipped)")
            continue

        angles_path = candidates[0]
        # 同目录下的 *_events_*.csv
        events_path = angles_path.parent / angles_path.name.replace("_angles_", "_events_")
        left_ev, right_ev = load_events_csv(events_path) if events_path.exists() else ([], [])

        # 提取特征
        angles_rows = load_angles_csv(angles_path)
        feat = extract_row_features(angles_rows, left_ev, right_ev, duration=10.0)

        results.append({
            "subject_name": subject,
            "video_name":   video,
            "segment":      segment_tag,
            "affected_side": affected,
            **pct_scores,
            **feat,
        })

        # 打印本片段摘要
        available = [k for k, v in pct_scores.items() if k.endswith("_pct") and math.isfinite(v)]
        scores_str = ", ".join(f"{k}={v:.1f}" for k, v in pct_scores.items() if k.endswith("_pct") and math.isfinite(v))
        print(f"  [{idx+1}] {subject}/{video} | {scores_str}")
        for k, v in feat.items():
            print(f"      {k}={v:.4f}" if isinstance(v, float) else f"      {k}={v}")

    print(f"\nExtracted {len(results)} segments ({skipped} skipped)\n")

    # ---- 3. 打印各评分可用情况 ----
    print("=" * 70)
    print("SCORE AVAILABILITY")
    print("=" * 70)
    pct_keys = [k for k in SCORE_MAX]
    available_counts = {k: 0 for k in pct_keys}
    composite_count = 0
    for r in results:
        for k in pct_keys:
            pk = f"{k}_pct"
            if pk in r and math.isfinite(r.get(pk, float("nan"))):
                available_counts[k] += 1
        if math.isfinite(r.get("composite_pct", float("nan"))):
            composite_count += 1
    for k, cnt in available_counts.items():
        max_val = SCORE_MAX[k]
        print(f"  {k:10s} (max={max_val:.0f}): {cnt} samples")
    print(f"  composite_pct: {composite_count} samples")

    # ---- 4. 相关性分析：各特征 vs 每个评分量表 + 综合评分 ----
    print("\n" + "=" * 70)
    print("CORRELATION WITH CLINICAL SCORES (Pearson r)")
    print("  *   |r| >= 0.3  **   |r| >= 0.5")
    print("=" * 70)

    feat_keys = [
        "bos_mean", "bos_std", "bos_cv", "bos_range", "bos_stability",  # BOS 指标
        "knee_width_std", "ankle_width_std", "hip_width_std",           # 宽度变异性
        "knee_width_cv", "ankle_width_cv",                               # 变异系数
        "left_heel_strikes", "right_heel_strikes", "event_count_diff", "cadence",  # 步态事件
        "left_heel_y_std", "right_heel_y_std", "vertical_oscillation_mean",          # 垂直摆动
    ]

    # 所有评分的 key（原始 + 百分比 + 综合）
    all_score_keys = (
        [f"{k}_pct" for k in SCORE_MAX] + ["composite_pct"]
    )

    for score_key in all_score_keys:
        score_filtered = [(r, r[score_key]) for r in results if math.isfinite(r.get(score_key, float("nan")))]
        if not score_filtered:
            print(f"\n--- {score_key.upper()} ---")
            print("  No valid data")
            continue

        print(f"\n--- {score_key.upper()} (n={len(score_filtered)}) ---")
        for feat_key in feat_keys:
            pairs = []
            for r, s in score_filtered:
                feat_val = r.get(feat_key, float("nan"))
                if isinstance(feat_val, float) and math.isfinite(feat_val):
                    pairs.append((feat_val, s))
            if len(pairs) < 3:
                continue

            # Pearson r = Cov(X,Y) / (std_X * std_Y)
            xs, ys = zip(*pairs)
            n = len(xs)
            mx, my = sum(xs) / n, sum(ys) / n
            cov = sum((x - mx) * (y - my) for x, y in pairs)
            sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
            sy = math.sqrt(sum((y - my) ** 2 for y in ys))
            if sx < 1e-12 or sy < 1e-12:
                continue
            r = cov / (sx * sy)
            marker = "**" if abs(r) >= 0.5 else ("*" if abs(r) >= 0.3 else "")
            print(f"  {feat_key:28s} r={r:+.3f} {marker}")

    # ---- 5. 保存结果到 CSV ----
    out_path = PROJECT_ROOT / "output/balance_features.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        if results:
            # 收集所有出现过的字段名（不同行可能有不同子集）
            fieldnames = list(results[0].keys())
            for r in results[1:]:
                for k in r.keys():
                    if k not in fieldnames:
                        fieldnames.append(k)
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    print(f"\nSaved to {out_path.as_posix()}")


if __name__ == "__main__":
    main()
