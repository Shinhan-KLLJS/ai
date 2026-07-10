"""트래커 설정 sweep: 같은 클립셋에 대해 config 항목을 바꿔가며 ID 지표를 비교한다.

OFAT(한 번에 한 요소) 원칙으로 baseline 대비 개별 항목을 흔든다.
검출기는 항상 완벽(GT 박스 주입)이므로, 지표 변화는 오롯이 트래커 설정 효과다.
"""
import copy

from loovi_vision.config import Settings
from loovi_vision.eval.track_id_eval import aggregate, evaluate_clips


def set_nested(config, dotted_key, value):
    """'tracker.match_thresh' 같은 점 경로에 값을 세팅한다(중간 dict 없으면 생성)."""
    node = config
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def apply_overrides(base_config, overrides):
    """base_config를 깊은 복사한 뒤 overrides(dict: 점경로->값)를 적용해 반환한다."""
    config = copy.deepcopy(base_config)
    for dotted_key, value in overrides.items():
        set_nested(config, dotted_key, value)
    return config


def run_sweep(base_config, data_root, clips, variants, log=print, workers=1):
    """variants=[{"name":..., "overrides":{점경로:값}}] 각각을 평가해 결과 리스트로 반환한다."""
    rows = []
    for variant in variants:
        name = variant["name"]
        overrides = variant.get("overrides", {})
        settings = Settings(apply_overrides(base_config, overrides))
        log(f"\n[sweep] {name}  {overrides or '(baseline)'}")
        per_clip = evaluate_clips(settings, data_root, clips, log=log, workers=workers)
        agg = aggregate(per_clip)
        rows.append({"name": name, "overrides": overrides, "aggregate": agg})
    return rows


def format_table(rows):
    """sweep 결과를 IDF1 내림차순 표 문자열로 만든다(baseline 대비 비교용)."""
    ordered = sorted(rows, key=lambda r: r["aggregate"].get("idf1_weighted", 0), reverse=True)
    header = f"{'variant':<18}{'IDF1':>8}{'MOTA':>8}{'IDSW':>7}{'Frag':>7}{'MT/PT/ML':>12}"
    lines = [header, "-" * len(header)]
    for row in ordered:
        a = row["aggregate"]
        mtml = f"{a['mt']}/{a['pt']}/{a['ml']}"
        lines.append(
            f"{row['name']:<18}{a['idf1_weighted']:>8.3f}{a['mota_weighted']:>8.3f}"
            f"{a['id_switches']:>7}{a['fragmentations']:>7}{mtml:>12}"
        )
    return "\n".join(lines)
