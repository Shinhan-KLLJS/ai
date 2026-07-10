"""VSCrowd 클립을 우리 트래커에 주입해 ID 일관성 지표를 계산한다.

설계: GT 머리박스를 detection으로 그대로 넣어(=완벽 검출 가정) 트래커의
연관(association) 품질만 격리 측정한다. 검출기 차이(머리 vs 몸통)는 배제되므로,
여기서 나오는 IDF1/MOTA는 "트래커 설정이 도달 가능한 상한"으로 해석한다.
"""
import os

import cv2

from loovi_vision.eval.id_metrics import IdConsistencyAccumulator
from loovi_vision.eval.vscrowd_loader import frame_image_path, load_clip
from loovi_vision.tracking.ultralytics_tracker import UltralyticsTracker


def evaluate_clip(settings, data_root, clip):
    """클립 하나를 평가해 지표 dict를 반환한다."""
    frames = load_clip(data_root, clip)
    tracker = UltralyticsTracker(settings)
    acc = IdConsistencyAccumulator()
    # 원본 프레임 이미지가 필요한 경우에만 읽는다(그 외엔 이미지 I/O 생략).
    #  - gmc != none: 카메라 움직임 보정에 프레임 필요
    #  - with_reid  : 외형 임베딩용 crop을 위해 프레임 필요(없으면 BoT-SORT가 img.shape에서 죽음)
    need_img = settings.tracker_gmc_method != "none" or settings.track_with_reid

    for fr in frames:
        heads = fr["heads"]
        # GT 머리박스를 트래커 입력(dict) 형식으로 변환. conf=1.0으로 모두 고신뢰 처리.
        dets = [{"bbox": list(bbox), "confidence": 1.0} for _, bbox in heads]
        img = None
        if need_img:
            img = cv2.imread(str(frame_image_path(data_root, clip, fr["frame"])))
        det_to_track = tracker.update(dets, img)
        # det 인덱스 순서 == heads 순서. 각 GT에 예측 track_id(없으면 None)를 붙인다.
        pairs = [(heads[i][0], det_to_track.get(i)) for i in range(len(heads))]
        acc.update(pairs)

    result = acc.result()
    result["frames"] = len(frames)
    return result


def auto_workers(requested):
    """워커 수를 정한다. requested=0이면 자동(코어-2, 최소 1), 그 외엔 코어수 상한으로 클램프."""
    cores = os.cpu_count() or 1
    if not requested:
        return max(1, cores - 2)
    return max(1, min(requested, cores))


def _format_line(clip, res):
    return (
        f"  {clip}: IDF1={res['idf1']:.3f} recall={res['recall']:.3f} "
        f"IDSW={res['id_switches']} Frag={res['fragmentations']} "
        f"MT/PT/ML={res['mt']}/{res['pt']}/{res['ml']} MOTA={res['mota']:.3f}"
    )


def _eval_one(packed):
    # ProcessPool 워커 진입점. 클립 하나를 평가해 (clip, 결과)를 돌려준다.
    settings, data_root, clip = packed
    return clip, evaluate_clip(settings, data_root, clip)


def evaluate_clips(settings, data_root, clips, log=print, workers=1):
    """여러 클립을 평가해 클립별 지표 dict를 반환한다.

    클립은 서로 독립이라 workers>1이면 프로세스 풀로 병렬 처리한다(트래커는 순수 CPU).
    ex.map은 입력 순서대로 결과를 내주므로 로그/집계 순서는 그대로 유지된다.
    """
    per_clip = {}
    if workers and workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        packed = [(settings, data_root, clip) for clip in clips]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for clip, res in pool.map(_eval_one, packed):
                per_clip[clip] = res
                log(_format_line(clip, res))
        return per_clip

    for clip in clips:
        res = evaluate_clip(settings, data_root, clip)
        per_clip[clip] = res
        log(_format_line(clip, res))
    return per_clip


def aggregate(per_clip):
    """클립 지표를 종합한다.

    클립마다 씬/ID 체계가 독립이므로 IDF1·MOTA는 등장 수(gt_total) 가중 평균으로 요약한다
    (여러 클립 ID를 한 풀로 합치는 것은 정의상 부적절).
    """
    if not per_clip:
        return {"clips": 0}
    sums = ["id_switches", "fragmentations", "mt", "pt", "ml", "gt_total", "matched", "frames"]
    agg = {k: sum(r[k] for r in per_clip.values()) for k in sums}
    total = agg["gt_total"] or 1
    agg["idf1_weighted"] = sum(r["idf1"] * r["gt_total"] for r in per_clip.values()) / total
    agg["mota_weighted"] = sum(r["mota"] * r["gt_total"] for r in per_clip.values()) / total
    agg["recall"] = agg["matched"] / total
    agg["clips"] = len(per_clip)
    return agg
