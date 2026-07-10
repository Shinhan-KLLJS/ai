"""트래킹 ID 일관성 지표: IDF1, ID switch, MT/PT/ML, Fragmentation, MOTA.

전제(중요): 본 평가는 GT 머리박스를 그대로 트래커에 주입하므로, 프레임마다
각 GT는 최대 하나의 예측 track_id에 대응하고 오검출(FP)은 발생하지 않는다.
이 폐쇄 구조 덕에 표준 CLEAR-MOT / IDF1 정의를 외부 라이브러리 없이 정확히 구현할 수 있다.
(py-motmetrics와 동일한 정의를 따르되, GT↔예측 대응이 이미 확정된 점만 활용한다.)
"""
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment


class IdConsistencyAccumulator:
    """프레임마다 (gt_id, pred_id) 쌍을 받아 트래킹 ID 지표를 누적한다."""

    def __init__(self):
        self._pair = defaultdict(int)          # (gt_id, pred_id) -> 동시 등장 프레임 수(IDF1용)
        self._gt_count = defaultdict(int)      # gt_id -> 등장 프레임 수
        self._pred_count = defaultdict(int)    # pred_id -> 매칭된 프레임 수
        self._last_pred = {}                   # gt_id -> 마지막으로 매칭된 pred_id(공백 넘어 유지)
        self._prev_matched = {}                # gt_id -> 직전 등장 프레임의 매칭 여부(Frag용)
        self._tracked_frames = defaultdict(int)  # gt_id -> 매칭 성공 프레임 수(MT/ML용)
        self.id_switches = 0
        self.fragmentations = 0
        self.gt_total = 0                      # 전체 GT 등장 수(프레임×인원)
        self.matched_total = 0                 # 매칭 성공 수(= 예측 등장 총합)

    def update(self, pairs):
        """pairs: 이번 프레임의 [(gt_id, pred_id 또는 None), ...]"""
        for gt_id, pred_id in pairs:
            self.gt_total += 1
            self._gt_count[gt_id] += 1
            matched = pred_id is not None
            if matched:
                self.matched_total += 1
                self._tracked_frames[gt_id] += 1
                self._pair[(gt_id, pred_id)] += 1
                self._pred_count[pred_id] += 1
                # ID switch: 이 GT의 직전 매칭 pred와 달라지면 스위치로 센다.
                last = self._last_pred.get(gt_id)
                if last is not None and last != pred_id:
                    self.id_switches += 1
                self._last_pred[gt_id] = pred_id
                # Fragmentation: 직전 등장 때 끊겼다가(미매칭) 다시 매칭되면 +1.
                if self._prev_matched.get(gt_id) is False:
                    self.fragmentations += 1
            self._prev_matched[gt_id] = matched

    def _idf1(self):
        """전역 1:1 매칭(Hungarian)으로 IDTP를 최대화해 IDF1/IDP/IDR을 구한다."""
        gts = list(self._gt_count)
        preds = list(self._pred_count)
        if not gts or not preds:
            return {"idf1": 0.0, "idp": 0.0, "idr": 0.0, "idtp": 0}
        gi = {g: i for i, g in enumerate(gts)}
        pj = {p: j for j, p in enumerate(preds)}
        weight = np.zeros((len(gts), len(preds)))
        for (g, p), c in self._pair.items():
            weight[gi[g], pj[p]] = c
        # 매칭 프레임 수(IDTP) 최대화 → 비용을 음수화해 최소화 문제로 푼다.
        rows, cols = linear_sum_assignment(-weight)
        idtp = int(weight[rows, cols].sum())
        idfn = self.gt_total - idtp            # 매칭 못 받은 GT 등장
        idfp = self.matched_total - idtp       # 최적 매칭에서 남은 예측 등장
        idp = idtp / (idtp + idfp) if idtp + idfp else 0.0
        idr = idtp / (idtp + idfn) if idtp + idfn else 0.0
        denom = 2 * idtp + idfp + idfn
        idf1 = 2 * idtp / denom if denom else 0.0
        return {"idf1": idf1, "idp": idp, "idr": idr, "idtp": idtp}

    def result(self):
        idf1 = self._idf1()
        # MT/PT/ML: GT 수명 대비 매칭 비율(>=0.8 MT, <=0.2 ML, 그 사이 PT).
        mt = pt = ml = 0
        for g, total in self._gt_count.items():
            ratio = self._tracked_frames.get(g, 0) / total
            if ratio >= 0.8:
                mt += 1
            elif ratio <= 0.2:
                ml += 1
            else:
                pt += 1
        fn = self.gt_total - self.matched_total
        # 이 평가에선 FP=0(모든 예측이 GT 박스에서 유래). MOTA = 1 - (FN + FP + IDSW)/GT.
        mota = 1 - (fn + self.id_switches) / self.gt_total if self.gt_total else 0.0
        return {
            "gt_ids": len(self._gt_count),
            "pred_ids": len(self._pred_count),
            "gt_total": self.gt_total,
            "matched": self.matched_total,
            "recall": self.matched_total / self.gt_total if self.gt_total else 0.0,
            "idf1": idf1["idf1"],
            "idp": idf1["idp"],
            "idr": idf1["idr"],
            "id_switches": self.id_switches,
            "fragmentations": self.fragmentations,
            "mt": mt,
            "pt": pt,
            "ml": ml,
            "mota": mota,
        }
