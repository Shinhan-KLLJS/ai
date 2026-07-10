"""검출 병렬화 유틸.

프레임은 서로 독립이라 구간을 나눠 여러 프로세스가 동시에 person 검출을 돌린다.
(트래킹은 순서가 필요하므로 병렬화하지 않는다 — 검출만 병렬로 캐시한 뒤, 프레임 순서대로
합쳐 트래커에 한 번 먹인다. 그래야 track이 청크 경계에서 끊겨 유동인구가 부풀지 않는다.)

모델 입력이 batch=1 고정이라 GPU 배치가 불가 → 다중 프로세스로 GPU 유휴시간(디코드/후처리)을 메운다.
"""
import json
from concurrent.futures import ProcessPoolExecutor

import cv2

from loovi_vision.detectors.person import PersonDetector


def save_detections(frames_dets, path):
    # 검출 결과를 디스크에 캐시한다(검출 1회 → 트래킹 설정 무한 재실험용).
    data = [[[*d["bbox"], round(float(d.get("confidence", 1.0)), 4)] for d in dets] for dets in frames_dets]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def load_detections(path):
    # 캐시된 검출 결과를 트래커 입력 형식으로 복원한다.
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [[{"bbox": (x, y, w, h), "confidence": c} for x, y, w, h, c in dets] for dets in data]


def _detect_chunk(packed):
    # 워커 진입점: 한 청크 구간을 자기 detector/decoder로 검출해 [(frame_idx, dets), ...] 반환.
    settings, video_path, start, end, step = packed
    detector = PersonDetector(settings)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = []
    idx = start
    while idx < end:
        ok, frame = cap.read()
        if not ok:
            break
        # 전역 프레임 번호 기준으로 step 샘플링(청크 경계와 무관하게 일관).
        if idx % step == 0:
            out.append((idx, detector.detect(frame)))
        idx += 1
    cap.release()
    return out


def _split_chunks(settings, video_path, start, end, step, workers):
    # 구간 [start,end)를 workers개 연속 청크로 균등 분할.
    span = end - start
    size = -(-span // workers)  # 올림 나눗셈
    chunks = []
    pos = start
    while pos < end:
        chunks.append((settings, video_path, pos, min(pos + size, end), step))
        pos += size
    return chunks


def detect_parallel(settings, video_path, start, end, step, workers, log=print):
    """구간을 병렬 검출해 프레임 순서대로 정렬된 detection 리스트를 반환한다."""
    chunks = _split_chunks(settings, video_path, start, end, step, workers)
    log(f"  검출 병렬: {len(chunks)}청크 × {workers}워커 · 구간 {start}~{end} step {step}")
    merged = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for part in pool.map(_detect_chunk, chunks):
            merged.extend(part)
    merged.sort(key=lambda x: x[0])          # 프레임 번호 순 정렬(트래킹 순서 보장)
    return [dets for _, dets in merged]
