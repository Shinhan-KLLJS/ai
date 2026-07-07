import argparse
import json
import time
from datetime import datetime

import cv2
import numpy as np

from loovi_vision.config import Settings, load_config
from loovi_vision.detectors import PersonDetector
from loovi_vision.tracking import create_tracker
from loovi_vision.pipelines.batch import PersonOnlyBatch
from loovi_vision.pipelines.capture_loops import live_loop, sync_loop
from loovi_vision.pipelines.detection_worker import DetectionWorker
from loovi_vision.pipelines.gaze_runtime import attach_performance, build_gaze_runtime
from loovi_vision.pipelines.session_io import (
    build_manifest,
    make_run_id,
    open_capture,
    output_path,
    session_path,
    video_path,
    write_session_manifest,
)


def append_row(out_path, payload):
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _warmup(settings, detector, enricher, gaze_runtime):
    # 사용하는 모든 모델의 첫 추론(커널 초기화/conv 알고리즘 선택)을 시작 시점에 미리 치른다.
    # enrich/gaze 모델은 없을 수도 있고 더미 입력에서 실패해도 무방하므로 개별 try로 감싼다.
    detector.detect(np.zeros((settings.frame_h, settings.frame_w, 3), dtype=np.uint8))
    if enricher is not None:
        try:
            enricher.analyzer.detect(np.zeros((*settings.face_det_size, 3), dtype=np.uint8))
        except Exception:
            pass
    if gaze_runtime is not None:
        try:
            gaze_runtime.gaze.pose_model.estimate(np.zeros((224, 224, 3), dtype=np.uint8))
        except Exception:
            pass


def run(config_path="loovi_vision/configs/person_only.yaml"):
    # person-only 파이프라인 실행 단위: 설정, 모델, 카메라, 저장, 루프를 묶는다.
    settings = Settings(load_config(config_path))
    run_id = make_run_id(settings)
    run_started_at = time.time()
    run_started_text = datetime.fromtimestamp(run_started_at).strftime("%Y-%m-%d %H:%M:%S")
    out_path = output_path(settings, run_id)
    rec_path = video_path(settings, run_id) if settings.record_video else None
    meta_path = session_path(settings, run_id)
    manifest = build_manifest(settings, run_id, out_path, rec_path, run_started_text)
    write_session_manifest(meta_path, manifest)
    # 라벨은 config의 experiment.name을 그대로 쓴다(예: attention / person_only).
    # 켜진 계층을 함께 표시해 어떤 모드로 도는지 로그만 봐도 알 수 있게 한다.
    stages = "+".join(["person"]
                      + (["face"] if settings.face_enable else [])
                      + (["gaze"] if settings.gaze_enable else []))
    label = f"[{settings.experiment_name}]"
    print(f"{label} config: {config_path}")
    print(f"{label} stages: {stages}")
    print(f"{label} run_id: {run_id}")
    print(f"{label} output: {out_path}")
    if rec_path:
        print(f"{label} video: {rec_path}")
    print(f"{label} session: {meta_path}")
    print(f"{label} model: {settings.person_onnx}")

    detector = PersonDetector(settings)

    enricher = None
    if settings.face_enable:
        # face.enable=false면 enricher를 만들지 않아 기존 동작과 100% 동일하다.
        from loovi_vision.enrich.face_enricher import FaceEnricher

        enricher = FaceEnricher(settings)

    # gaze.enable=false면 None을 반환해 1차와 100% 동일하게 동작한다.
    gaze_runtime = build_gaze_runtime(settings, enricher, run_id)

    # CUDA 첫 추론은 커널 초기화/알고리즘 선택 때문에 느리다. 쓰는 모델(person+face+headpose)을
    # 모두 더미 입력으로 미리 돌려, 그 비용을 라이브 중이 아니라 시작 시점에 치른다.
    if settings.enable_cuda:
        print("  CUDA warmup started...", flush=True)
        _warmup(settings, detector, enricher, gaze_runtime)
        print("  CUDA warmup done", flush=True)

    tracker = create_tracker(settings)
    # 입력 소스: video_path 있으면 사전 촬영 영상(source_fps>0), 없으면 웹캠(source_fps=0).
    cap, source_fps = open_capture(settings)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open input: {settings.camera_video or settings.camera_id}")
        return

    batch = PersonOnlyBatch(settings, run_id, run_started_at, face_enabled=settings.face_enable)

    def on_flush(row):
        # batch_sec 경계마다 집계 row를 JSONL에 append한다(워커가 호출).
        append_row(out_path, row)
        print(f"  saved: {out_path}")

    worker = DetectionWorker(settings, detector, tracker, enricher, gaze_runtime, batch, on_flush)

    # 라이브 웹캠은 검출을 워커 스레드로 분리(화면 부드러움 우선). 영상 파일은 결정론적 동기 처리.
    live = settings.threaded_capture and not settings.camera_video
    print(f"{label} loop: {'live(threaded detection)' if live else 'sync'}")
    if live:
        writer = live_loop(settings, cap, worker, rec_path, run_started_at)
    else:
        writer = sync_loop(settings, cap, worker, rec_path, run_started_at, source_fps)

    # 종료 직전 남은 partial batch도 버리지 않고 저장한다.
    worker.flush_final(time.time() - run_started_at)
    if writer:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()

    if enricher:
        # 세션 종료 시 best_face로 성별/연령을 1회 판정하고 세션 요약을 만든다.
        from loovi_vision.enrich.session_summary import build_summary

        manifest.update(build_summary(enricher.finalize(), settings.track_min_hits))

    # 성능 로그 + (gaze 활성 시) COLD 응시 분석/전송 지표를 manifest에 병합한다.
    attach_performance(manifest, worker.proc_frames, max(1e-6, time.time() - run_started_at),
                       worker.max_dt, gaze_runtime)

    manifest["status"] = "completed"
    manifest["ended_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_session_manifest(meta_path, manifest)
    print(f"\nDone. {settings.experiment_name} log saved.")
    print("Review with: python -m loovi_vision.review.server --port 8765")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="loovi_vision/configs/person_only.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
