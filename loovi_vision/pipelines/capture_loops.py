import time

import cv2

from loovi_vision.pipelines.frame_render import render_from_snapshot
from loovi_vision.pipelines.session_io import pace_realtime


def _display_and_record(settings, frame, snapshot, recorders):
    # 스냅샷으로 오버레이 프레임을 만들어 창 표시 + 영상 저장(오버레이/원본)을 처리한다.
    display_frame = frame
    # 창 표시나 오버레이 저장이 필요할 때만 렌더한다(둘 다 아니면 원본 프레임 그대로 사용).
    if settings.show_window or recorders.wants_overlay:
        display_frame = render_from_snapshot(frame, snapshot, settings)
        if settings.show_window:
            cv2.imshow("Loovi Person Only (q: quit)", display_frame)
    if recorders.active:
        # 오버레이엔 렌더 결과를, 원본엔 캡처 프레임을 각각 기록한다.
        recorders.write(display_frame, frame)


def _tick_display_fps(settings, state):
    # 메인 루프(=표시/submit) 실효 FPS를 2초마다 출력. worker FPS와 비교해 병목 위치를 가른다.
    if not settings.perf_log:
        return
    state["n"] += 1
    elapsed = time.perf_counter() - state["t0"]
    if elapsed >= 2.0:
        print(f"  [perf/display] fps:{state['n'] / elapsed:5.1f}", flush=True)
        state["n"], state["t0"] = 0, time.perf_counter()


def live_loop(settings, cap, worker, recorders, run_started_at):
    # 라이브(웹캠): 검출은 워커 스레드가 최신 프레임으로 돌리고, 메인은 캡처·표시에 집중한다.
    # → 화면은 카메라 FPS로 부드럽게 흐르고, bbox는 마지막 검출 결과라 빠른 움직임엔 살짝 지연된다.
    worker.start()
    frame_id = 0
    fps_state = {"n": 0, "t0": time.perf_counter()}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        now_sec = time.time() - run_started_at
        worker.submit(frame, now_sec, frame_id)          # 검출은 워커에 맡기고
        _display_and_record(settings, frame, worker.snapshot(), recorders)  # 최신 스냅샷으로 즉시 표시
        _tick_display_fps(settings, fps_state)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    worker.stop()


def sync_loop(settings, cap, worker, recorders, run_started_at, source_fps):
    # 영상 파일/스레드 비활성: 기존처럼 process_every_n 마다 동기로 검출한다(결정론적 재생 유지).
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        pace_realtime(frame_id, source_fps, run_started_at)  # 사전 촬영 영상은 원래 속도로 재생
        now_sec = time.time() - run_started_at
        if frame_id % settings.process_every_n == 0:
            worker.process(frame, now_sec, frame_id)
        _display_and_record(settings, frame, worker.snapshot(), recorders)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
