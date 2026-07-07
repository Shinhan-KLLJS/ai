import time


class PerfMeter:
    """워커 단계별 처리시간을 모아 주기적으로 평균을 출력하는 경량 계측기.

    성능 진단용: 검출/추적/얼굴/gaze 각 단계가 얼마나 걸리는지, 워커 실효 FPS와
    평균 인원을 콘솔에 찍어 어디가 병목인지 실측으로 확인한다. enabled=False면 무동작.
    """

    def __init__(self, label="worker", interval_sec=2.0, enabled=True):
        self.label = label
        self.interval = interval_sec
        self.enabled = enabled
        self._reset()

    def _reset(self):
        self._sums = {}       # stage -> 누적 ms
        self._n = 0           # 집계 구간 처리 프레임 수
        self._persons = 0     # 누적 인원(평균용)
        self._t0 = time.perf_counter()

    def add(self, stages_ms, persons):
        # stages_ms: {"detect": ms, "track": ms, ...}, persons: 이번 프레임 검출 인원.
        if not self.enabled:
            return
        for key, value in stages_ms.items():
            self._sums[key] = self._sums.get(key, 0.0) + value
        self._n += 1
        self._persons += persons
        if time.perf_counter() - self._t0 >= self.interval:
            self._flush()

    def _flush(self):
        # 구간 평균을 한 줄로 출력하고 누적값을 초기화한다.
        elapsed = time.perf_counter() - self._t0
        if self._n == 0 or elapsed <= 0:
            self._reset()
            return
        fps = self._n / elapsed
        avg_persons = self._persons / self._n
        parts = " ".join(f"{k}:{self._sums[k] / self._n:5.1f}ms" for k in sorted(self._sums))
        print(f"  [perf/{self.label}] fps:{fps:4.1f} persons:{avg_persons:3.1f} {parts}", flush=True)
        self._reset()
