import json
import os
import threading
import time
from collections import deque


def resolve_sqs_target(settings):
    # 큐 URL/리전을 config -> 환경변수 순으로 해석한다(SqsSender와 스모크 테스트 공용).
    # 리전: 설정값 우선, 없으면 표준 환경변수(AWS_REGION -> AWS_DEFAULT_REGION)로 폴백.
    queue_url = (settings.rt_sqs_queue_url
                 or os.environ.get("SQS_QUEUE_URL")
                 or os.environ.get("LOOVI_SQS_QUEUE_URL", ""))
    region = (settings.rt_sqs_region
              or os.environ.get("AWS_REGION")
              or os.environ.get("AWS_DEFAULT_REGION", ""))
    return queue_url, region


class SqsSender:
    # 스냅샷을 비동기로 SQS 전송한다 (프레임 처리 블로킹 금지).
    # 실패 시 버퍼 유지·지수 백오프 재시도, 버퍼 상한 초과/종료 시 디스크 스필.
    def __init__(self, settings):
        self.settings = settings
        self.queue_url, self.region = resolve_sqs_target(settings)
        self.spill_dir = settings.rt_spill_dir
        self.buffer_max = settings.rt_buffer_max
        self._buf = deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._client = None
        self._logged_send_error = False   # 첫 전송 실패 원인만 1회 출력(로그 폭주 방지)
        self._client_ready = self._init_client()
        # 성능/전송 지표.
        self.sent = self.failures = self.spilled = self.max_depth = 0
        self.last_latency_ms = 0.0

    def _init_client(self):
        if not self.queue_url:
            print("  WARNING: SQS queue_url 미설정 -> 스냅샷은 디스크 스필만 됨")
            return False
        try:
            import boto3

            # region_name 을 명시하면 자격증명 서명에 사용. 빈 값이면 boto3 기본 해석(env/설정파일).
            self._client = boto3.client("sqs", region_name=self.region or None)
            return True
        except Exception as exc:
            print(f"  WARNING: SQS client 생성 실패({exc}) -> 스냅샷은 디스크 스필만 됨")
            return False

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def enqueue(self, message):
        with self._lock:
            self._buf.append(message)
            self.max_depth = max(self.max_depth, len(self._buf))
            while len(self._buf) > self.buffer_max:
                self._spill(self._buf.popleft())  # 상한 초과: 오래된 것부터 스필

    def _run(self):
        backoff = 1.0
        while not self._stop.is_set() or self._buf:
            with self._lock:
                msg = self._buf[0] if self._buf else None
            if msg is None:
                self._stop.wait(0.05)
                continue
            if not self._client_ready:
                # 오프라인: 재시도 의미 없으므로 바로 스필해 버퍼를 비운다.
                self._drop_front(msg, spill=True)
                continue
            if self._send(msg):
                self._drop_front(msg, spill=False)
                backoff = 1.0
            else:
                self.failures += 1
                self._stop.wait(min(backoff, 30.0))  # 지수 백오프
                backoff = min(backoff * 2, 30.0)

    def _drop_front(self, msg, spill):
        with self._lock:
            if self._buf and self._buf[0] is msg:
                self._buf.popleft()
        if spill:
            self._spill(msg)

    def _send(self, message):
        try:
            t0 = time.time()
            self._client.send_message(
                QueueUrl=self.queue_url,
                MessageBody=json.dumps(message, ensure_ascii=False),
            )
            self.last_latency_ms = round((time.time() - t0) * 1000.0, 2)
            self.sent += 1
            return True
        except Exception as exc:
            # 첫 실패만 원인을 남긴다(리전/자격증명/큐URL 오설정 진단용). 이후엔 조용히 재시도.
            if not self._logged_send_error:
                print(f"  WARNING: SQS 전송 실패({exc}) -> 재시도/디스크 스필 진행")
                self._logged_send_error = True
            return False

    def _spill(self, message):
        # COLD raw 가 최종 안전망이지만, 미전송 스냅샷도 디스크에 보존한다.
        self.spill_dir.mkdir(parents=True, exist_ok=True)
        path = self.spill_dir / f"snapshot_{int(message.get('seq', 0)):08d}.json"
        path.write_text(json.dumps(message, ensure_ascii=False), encoding="utf-8")
        self.spilled += 1

    def stop(self, drain_sec=3.0):
        # 종료: 잠깐 전송 시도 후 남은 버퍼는 전부 디스크로 스필 (미손실 보장).
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=drain_sec)
        with self._lock:
            while self._buf:
                self._spill(self._buf.popleft())

    def metrics(self):
        return {
            "snapshots_sent": self.sent,
            "send_failures": self.failures,
            "snapshots_spilled": self.spilled,
            "buffer_max_depth": self.max_depth,
            "last_send_latency_ms": self.last_latency_ms,
        }
