import queue
import threading


class ThreadedVideoWriter:
    """VideoWriter 인코딩(1080p mp4v)을 백그라운드 스레드로 분리하는 래퍼.

    프레임당 무거운 인코딩을 메인 루프에서 빼내 루프 반복 속도(=화면 표시 FPS)를 높인다.
    빠르게 움직이는 사물이 끊겨 보이는 건 표시 FPS가 낮기 때문인데, 인코딩을 옮기면
    메인 스레드가 read→표시에만 집중해 라이브가 부드러워진다.

    cv2.VideoWriter와 동일한 write()/release() 인터페이스를 제공해 드롭인 교체가 된다.
    주의: write()에 넘기는 프레임은 호출자가 이후 in-place로 수정하지 않는 것이어야 한다
    (현재 파이프라인은 오버레이 렌더가 매번 새 배열을 만들고, raw 프레임도 캡처가 새 배열을 반환).
    """

    def __init__(self, writer, queue_size=120):
        self._writer = writer
        self._queue = queue.Queue(maxsize=queue_size)  # 약 4초(30fps) 버퍼. 초과 시 프레임 드롭
        self._dropped = 0                              # 인코더가 못 따라가 버려진 프레임 수
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        # 백그라운드: 큐에서 프레임을 꺼내 순서대로 인코딩한다. None은 종료 신호.
        while True:
            frame = self._queue.get()
            if frame is None:
                break
            self._writer.write(frame)

    def write(self, frame):
        # 논블로킹 적재: 큐가 가득 차면(인코더가 뒤처지면) 프레임을 버려 실시간성을 지킨다.
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            self._dropped += 1

    def release(self):
        # 남은 프레임을 모두 인코딩한 뒤 스레드와 원본 writer를 정리한다.
        self._queue.put(None)              # sentinel: 앞선 프레임들을 다 쓰고 종료
        self._thread.join(timeout=10.0)
        self._writer.release()
        if self._dropped:
            print(f"  WARNING: 인코더 지연으로 {self._dropped} 프레임 드롭됨(저장물만 영향, 라이브 무관)")
