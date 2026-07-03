import json
from pathlib import Path

import cv2

from .paths import stored_path_text


class ReviewState:
    # 리뷰 서버가 접근할 data/jsonl, data/videos, data/sessions 위치를 관리한다.
    def __init__(self, root, data_dir, video_dir, session_dir):
        self.root = Path(root).resolve()
        self.data_dir = self.resolve_dir(data_dir)
        self.video_dir = self.resolve_dir(video_dir)
        self.session_dir = self.resolve_dir(session_dir)
        # 구조 변경 전 생성된 파일도 웹뷰에서 계속 확인할 수 있도록 fallback 경로를 둔다.
        self.legacy_data_dir = self.resolve_dir("data")
        self.legacy_video_dir = self.resolve_dir("recordings")
        self.legacy_session_dir = self.resolve_dir("sessions")

    def resolve_dir(self, value):
        path = Path(value)
        if not path.is_absolute():
            path = self.root / path
        return path.resolve()

    def resolve_manifest_path(self, run_id):
        # 새 data/sessions를 우선 보고, 없으면 이전 sessions/ 위치를 확인한다.
        primary = self.session_dir / f"{run_id}.json"
        if primary.exists():
            return primary
        legacy = self.legacy_session_dir / f"{run_id}.json"
        return legacy if legacy.exists() else primary

    def resolve_stored_path(self, value):
        if not value:
            return None
        path = Path(value)
        if not path.is_absolute():
            path = self.root / path
        return path.resolve()

    def load_manifest(self, run_id):
        # session manifest는 JSONL/영상 파일을 연결하고 실행 설정을 담는다.
        path = self.resolve_manifest_path(run_id)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def fallback_manifest(self, run_id):
        # manifest가 없는 옛 JSONL/영상도 최소한의 가상 manifest로 목록에 노출한다.
        jsonl_path = self.data_dir / f"{run_id}.jsonl"
        if not jsonl_path.exists():
            legacy_jsonl = self.legacy_data_dir / f"{run_id}.jsonl"
            if legacy_jsonl.exists():
                jsonl_path = legacy_jsonl
        video_path = self.video_dir / f"{run_id}.mp4"
        if not video_path.exists():
            legacy_video = self.legacy_video_dir / f"{run_id}.mp4"
            if legacy_video.exists():
                video_path = legacy_video
        if not jsonl_path.exists() and not video_path.exists():
            return None
        return {
            "run_id": run_id,
            "status": "unknown",
            "mode": "person_only",
            "jsonl_path": stored_path_text(jsonl_path, self.root) if jsonl_path.exists() else None,
            "video_path": stored_path_text(video_path, self.root) if video_path.exists() else None,
            "record_video": video_path.exists(),
        }

    def manifest_for(self, run_id):
        return self.load_manifest(run_id) or self.fallback_manifest(run_id)

    def list_sessions(self):
        # session manifest와 JSONL 파일명을 합쳐 리뷰 가능한 run_id 목록을 만든다.
        run_ids = set()
        if self.session_dir.exists():
            run_ids.update(path.stem for path in self.session_dir.glob("*.json"))
        if self.legacy_session_dir.exists():
            run_ids.update(path.stem for path in self.legacy_session_dir.glob("*.json"))
        if self.data_dir.exists():
            run_ids.update(path.stem for path in self.data_dir.glob("*.jsonl"))
        if self.legacy_data_dir.exists():
            run_ids.update(path.stem for path in self.legacy_data_dir.glob("*.jsonl"))

        sessions = []
        for run_id in sorted(run_ids, reverse=True):
            manifest = self.manifest_for(run_id)
            if not manifest:
                continue
            jsonl_path = self.resolve_stored_path(manifest.get("jsonl_path"))
            video_path = self.resolve_stored_path(manifest.get("video_path"))
            sessions.append({
                "run_id": run_id,
                "status": manifest.get("status"),
                "started_at": manifest.get("started_at"),
                "ended_at": manifest.get("ended_at"),
                "mode": manifest.get("mode", "person_only"),
                "jsonl_exists": bool(jsonl_path and jsonl_path.exists()),
                "video_exists": bool(video_path and video_path.exists()),
            })
        return sessions

    def load_rows(self, manifest):
        # JSONL은 한 줄이 하나의 집계 window다. 깨진 줄은 error row로 보존한다.
        jsonl_path = self.resolve_stored_path(manifest.get("jsonl_path"))
        if not jsonl_path or not jsonl_path.exists():
            return []

        rows = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    row = {"line_no": line_no, "error": str(exc), "raw": line}
                rows.append(row)
        self.add_review_seconds(rows)
        return rows

    def add_review_seconds(self, rows):
        # UI에서는 첫 유효 row를 0초로 정규화한 review_* 시간축을 사용한다.
        starts = []
        for row in rows:
            if not isinstance(row, dict) or row.get("error"):
                continue
            try:
                starts.append(float(row.get("elapsed_start_sec", 0.0)))
            except (TypeError, ValueError):
                continue
        base = min(starts) if starts else 0.0
        for row in rows:
            if not isinstance(row, dict) or row.get("error"):
                continue
            try:
                start = float(row.get("elapsed_start_sec", 0.0))
                end = float(row.get("elapsed_end_sec", start))
            except (TypeError, ValueError):
                continue
            row["review_start_sec"] = round(max(0.0, start - base), 3)
            row["review_end_sec"] = round(max(0.0, end - base), 3)

    def timeline_duration(self, rows):
        # 그래프와 fallback 플레이어의 최대 시간은 review_end_sec 기준이다.
        values = [
            float(row.get("review_end_sec", 0.0))
            for row in rows
            if isinstance(row, dict) and not row.get("error")
        ]
        return max(values) if values else 0.0

    def video_metadata(self, manifest):
        # OpenCV로 MP4의 내부 FPS/프레임 수를 읽어 JSONL 시간축과 매핑한다.
        video_path = self.resolve_stored_path(manifest.get("video_path"))
        if not video_path or not video_path.exists():
            return {"fps": 0.0, "frame_count": 0, "duration_sec": 0.0}
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"fps": 0.0, "frame_count": 0, "duration_sec": 0.0}
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
        return {"fps": fps, "frame_count": frame_count, "duration_sec": duration}
