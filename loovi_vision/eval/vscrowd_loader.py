"""VSCrowd 데이터셋 로더.

구조: data/VSCrowd/videos/<clip>/NNNNNN.jpg (프레임)
      data/VSCrowd/annotations.zip -> annotations/<clip>.txt (프레임별 GT)

어노테이션 포맷(공식): 한 줄 = 한 프레임
  FrameID  HeadID x1 y1 x2 y2 cx cy  [다음 사람 반복...]
  - HeadID     : 추적 ID
  - x1 y1 x2 y2: 머리 바운딩박스(좌,상,우,하)  ※ 몸통이 아니라 머리 기준
  - cx cy      : 머리 중심점(crowd counting/localization용)
"""
from pathlib import Path
import zipfile

# 사람 한 명당 토큰 수: HeadID + bbox(4) + 중심점(2)
TOKENS_PER_HEAD = 7


def list_clips(data_root, split=None):
    """videos/ 하위 클립 폴더 이름 목록을 정렬해 반환한다.
    split="test"/"train"이면 접두사로 거른다."""
    videos = Path(data_root) / "videos"
    names = sorted(d.name for d in videos.iterdir() if d.is_dir())
    if split:
        names = [n for n in names if n.startswith(split)]
    return names


def _read_annotation_text(data_root, clip):
    """클립 어노테이션 원문을 반환한다. annotations.zip 우선, 없으면 annotations/ 폴더."""
    root = Path(data_root)
    zip_path = root / "annotations.zip"
    member = f"annotations/{clip}.txt"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as z:
            return z.read(member).decode("utf-8")
    return (root / "annotations" / f"{clip}.txt").read_text(encoding="utf-8")


def parse_frames(text):
    """어노테이션 원문을 프레임 리스트로 파싱한다.

    반환: [{"frame": int, "heads": [(head_id, (x, y, w, h)), ...]}, ...]
          bbox는 좌상단 x,y + 너비/높이 (트래커 입력 형식과 동일).
    """
    frames = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        frame_id = int(parts[0])
        rest = parts[1:]
        heads = []
        # 남는 토큰을 7개씩 끊어 사람별로 해석한다(끝의 잔여 토큰은 버린다).
        usable = len(rest) - len(rest) % TOKENS_PER_HEAD
        for k in range(0, usable, TOKENS_PER_HEAD):
            g = rest[k:k + TOKENS_PER_HEAD]
            head_id = int(float(g[0]))
            x1, y1, x2, y2 = (float(v) for v in g[1:5])
            heads.append((head_id, (x1, y1, x2 - x1, y2 - y1)))
        frames.append({"frame": frame_id, "heads": heads})
    return frames


def load_clip(data_root, clip):
    """클립 하나의 프레임별 GT를 로드한다."""
    return parse_frames(_read_annotation_text(data_root, clip))


def frame_image_path(data_root, clip, frame_id):
    """프레임 이미지 경로. gmc 등 원본 이미지가 필요한 경우에만 사용한다."""
    return Path(data_root) / "videos" / clip / f"{frame_id:06d}.jpg"
