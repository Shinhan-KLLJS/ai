from pathlib import Path


def to_posix(path):
    # 윈도우 역슬래시 경로를 웹/manifest 공통의 posix 형식으로 통일한다.
    return str(path).replace("\\", "/")


def stored_path_text(path, root):
    # manifest에는 가능하면 workspace 기준 상대 경로를 저장/표시한다.
    try:
        return to_posix(path.relative_to(root))
    except ValueError:
        return to_posix(path)


def safe_run_id(value):
    # run_id는 파일명으로만 쓰이게 제한해 path traversal을 막는다.
    name = Path(value).name
    return name == value and "/" not in value and "\\" not in value and value not in {"", ".", ".."}
