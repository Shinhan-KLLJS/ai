import argparse
import statistics
from pathlib import Path

from loovi_vision.analysis.gaze_sessions import load_pose_records


def text_histogram(values, lo, hi, bins=20, width=40):
    # 콘솔 텍스트 히스토그램. center 추정과 분포 확인용.
    if not values:
        return "  (no data)"
    step = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        idx = int((v - lo) / step) if step else 0
        counts[min(bins - 1, max(0, idx))] += 1
    peak = max(counts) or 1
    lines = []
    for i, c in enumerate(counts):
        edge = lo + i * step
        bar = "#" * int(width * c / peak)
        lines.append(f"  {edge:7.1f} | {bar} {c}")
    return "\n".join(lines)


def summarize(name, values):
    if not values:
        print(f"\n[{name}] no samples")
        return
    print(f"\n[{name}] n={len(values)} "
          f"min={min(values):.1f} max={max(values):.1f} "
          f"mean={statistics.mean(values):.2f} median={statistics.median(values):.2f}")
    print(text_histogram(values, min(values), max(values)))


def distance_bins(records, q=3):
    # face_px_size(거리 proxy) 구간별 yaw 분포 → 단일 center 충분 여부 판단(C안).
    sized = [r for r in records if r.get("face_px_size")]
    if not sized:
        print("\n[distance bins] face_px_size 없음")
        return
    sized.sort(key=lambda r: r["face_px_size"])
    n = len(sized)
    print("\n[distance bins] face_px_size 구간별 yaw median (단일 center 충분 여부)")
    for i in range(q):
        chunk = sized[i * n // q:(i + 1) * n // q]
        if not chunk:
            continue
        yaws = [r["yaw"] for r in chunk]
        px = [r["face_px_size"] for r in chunk]
        print(f"  px[{min(px):.0f}-{max(px):.0f}] n={len(chunk)} "
              f"yaw_median={statistics.median(yaws):.2f} yaw_mean={statistics.mean(yaws):.2f}")


def latest_poses(poses_dir):
    files = sorted(Path(poses_dir).glob("*.jsonl"))
    return files[-1] if files else None


def main():
    parser = argparse.ArgumentParser(description="Loovi head pose 캘리브레이션 보조")
    parser.add_argument("--poses", help="poses jsonl 경로 (생략 시 data/poses 최신)")
    parser.add_argument("--poses-dir", default="data/poses")
    args = parser.parse_args()

    path = Path(args.poses) if args.poses else latest_poses(args.poses_dir)
    if not path or not Path(path).exists():
        print("poses jsonl 을 찾지 못했습니다. gaze.enable=true 로 먼저 수집하세요.")
        return
    records = load_pose_records(path)
    print(f"loaded {len(records)} pose records from {path}")
    yaws = [r["yaw"] for r in records if "yaw" in r]
    pitches = [r["pitch"] for r in records if "pitch" in r]
    summarize("yaw", yaws)
    summarize("pitch", pitches)
    distance_bins(records)
    if yaws and pitches:
        print(f"\n추정 center -> yaw_center≈{statistics.median(yaws):.1f}, "
              f"pitch_center≈{statistics.median(pitches):.1f}")
        print("부호 규약: 사람이 화면을 정면으로 볼 때 위 median 근처면 OK. "
              "고개 좌/우로 돌릴 때 yaw 부호가 기대대로 바뀌는지 확인.")


if __name__ == "__main__":
    main()
