"""VSCrowd로 트래커 ID 일관성을 평가하는 CLI.

사용 예:
  python -m loovi_vision.tools.eval_vscrowd \
      --config loovi_vision/configs/person_only.yaml \
      --data-root data/VSCrowd --split test --limit 5 \
      --out data/eval/vscrowd_test.json
"""
import argparse
import json
from pathlib import Path

from loovi_vision.config import Settings, load_config
from loovi_vision.eval.track_id_eval import aggregate, auto_workers, evaluate_clips
from loovi_vision.eval.vscrowd_loader import list_clips


def parse_args():
    p = argparse.ArgumentParser(description="VSCrowd 트래킹 ID 일관성 평가")
    p.add_argument("--config", default="loovi_vision/configs/person_only.yaml")
    p.add_argument("--data-root", default="data/VSCrowd")
    p.add_argument("--split", default="test", choices=["test", "train", "all"])
    p.add_argument("--clips", nargs="*", help="특정 클립만 지정 (예: test_001 test_002)")
    p.add_argument("--limit", type=int, default=0, help="split 앞에서 N개만 (0=전체)")
    p.add_argument("--workers", type=int, default=0, help="병렬 프로세스 수 (0=자동 코어-2, 1=순차)")
    p.add_argument("--out", default="", help="결과 JSON 저장 경로")
    return p.parse_args()


def resolve_clips(args):
    """--clips 우선, 없으면 split에서 목록을 뽑고 --limit로 자른다."""
    if args.clips:
        return args.clips
    split = None if args.split == "all" else args.split
    clips = list_clips(args.data_root, split)
    return clips[: args.limit] if args.limit else clips


def main():
    args = parse_args()
    settings = Settings(load_config(args.config))
    clips = resolve_clips(args)
    workers = auto_workers(args.workers)
    print(
        f"[VSCrowd] {len(clips)}개 클립 평가 · workers={workers} · "
        f"tracker={settings.tracker_backend} gmc={settings.tracker_gmc_method} "
        f"match_thresh={settings.track_match_thresh} buffer={settings.track_buffer}"
    )
    per_clip = evaluate_clips(settings, args.data_root, clips, workers=workers)
    agg = aggregate(per_clip)

    print("\n=== 종합 ===")
    print(f"클립 {agg['clips']}개 | GT 등장 {agg['gt_total']} | recall {agg['recall']:.3f}")
    print(f"IDF1(가중평균) {agg['idf1_weighted']:.3f} | MOTA(가중평균) {agg['mota_weighted']:.3f}")
    print(
        f"ID switch {agg['id_switches']} | Frag {agg['fragmentations']} | "
        f"MT/PT/ML {agg['mt']}/{agg['pt']}/{agg['ml']}"
    )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"config": args.config, "per_clip": per_clip, "aggregate": agg}
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
