"""VSCrowd로 트래커 설정 sweep을 돌리는 CLI.

baseline(현재 config) 대비 match_thresh / track_buffer / with_reid를 OFAT로 흔들어
IDF1·MOTA·IDSW를 비교한다. Re-ID 정량 효과(로드맵 #7)를 여기서 확인할 수 있다.

사용 예:
  python -m loovi_vision.tools.sweep_vscrowd --split test --limit 20 \
      --out data/eval/vscrowd_sweep.json
"""
import argparse
import json
from pathlib import Path

from loovi_vision.config import load_config
from loovi_vision.eval.config_sweep import format_table, run_sweep
from loovi_vision.eval.track_id_eval import auto_workers
from loovi_vision.eval.vscrowd_loader import list_clips


# baseline 대비 한 항목씩만 바꾸는 OFAT 변형 격자를 preset별로 제공한다.
#  - full : 전체(match/buffer/reid) — 소량 클립 탐색용
#  - match: match_thresh 만 스윕 — 대량 클립(train 등)에서 최적값 튜닝용(느린 reid 제외)
VARIANT_PRESETS = {
    "full": [
        {"name": "baseline", "overrides": {}},
        {"name": "match_0.70", "overrides": {"tracker.match_thresh": 0.70}},
        {"name": "match_0.90", "overrides": {"tracker.match_thresh": 0.90}},
        {"name": "buffer_15", "overrides": {"tracker.track_buffer": 15}},
        {"name": "buffer_60", "overrides": {"tracker.track_buffer": 60}},
        {"name": "reid_on", "overrides": {"tracker.with_reid": True}},
    ],
    "match": [
        {"name": "m_0.80", "overrides": {"tracker.match_thresh": 0.80}},
        {"name": "m_0.85", "overrides": {"tracker.match_thresh": 0.85}},
        {"name": "m_0.90", "overrides": {"tracker.match_thresh": 0.90}},
        {"name": "m_0.95", "overrides": {"tracker.match_thresh": 0.95}},
    ],
}


def parse_args():
    p = argparse.ArgumentParser(description="VSCrowd 트래커 설정 sweep")
    p.add_argument("--config", default="loovi_vision/configs/person_only.yaml")
    p.add_argument("--data-root", default="data/VSCrowd")
    p.add_argument("--split", default="test", choices=["test", "train", "all"])
    p.add_argument("--clips", nargs="*", help="특정 클립만 (예: test_001 test_002)")
    p.add_argument("--limit", type=int, default=20, help="split 앞에서 N개만 (0=전체, 기본 20)")
    p.add_argument("--preset", default="full", choices=list(VARIANT_PRESETS),
                   help="변형 격자 preset (full=전체, match=match_thresh만)")
    p.add_argument("--workers", type=int, default=0, help="병렬 프로세스 수 (0=자동 코어-2, 1=순차)")
    p.add_argument("--out", default="", help="결과 JSON 저장 경로")
    return p.parse_args()


def resolve_clips(args):
    if args.clips:
        return args.clips
    split = None if args.split == "all" else args.split
    clips = list_clips(args.data_root, split)
    return clips[: args.limit] if args.limit else clips


def main():
    args = parse_args()
    base_config = load_config(args.config)
    clips = resolve_clips(args)
    variants = VARIANT_PRESETS[args.preset]
    workers = auto_workers(args.workers)
    print(f"[sweep] 클립 {len(clips)}개 × 변형 {len(variants)}개 preset={args.preset} workers={workers} (base={args.config})")

    rows = run_sweep(base_config, args.data_root, clips, variants, workers=workers)

    print("\n=== sweep 비교 (IDF1 내림차순) ===")
    print(format_table(rows))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"config": args.config, "clips": clips, "variants": rows}
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
