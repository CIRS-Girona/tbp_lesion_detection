import os
#!/usr/bin/env python
"""
tide_analysis.py
----------------
Optional TIDE error decomposition: breaks detection errors into
classification / localization / duplicate / background / missed components.
Feed it the GT COCO JSON + the predictions JSON produced by eval_f1_pr.py.

Install once:  pip install tidecv

Run (from anywhere, paths are explicit):
  python tide_analysis.py \
      --gt /path/to/annotations/test.json \
      --pred eval_out/pred_baseline_test.json \
      --tag baseline_test --outdir eval_out

Author: Praveen Kumar Murali | June 2026
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--tag", default="rtdetrv2")
    ap.add_argument("--outdir", default="eval_out")
    a = ap.parse_args()

    from tidecv import TIDE, datasets

    gt = datasets.COCO(a.gt)
    res = datasets.COCOResult(a.pred)

    tide = TIDE()
    tide.evaluate(gt, res, mode=TIDE.BOX)   # bbox task
    tide.summarize()                        # prints the dAP error table
    tide.plot(out_dir=a.outdir)             # saves a summary bar plot
    print(f"[tide] plot saved to {a.outdir}/  (tag={a.tag})")


if __name__ == "__main__":
    main()
