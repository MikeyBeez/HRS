"""Phase 29b: composition retest at rank 128.

Phase 29 found that naive merging of two rank-512 LoRA adapters fails
0/5 on compositional queries where each adapter alone scores 100%. The
paper now defaults to rank 128. With smaller adapters, each adapter
occupies a quarter of the directions in the residual stream — naive
merging might still fail, or it might fail less catastrophically.
This wrapper reruns Phase 29's exact procedure at rank 128 to find out.

Implementation: same monkey-patch trick as Phase 30b. Results land in
results/identity_ae/phase29_rank128/.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase29b_rank128.py
"""

import experiments.identity_ae.phase29_composition as p29


def main():
    p29.RANK = 128

    original_Path = p29.Path
    def patched_Path(s):
        if str(s) == "results/identity_ae/phase29":
            return original_Path("results/identity_ae/phase29_rank128")
        return original_Path(s)
    p29.Path = patched_Path

    print("=" * 64)
    print("Phase 29b: composition retest at rank 128 (Phase 29 rerun)")
    print("=" * 64)
    p29.main()


if __name__ == "__main__":
    main()
