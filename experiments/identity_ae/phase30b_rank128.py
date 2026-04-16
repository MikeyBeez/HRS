"""Phase 30b: int8 adapter quantization at rank 128 (Phase 30 confirmation).

Phase 30 measured 4× int8 compression with zero retrieval loss at the
rank-512 prototype. The paper now defaults to rank 128 (Phase 38a–b).
This wrapper reruns Phase 30's exact procedure at rank 128 to confirm
the analytical claim in §4.8 (Table 8) that the int8 ratio is rank-
independent and gives 52 MB / 20-passage library at rank 128.

Implementation: monkey-patch RANK and Path inside the imported phase30
module before calling its main(). Results land in results/identity_ae/
phase30_rank128/ instead of overwriting the rank-512 measurement.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase30b_rank128.py
"""

import experiments.identity_ae.phase30_quantized as p30


def main():
    p30.RANK = 128

    original_Path = p30.Path
    def patched_Path(s):
        if str(s) == "results/identity_ae/phase30":
            return original_Path("results/identity_ae/phase30_rank128")
        return original_Path(s)
    p30.Path = patched_Path

    print("=" * 64)
    print("Phase 30b: int8 quantization at rank 128 (Phase 30 rerun)")
    print("=" * 64)
    p30.main()


if __name__ == "__main__":
    main()
