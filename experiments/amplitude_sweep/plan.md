# Amplitude Sweep (α-Sweep) — Specification

## Hypothesis

L2 H0 is a high-amplitude beacon; downstream attention reads it via effective SNR thresholding. Scaling L2 H0's output by α should reveal a sharp threshold α* below which passkey collapses. If the transition is gradual, amplitude matters but is not threshold-gated.

## Intervention

After attention computation and before W_O projection, multiply head_h's output tensor by a scalar α per (layer, head). Nothing else changes; no fine-tuning.

## Phases

1. Sweep α ∈ {1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0} on L2 H0. Report passkey exact, passkey digit, val PPL per α.
2. Rescue: zero L2 H0, amplify its siblings. 2a = one sibling at a time × β ∈ {1, 2, 3, 5, 10, 20}. 2b = all three siblings together × same β.
3. Same α sweep on L0 H3 (find head) and L3 H0 (read head). Passkey only.

## Derived quantities

- α* = α where passkey = 0.5.
- Transition width: α range where passkey transitions 0.9 → 0.1.
- Rescue threshold: largest passkey reached in Phase 2.
- Cross-layer comparison: are other nodes equally sharply gated?

## Outcomes

1. Sharp threshold at L2 H0, no sibling rescue → amplitude-gated mechanism confirmed.
2. Gradual degradation at L2 H0, partial rescue → amplitude-sensitive, not gated.
3. No threshold, full rescue → information matters more than amplitude.
