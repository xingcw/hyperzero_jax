
DMC and Brax handle rewards differently.

## DMC (DeepMind Control Suite)

DMC uses a **`tolerance()`**-style reward that maps raw quantities to **bounded rewards in [0, 1]** via a sigmoid.

- **Mechanism**: Raw values (speed, distance, etc.) are passed through a sigmoid (e.g. gaussian, linear) with `bounds`, `margin`, and `value_at_margin`.
- **Output**: Per-step reward is always in **[0, 1]**.
- **Cheetah example** (`cheetah_default.yaml`): `bounds: [10, .inf]`, `margin: 10`, `sigmoid: linear` → speed ≥ 10 gives reward 1; below that, linear decay toward 0.
- **Episode return**: Up to ~1000 for a 1000-step episode if the agent stays at reward 1.

This is not “return normalization” (no mean/std of returns). It’s a different reward definition that maps raw quantities to [0, 1].

## Brax (HalfCheetah)

Brax uses **raw MuJoCo-style rewards**:

- **Mechanism**: Reward is typically forward velocity (and possibly a small control penalty).
- **Output**: Unbounded; per-step reward can be ~5–8 when running fast.
- **Episode return**: Often 5000–7000 for good policies over 1000 steps.

## Summary

| Aspect | DMC | Brax |
|--------|-----|------|
| Reward computation | Sigmoid/tolerance → [0, 1] | Raw velocity (unbounded) |
| Per-step reward range | [0, 1] | Unbounded (often 5–8 when fast) |
| Episode return scale | ~0–1000 | ~5000–7000 |
| Return normalization | No | No |

So DMC does not normalize returns; it uses a different reward function that is inherently bounded. Brax uses raw velocity and does not normalize either.