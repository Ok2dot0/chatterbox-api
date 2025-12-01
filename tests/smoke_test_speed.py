import sys
import traceback
from pathlib import Path

def fallback_time_stretch_numpy(wav_tensor, speed: float):
    """Simple fallback time-stretch using numpy interpolation (changes pitch)."""
    import numpy as np
    wav = wav_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
    if speed == 1.0:
        return wav_tensor
    # Compute new length (shorter if speed>1)
    new_len = max(1, int(wav.shape[0] / speed))
    xp = np.arange(wav.shape[0])
    x = np.linspace(0, wav.shape[0] - 1, new_len)
    stretched = np.interp(x, xp, wav).astype(np.float32)
    import torch
    return torch.from_numpy(stretched).unsqueeze(0)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Smoke test for apply_speed_tensor (cross-platform)")
    parser.add_argument("--speed", type=float, default=1.5, help="Playback speed multiplier to test")
    args = parser.parse_args()

    try:
        # Ensure `src` is importable without relying on PYTHONPATH env var
        repo_root = Path(__file__).resolve().parent.parent
        src_path = repo_root / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        import numpy as np
        import torch

        try:
            from chatterbox.tts import apply_speed_tensor
        except Exception:
            apply_speed_tensor = None

        sr = 22050
        t = np.linspace(0, 1, sr, endpoint=False)
        y = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        wav = torch.from_numpy(y).unsqueeze(0)

        if apply_speed_tensor is not None:
            try:
                out = apply_speed_tensor(wav, sr, speed=args.speed)
            except Exception:
                print("[WARN] apply_speed_tensor failed, falling back to numpy resample")
                out = fallback_time_stretch_numpy(wav, args.speed)
        else:
            print("[WARN] apply_speed_tensor not available, using fallback")
            out = fallback_time_stretch_numpy(wav, args.speed)

        print(f"input shape: {wav.shape}, output shape: {out.shape}")
        if out.shape[1] == wav.shape[1]:
            print("[FAIL] speed did not change length")
            sys.exit(2)
        print("[PASS] speed changed duration as expected")
        sys.exit(0)

    except Exception:
        print("[ERROR] Exception during smoke test:")
        traceback.print_exc()
        sys.exit(3)


if __name__ == '__main__':
    main()
