# check_checkpoint_load.py
import torch, sys
from pathlib import Path
import torch.nn as nn

CKPT = Path("experiments/iterations_0201-0250/iter_0206_20260520_141015/model.pt")
NUM_SPECIES = 206

if not CKPT.exists():
    print("Checkpoint not found:", CKPT); sys.exit(1)

raw = torch.load(CKPT, map_location="cpu")
state = raw.get("model", raw.get("state_dict", raw)) if isinstance(raw, dict) else raw
# unwrap common wrappers
state = {k.replace("module.", ""): v for k, v in state.items()}

print("Sample keys (first 12):")
for k in list(state.keys())[:12]:
    print(" ", k)
print("Total keys:", len(state))

def try_timm():
    try:
        import timm
    except Exception as e:
        return ("timm_missing", str(e))
    base = timm.create_model("efficientnet_b1", pretrained=False, in_chans=1, num_classes=0)
    in_features = getattr(base, "num_features")
    classifier = nn.Sequential(nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, NUM_SPECIES))
    model = nn.Sequential(base, classifier)
    try:
        model.load_state_dict(state)
        return ("timm_ok", None)
    except Exception as e:
        # attempt non-strict to see missing/unexpected
        res = model.load_state_dict(state, strict=False)
        return ("timm_partial", {"exception": str(e), "missing": res.missing_keys, "unexpected": res.unexpected_keys})

def try_simple_cnn():
    try:
        from backups.models import build_simple_cnn_torch
    except Exception as e:
        return ("simplecnn_missing", str(e))
    model = build_simple_cnn_torch(num_classes=NUM_SPECIES)
    try:
        model.load_state_dict(state)
        return ("simplecnn_ok", None)
    except Exception as e:
        res = model.load_state_dict(state, strict=False) if hasattr(model, 'load_state_dict') else None
        return ("simplecnn_partial", {"exception": str(e), "missing": getattr(res, "missing_keys", None), "unexpected": getattr(res, "unexpected_keys", None)})

print("\nTrying timm EfficientNet-B1 load...")
r_timm = try_timm()
print("timm result:", r_timm)

print("\nTrying repo SimpleCNN load...")
r_cnn = try_simple_cnn()
print("simple_cnn result:", r_cnn)