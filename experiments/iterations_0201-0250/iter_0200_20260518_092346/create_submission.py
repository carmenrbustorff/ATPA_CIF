import os
import gc
import torch
import torch.nn as nn
import torchaudio.transforms as T
import numpy as np
import soundfile as sf
import pandas as pd
import timm
from pathlib import Path

# ==========================================
# PASTE YOUR 206 CLASSES HERE FROM YOUR DATALOADER
# Must be in the exact order they were trained!
# ==========================================
LOCAL_CLASSES = ['1161364', '116570', '1176823', '1595929', '209233', '22930', '22956', '22961', '22967', '22973', '22983', '22985', '23150', '23154', '23158', '23176', '23724', '24279', '24285', '24287', '24321', '244024', '25092', '25214', '326272', '41970', '43435', '47144', '476521', '516975', '555123', '555145', '555146', '64898', '65377', '65380', '66971', '67107', '67252', '70711', '738183', '74113', '74580', '760266', 'ashgre1', 'astcra1', 'bafcur1', 'baffal1', 'banana', 'barant1', 'batbel1', 'baymac', 'bbwduc', 'bcwfin2', 'bkcdon', 'bkhpar', 'blchaw1', 'blheag1', 'blttit1', 'bncfly', 'bobfly1', 'brcmar1', 'brnowl', 'bucmot4', 'bucpar', 'bufpar', 'bunibi1', 'burowl', 'camfli1', 'chacha1', 'chbmoc1', 'chobla1', 'chvcon1', 'cibspi1', 'coffal1', 'compau', 'compot1', 'crbthr1', 'crebec1', 'dwatin1', 'epaori4', 'eulfly1', 'fabwre1', 'fepowl', 'ficman1', 'flawar1', 'fotfly', 'fusfly1', 'gilhum1', 'giwrai1', 'glteme1', 'grasal3', 'greani1', 'greant1', 'greela', 'grekis', 'grepot1', 'gretho2', 'greyel', 'grfdov1', 'grhtan1', 'gycwor1', 'horscr1', 'houspa', 'hyamac1', 'larela1', 'lesela1', 'lesgrf1', 'limpki', 'linwoo1', 'litcuc2', 'litnig1', 'mabpar', 'magant1', 'magtan2', 'masgna1', 'nacnig1', 'ocecra1', 'oliwoo1', 'orbtro3', 'orwpar', 'osprey', 'pabspi1', 'palhor3', 'paltan1', 'phecuc1', 'picpig2', 'pirfly1', 'plasla1', 'platyr1', 'plcjay1', 'pluibi1', 'purjay1', 'pvttyr1', 'ragmac1', 'rebscy1', 'recfin1', 'redjun', 'relser1', 'rinkin1', 'rivwar1', 'roahaw', 'rubthr1', 'rufcac2', 'rufcas2', 'rufgna3', 'rufhor2', 'rufnig1', 'ruftho1', 'ruftof1', 'rumfly1', 'ruther1', 'rutjac1', 'sabspa1', 'saffin', 'saytan1', 'scadov1', 'schpar1', 'scther1', 'shcfly1', 'shshaw', 'shtnig1', 'sibtan2', 'smbani', 'smbtin1', 'sobcac1', 'sobtyr1', 'socfly1', 'sofspi1', 'souant1', 'soulap1', 'souscr1', 'spbant3', 'spispi1', 'sptnig1', 'squcuc1', 'stbwoo2', 'strcuc1', 'strher2', 'strowl1', 'swthum1', 'swtman1', 'tattin1', 'thlwre1', 'toctou1', 'trokin', 'trsowl', 'undtin1', 'varant1', 'watjac1', 'wesfie1', 'wfwduc1', 'whbant2', 'whbwar2', 'whiwoo1', 'whlspi1', 'whnjay1', 'whtdov', 'whwpic1', 'y00678', 'yebcar', 'yebela1', 'yecmac', 'yecpar', 'yehcar1', 'yeofly1']
len(LOCAL_CLASSES)

def generate_submission_csv(test_audio_dir, model_path, output_csv, sample_submission_path="sample_submission.csv"):
    print("[Inference] Loading model…")

    if len(LOCAL_CLASSES) != 206:
        raise ValueError(f"You must define exactly 206 classes in LOCAL_CLASSES. Found {len(LOCAL_CLASSES)}")

    import inspect
    model_class = None
    for name, obj in globals().items():
        if inspect.isclass(obj) and issubclass(obj, nn.Module) and name not in ('nn', 'Module') and not name.startswith('_'):
            model_class = obj
            break

    if model_class is None:
        raise RuntimeError("No nn.Module subclass found")

    model = model_class(num_classes=206)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model = model.to(DEVICE)
    model.eval()

    print(f"[Inference] Model loaded")

    audio_dir = Path(test_audio_dir)
    audio_exts = {".ogg", ".wav", ".flac", ".mp3"}
    audio_files = sorted([f for f in audio_dir.rglob("*") if f.suffix.lower() in audio_exts])

    # Hardcoded Kaggle columns as a bulletproof fallback
    kaggle_fallback_cols = ['row_id', '1161364', '116570', '1176823', '1491113', '1595929', '209233', '22930', '22956', '22961', '22967', '22973', '22983', '22985', '23150', '23154', '23158', '23176', '23724', '24279', '24285', '24287', '24321', '244024', '25073', '25092', '25214', '326272', '41970', '43435', '47144', '47158son01', '47158son02', '47158son03', '47158son04', '47158son05', '47158son06', '47158son07', '47158son08', '47158son09', '47158son10', '47158son11', '47158son12', '47158son13', '47158son14', '47158son15', '47158son16', '47158son17', '47158son18', '47158son19', '47158son20', '47158son21', '47158son22', '47158son23', '47158son24', '47158son25', '476521', '516975', '517063', '555123', '555145', '555146', '64898', '65377', '65380', '66971', '67107', '67252', '70711', '738183', '74113', '74580', '760266', 'ashgre1', 'astcra1', 'bafcur1', 'baffal1', 'banana', 'barant1', 'batbel1', 'baymac', 'bbwduc', 'bcwfin2', 'bkcdon', 'bkhpar', 'blchaw1', 'blheag1', 'blttit1', 'bncfly', 'bobfly1', 'brcmar1', 'brnowl', 'bucmot4', 'bucpar', 'bufpar', 'bunibi1', 'burowl', 'camfli1', 'chacha1', 'chbmoc1', 'chobla1', 'chvcon1', 'cibspi1', 'coffal1', 'compau', 'compot1', 'crbthr1', 'crebec1', 'dwatin1', 'epaori4', 'eulfly1', 'fabwre1', 'fepowl', 'ficman1', 'flawar1', 'fotfly', 'fusfly1', 'gilhum1', 'giwrai1', 'glteme1', 'grasal3', 'greani1', 'greant1', 'greela', 'grekis', 'grepot1', 'gretho2', 'greyel', 'grfdov1', 'grhtan1', 'gycwor1', 'horscr1', 'houspa', 'hyamac1', 'larela1', 'lesela1', 'lesgrf1', 'limpki', 'linwoo1', 'litcuc2', 'litnig1', 'mabpar', 'magant1', 'magtan2', 'masgna1', 'nacnig1', 'ocecra1', 'oliwoo1', 'orbtro3', 'orwpar', 'osprey', 'pabspi1', 'palhor3', 'paltan1', 'phecuc1', 'picpig2', 'pirfly1', 'plasla1', 'platyr1', 'plcjay1', 'pluibi1', 'purjay1', 'pvttyr1', 'ragmac1', 'rebscy1', 'recfin1', 'redjun', 'relser1', 'rinkin1', 'rivwar1', 'roahaw', 'rubthr1', 'rufcac2', 'rufcas2', 'rufgna3', 'rufhor2', 'rufnig1', 'ruftho1', 'ruftof1', 'rumfly1', 'ruther1', 'rutjac1', 'sabspa1', 'saffin', 'saytan1', 'scadov1', 'schpar1', 'scther1', 'shcfly1', 'shshaw', 'shtnig1', 'sibtan2', 'smbani', 'smbtin1', 'sobcac1', 'sobtyr1', 'socfly1', 'sofspi1', 'souant1', 'soulap1', 'souscr1', 'spbant3', 'spispi1', 'sptnig1', 'squcuc1', 'stbwoo2', 'strcuc1', 'strher2', 'strowl1', 'swthum1', 'swtman1', 'tattin1', 'thlwre1', 'toctou1', 'trokin', 'trsowl', 'undtin1', 'varant1', 'watjac1', 'wesfie1', 'wfwduc1', 'whbant2', 'whbwar2', 'whiwoo1', 'whlspi1', 'whnjay1', 'whtdov', 'whwpic1', 'y00678', 'yebcar', 'yebela1', 'yecmac', 'yecpar', 'yehcar1', 'yeofly1']

    if not audio_files:
        print(f"[Warning] No audio files in {test_audio_dir}")
        print("[Warning] Creating dummy submission.")
        empty_df = pd.DataFrame(columns=kaggle_fallback_cols)
        empty_df.to_csv(output_csv, index=False) # Fixed NameError here
        return 

    print(f"[Inference] Found {len(audio_files)} audio files")

    all_results = []
    for i, audio_file in enumerate(audio_files):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"[Inference] {i+1}/{len(audio_files)}…")
        chunk_results = infer_on_audio_file(str(audio_file), model, DEVICE)
        all_results.extend(chunk_results)

    print(f"[Inference] Total chunks: {len(all_results)}")

    if not all_results:
        empty_df = pd.DataFrame(columns=kaggle_fallback_cols)
        empty_df.to_csv(output_csv, index=False)
        return

    # Map by NAME, not by index
    local_data = {"row_id": [r["row_id"] for r in all_results]}
    for local_idx, class_name in enumerate(LOCAL_CLASSES):
        local_data[class_name] = np.array([r["predictions"][local_idx] for r in all_results], dtype=np.float32)

    local_df = pd.DataFrame(local_data)

    try:
        kaggle_df = pd.read_csv(sample_submission_path)
        kaggle_cols = [c for c in kaggle_df.columns if c != "row_id"]
    except FileNotFoundError:
        kaggle_cols = [c for c in kaggle_fallback_cols if c != "row_id"]

    print(f"[Inference] Mapping {len(LOCAL_CLASSES)} local classes to {len(kaggle_cols)} Kaggle classes")

    final_data = {"row_id": local_df["row_id"].values}
    
    # Safe string-to-string dictionary merge
    for k_col in kaggle_cols:
        if k_col in local_df.columns:
            final_data[k_col] = local_df[k_col].values
        else:
            final_data[k_col] = np.zeros(len(local_df), dtype=np.float32)

    final_df = pd.DataFrame(final_data)
    final_df = final_df[["row_id"] + kaggle_cols]

    for col in kaggle_cols:
        final_df[col] = final_df[col].astype(np.float32)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)

    print(f"[Inference] Saved: {output_path}")
    print(f"[Inference] Shape: {final_df.shape} ({len(final_df)} rows = 1 per chunk)")

    return final_df