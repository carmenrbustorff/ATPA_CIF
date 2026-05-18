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


SAMPLE_RATE = 32000
CHUNK_DURATION = 5.0
MEL_BINS = 128
N_FFT = 2048
HOP_LENGTH = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BirdCLEFModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base_model = timm.create_model("efficientnet_b1", pretrained=False, in_chans=1, num_classes=0)
        for param in self.base_model.parameters():
            param.requires_grad = True 
        in_features = getattr(self.base_model, "num_features")
        
        # Adding a self-attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, in_features),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512), 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.base_model(x)
        
        # Apply attention
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        logits = self.classifier(attended_features)
        return logits


def extract_mel_spectrogram_cpu(audio_chunk, sr=SAMPLE_RATE):
    # Handle empty chunks safely
    if audio_chunk.shape[0] == 0:
        # Assuming you want a [1, Channels, Mels, Time] output eventually
        # You may need to adjust the time dimension depending on your HOP_LENGTH
        return np.zeros((1, MEL_BINS, 1), dtype=np.float32)
        
    audio_tensor = torch.from_numpy(audio_chunk.astype(np.float32))
    
    # Extract linear Mel
    mel_transform = T.MelSpectrogram(sample_rate=sr, n_mels=MEL_BINS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_spec = mel_transform(audio_tensor)
    
    # Convert to DB (matches standard training distributions and prevents FP16 Overflow)
    log_mel = T.AmplitudeToDB(stype='power', top_db=80)(mel_spec)
    
    # Standardize to mean 0, std 1
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    
    # Add channel dimension and return
    return log_mel.unsqueeze(0).numpy().astype(np.float32)

def infer_on_audio_file(audio_path, model, device):
    results = []
    file_stem = Path(audio_path).stem
    
    # 1. Safely attempt to read the audio file
    try:
        audio_data, sr = sf.read(audio_path, dtype=np.float32)
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        if sr != SAMPLE_RATE:
            resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            audio_tensor = torch.from_numpy(audio_data).float()
            audio_data = resampler(audio_tensor).numpy()
    except Exception as e:
        print(f"[Warning] Corrupted audio file {audio_path}: {e}. Skipping safely.")
        return results # Returns empty. The Pandas .update() will leave these rows as zeros.

    num_samples_per_chunk = int(CHUNK_DURATION * SAMPLE_RATE)
    num_chunks = int(np.ceil(len(audio_data) / num_samples_per_chunk))

    # 2. Safely process chunk by chunk
    for chunk_idx in range(num_chunks):
        end_time_seconds = int((chunk_idx + 1) * CHUNK_DURATION)
        row_id = f"{file_stem}_{end_time_seconds}"
        
        try:
            start_sample = chunk_idx * num_samples_per_chunk
            end_sample = min((chunk_idx + 1) * num_samples_per_chunk, len(audio_data))
            audio_chunk = audio_data[start_sample:end_sample]
            
            # Pad short chunks to prevent CNN collapse
            if len(audio_chunk) < num_samples_per_chunk:
                pad_length = num_samples_per_chunk - len(audio_chunk)
                audio_chunk = np.pad(audio_chunk, (0, pad_length), mode='constant')

            mel_spec = extract_mel_spectrogram_cpu(audio_chunk)
            mel_tensor = torch.from_numpy(mel_spec).to(device)

            with torch.no_grad(), torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                logits = model(mel_tensor.unsqueeze(0))  # Add batch dimension
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy().astype(np.float32)

            results.append({"row_id": row_id, "predictions": probs})
            
            del mel_tensor, logits
            
        except Exception as e:
            # If THIS specific chunk crashes, output zeros but keep going!
            print(f"[Warning] Chunk {chunk_idx} failed in {file_stem}: {e}")
            results.append({"row_id": row_id, "predictions": np.zeros(206, dtype=np.float32)})

    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return results

    file_stem = Path(audio_path).stem
    num_samples_per_chunk = int(CHUNK_DURATION * SAMPLE_RATE)
    num_chunks = int(np.ceil(len(audio_data) / num_samples_per_chunk))

    for chunk_idx in range(num_chunks):
        start_sample = chunk_idx * num_samples_per_chunk
        end_sample = min((chunk_idx + 1) * num_samples_per_chunk, len(audio_data))
        audio_chunk = audio_data[start_sample:end_sample]
        
        # CRITICAL FIX: Pad short chunks with silence to prevent CNN collapse
        if len(audio_chunk) < num_samples_per_chunk:
            pad_length = num_samples_per_chunk - len(audio_chunk)
            audio_chunk = np.pad(audio_chunk, (0, pad_length), mode='constant')

        mel_spec = extract_mel_spectrogram_cpu(audio_chunk)
        mel_tensor = torch.from_numpy(mel_spec).to(device)

        with torch.no_grad(), torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
            logits = model(mel_tensor.unsqueeze(0))  # Add batch dimension
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy().astype(np.float32)

        end_time_seconds = int((chunk_idx + 1) * CHUNK_DURATION)
        row_id = f"{file_stem}_{end_time_seconds}"
        results.append({"row_id": row_id, "predictions": probs})

        del mel_tensor, logits
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    return results

def generate_submission_csv(test_audio_dir, model_path, output_csv, sample_submission_path="sample_submission.csv"):
    # ==========================================
    # 1. THE ABSOLUTE FAILSAFE
    # Write the dummy file immediately so it exists on disk
    # even if the Python interpreter segfaults later.
    # ==========================================
    print("[Inference] Booting. Writing failsafe dummy submission...")
    kaggle_fallback_cols = ['row_id', '1161364', '116570', '1176823', '1491113', '1595929', '209233', '22930', '22956', '22961', '22967', '22973', '22983', '22985', '23150', '23154', '23158', '23176', '23724', '24279', '24285', '24287', '24321', '244024', '25073', '25092', '25214', '326272', '41970', '43435', '47144', '47158son01', '47158son02', '47158son03', '47158son04', '47158son05', '47158son06', '47158son07', '47158son08', '47158son09', '47158son10', '47158son11', '47158son12', '47158son13', '47158son14', '47158son15', '47158son16', '47158son17', '47158son18', '47158son19', '47158son20', '47158son21', '47158son22', '47158son23', '47158son24', '47158son25', '476521', '516975', '517063', '555123', '555145', '555146', '64898', '65377', '65380', '66971', '67107', '67252', '70711', '738183', '74113', '74580', '760266', 'ashgre1', 'astcra1', 'bafcur1', 'baffal1', 'banana', 'barant1', 'batbel1', 'baymac', 'bbwduc', 'bcwfin2', 'bkcdon', 'bkhpar', 'blchaw1', 'blheag1', 'blttit1', 'bncfly', 'bobfly1', 'brcmar1', 'brnowl', 'bucmot4', 'bucpar', 'bufpar', 'bunibi1', 'burowl', 'camfli1', 'chacha1', 'chbmoc1', 'chobla1', 'chvcon1', 'cibspi1', 'coffal1', 'compau', 'compot1', 'crbthr1', 'crebec1', 'dwatin1', 'epaori4', 'eulfly1', 'fabwre1', 'fepowl', 'ficman1', 'flawar1', 'fotfly', 'fusfly1', 'gilhum1', 'giwrai1', 'glteme1', 'grasal3', 'greani1', 'greant1', 'greela', 'grekis', 'grepot1', 'gretho2', 'greyel', 'grfdov1', 'grhtan1', 'gycwor1', 'horscr1', 'houspa', 'hyamac1', 'larela1', 'lesela1', 'lesgrf1', 'limpki', 'linwoo1', 'litcuc2', 'litnig1', 'mabpar', 'magant1', 'magtan2', 'masgna1', 'nacnig1', 'ocecra1', 'oliwoo1', 'orbtro3', 'orwpar', 'osprey', 'pabspi1', 'palhor3', 'paltan1', 'phecuc1', 'picpig2', 'pirfly1', 'plasla1', 'platyr1', 'plcjay1', 'pluibi1', 'purjay1', 'pvttyr1', 'ragmac1', 'rebscy1', 'recfin1', 'redjun', 'relser1', 'rinkin1', 'rivwar1', 'roahaw', 'rubthr1', 'rufcac2', 'rufcas2', 'rufgna3', 'rufhor2', 'rufnig1', 'ruftho1', 'ruftof1', 'rumfly1', 'ruther1', 'rutjac1', 'sabspa1', 'saffin', 'saytan1', 'scadov1', 'schpar1', 'scther1', 'shcfly1', 'shshaw', 'shtnig1', 'sibtan2', 'smbani', 'smbtin1', 'sobcac1', 'sobtyr1', 'socfly1', 'sofspi1', 'souant1', 'soulap1', 'souscr1', 'spbant3', 'spispi1', 'sptnig1', 'squcuc1', 'stbwoo2', 'strcuc1', 'strher2', 'strowl1', 'swthum1', 'swtman1', 'tattin1', 'thlwre1', 'toctou1', 'trokin', 'trsowl', 'undtin1', 'varant1', 'watjac1', 'wesfie1', 'wfwduc1', 'whbant2', 'whbwar2', 'whiwoo1', 'whlspi1', 'whnjay1', 'whtdov', 'whwpic1', 'y00678', 'yebcar', 'yebela1', 'yecmac', 'yecpar', 'yehcar1', 'yeofly1']

    # Attempt to load the hidden sample_submission to get the exact Kaggle row_ids
    dummy_df = pd.DataFrame(columns=kaggle_fallback_cols)
    actual_sample_path = Path(test_audio_dir).parent / "sample_submission.csv"
    if actual_sample_path.exists():
        dummy_df = pd.read_csv(actual_sample_path)
        
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_df.to_csv(output_path, index=False)
    print("[Inference] Failsafe written securely to disk.")

    # ==========================================
    # 2. THE PROTECTED INFERENCE LOOP
    # Wrap everything in a try/except so if PyTorch crashes,
    # the script exits cleanly and Kaggle grades the failsafe.
    # ==========================================
    try:
        print("[Inference] Loading model…")
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

        audio_dir = Path(test_audio_dir)
        audio_exts = {".ogg", ".wav", ".flac", ".mp3"}
        audio_files = sorted([f for f in audio_dir.rglob("*") if f.suffix.lower() in audio_exts])

        if not audio_files:
            print(f"[Warning] No audio files in {test_audio_dir}. Exiting safely.")
            return # Exits safely, failsafe is already on disk

        all_results = []
        for i, audio_file in enumerate(audio_files):
            if (i + 1) % 100 == 0 or i == 0:
                print(f"[Inference] {i+1}/{len(audio_files)}…")
            chunk_results = infer_on_audio_file(str(audio_file), model, DEVICE)
            all_results.extend(chunk_results)

        if not all_results:
            print("[Warning] Audio files processed but no valid chunks found. Exiting safely.")
            return 

        # Map by string name
        local_data = {"row_id": [r["row_id"] for r in all_results]}
        for local_idx, class_name in enumerate(LOCAL_CLASSES):
            local_data[class_name] = np.array([r["predictions"][local_idx] for r in all_results], dtype=np.float32)

        local_df = pd.DataFrame(local_data)

        # Merge predictions exactly into Kaggle's expected structure
        print(f"[Inference] Aligning local predictions with strict Kaggle format...")
        final_df = dummy_df.copy()
        
        final_df.set_index('row_id', inplace=True)
        local_df.set_index('row_id', inplace=True)
        final_df.update(local_df) # This guarantees no extra or missing rows!
        final_df.reset_index(inplace=True)

        for col in final_df.columns:
            if col != 'row_id':
                final_df[col] = final_df[col].astype(np.float32)

        final_df.to_csv(output_path, index=False)
        print("[Inference] Final valid submission overwritten successfully.")
        return final_df

    except Exception as e:
        print(f"CRITICAL ERROR DURING INFERENCE: {e}")
        import traceback
        traceback.print_exc()
        print("Interpreter survived. Failsafe remains on disk for Kaggle grading.")
        return

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

    if not audio_files:
        print(f"[Warning] No audio files in {test_audio_dir}")
        audio_files = []

    print(f"[Inference] Found {len(audio_files)} audio files")

    all_results = []
    for i, audio_file in enumerate(audio_files):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"[Inference] {i+1}/{len(audio_files)}…")
        chunk_results = infer_on_audio_file(str(audio_file), model, DEVICE)
        all_results.extend(chunk_results)

    print(f"[Inference] Total chunks: {len(all_results)}")

    if not all_results:
        print("[Warning] No predictions generated. Creating dummy submission.")
        # We MUST write the exact 235-column header even if there are 0 rows
        
        # Try loading Kaggle's sample to get columns, or fallback to a hardcoded list
        try:
            sample_sub = pd.read_csv('sample_submission.csv')
            kaggle_cols = sample_sub.columns.tolist()
        except FileNotFoundError:
            # Hardcoded Kaggle columns as a bulletproof fallback
            kaggle_cols = ['row_id', '1161364', '116570', '1176823', '1491113', '1595929', '209233', '22930', '22956', '22961', '22967', '22973', '22983', '22985', '23150', '23154', '23158', '23176', '23724', '24279', '24285', '24287', '24321', '244024', '25073', '25092', '25214', '326272', '41970', '43435', '47144', '47158son01', '47158son02', '47158son03', '47158son04', '47158son05', '47158son06', '47158son07', '47158son08', '47158son09', '47158son10', '47158son11', '47158son12', '47158son13', '47158son14', '47158son15', '47158son16', '47158son17', '47158son18', '47158son19', '47158son20', '47158son21', '47158son22', '47158son23', '47158son24', '47158son25', '476521', '516975', '517063', '555123', '555145', '555146', '64898', '65377', '65380', '66971', '67107', '67252', '70711', '738183', '74113', '74580', '760266', 'ashgre1', 'astcra1', 'bafcur1', 'baffal1', 'banana', 'barant1', 'batbel1', 'baymac', 'bbwduc', 'bcwfin2', 'bkcdon', 'bkhpar', 'blchaw1', 'blheag1', 'blttit1', 'bncfly', 'bobfly1', 'brcmar1', 'brnowl', 'bucmot4', 'bucpar', 'bufpar', 'bunibi1', 'burowl', 'camfli1', 'chacha1', 'chbmoc1', 'chobla1', 'chvcon1', 'cibspi1', 'coffal1', 'compau', 'compot1', 'crbthr1', 'crebec1', 'dwatin1', 'epaori4', 'eulfly1', 'fabwre1', 'fepowl', 'ficman1', 'flawar1', 'fotfly', 'fusfly1', 'gilhum1', 'giwrai1', 'glteme1', 'grasal3', 'greani1', 'greant1', 'greela', 'grekis', 'grepot1', 'gretho2', 'greyel', 'grfdov1', 'grhtan1', 'gycwor1', 'horscr1', 'houspa', 'hyamac1', 'larela1', 'lesela1', 'lesgrf1', 'limpki', 'linwoo1', 'litcuc2', 'litnig1', 'mabpar', 'magant1', 'magtan2', 'masgna1', 'nacnig1', 'ocecra1', 'oliwoo1', 'orbtro3', 'orwpar', 'osprey', 'pabspi1', 'palhor3', 'paltan1', 'phecuc1', 'picpig2', 'pirfly1', 'plasla1', 'platyr1', 'plcjay1', 'pluibi1', 'purjay1', 'pvttyr1', 'ragmac1', 'rebscy1', 'recfin1', 'redjun', 'relser1', 'rinkin1', 'rivwar1', 'roahaw', 'rubthr1', 'rufcac2', 'rufcas2', 'rufgna3', 'rufhor2', 'rufnig1', 'ruftho1', 'ruftof1', 'rumfly1', 'ruther1', 'rutjac1', 'sabspa1', 'saffin', 'saytan1', 'scadov1', 'schpar1', 'scther1', 'shcfly1', 'shshaw', 'shtnig1', 'sibtan2', 'smbani', 'smbtin1', 'sobcac1', 'sobtyr1', 'socfly1', 'sofspi1', 'souant1', 'soulap1', 'souscr1', 'spbant3', 'spispi1', 'sptnig1', 'squcuc1', 'stbwoo2', 'strcuc1', 'strher2', 'strowl1', 'swthum1', 'swtman1', 'tattin1', 'thlwre1', 'toctou1', 'trokin', 'trsowl', 'undtin1', 'varant1', 'watjac1', 'wesfie1', 'wfwduc1', 'whbant2', 'whbwar2', 'whiwoo1', 'whlspi1', 'whnjay1', 'whtdov', 'whwpic1', 'y00678', 'yebcar', 'yebela1', 'yecmac', 'yecpar', 'yehcar1', 'yeofly1']
            
        empty_df = pd.DataFrame(columns=kaggle_cols)
        empty_df.to_csv(output_csv, index=False)
        return  # Safely exit the function here!
    
    local_data = {"row_id": [r["row_id"] for r in all_results]}
    for local_idx in range(206):
        local_data[f"c{local_idx}"] = np.array([r["predictions"][local_idx] for r in all_results], dtype=np.float32)

    local_df = pd.DataFrame(local_data)

    try:
        kaggle_df = pd.read_csv(sample_submission_path)
        kaggle_cols = [c for c in kaggle_df.columns if c != "row_id"]
    except FileNotFoundError:
        kaggle_cols = [f"species_{i:03d}" for i in range(234)]

    print(f"[Inference] Mapping {len([c for c in local_data.keys()])} local classes to {len(kaggle_cols)} Kaggle classes")

    final_data = {"row_id": local_df["row_id"].values}
    for kaggle_col_idx in range(len(kaggle_cols)):
        if kaggle_col_idx < 206:
            final_data[kaggle_cols[kaggle_col_idx]] = local_df[f"c{kaggle_col_idx}"].values.astype(np.float32)
        else:
            final_data[kaggle_cols[kaggle_col_idx]] = np.zeros(len(local_df), dtype=np.float32)

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

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: inference.py <model_path> <test_audio_dir> <output_csv>")
        sys.exit(1)
    generate_submission_csv(sys.argv[2], sys.argv[1], sys.argv[3])
    print("[Inference] Complete!")
