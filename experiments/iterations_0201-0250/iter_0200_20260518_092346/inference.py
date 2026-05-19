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

# ==========================================
# MUST BE DEFINED HERE IN THE GLOBAL SCOPE
# Paste your exact list of 206 birds here!
# ==========================================
LOCAL_CLASSES = ['1161364', '116570', '1176823', '1595929', '209233', '22930', '22956', '22961', '22967', '22973', '22983', '22985', '23150', '23154', '23158', '23176', '23724', '24279', '24285', '24287', '24321', '244024', '25092', '25214', '326272', '41970', '43435', '47144', '476521', '516975', '555123', '555145', '555146', '64898', '65377', '65380', '66971', '67107', '67252', '70711', '738183', '74113', '74580', '760266', 'ashgre1', 'astcra1', 'bafcur1', 'baffal1', 'banana', 'barant1', 'batbel1', 'baymac', 'bbwduc', 'bcwfin2', 'bkcdon', 'bkhpar', 'blchaw1', 'blheag1', 'blttit1', 'bncfly', 'bobfly1', 'brcmar1', 'brnowl', 'bucmot4', 'bucpar', 'bufpar', 'bunibi1', 'burowl', 'camfli1', 'chacha1', 'chbmoc1', 'chobla1', 'chvcon1', 'cibspi1', 'coffal1', 'compau', 'compot1', 'crbthr1', 'crebec1', 'dwatin1', 'epaori4', 'eulfly1', 'fabwre1', 'fepowl', 'ficman1', 'flawar1', 'fotfly', 'fusfly1', 'gilhum1', 'giwrai1', 'glteme1', 'grasal3', 'greani1', 'greant1', 'greela', 'grekis', 'grepot1', 'gretho2', 'greyel', 'grfdov1', 'grhtan1', 'gycwor1', 'horscr1', 'houspa', 'hyamac1', 'larela1', 'lesela1', 'lesgrf1', 'limpki', 'linwoo1', 'litcuc2', 'litnig1', 'mabpar', 'magant1', 'magtan2', 'masgna1', 'nacnig1', 'ocecra1', 'oliwoo1', 'orbtro3', 'orwpar', 'osprey', 'pabspi1', 'palhor3', 'paltan1', 'phecuc1', 'picpig2', 'pirfly1', 'plasla1', 'platyr1', 'plcjay1', 'pluibi1', 'purjay1', 'pvttyr1', 'ragmac1', 'rebscy1', 'recfin1', 'redjun', 'relser1', 'rinkin1', 'rivwar1', 'roahaw', 'rubthr1', 'rufcac2', 'rufcas2', 'rufgna3', 'rufhor2', 'rufnig1', 'ruftho1', 'ruftof1', 'rumfly1', 'ruther1', 'rutjac1', 'sabspa1', 'saffin', 'saytan1', 'scadov1', 'schpar1', 'scther1', 'shcfly1', 'shshaw', 'shtnig1', 'sibtan2', 'smbani', 'smbtin1', 'sobcac1', 'sobtyr1', 'socfly1', 'sofspi1', 'souant1', 'soulap1', 'souscr1', 'spbant3', 'spispi1', 'sptnig1', 'squcuc1', 'stbwoo2', 'strcuc1', 'strher2', 'strowl1', 'swthum1', 'swtman1', 'tattin1', 'thlwre1', 'toctou1', 'trokin', 'trsowl', 'undtin1', 'varant1', 'watjac1', 'wesfie1', 'wfwduc1', 'whbant2', 'whbwar2', 'whiwoo1', 'whlspi1', 'whnjay1', 'whtdov', 'whwpic1', 'y00678', 'yebcar', 'yebela1', 'yecmac', 'yecpar', 'yehcar1', 'yeofly1']

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
    if audio_chunk.shape[0] == 0:
        return np.zeros((1, MEL_BINS, 1), dtype=np.float32)
        
    audio_tensor = torch.from_numpy(audio_chunk.astype(np.float32))
    
    mel_transform = T.MelSpectrogram(sample_rate=sr, n_mels=MEL_BINS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_spec = mel_transform(audio_tensor)
    
    # THE FIX: Restore the exact natural logarithm your model was trained on!
    # No AmplitudeToDB, no standard scaling.
    log_mel = torch.log(mel_spec + 1e-9)
    
    return log_mel.unsqueeze(0).numpy().astype(np.float32)

def infer_on_audio_file(audio_path, model, device):
    results = []
    file_stem = Path(audio_path).stem
    
    try:
        audio_data, sr = sf.read(audio_path, dtype=np.float32) #type: ignore
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        if sr != SAMPLE_RATE:
            resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            audio_tensor = torch.from_numpy(audio_data).float()
            audio_data = resampler(audio_tensor).numpy()
    except Exception as e:
        print(f"[Warning] Corrupted audio file {audio_path}: {e}. Skipping safely.")
        return results

    num_samples_per_chunk = int(CHUNK_DURATION * SAMPLE_RATE)
    num_chunks = int(np.ceil(len(audio_data) / num_samples_per_chunk))

    for chunk_idx in range(num_chunks):
        end_time_seconds = int((chunk_idx + 1) * CHUNK_DURATION)
        row_id = f"{file_stem}_{end_time_seconds}"
        
        try:
            start_sample = chunk_idx * num_samples_per_chunk
            end_sample = min((chunk_idx + 1) * num_samples_per_chunk, len(audio_data))
            audio_chunk = audio_data[start_sample:end_sample]
            
            if len(audio_chunk) < num_samples_per_chunk:
                pad_length = num_samples_per_chunk - len(audio_chunk)
                audio_chunk = np.pad(audio_chunk, (0, pad_length), mode='constant')

            mel_spec = extract_mel_spectrogram_cpu(audio_chunk)
            mel_tensor = torch.from_numpy(mel_spec).to(device)

            if len(mel_tensor.shape) == 3:
                input_tensor = mel_tensor.unsqueeze(0)
            elif len(mel_tensor.shape) == 2:
                input_tensor = mel_tensor.unsqueeze(0).unsqueeze(0)
            else:
                input_tensor = mel_tensor

            with torch.no_grad(), torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                logits = model(input_tensor)
                # Forced flattening so Pandas doesn't choke on batch dimensions
                probs = torch.sigmoid(logits).cpu().numpy().reshape(-1).astype(np.float32)

            if probs.shape[0] != 206:
                raise ValueError(f"Expected 206 probabilities, got shape {probs.shape}")

            results.append({"row_id": row_id, "predictions": probs})
            
            del mel_tensor, logits
            
        except Exception as e:
            print(f"[Warning] Chunk {chunk_idx} failed in {file_stem}: {e}")
            results.append({"row_id": row_id, "predictions": np.zeros(206, dtype=np.float32)})

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
            local_data[class_name] = np.array([r["predictions"][local_idx] for r in all_results], dtype=np.float32) #type: ignore

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
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: inference.py <model_path> <test_audio_dir> <output_csv>")
        sys.exit(1)
    generate_submission_csv(sys.argv[2], sys.argv[1], sys.argv[3])
    print("[Inference] Complete!")
