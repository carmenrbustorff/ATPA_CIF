import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Import existing logic
from preprocessing import file_to_melspec, spec_to_cnn_input
from config import NUM_SPECIES

class BirdCLEFDataset(Dataset):
    def __init__(self, data_dir, metadata_file, backend="torchaudio"):
        """
        Initializes the Dataset using the ATPA_CIF team's preprocessing logic.
        """
        self.audio_dir = os.path.join(data_dir, "train_audio")
        self.metadata = pd.read_csv(metadata_file)
        self.backend = backend
        self.num_classes = NUM_SPECIES
        
        # Map species to integers
        self.species_list = sorted(self.metadata['primary_label'].unique())
        self.label_to_int = {species: i for i, species in enumerate(self.species_list)}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        file_path = os.path.join(self.audio_dir, row['filename'])
        label_str = row['primary_label']
        
        # 1. Generate Mel-Spectrogram using your torchaudio backend
        spec = file_to_melspec(
            path=file_path,
            backend=self.backend
        )
        
        # 2. Format for CNN (adds channel dimension and pads/trims to 216 frames)
        spec_cnn = spec_to_cnn_input(spec, target_frames=216, add_channel_dim=True)
        
        # 3. Convert NumPy array to PyTorch Tensor
        # PyTorch expects channels first: (Channels, n_mels, time_frames)
        spec_tensor = torch.from_numpy(spec_cnn).permute(2, 0, 1)

        # 4. Create One-Hot Label
        label_idx = self.label_to_int[label_str]
        label_tensor = torch.zeros(self.num_classes)
        label_tensor[label_idx] = 1.0

        return spec_tensor, label_tensor

def get_dataloader(data_dir, metadata_file, batch_size=32, num_workers=4):
    """
    Creates the DataLoader using 4 workers for the g2-standard-4 machine.
    """
    dataset = BirdCLEFDataset(data_dir, metadata_file, backend="torchaudio")
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True # Essential for fast transfer to the L4 GPU
    )