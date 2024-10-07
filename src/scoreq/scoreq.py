import fairseq
import torch
import torch.nn as nn
import numpy as np
import torchaudio
import os
from urllib.request import urlretrieve

class Scoreq():
    """
    Main class for handling the SCOREQ audio quality assessment model.

    This class loads the pre-trained SCOREQ model, processes audio files, and makes predictions in both
    no-reference (NR) and reference-based (FR/NMR) modes. It supports both natural and synthetic speech
    data domains.
    """
    def __init__(self, device=None, data_domain='natural', mode='nr'):
        """
        Initializes the Scoreq object.

        Args:
            device: Device to run the model on ('cuda' or 'cpu'). If None, automatically detects GPU availability.
            data_domain: Domain of the audio data ('natural' or 'synthetic').
            mode: Mode of operation ('nr' for no-reference or 'ref' for either full-reference or non-matching-reference modes).
        """
        
        # Store variables
        self.data_domain = data_domain
        self.mode = mode
        
        # *** DEVICE SETTINGS ***
        # Automatically set based on GPU detection
        if torch.cuda.is_available():
            self.DEVICE = 'cuda'
        else:
            self.DEVICE = 'cpu'
        
        # Overwrite user choice
        if device is not None:
            self.DEVICE = device
        
        print(f'SCOREQ running on: {self.DEVICE}')
        
        # *** LOAD MODEL ***
        # *** Pytorch models directory ****
        if not os.path.isdir('./pt-models'):
            print('Creating pt-models directory')
            os.makedirs('./pt-models')

        # Download wav2vec 2.0
        url_w2v = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt"
        CHECKPOINT_PATH = './pt-models/wav2vec_small.pt'
        if not os.path.isfile(CHECKPOINT_PATH):
            print('Downloading wav2vec 2.0')
            urlretrieve(url_w2v, CHECKPOINT_PATH)
            print('Completed')
        
        # w2v BASE parameters
        W2V_OUT_DIM = 768
        EMB_DIM = 256

        # Load w2v BASE
        w2v_model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([CHECKPOINT_PATH])
        ssl_model = w2v_model[0] 
        ssl_model.remove_pretraining_modules()
    
        # Create SCOREQ model
        pt_model = TripletModel(ssl_model, W2V_OUT_DIM, EMB_DIM)
        
        # Add mos projection layer for no-reference mode
        if mode == 'nr':
            model = MosPredictor(pt_model, emb_dim=W2V_OUT_DIM)
        elif mode == 'ref':
            model = pt_model

        # Load weights
        if data_domain == 'natural':
            if mode == 'nr':
                MODEL_PATH = './pt-models/adapt_nr_telephone.pt'
                url_scoreq = 'https://zenodo.org/records/13860326/files/adapt_nr_telephone.pt'
                if not os.path.isfile(MODEL_PATH):
                    print('Downloading PyTorch weights from Zenodo')
                    print('SCOREQ | Mode: No-Reference | Data: Natural speech')
                    urlretrieve(url_scoreq, MODEL_PATH)
                    print('Download completed')
            elif mode == 'ref':
                MODEL_PATH = './pt-models/fixed_nmr_telephone.pt'
                url_scoreq = 'https://zenodo.org/records/13860326/files/fixed_nmr_telephone.pt'
                if not os.path.isfile(MODEL_PATH):
                    print('Downloading PyTorch weights from Zenodo')
                    print('SCOREQ | Mode: Full-Reference/NMR | Data: Natural speech')
                    urlretrieve(url_scoreq, MODEL_PATH)
                    print('Download completed')
            else:
                raise Exception('Mode must be either "nr" for no-reference or "ref" for full-reference and non-matching reference.')
        elif data_domain == 'synthetic':
            if mode == 'nr':
                MODEL_PATH = './pt-models/adapt_nr_synthetic.pt'
                url_scoreq = 'https://zenodo.org/records/13860326/files/adapt_nr_synthetic.pt'
                if not os.path.isfile(MODEL_PATH):
                    print('Downloading PyTorch weights from Zenodo')
                    print('SCOREQ | Mode: No-Reference | Data: Synthetic speech')
                    urlretrieve(url_scoreq, MODEL_PATH)
                    print('Download completed')
            elif mode == 'ref':
                MODEL_PATH = './pt-models/fixed_nmr_synthetic.pt'
                url_scoreq = 'https://zenodo.org/records/13860326/files/fixed_nmr_synthetic.pt'
                if not os.path.isfile(MODEL_PATH):
                    print('Downloading PyTorch weights from Zenodo')
                    print('SCOREQ | Mode: Full-reference/NMR | Data: Synthetic speech')
                    urlretrieve(url_scoreq, MODEL_PATH)
                    print('Download completed')
            else:
                raise Exception('Mode must be either "nr" for no-reference or "ref" for full-reference and non-matching reference.')
        else:
            raise Exception('Invalid data domain, you must select either "natural" or "synthetic".')

        model.load_state_dict(torch.load(MODEL_PATH, map_location=self.DEVICE))  
        self.model = model
        self.model.to(self.DEVICE)
        self.model.eval()


    def predict(self, test_path, ref_path=None):
        """"
        Makes predictions on audio files.

        Args:
            test_path: Path to the test audio file.
            ref_path: Path to the reference audio file (required in 'ref' mode). If ref_path is the clean counterpart the model will work in full-reference mode. If it's any clean speech, it will work in non-matching-reference mode.
            results_path: Optional path to save the results.

        Returns:
            The predicted quality score (MOS (1-5) in 'nr' mode, euclidean distance w.r.t to ref_path in 'ref' mode).
        """

        # Check invalid input
        if test_path is None:
            raise Exception('test_path not specified, you need to pass a valid path to an audio file')
        
        if self.mode == 'ref':
            if ref_path is None:
                raise Exception('ref_path must be a wav file in ref mode, found None')

        # *** CHOOSE MODE ***        
        # No-Reference (NR) mode
        if self.mode == 'nr':
            pred = np.round(self.nr_scoreq(test_path), 4)
            print(f'SCOREQ | No-Reference Mode | Domain {self.data_domain} | {test_path}: {pred}')

        elif self.mode == 'ref':
            # Full-reference (FR) mode or Non-Matching Reference (NMR) mode depending on which reference audio is used
            pred = self.ref_scoreq(test_path, ref_path)      
            print(f'SCOREQ | Fr/Nmr-Reference Mode | Domain {self.data_domain} | Ref-> {ref_path}, Test-> {test_path}: {pred}')
        
        else:
            raise Exception('Selected mode is not valid, choose between nr and ref')
        
        return pred

    def nr_scoreq(self, test_path):
        """
        Performs no-reference quality prediction.

        Args:
            test_path: Path to the test audio file.

        Returns:
            The predicted MOS.
        """

        wave = self.load_processing(test_path).to(self.DEVICE)
        with torch.no_grad():
            pred_mos = self.model(wave).item()
        
        return pred_mos

    def ref_scoreq(self, test_path, ref_path):
        """
        Performs reference-based quality prediction.

        Args:
            test_path: Path to the test audio file.
            ref_path: Path to the reference audio file. It can either be the clean counterpart (Full-reference) or any clean speech (Non-matching reference).
            phead: Choose whether you want to use linear projection head for predictions.

        Returns:
            The euclidean distance between the embeddings of the test and reference audio files.
        """
        test_wave = self.load_processing(test_path).to(self.DEVICE)
        ref_wave = self.load_processing(ref_path).to(self.DEVICE)

        # Get embeddings
        with torch.no_grad():
            test_emb = self.model(test_wave)
            ref_emb = self.model(ref_wave)

        # Get euclidean distance
        scoreq_dist = torch.cdist(test_emb, ref_emb).item()
        return scoreq_dist

    # Load wave file
    def load_processing(self, filepath, target_sr=16000, trim=False):
        """
        Loads and preprocesses an audio file.

        Args:
            filepath: Path to the audio file or a numpy array containing the audio data.
            target_sr: Target sample rate (default: 16000 Hz).
            trim: Whether to trim the audio to 10 seconds (default: False).

        Returns:
            The preprocessed audio waveform as a PyTorch tensor.
        """

        # Load waveform
        if isinstance(filepath, np.ndarray):
            filepath = filepath[0]
        wave, sr = torchaudio.load(filepath)
        
        # Check number of channels (MONO)
        if wave.shape[0] > 1:
            wave = ((wave[0,:] + wave[1,:])/2).unsqueeze(0)
        
        # Check resampling (16 khz)
        if sr != target_sr:
            wave = torchaudio.transforms.Resample(sr,  target_sr)(wave)
            sr = target_sr
        
        # Trim audio to 10 secs
        if trim:
            if wave.shape[1] > sr*10:
                wave = wave[:, :sr*10]
        
        return wave

class TripletModel(nn.Module):
    """
    Helper class defining the underlying neural network architecture for the SCOREQ model.
    """
    
    def __init__(self, ssl_model, ssl_out_dim, emb_dim=256):
        """
        Initializes the TripletModel.

        Args:
            ssl_model: The pre-trained self-supervised learning model (e.g., wav2vec).
            ssl_out_dim: Output dimension of the SSL model.
            emb_dim: Dimension of the final embedding (default: 256).
        """

        super(TripletModel, self).__init__()
        self.ssl_model = ssl_model
        self.ssl_features = ssl_out_dim
        self.embedding_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.ssl_features, emb_dim)
        )
    
    def forward(self, wav, phead=False):
        """
        Defines the forward pass of the model.

        Args:
            wav: Input audio waveform.
            phead: Attach embedding layer for reference mode prei

        Returns:
            The normalized embedding of the input audio.
        """

        wav = wav.squeeze(1)
        res = self.ssl_model(wav, mask=False, features_only=True)
        x = res['x']
        x = torch.mean(x, 1)

        # Choose if you want to keep projection head, remove for NR mode. Const model shows better performance in ODM without phead.
        if phead:
            x = self.embedding_layer(x)
        x = torch.nn.functional.normalize(x, dim=1)
        return x

# ******** MOS PREDICTOR **********
class MosPredictor(nn.Module):
    """
    Helper class that adds a layer for predicting Mean Opinion Scores (MOS) in the no-reference mode.
    """

    def __init__(self, pt_model, emb_dim=768):
        """
        Initializes the MosPredictor.

        Args:
            pt_model: The pre-trained triplet model.
            emb_dim: Dimension of the embedding (default: 768).
        """
        super(MosPredictor, self).__init__()
        self.pt_model = pt_model
        self.mos_layer = nn.Linear(emb_dim, 1)
        
    def forward(self, wav):
        """
        Defines the forward pass of the MOS predictor.

        Args:
            wav: Input audio waveform.

        Returns:
            The predicted MOS and the embedding.
        """
        x = self.pt_model(wav, phead=False)
        if len(x.shape) == 3:
            x.squeeze_(2)
        out = self.mos_layer(x)
        return out