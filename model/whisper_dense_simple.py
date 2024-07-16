import torch.nn as nn
import torchaudio
from transformers import WhisperModel
from tqdm import tqdm

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class WhisperDenseModelSimple(nn.Module):
    def __init__(self, input_dim=1024):
        super().__init__()
        self.dense1 = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 2048)
        )
        self.dense2 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.dense1(x)  # [batch, time, 1]
        x = self.dense2(x)  # [batch, time, 1]
        #x = x.mean(dim=[1, 2], keepdims=True)  # [batch, 1, 1]
        return x


class WhisperDenseFullModel(nn.Module):
    def __init__(self, checkpoint_path, input_dim=1024, model_name="whisper-base", freeze=True, cuda=True):
        super().__init__()
        self.cuda_flag = cuda
        self.encoder, self.processor = self._load_encoder(model_name)
        self.classficator = _load_classificator(input_dim, checkpoint_path)

        self.device = torch.device('cuda' if cuda else 'cpu')

        if self.freeze:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    def _load_encoder(self, model_name="whisper-base"):
        model_path = None
        if (model_name == "whisper-tiny"): # 39 M parameters
            model_path = "openai/whisper-tiny"
        elif (model_name == "whisper-base"): # 74 M parameters
            model_path = "openai/whisper-base"
        elif (model_name == "whisper-small"): # 244 M parameters
            model_path = "openai/whisper-small"
        elif (model_name == "whisper-medium"): # 769 M parameters
            model_path = "openai/whisper-medium"
        elif (model_name == "whisper-large"): # 1550 M parameters
            model_path = "openai/whisper-large"
        model = WhisperModel.from_pretrained(model_path)
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
        model = model.encoder
        model = model.to(self.device)
        model.eval()
        return model, feature_extractor

    def _load_classificator(self, input_dim, checkpoint_path):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)

        model = WhisperDenseModel(input_dim)
        model.load_state_dict(checkpoint['state_dict'])
        print("Checkpoint loaded.")
        return model

    def forward(self, x):
        x = self.encoder(x).last_hidden_state
        x = self.classficator(x)
        return x

    def _get_input_feature(self, filepath):
        signal, sr = torchaudio.load(filepath)
        feature = self.feature_extractor(
            signal.squeeze(), sampling_rate=16000, return_tensors="pt"
        )
        input_features = feature.input_features
        input_features = input_features.to(self.device)

        if self.cuda_flag:
            input_features = input_features.to(self.device)
        return input_features

    def calculate_dir(self, path):
        predictions = []
        for filepath in tqdm.tqdm(sorted(glob.glob(f"{path}/*.wav"))):
            input_features = self._get_input_feature(filepath)
            with torch.no_grad():
                pred = self.forward(input_features)

            predictions.append(pred.item())
            return predictions
        
    def calculate_one(self, filepath):
        input_features = self._get_input_feature(filepath)

        with torch.no_grad():
            pred = self.forward(input_features)

        return pred.cpu().item()