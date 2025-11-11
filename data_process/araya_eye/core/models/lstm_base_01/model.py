import torch
import torch.nn as nn
from typing import Union, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import yaml

from .config import LstmBase01Config

@dataclass
class ModelConfig:
    input_size: int
    output_size: int
    num_layers: int
    hidden_size: int
    bidirectional: bool

class LSTMBase01Model(nn.Module):    
    def __init__(self, config: LstmBase01Config):
        super().__init__()
        self.config = config

        with open(config.config_path, "r", encoding="utf-8") as f:
            model_config = ModelConfig(**yaml.safe_load(f))
        self.input_size = model_config.input_size
        self.hidden_size = model_config.hidden_size
        self.output_size = model_config.output_size
        self.num_layers = model_config.num_layers
        self.device = config.device
        self.bidirectional = model_config.bidirectional
        assert self.bidirectional == True

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=config.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional,
            batch_first=True
        )
        lstm_output_size = self.hidden_size * (2 if self.bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(lstm_output_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_size, self.output_size)
        )
        self._init_weights()
    
    def forward(self, x: torch.Tensor, h: torch.Tensor, c:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 入力テンソル [batch_size, seq_len, input_size]
            hidden: オプション - 前のhidden state (h_0, c_0)
            
        Returns:
            クラス確率 [batch_size, 3], 新しいhidden state (h_n, c_n)
        """
        # LSTM処理 - hidden stateが与えられた場合は使用、そうでなければNoneで初期化
        output, (h_n, c_n) = self.lstm(x, (h, c))
        output = output[:, -1, :]
        logits = self.classifier(output)
        return logits, h_n, c_n
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.0)

    @classmethod
    def load_model(cls, model_path:Union[str,Path], config: LstmBase01Config):
        model_path = Path(model_path)        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if model_path.stat().st_size == 0:
            raise ValueError(f"Model file is empty: {model_path}")
        
        model = cls(config)
        checkpoint = torch.load(str(model_path), map_location=config.device, weights_only=False)
        
        # 状態辞書取得
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("INFO: Loading 3-class LSTM model from checkpoint")
        else:
            state_dict = checkpoint
        
        # LSTM構造だが'rnn'キーで保存されたモデルの自動変換
        if 'rnn.weight_ih_l0' in state_dict:
            converted_state = {}
            for key, value in state_dict.items():
                if key.startswith('rnn.'):
                    # 'rnn.*' -> 'lstm.*' キー名修正
                    new_key = key.replace('rnn.', 'lstm.')
                    converted_state[new_key] = value
                else:
                    converted_state[key] = value
            state_dict = converted_state
            print("INFO: LSTM model with 'rnn' keys automatically converted")
        
        # モデル読み込み
        model.load_state_dict(state_dict)
        model.to(config.device)
        model.eval()
        return model