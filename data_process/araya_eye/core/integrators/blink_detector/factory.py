from . import BlinkDetectorConfig, BlinkDetector

def create_blink_detector(config: BlinkDetectorConfig)->BlinkDetector:
    if config.type == "lstm_base_01":
        from .wrappers.lstm_base_01 import LstmBase01BlinkDetector
        assert config.lstm_base_01_blink_detector
        return LstmBase01BlinkDetector(config.lstm_base_01_blink_detector)
    
    raise ValueError(f"Unsupported detector type: {config.type}")