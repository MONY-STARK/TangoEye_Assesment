import torchreid
import torch

class Person_Re_ID_Model:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def choose_model(self, model_name: str):
        
        model_dict = {
            "mobilenet": "mobilenetv2_x1_4",
            "osnet": "osnet_x1_0",
            "resnet": "resnet50"
        }
        
        if model_name.lower() in model_dict:
            chosen_model = model_dict[model_name.lower()]
            model = torchreid.utils.FeatureExtractor(
                model_name=chosen_model,
                model_path=None,  
                device=self.device
            )
            return model
        else:
            raise ValueError(f"Model '{model_name}' is not supported. Choose from {list(model_dict.keys())}.")
