from robot.config.finetune_config import ModelConfig

class BaseModel:
    def __init__(self, config: ModelConfig):
        self.config = config
