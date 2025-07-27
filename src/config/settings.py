import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    base_model: str = "1TuanPham/T-VisStar-7B-v0.1"
    max_seq_length: int = 4096
    dtype: Optional[str] = None
    load_in_4bit: bool = True
    r: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    bias: str = "none"
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 3407
    use_rslora: bool = False
    loftq_config: Optional[dict] = None
    target_modules: list = None

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]


@dataclass
class TrainingConfig:
    dataset_id: str = "PhanDai/luat-viet-nam-qa_small"
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    max_steps: int = 150
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    seed: int = 3407
    output_dir: str = "outputs"
    logging_steps: int = 1
    optim: str = "adamw_8bit"
    dataset_num_proc: int = 2
    packing: bool = True


@dataclass
class InferenceConfig:
    max_new_tokens: int = 256
    use_cache: bool = True
    streaming: bool = True


@dataclass
class APIConfig:
    hf_token: str = ""
    comet_api_key: str = ""
    comet_project_name: str = "second-brain-course"
    # model_name: str = "Chatbot_VietNamese_Law"
    model_name: str = "1TuanPham/T-VisStar-7B-v0.1"

    def __post_init__(self):
        self.hf_token = os.getenv("HF_TOKEN", self.hf_token)
        self.comet_api_key = os.getenv("COMET_API_KEY", self.comet_api_key)
        self.comet_project_name = os.getenv("COMET_PROJECT_NAME", self.comet_project_name)

    @property
    def enable_hf(self) -> bool:
        return bool(self.hf_token)

    @property
    def enable_comet(self) -> bool:
        return bool(self.comet_api_key)


@dataclass
class Config:
    model: ModelConfig = None
    training: TrainingConfig = None
    inference: InferenceConfig = None
    api: APIConfig = None

    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.inference is None:
            self.inference = InferenceConfig()
        if self.api is None:
            self.api = APIConfig()


def get_config() -> Config:
    """Get the application configuration."""
    return Config()


def configure_gpu_settings(config: Config, gpu_name: str) -> None:
    """Configure GPU-specific settings based on detected GPU."""
    if gpu_name and "T4" in gpu_name:
        config.model.load_in_4bit = True
        config.training.max_steps = 25
    elif gpu_name and ("A100" in gpu_name or "L4" in gpu_name):
        config.model.load_in_4bit = False
        config.training.max_steps = 250
    elif gpu_name:
        config.model.load_in_4bit = False
        config.training.max_steps = 150
    else:
        raise ValueError("No Nvidia GPU found.")


def setup_environment(config: Config) -> None:
    """Set up environment variables."""
    if config.api.enable_hf:
        os.environ["HF_TOKEN"] = config.api.hf_token
    if config.api.enable_comet:
        os.environ["COMET_API_KEY"] = config.api.comet_api_key
        os.environ["COMET_PROJECT_NAME"] = config.api.comet_project_name


ALPACA_PROMPT = """Dưới đây là hướng dẫn mô tả một nhiệm vụ, kết hợp với thông tin đầu vào cung cấp thêm ngữ cảnh. Hãy viết phản hồi hoàn thành yêu cầu một cách phù hợp.

### Instruction:
Bạn là một trợ lý thông minh, hãy trả lời câu hỏi hiện tại của user dựa trên lịch sử chat và các tài liệu liên quan. Câu trả lời phải ngắn gọn, chính xác nhưng vẫn đảm bảo đầy đủ các ý chính.
### Input:
{}

### Response:
{}"""