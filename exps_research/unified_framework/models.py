"""
Unified model setup for experiments
"""

from typing import Dict, Any, Optional, Union

from smolagents import VLLMServerModel, VLLMModel


def setup_model(
    model_type: str = "vllm", 
    model_id: str = None, 
    fine_tuned: bool = False,
    local_device_id: int = -1,
    lora_path: str = None,
    **kwargs
) -> Union[VLLMServerModel, VLLMModel]:
    """
    Initialize a model for experiments
    
    Args:
        model_type: Type of model to use ("vllm")
        model_id: Model ID to use (e.g., Qwen/Qwen2.5-7B-Instruct)
        fine_tuned: Whether to use a fine-tuned model
        **kwargs: Additional keyword arguments for model initialization
    
    Returns:
        Initialized model
    """
    default_models = {
        "vllm": "Qwen/Qwen2.5-7B-Instruct",
    }
    model_id = model_id or default_models.get(model_type)    
    if model_type == "openai":
        raise ValueError("OpenAI-backed models are disabled to avoid OpenAI API calls. Use model_type='vllm'.")
    elif model_type == "vllm":
        if fine_tuned:
            if int(local_device_id) >= 0:
                return VLLMModel(
                    model_id=model_id,
                    lora_path=lora_path,
                    local_device_id=local_device_id,
                    **kwargs
                )
            else:
                return VLLMServerModel(
                    model_id=model_id,
                    # api_base="http://0.0.0.0:8000/v1",
                    # api_key="token-abc",
                    lora_name="finetune",
                    **kwargs
                )
        else:
            if int(local_device_id) >= 0:
                return VLLMModel(
                    model_id=model_id,
                    local_device_id=local_device_id,
                    **kwargs
                )
            else:
                return VLLMServerModel(
                    model_id=model_id,
                    # api_base="http://0.0.0.0:8000/v1",
                    # api_key="token-abc",
                    **kwargs
                )
    else:
        raise ValueError(f"Unsupported model type: {model_type}") 
