from . import api
from .llm_nodes import (
    llama_cpp_model_loader, llama_cpp_instruct_adv, llama_cpp_parameters,
    llama_cpp_unload_model, llama_cpp_clean_states
)
from .utils_nodes import (
    parse_json_node, json_to_bbox, bbox_to_segs, bbox_to_mask,
    bboxes_to_bbox, remove_code_block
)
from .omni_nodes import LlamaOmniTaskPrompter
from .preset_nodes import PromptEnhancerPreset

NODE_CLASS_MAPPINGS = {
    "llama_cpp_model_loader": llama_cpp_model_loader,
    "llama_cpp_instruct_adv": llama_cpp_instruct_adv,
    "llama_cpp_parameters": llama_cpp_parameters,
    "llama_cpp_unload_model": llama_cpp_unload_model,
    "llama_cpp_clean_states": llama_cpp_clean_states,
    "parse_json_node": parse_json_node,
    "json_to_bbox": json_to_bbox,
    "bbox_to_segs": bbox_to_segs,
    "bbox_to_mask": bbox_to_mask,
    "bboxes_to_bbox": bboxes_to_bbox,
    "remove_code_block": remove_code_block,
    "PromptEnhancerPreset": PromptEnhancerPreset,
    "LlamaOmniTaskPrompter": LlamaOmniTaskPrompter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "llama_cpp_model_loader": "Llama-cpp 模型加载器 (Model Loader)",
    "llama_cpp_instruct_adv": "Llama-cpp 智能提示词 (Instruct)",
    "llama_cpp_parameters": "Llama-cpp 参数设置 (Parameters)",
    "llama_cpp_unload_model": "Llama-cpp 卸载模型 (Unload Model)",
    "llama_cpp_clean_states": "Llama-cpp 清除状态 (Clean States)",
    "parse_json_node": "解析 JSON (Parse JSON)",
    "json_to_bbox": "JSON 转边界框 (JSON to BBoxes)",
    "bbox_to_segs": "边界框转 SEGS (BBoxes to SEGS)",
    "bbox_to_mask": "边界框转遮罩 (BBoxes to MASK)",
    "bboxes_to_bbox": "边界框选择 (BBoxes to BBox)",
    "remove_code_block": "解包代码块 (Unpack Code Block)",
    "PromptEnhancerPreset": "提示词增强预设 (Prompt Enhancer Preset)",
    "LlamaOmniTaskPrompter": "Llama 全能任务提示词中枢 (Omni Prompter)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
