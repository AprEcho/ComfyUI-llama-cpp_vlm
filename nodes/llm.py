import os
import gc
import json
import random
import torch
import numpy as np

import folder_paths
import comfy.model_management as mm
from llama_cpp import Llama

from ..core.common import any_type, base_path
from ..core.cqdm import cqdm
from ..core.gguf_layers import get_layer_count
from .utils import image2base64, scale_image, extract_thought

# 导入 llama-cpp-python 内置的对话模板处理器
from llama_cpp.llama_chat_format import (
    Llava15ChatHandler, Llava16ChatHandler, MoondreamChatHandler,
    NanoLlavaChatHandler, Llama3VisionAlphaChatHandler, MiniCPMv26ChatHandler
)

# 尝试导入更先进的 ChatHandler (需要最新版 llama-cpp-python)
try:
    from llama_cpp.llama_chat_format import MTMDChatHandler
    _MTMD = True
except:
    _MTMD = False

# 节点界面上显示的对话模型类型列表
chat_handlers = ["None", "LLaVA-1.5", "LLaVA-1.6", "Moondream2", "nanoLLaVA", "llama3-Vision-Alpha", "MiniCPM-v2.6", "MiniCPM-v4.5", "MiniCPM-v4.5-Thinking"]

# 动态探测并添加更多模型支持
try:
    from llama_cpp.llama_chat_format import Gemma3ChatHandler
    chat_handlers += ["Gemma3"]
except:
    Gemma3ChatHandler = None

try:
    from llama_cpp.llama_chat_format import Gemma4ChatHandler
    chat_handlers += ["Gemma4"]
except:
    Gemma4ChatHandler = None

try:
    from llama_cpp.llama_chat_format import Qwen25VLChatHandler
    chat_handlers += ["Qwen2.5-VL"]
except:
    Qwen25VLChatHandler = None

try:
    from llama_cpp.llama_chat_format import Qwen3VLChatHandler
    chat_handlers += ["Qwen3-VL", "Qwen3-VL-Thinking"]
except:
    Qwen3VLChatHandler = None

try:
    from llama_cpp.llama_chat_format import Qwen35ChatHandler
    chat_handlers += ["Qwen3.5", "Qwen3.5-Thinking"]
except:
    Qwen35ChatHandler = None

try:
    from llama_cpp.llama_chat_format import (GLM46VChatHandler, LFM2VLChatHandler, GLM41VChatHandler)
    chat_handlers += ["GLM-4.6V", "GLM-4.6V-Thinking", "GLM-4.1V-Thinking", "LFM2-VL"]
except:
    GLM46VChatHandler = None
    LFM2VLChatHandler = None
    GLM41VChatHandler = None

try:
    from llama_cpp.llama_chat_format import LFM25VLChatHandler
    chat_handlers += ["LFM2.5-VL"]
except:
    LFM25VLChatHandler = None

try:
    from llama_cpp.llama_chat_format import GraniteDoclingChatHandler
    chat_handlers += ["Granite-Docling"]
except:
    GraniteDoclingChatHandler = None

class LLAMA_CPP_STORAGE:
    """模型存储单例：负责 Llama 模型的持久化加载、显存管理和状态缓存"""
    llm = None
    chat_handler = None
    current_config = None
    messages = {}  # 存储多轮对话历史
    sys_prompts = {} # 存储每个 Session 的系统提示词

    @classmethod
    def clean_state(cls, id=-1):
        """清理特定或全部 Session 的对话记录"""
        if id == -1:
            cls.messages.clear()
            cls.sys_prompts.clear()
        else:
            cls.messages.pop(f"{id}", None)
            cls.sys_prompts.pop(f"{id}", None)

    @classmethod
    def clean(cls, all=False):
        """释放模型资源并清理显存"""
        try:
            cls.llm.close()
        except Exception:
            pass

        try:
            cls.chat_handler._exit_stack.close()
        except Exception:
            pass

        cls.llm = None
        cls.chat_handler = None
        cls.current_config = None
        if all:
            cls.clean_state()

        gc.collect()
        mm.soft_empty_cache()

    @classmethod
    def load_model(cls, config):
        """根据配置动态加载 GGUF 模型和对应的 Vision Handler"""
        def get_chat_handler(chat_handler_name):
            # 映射 UI 名称到对应的处理器类
            match chat_handler_name:
                case "Qwen3.5"|"Qwen3.5-Thinking": return Qwen35ChatHandler
                case "Qwen3-VL"|"Qwen3-VL-Thinking": return Qwen3VLChatHandler
                case "Qwen2.5-VL": return Qwen25VLChatHandler
                case "LLaVA-1.5": return Llava15ChatHandler
                case "LLaVA-1.6": return Llava16ChatHandler
                case "Moondream2": return MoondreamChatHandler
                case "nanoLLaVA": return NanoLlavaChatHandler
                case "llama3-Vision-Alpha": return Llama3VisionAlphaChatHandler
                case "MiniCPM-v2.6": return MiniCPMv26ChatHandler
                case "MiniCPM-v4.5"|"MiniCPM-v4.5-Thinking": return MiniCPMv26ChatHandler
                case "Gemma3": return Gemma3ChatHandler
                case "Gemma4": return Gemma4ChatHandler
                case "GLM-4.6V"|"GLM-4.6V-Thinking": return GLM46VChatHandler
                case "GLM-4.1V-Thinking": return GLM41VChatHandler
                case "LFM2-VL": return LFM2VLChatHandler
                case "LFM2.5-VL": return LFM25VLChatHandler
                case "Granite-Docling": return GraniteDoclingChatHandler
                case "None": return None
                case _: raise ValueError(f'Unknown model type: "{chat_handler_name}"')

        cls.clean(all=True)
        cls.current_config = config.copy()
        model = config["model"]
        mmproj = config["mmproj"]
        chat_handler_name = config["chat_handler"]
        n_ctx = config["n_ctx"]
        vram_limit = config["vram_limit"]
        image_max_tokens = config["image_max_tokens"]
        image_min_tokens = config["image_min_tokens"]
        n_gpu_layers = -1

        model_path = os.path.join(folder_paths.models_dir, 'LLM', model)
        handler = get_chat_handler(chat_handler_name)

        # 智能显存估算：根据限额计算 GPU 层数
        if vram_limit != -1:
            gguf_layers = get_layer_count(model_path) or 32
            gguf_size = os.path.getsize(model_path) * 1.55 / (1024 ** 3)
            gguf_layer_size = gguf_size / gguf_layers

        # 加载 Vision 组件 (mmproj)
        if mmproj and mmproj != "None":
            mmproj_path = os.path.join(folder_paths.models_dir, 'LLM', mmproj)
            if chat_handler_name == "None":
                raise ValueError('"chat_handler" 不能为空，请选择正确的模型类型！')
            if vram_limit != -1:
                mmproj_size = os.path.getsize(mmproj_path) * 1.55 / (1024 ** 3)
                n_gpu_layers = max(1, int((vram_limit - mmproj_size) / gguf_layer_size))

            think_mode = "Thinking" in chat_handler_name
            kwargs = {"clip_model_path": mmproj_path, "verbose": False}
            
            # 特殊模型的特殊参数
            if chat_handler_name in ["Qwen3-VL", "Qwen3-VL-Thinking"]:
                kwargs["force_reasoning"] = think_mode
                kwargs["image_max_tokens"] = image_max_tokens
                kwargs["image_min_tokens"] = image_min_tokens
            elif chat_handler_name in ["MiniCPM-v4.5", "GLM-4.6V", "Qwen3.5"]:
                kwargs["enable_thinking"] = think_mode
            if _MTMD:
                kwargs["image_max_tokens"] = image_max_tokens
                kwargs["image_min_tokens"] = image_min_tokens
            
            try:
                cls.chat_handler = handler(**kwargs)
            except Exception as e:
                raise RuntimeError(f"{e}\n请更新 llama-cpp-python 到最新版本。")
        else:
            # 纯文本模式
            if vram_limit != -1:
                n_gpu_layers = max(1, int(vram_limit / gguf_layer_size))
            cls.chat_handler = handler(verbose=False) if handler else None

        # 实例化 Llama
        cls.llm = Llama(model_path, chat_handler=cls.chat_handler, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, verbose=False)

# 应用全局资源回收钩子：当 ComfyUI 卸载模型时，同步释放 Llama 显存
if not hasattr(mm, "unload_all_models_backup"):
    mm.unload_all_models_backup = mm.unload_all_models
    def patched_unload_all_models(*args, **kwargs):
        LLAMA_CPP_STORAGE.clean(all=True)
        return mm.unload_all_models_backup(*args, **kwargs)
    mm.unload_all_models = patched_unload_all_models

# 注册 LLM 模型路径
llm_extensions = ['.ckpt', '.pt', '.bin', '.pth', '.safetensors', '.gguf']
folder_paths.folder_names_and_paths["LLM"] = ([os.path.join(folder_paths.models_dir, "LLM")], llm_extensions)

# 节点内置的快捷提示词模板
preset_prompts = {
    "Empty - Nothing": "",
    "Normal - Describe": "Describe this @.",
    "Prompt Style - Tags": "Your task is to generate a list of comma-separated tags for a text-to-@ AI...",
    "Prompt Style - Simple": "Analyze the @ and generate a simple, single-sentence text-to-@ prompt.",
    "Prompt Style - Detailed": "Generate a detailed, artistic text-to-@ prompt based on the @.",
    "Prompt Style - Extreme Detailed": "Generate an extremely detailed and descriptive text-to-@ prompt from the @.",
    "Prompt Style - Cinematic": "Act as a master prompt engineer. Create a highly detailed and evocative prompt for an @ generation AI.",
    "Creative - Detailed Analysis": "Describe this @ in detail, breaking down the subject, attire, background, etc.",
    "Creative - Summarize Video": "Summarize the key events and narrative points in this video.",
    "Creative - Short Story": "Write a short, imaginative story inspired by this @ or video.",
    "Creative - Refine & Expand Prompt": "Refine and enhance the following user prompt for creative text-to-@ generation.",
    "Vision - *Bounding Box": 'Locate every instance that belongs to the following categories: "#". Report bbox coordinates in JSON format.'
}
preset_tags = list(preset_prompts.keys())

class llama_cpp_model_loader:
    """模型加载器节点：负责配置并触发 LLM 模型加载"""
    @classmethod
    def INPUT_TYPES(s):
        all_llms = folder_paths.get_filename_list("LLM")
        model_list = [f for f in all_llms if "mmproj" not in f.lower()]
        mmproj_list = ["None"]+[f for f in all_llms if "mmproj" in f.lower()]
        return {"required": {
            "model": (model_list,),
            "mmproj": (mmproj_list, {"default": "None"}),
            "chat_handler": (chat_handlers, {"default": "None"}),
            "n_ctx": ("INT", {"default": 8192, "min": 1024, "max": 327680}),
            "vram_limit": ("INT", {"default": -1, "min": -1, "max": 1024, "tooltip": "VRAM 使用上限 (GB)，-1 为不限制"}),
            "image_min_tokens": ("INT", {"default": 0, "min": 0, "max": 4096}),
            "image_max_tokens": ("INT", {"default": 0, "min": 0, "max": 4096}),
        }}
    RETURN_TYPES = ("LLAMACPPMODEL",)
    RETURN_NAMES = ("llama_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "llama-cpp-vlm"
    def loadmodel(self, **kwargs): return (kwargs,)

class llama_cpp_instruct_adv:
    """智能提示词节点：执行 LLM/VLM 推理的核心节点"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "llama_model": ("LLAMACPPMODEL",),
                "preset_prompt": (preset_tags, {"default": preset_tags[1]}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True}),
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "inference_mode": (["one by one", "images", "video"], {"default": "one by one"}),
                "max_frames": ("INT", {"default": 24, "min": 2, "max": 1024, "tooltip": "视频模式下的最大采样帧数"}),
                "max_size": ("INT", {"default": 256, "min": 128, "max": 16384, "tooltip": "图像缩放大小"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_offload": ("BOOLEAN", {"default": False, "tooltip": "推理完成后立即释放显存"}),
                "save_states": ("BOOLEAN", {"default": False, "tooltip": "启用多轮对话历史记录"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
            "optional": {
                "parameters": ("LLAMACPPARAMS",),
                "images": ("IMAGE",),
                "queue_handler": (any_type,),
            },
        }
    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("output", "thought", "output_list", "state_uid")
    OUTPUT_IS_LIST = (False, False, True, False)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"

    def sanitize_messages(self, messages):
        """清理历史记录，将 base64 图像替换为占位符以节省内存/显存"""
        clean_messages = []
        for msg in messages:
            new_msg = msg.copy()
            content = msg.get("content")
            if isinstance(content, list):
                new_content = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        new_item = item.copy()
                        new_item["image_url"] = {"url": "data:image/png;base64,...(cleaned)"}
                        new_content.append(new_item)
                    else:
                        new_content.append(item)
                new_msg["content"] = new_content
            clean_messages.append(new_msg)
        return clean_messages

    def process(self, llama_model, preset_prompt, custom_prompt, system_prompt, inference_mode, max_frames, max_size, seed, force_offload, save_states, unique_id, parameters=None, images=None, queue_handler=None):
        # 1. 模型热切换检查
        if not llama_model: raise ValueError("需要连接 Llama 模型配置！")
        if not LLAMA_CPP_STORAGE.llm or LLAMA_CPP_STORAGE.current_config != llama_model:
            LLAMA_CPP_STORAGE.load_model(llama_model)

        # 2. 参数准备
        if parameters is None:
            parameters = {"max_tokens": 1024, "top_k": 30, "top_p": 0.9, "temperature": 0.8}
        
        _parameters = parameters.copy()
        uid = unique_id.rpartition('.')[-1] if parameters.get("state_uid", -1) == -1 else parameters.get("state_uid")
        _parameters.pop("state_uid", None)

        # 3. 对话历史与系统提示词管理
        last_sys_prompt = LLAMA_CPP_STORAGE.sys_prompts.get(f"{uid}")
        video_input = inference_mode == "video"
        sys_p = "Treat sequence as video, " + system_prompt if video_input else system_prompt
        
        if last_sys_prompt != sys_p:
            messages = []
            LLAMA_CPP_STORAGE.clean_state()
            LLAMA_CPP_STORAGE.sys_prompts[f"{uid}"] = sys_p
            if sys_p.strip(): messages.append({"role": "system", "content": sys_p})
        else:
            messages = LLAMA_CPP_STORAGE.messages.get(f"{uid}", []) if save_states else []

        # 4. 构建用户提示词
        user_content = []
        p_text = preset_prompts[preset_prompt].replace("#", custom_prompt.strip()).replace("@", "video" if video_input else "image")
        user_content.append({"type": "text", "text": p_text if "*" in preset_prompt or not custom_prompt.strip() else custom_prompt})

        out_text, out_thought = "", ""
        out_list, out_thought_list = [], []

        # 5. 执行推理
        if images is not None:
            # 5.1 视觉模式
            frames = [images[i] for i in np.linspace(0, len(images)-1, max_frames, dtype=int)] if video_input else images
            if inference_mode == "one by one":
                # 逐张图片推理流程
                for img in cqdm(frames):
                    if mm.processing_interrupted(): raise mm.InterruptProcessingException()
                    u_c = user_content + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image2base64(np.clip(255.0*img.cpu().numpy().squeeze(),0,255).astype(np.uint8))}"}}]
                    res = LLAMA_CPP_STORAGE.llm.create_chat_completion(messages=messages+[{"role":"user","content":u_c}], seed=seed, **_parameters)
                    msg = res['choices'][0]['message']
                    txt, tht = msg['content'], msg.get('thought', '')
                    if not tht: txt, tht = extract_thought(txt)
                    out_list.append(txt); out_thought_list.append(tht)
                out_text, out_thought = "\n\n".join(out_list), "\n\n".join(out_thought_list)
            else:
                # 图像组/视频推理流程 (所有图一起发给模型)
                for img in frames:
                    u_c_img = image2base64(scale_image(img, max_size)) if len(frames)>1 else image2base64(np.clip(255.0*img.cpu().numpy().squeeze(),0,255).astype(np.uint8))
                    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{u_c_img}"}})
                res = LLAMA_CPP_STORAGE.llm.create_chat_completion(messages=messages+[{"role":"user","content":user_content}], seed=seed, **_parameters)
                msg = res['choices'][0]['message']
                out_text, out_thought = msg['content'], msg.get('thought', '')
                if not out_thought: out_text, out_thought = extract_thought(out_text)
                out_list, out_thought_list = [out_text], [out_thought]
        else:
            # 5.2 纯文本模式
            res = LLAMA_CPP_STORAGE.llm.create_chat_completion(messages=messages+[{"role":"user","content":user_content}], seed=seed, **_parameters)
            msg = res['choices'][0]['message']
            out_text, out_thought = msg['content'], msg.get('thought', '')
            if not out_thought: out_text, out_thought = extract_thought(out_text)
            out_list, out_thought_list = [out_text], [out_thought]

        # 6. 保存状态并回收资源
        if save_states:
            full = f"<think>\n{out_thought}\n</think>\n\n{out_text}" if out_thought and "<think>" not in out_text else out_text
            messages.append({"role": "assistant", "content": full})
            LLAMA_CPP_STORAGE.messages[f"{uid}"] = self.sanitize_messages(messages)

        if force_offload: LLAMA_CPP_STORAGE.clean()
        gc.collect()
        
        try:
            return_uid = int(uid)
        except (ValueError, TypeError):
            return_uid = 0
            
        return (out_text, out_thought, out_list, return_uid)

class llama_cpp_parameters:
    """参数设置节点：配置 LLM 推理的超参数 (Temperature, Top-P 等)"""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "max_tokens": ("INT", {"default": 1024, "min": 0, "max": 32768}),
            "top_k": ("INT", {"default": 30}),
            "top_p": ("FLOAT", {"default": 0.9}),
            "temperature": ("FLOAT", {"default": 0.8}),
            "repeat_penalty": ("FLOAT", {"default": 1.0}),
            "frequency_penalty": ("FLOAT", {"default": 0.0}),
            "presence_penalty": ("FLOAT", {"default": 1.0}),
            "mirostat_mode": ("INT", {"default": 0}),
            "mirostat_eta": ("FLOAT", {"default": 0.1}),
            "mirostat_tau": ("FLOAT", {"default": 5.0}),
            "state_uid": ("INT", {"default": -1, "tooltip": "Session ID，-1 为自动生成"}),
        }}
    RETURN_TYPES = ("LLAMACPPARAMS",)
    RETURN_NAMES = ("parameters",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    def process(self, **kwargs): return (kwargs,)

class llama_cpp_clean_states:
    """状态清理节点：手动清除特定 Session 的上下文历史"""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"any": (any_type,), "state_uid": ("INT", {"default": -1})}}
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("any",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    def process(self, any_val, state_uid):
        LLAMA_CPP_STORAGE.clean_state(state_uid)
        return (any_val,)

class llama_cpp_unload_model:
    """强制卸载节点：立即释放 Llama 模型的显存占用"""
    @classmethod
    def INPUT_TYPES(s): return {"required": {"any": (any_type,)}}
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("any",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    def process(self, any_val):
        LLAMA_CPP_STORAGE.clean()
        return (any_val,)
