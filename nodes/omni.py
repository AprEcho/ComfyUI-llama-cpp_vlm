import random
from ..core.common import prompt_lib

class LlamaOmniTaskPrompter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "user_prompt (意图/灵感)": ("STRING", {"multiline": True, "placeholder": "在此输入您的创意点子，支持中文..."}),
                "library_tags (已选词库标签)": ("STRING", {"multiline": True, "default": ""}),
                "negative_library_tags (已选负面词)": ("STRING", {"multiline": True, "default": ""}),
                "random_sampling (开启随机抽卡)": ("BOOLEAN", {"default": False}),
                "seed (种子)": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "sampling_count (抽卡数量)": ("INT", {"default": 3, "min": 1, "max": 10}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt (英文提示词)", "negative (负面词)", "description (中文描述)")
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm/prompts"

    def process(self, **kwargs):
        user_prompt = kwargs.get("user_prompt (意图/灵感)", "")
        lib_tags_raw = kwargs.get("library_tags (已选词库标签)", "")
        neg_lib_tags_raw = kwargs.get("negative_library_tags (已选负面词)", "")
        random_sampling = kwargs.get("random_sampling (开启随机抽卡)")
        seed = kwargs.get("seed (种子)")
        sampling_count = kwargs.get("sampling_count (抽卡数量)", 3)

        # 1. 标签翻译 (中文 -> 英文)
        tag_map = prompt_lib.get_full_mapping()
        def translate(raw_str):
            if not raw_str: return ""
            tags = [t.strip() for t in raw_str.split(",") if t.strip()]
            return ", ".join([tag_map.get(t, t) for t in tags])

        lib_tags = translate(lib_tags_raw)
        neg_lib_tags = translate(neg_lib_tags_raw)

        # 2. 随机抽卡逻辑
        sampled_tags = ""
        if random_sampling:
            random.seed(seed)
            # 默认使用通用分类进行抽卡
            cats = ["01起手式", "02人物", "03服饰", "04人物发型", "05动作", "06面部表情", "场景道具", "景象", "艺术、魔法"]
            sampled_tags = prompt_lib.random_sample(cats, sampling_count)

        # 3. 最终拼接
        pos_content = ", ".join([p for p in [sampled_tags, lib_tags, user_prompt] if p])
        
        default_neg = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
        neg_content = ", ".join([p for p in [default_neg, neg_lib_tags] if p])

        return (pos_content, neg_content, f"模式：提示词工作室\n内容：{pos_content}")
