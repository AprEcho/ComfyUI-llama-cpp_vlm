from ..core.preset_strings import *

class PromptEnhancerPreset:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "preset": (["Qwen-Image [EN]", "Qwen-Image [ZH]", "Qwen-Image 2512 [EN]", "Qwen-Image 2512 [ZH]", "Qwen-Image-Edit", "Qwen-Image-Edit 2509", "Qwen-Image-Edit 2511", "Z-Image Turbo", "Flux.2 T2I", "Flux.2 I2I", "Wan T2V [EN]", "Wan T2V [ZH]", "Wan I2V [EN]", "Wan I2V [ZH]", "Wan I2V Full-Auto [EN]", "Wan I2V Full-Auto [ZH]", "Wan FLF2V [EN]", "Wan FLF2V [ZH]"], )
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("system_prompt",)
    FUNCTION = "main"
    CATEGORY = "llama-cpp-vlm"

    def main(self, preset):
        match preset:
            case "Qwen-Image [EN]": return (QWEN_IMAGE_EN,)
            case "Qwen-Image [ZH]": return (QWEN_IMAGE_ZH,)
            case "Qwen-Image 2512 [EN]": return (QWEN_IMAGE_2512_EN,)
            case "Qwen-Image 2512 [ZH]": return (QWEN_IMAGE_2512_ZH,)
            case "Qwen-Image-Edit": return (QWEN_IMAGE_EDIT,)
            case "Qwen-Image-Edit 2509": return (QWEN_IMAGE_EDIT_2509,)
            case "Qwen-Image-Edit 2511": return (QWEN_IMAGE_EDIT_2511,)
            case "Z-Image Turbo": return (ZIMAGE_TURBO,)
            case "Flux.2 T2I": return (FLUX2_T2I,)
            case "Flux.2 I2I": return (FLUX2_I2I,)
            case "Wan T2V [EN]": return (WAN_T2V_EN,)
            case "Wan T2V [ZH]": return (WAN_T2V_ZH,)
            case "Wan I2V [EN]": return (WAN_I2V_EN,)
            case "Wan I2V [ZH]": return (WAN_I2V_ZH,)
            case "Wan I2V Full-Auto [EN]": return (WAN_I2V_EMPTY_EN,)
            case "Wan I2V Full-Auto [ZH]": return (WAN_I2V_EMPTY_ZH,)
            case "Wan FLF2V [EN]": return (WAN_FLF2V_EN,)
            case "Wan FLF2V [ZH]": return (WAN_FLF2V_ZH,)
            case _: raise ValueError(f'Unknown preset: "{preset}"')
