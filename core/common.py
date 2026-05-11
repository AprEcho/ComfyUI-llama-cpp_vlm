import os
from .prompt_lib import PromptLibrary

# 1. 路径配置
# 获取项目根目录 (当前文件在 core/ 目录下，所以取两层 parent)
base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# 2. 全局单例实例化
# prompt_lib 负责管理本地提示词词库的加载、查询和保存
prompt_lib = PromptLibrary(base_path)

class AnyType(str):
    """
    特殊的“万能类型”定义：
    在 ComfyUI 中，这种类型的挂件可以连接到任何其他的输入/输出点，
    因为它在相等性判断中永远返回 True。
    """
    def __ne__(self, __value: object) -> bool:
        return False

# 实例化万能类型，供各节点定义 RETURN_TYPES 或 INPUT_TYPES 使用
any_type = AnyType("*")
