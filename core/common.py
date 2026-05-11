import os
from .prompt_lib import PromptLibrary

# 获取当前文件所在目录的上一级，即项目根目录
base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
prompt_lib = PromptLibrary(base_path)

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")
