import os
from .support.prompt_library import PromptLibrary

base_path = os.path.dirname(os.path.realpath(__file__))
prompt_lib = PromptLibrary(base_path)

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")
