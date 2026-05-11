# ComfyUI-llama-cpp_vlm
Native LLM & VLM model execution in ComfyUI based on the llama.cpp framework, providing a professional prompt engineering workflow.

**[[🇨🇳中文版](./README_zh.md)]**

## Preview
![](./img/preview.jpg)

## Core Features
*   **Llama Omni Task Prompter**:
    *   Built-in graphical Prompt Studio for easy tag selection.
    *   Automatic mapping of Chinese tags to English.
    *   Random sampling feature for quick creative inspiration.
*   **Llama-cpp Model Loading & Inference**:
    *   Native support for LLM and VLM models in GGUF format.
    *   Support for multi-image input and video sequence inference.
    *   Chain-of-thought (Thought) extraction.
*   **Prompt Enhancer Preset**: System prompt presets optimized for models like Flux, Wan2.1, etc.
*   **VLM Utility Tools**: JSON parsing, BBox drawing, BBox to Mask/SEGS conversion, etc.

## Installation

#### Install the node:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lihaoyun6/ComfyUI-llama-cpp.git
python -m pip install -r ComfyUI-llama-cpp/requirements.txt
```

### Model Paths:
- Place your downloaded `.gguf` model files in the `ComfyUI/models/LLM` directory.
- **VLM Tip**: Before performing image inference with VLM models, ensure you have downloaded and selected the corresponding `mmproj` weight file for the main model.

## Project Structure (Modular)
*   `nodes/`: ComfyUI node wrappers (UI Layer).
*   `core/`: Core business logic, including library management and model drivers.
*   `api/`: Backend routes for communication.
*   `js/`: Frontend interface for Prompt Studio.

## Credits
- [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) @JamePeng
- [ComfyUI-llama-cpp](https://github.com/kijai/ComfyUI-llama-cpp) @kijai
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous
