# ComfyUI-llama-cpp_vlm
在 ComfyUI 中基于 llama.cpp 框架原生运行 LLM & VLM 模型，提供专业的提示词工程工作流。

**[[🌐English](./README.md)]**

## 预览
![](./img/preview.jpg)

## 核心功能
*   **Llama 全能任务提示词中枢 (Omni Prompter)**：
    *   内置图形化提示词工作室（Prompt Studio），点击即可选择标签。
    *   支持中文标签自动映射为英文。
    *   支持随机抽卡功能，快速获取创作灵感。
*   **Llama-cpp 模型加载与推理**：
    *   原生支持 GGUF 格式的 LLM 和 VLM 模型。
    *   支持多图像输入、视频序列推理。
    *   支持思维链（Thought）提取。
*   **提示词增强预设 (Prompt Enhancer Preset)**：针对 Flux, Wan2.1 等模型优化的系统提示词预设。
*   **VLM 辅助工具**：JSON 解析、坐标框（BBox）绘制、BBox 转 Mask/SEGS 等。

## 安装步骤

#### 安装节点:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lihaoyun6/ComfyUI-llama-cpp.git
python -m pip install -r ComfyUI-llama-cpp/requirements.txt
```

### 模型路径:
- 请将下载的 `.gguf` 模型放置在 `ComfyUI/models/LLM` 目录下。
- **VLM 提示**：在使用 VLM 模型进行图像推理之前，请确保已下载并选择了主模型对应的 `mmproj` 权重文件。

## 项目结构 (模块化)
*   `nodes/`: ComfyUI 节点包装（UI 层）。
*   `core/`: 核心业务逻辑，包括词库管理、模型加载驱动等。
*   `api/`: 前后端通信路由。
*   `js/`: 提示词工作室前端界面。

## 鸣谢
- [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) @JamePeng
- [ComfyUI-llama-cpp](https://github.com/kijai/ComfyUI-llama-cpp) @kijai
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous
