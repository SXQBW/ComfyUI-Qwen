<div align="center">

# ComfyUI-Qwen 🐼
<p align="center">
        <a href="README_zh.md">中文</a> &nbsp｜ &nbsp English
</p>

**Where Figma meets VSCode: Artistic vision meets engineering precision —— a romantic manifesto from designers to the code world.**  
✨ Qwen3 ComfyUI 集成组件是一个强大的工具，专为ComfyUI工作流设计，旨在无缝集成Qwen系列大型语言模型（LLM）。这个组件提供了单轮生成和多轮对话两种节点，支持自动模型下载、智能内存管理和"思考模式"等高级功能。 ✨
  
[![Star History](https://img.shields.io/github/stars/SXQBW/ComfyUI-Qwen?style=for-the-badge&logo=starship&color=FE428E&labelColor=0D1117)](https://github.com/SXQBW/ComfyUI-Qwen/stargazers)
[![Model Download](https://img.shields.io/badge/Model_Download-6DB33F?style=for-the-badge&logo=ipfs&logoColor=white)](https://huggingface.co/Qwen)
</div>
<div align="center">
  <img src="pic/screenshot-20250523-180706.png" width="90%">
</div>

---



### 主要特点

- 支持Qwen3系列模型（包括Qwen3-7B、Qwen3-14B、Qwen3-72B等）
- 智能设备检测和内存优化，自动选择最佳运行设备
- 内置模型注册表，自动选择最快下载源（Hugging Face或ModelScope）
- 支持4-bit/8-bit量化，显著降低显存需求
- 思考模式功能，提升复杂任务的处理能力
- 中英文双语支持，代码和界面均为双语设计


### 安装

1. 打开ComfyUI的custom_nodes目录
2. 克隆此仓库：
   ```bash
   git clone https://github.com/SXQBW/ComfyUI-Qwen3.git
   ```
3. 安装依赖：
   ```bash
   cd ComfyUI-Qwen3
   pip install -r requirements.txt
   ```
4. 重启ComfyUI


## 使用方法

### 多轮对话节点 (Qwen Conversation)

此节点支持完整的多轮对话，维护对话历史，适合文生图如Flux提示词扩展完善，聊天机器人等应用：

![alt text](pic/demo1_screenshot-20250526-091723.png)
对Flux提示词扩展完善生图效果1
![alt text](pic/ComfyUI_00030_.png)
对Flux提示词扩展完善生图效果2
![alt text](pic/ComfyUI_00080_.png)

1. 选择模型版本（默认推荐Qwen3-7B）
2. 选择量化级别（4-bit适合低显存设备，8-bit平衡精度和性能）
3. 启用或禁用"思考模式"（适合复杂推理任务）
4. 输入您的提示文本
5. 调整最大生成长度
6. 选择是否在生成后卸载模型以释放资源

### 单轮生成节点 (Qwen Generation)

此节点专注于单次文本生成，适合文生图如Flux提示词优化、文本扩展等任务：
![alt text](pic/demo-screenshot-20250523-124950.png)
![alt text](pic/demo3_screenshot-20250526-100723.png)

1. 选择模型版本
2. 选择量化级别
3. 启用或禁用"思考模式"
4. 输入您的提示文本
5. 调整最大生成长度
6. 选择是否在生成后卸载模型以释放资源

## 技术细节

### 内存管理

组件会自动检测您的设备（GPU/CPU/MPS）并选择最佳运行配置：

- NVIDIA GPU用户：自动使用CUDA并根据显存大小调整量化级别
- Apple Silicon用户：自动使用MPS加速
- 低内存设备：自动降级到4-bit量化以节省资源

### 模型下载

组件会自动测试Hugging Face和ModelScope的下载速度，选择最快的源进行下载。如果下载失败，会自动尝试另一个源，最多重试3次。

### 思考模式

"思考模式"通过特殊的标签机制实现，在生成过程中会在输出中添加思考过程标记：

- 启用时：会显示完整的思考过程
- 禁用时：会自动过滤思考内容，只保留最终结果

## 支持的模型

当前支持以下Qwen模型版本：

- Qwen3-0.6B-FP8
- Qwen3-0.6B-Base
- Qwen3-0.6B
- Qwen3-1.7B-FP8
- Qwen3-1.7B-Base
- Qwen3-1.7B
- Qwen3-4B-FP8
- Qwen3-4B-Base
- Qwen3-4B
- Qwen3-8B-FP8
- Qwen3-8B-Base
- Qwen3-8B
- Qwen3-14B-FP8
- Qwen3-14B-AWQ
- Qwen3-14B-Base
- Qwen3-14B
- Qwen3-14B-GGUF
- Qwen3-30B-A3B-FP8
- Qwen3-30B-A3B
- Qwen3-30B-A3B-Base
- Qwen3-32B-FP8
- Qwen3-32B-AWQ
- Qwen3-32B
- Qwen3-32B-GGUF
- Qwen3-235B-A22B-FP8
- Qwen3-235B-A22B


## 贡献

我们欢迎社区贡献！如果您发现问题或有改进建议，请提交issue或pull request。
