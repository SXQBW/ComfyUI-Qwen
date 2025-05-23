from __future__ import annotations
import torch
import os
import json
import re
import time
import requests
from transformers import pipeline, TextStreamer, TextIteratorStreamer, BitsAndBytesConfig
from threading import Thread
from huggingface_hub import snapshot_download
from modelscope.hub.snapshot_download import snapshot_download as modelscope_snapshot_download
import folder_paths
from pathlib import Path

# 双语分隔符
BILINGUAL_SEPARATOR = " | "

def replace_thinking_tags(text):
    """将文本中的<think>和</think>标签替换为双语描述
    Replace <think> and </think> tags in the text with bilingual descriptions"""

    # 确保输入是字符串
    if not isinstance(text, str):
        if isinstance(text, list):
            text = " ".join(map(str, text))
        else:
            text = str(text)

    # 替换开始标签，使用美观的分隔符
    text = re.sub(r'<think>', 
                  f'🌐 深度思考启动 | Deep Thinking Start ------------------------------', 
                  text)
    # 替换结束标签
    text = re.sub(r'</think>', 
                  f'✅ 深度思考完成 | Deep Thinking Complete ----------------------------', 
                  text)
    return text

def remove_thinking_tags(text):
    """移除文本中的思考内容标签和内容
    Remove thinking tags and their content from the text"""
    # 简单的正则表达式匹配
    clean_text = re.sub(r'🌐 深度思考启动.*?✅ 深度思考完成', 
                        '', text, flags=re.DOTALL)
    
    # 额外处理可能的变体，确保所有思考标签都被移除
    clean_text = re.sub(r'\| Deep Thinking Complete\s*----------------------------', 
                        '', clean_text)

    return clean_text.strip()

def markdown_to_plaintext(text):
    """将Markdown文本转换为纯文本
    Convert Markdown text to plain text"""
    # 移除标题标记
    text = re.sub(r'^(#+)\s*(.*)$', r'\2', text, flags=re.MULTILINE)
    
    # 移除粗体/斜体标记
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    
    # 移除列表标记
    text = re.sub(r'^[\s]*[-*+]\s+(.*)$', r'\1', text, flags=re.MULTILINE)
    
    # 移除链接标记
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1', text)
    
    return text

# 模型注册表JSON文件路径 - 保持在原目录，不移动到Qwen/Qwen目录
MODEL_REGISTRY_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen_model_registry.json")

def load_model_registry():
    """从JSON文件加载模型注册表
    Load model registry from JSON file"""
    try:
        if os.path.exists(MODEL_REGISTRY_JSON):
            with open(MODEL_REGISTRY_JSON, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 如果文件不存在，创建一个空的注册表
            print(f"模型注册表文件 {MODEL_REGISTRY_JSON} 不存在，将使用空注册表 | "
                  f"Model registry file {MODEL_REGISTRY_JSON} does not exist, using empty registry")
            return {}
    except json.JSONDecodeError as e:
        print(f"错误: 解析模型注册表JSON文件时出错: {e} | "
              f"Error: Failed to parse model registry JSON file: {e}")
        return {}

# 加载模型注册表
MODEL_REGISTRY = load_model_registry()

def get_gpu_info():
    """获取GPU信息，包括显存使用情况
    Get GPU information, including memory usage"""
    try:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            total_memory = props.total_memory / 1024**3  # GB
            allocated_memory = torch.cuda.memory_allocated(device) / 1024**3  # GB
            free_memory = total_memory - allocated_memory
            
            return {
                "available": True,
                "count": gpu_count,
                "name": props.name,
                "total_memory": total_memory,
                "allocated_memory": allocated_memory,
                "free_memory": free_memory
            }
        else:
            return {
                "available": False,
                "count": 0,
                "name": "None",
                "total_memory": 0,
                "allocated_memory": 0,
                "free_memory": 0
            }
    except Exception as e:
        print(f"获取GPU信息时出错: {e} | "
              f"Error getting GPU information: {e}")
        return {
            "available": False,
            "count": 0,
            "name": "None",
            "total_memory": 0,
            "allocated_memory": 0,
            "free_memory": 0
        }

def get_system_memory_info():
    """获取系统内存信息，包括总内存和可用内存
    Get system memory information, including total and available memory"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            "total": mem.total / 1024**3,  # GB
            "available": mem.available / 1024**3,  # GB
            "used": mem.used / 1024**3,  # GB
            "percent": mem.percent
        }
    except ImportError:
        print("警告: 无法导入psutil库，系统内存检测功能将不可用 | "
              "Warning: Failed to import psutil library, system memory detection disabled")
        return {
            "total": 0,
            "available": 0,
            "used": 0,
            "percent": 0
        }

def get_device_info():
    """获取设备信息，包括GPU和CPU，并分析最佳运行设备
    Get device information, including GPU and CPU, and analyze optimal running device"""
    device_info = {
        "device_type": "unknown",
        "gpu": get_gpu_info(),
        "system_memory": get_system_memory_info(),
        "recommended_device": "cpu",  # 默认推荐CPU | Default to CPU
        "memory_sufficient": True,
        "warning_message": None
    }
    
    # 检查是否为Apple Silicon
    try:
        import platform
        if platform.system() == "Darwin" and platform.processor() == "arm":
            device_info["device_type"] = "apple_silicon"
            # M1/M2芯片有统一内存，检查总内存是否充足
            if device_info["system_memory"]["total"] >= 16:  # 至少16GB内存 | At least 16GB RAM
                device_info["recommended_device"] = "mps"
            else:
                device_info["memory_sufficient"] = False
                device_info["warning_message"] = "Apple Silicon芯片内存不足，建议使用至少16GB内存的设备 | Insufficient memory on Apple Silicon, recommend at least 16GB RAM"

            return device_info
    except:
        pass
    
    # 检查是否有NVIDIA GPU
    if device_info["gpu"]["available"]:
        device_info["device_type"] = "nvidia_gpu"
        # 检查GPU内存是否充足
        if device_info["gpu"]["total_memory"] >= 8:  # 至少8GB显存 | At least 8GB VRAM
            device_info["recommended_device"] = "cuda"
        else:
            # 显存不足，但仍可使用，只是性能会受影响
            device_info["memory_sufficient"] = False
            device_info["warning_message"] = "NVIDIA GPU显存不足，可能会使用系统内存，性能会下降 | Insufficient NVIDIA GPU memory, may use system RAM with reduced performance"
            device_info["recommended_device"] = "cuda"  # 仍推荐使用GPU，但会启用内存优化 | Still recommend GPU with memory optimization
        return device_info
    
    # 检查是否有AMD GPU (ROCm)
    try:
        import torch
        if hasattr(torch, 'device') and torch.device('cuda' if torch.cuda.is_available() else 'cpu').type == 'cuda':
            device_info["device_type"] = "amd_gpu"
            # AMD GPU内存检查
            if device_info["gpu"]["total_memory"] >= 8:
                device_info["recommended_device"] = "cuda"
            else:
                device_info["memory_sufficient"] = False
                device_info["warning_message"] = "AMD GPU显存不足，可能会使用系统内存，性能会下降 | Insufficient AMD GPU memory, may use system RAM with reduced performance"

                device_info["recommended_device"] = "cuda"
            return device_info
    except:
        pass
    
    # 默认为CPU
    device_info["device_type"] = "cpu"
    # 检查系统内存是否充足
    if device_info["system_memory"]["total"] < 8:
        device_info["memory_sufficient"] = False
        device_info["warning_message"] = "系统内存不足，模型运行可能会非常缓慢 | Insufficient system memory, model may run very slowly"
    return device_info

def calculate_required_memory(model_name, quantization, use_cpu=False, use_mps=False):
    """根据模型名称、量化方式和设备类型计算所需内存
    Calculate required memory based on model name, quantization, and device type"""
    model_info = MODEL_REGISTRY.get(model_name, {})
    vram_config = model_info.get("vram_requirement", {})
    
    # 检查模型是否已经量化
    is_quantized_model = model_info.get("quantized", False)
    
    # 基础内存需求计算
    if is_quantized_model:
        base_memory = vram_config.get("full", 0)
    else:
        if quantization == "👍 4-bit (VRAM-friendly)":
            base_memory = vram_config.get("4bit", 0)
        elif quantization == "⚖️ 8-bit (Balanced Precision)":
            base_memory = vram_config.get("8bit", 0)
        else:
            base_memory = vram_config.get("full", 0)
    
    # 调整内存需求（CPU和MPS通常需要更多内存）
    if use_cpu or use_mps:
        # CPU和MPS通常需要更多内存用于内存交换
        memory_factor = 1.5 if use_cpu else 1.2
        return base_memory * memory_factor
    
    return base_memory

def check_flash_attention():
    """检测Flash Attention 2支持（需Ampere架构及以上）
    Check for Flash Attention 2 support (requires Ampere architecture or higher)"""
    try:
        from flash_attn import flash_attn_func
        major, _ = torch.cuda.get_device_capability()
        return major >= 8  # 仅支持计算能力8.0+的GPU | Only support GPUs with compute capability 8.0+
    except ImportError:
        return False

FLASH_ATTENTION_AVAILABLE = check_flash_attention()

def init_qwen_paths(model_name):
    """初始化模型路径，支持动态生成不同模型版本的路径
    Initialize model paths, supporting dynamic generation of paths for different model versions"""
    base_dir = Path(folder_paths.models_dir).resolve()
    qwen_dir = base_dir / "Qwen" / "Qwen"  # 添加Qwen子目录 | Add Qwen subdirectory
    model_dir = qwen_dir / model_name  # 使用模型名称作为子目录 | Use model name as subdirectory
    
    # 创建目录
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 注册到ComfyUI
    if hasattr(folder_paths, "add_model_folder_path"):
        folder_paths.add_model_folder_path("Qwen", str(qwen_dir))
    else:
        folder_paths.folder_names_and_paths["Qwen"] = ([str(qwen_dir)], {'.safetensors', '.bin', '.gguf' })
    
    print(f"模型路径已初始化: {model_dir} | "
          f"Model path initialized: {model_dir}")
    return str(model_dir)  # 修改：返回模型目录路径，而不是父目录 | Return model directory path instead of parent directory

def test_download_speed(url):
    """测试下载速度，下载 5 秒
    Test download speed by downloading for 5 seconds"""
    try:
        start_time = time.time()
        response = requests.get(url, stream=True, timeout=10)
        downloaded_size = 0
        for data in response.iter_content(chunk_size=1024):
            if time.time() - start_time > 5:
                break
            downloaded_size += len(data)
        end_time = time.time()
        speed = downloaded_size / (end_time - start_time) / 1024  # KB/s
        return speed
    except Exception as e:
        print(f"测试下载速度时出现错误: {e} | "
              f"Error testing download speed: {e}")
        return 0

def validate_model_path(model_path, model_name):
    """验证模型路径的有效性和模型文件是否齐全
    Validate the validity of the model path and check if model files are complete"""
    path_obj = Path(model_path)
    
    # 基本路径检查
    if not path_obj.is_absolute():
        print(f"错误: {model_path} 不是绝对路径 | "
              f"Error: {model_path} is not an absolute path")
        return False
    
    if not path_obj.exists():
        print(f"模型目录不存在: {model_path} | "
              f"Model directory does not exist: {model_path}")
        return False
    
    if not path_obj.is_dir():
        print(f"错误: {model_path} 不是目录 | "
              f"Error: {model_path} is not a directory")
        return False
    
    # 检查模型文件是否齐全
    if not check_model_files_exist(model_path, model_name):
        print(f"模型文件不完整: {model_path} | "
              f"Model files incomplete: {model_path}")
        return False
    
    return True

def check_model_files_exist(model_dir, model_name):
    """检查特定模型版本所需的文件是否齐全
    Check if required files for a specific model version are complete"""
    if model_name not in MODEL_REGISTRY:
        print(f"错误: 未知模型版本 {model_name} | "
              f"Error: Unknown model version {model_name}")
        return False
    
    required_files = MODEL_REGISTRY[model_name]["required_files"]
    for file in required_files:
        if not os.path.exists(os.path.join(model_dir, file)):
            return False
    return True

class QwenTextProcessor:
    def __init__(self):
        # 默认使用注册表中的第一个默认模型
        default_model = next((name for name, info in MODEL_REGISTRY.items() if info.get("default", False)), 
                            list(MODEL_REGISTRY.keys())[0])       

        # 重置环境变量，避免干扰
        os.environ.pop("HUGGINGFACE_HUB_CACHE", None)   
        self.current_model_name = default_model
        self.current_quantization = None  # 记录当前的量化配置
        self.model_path = init_qwen_paths(self.current_model_name)
        self.cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        print(f"模型路径: {self.model_path} | "
              f"Model path: {self.model_path}")
        print(f"缓存路径: {self.cache_dir} | "
              f"Cache path: {self.cache_dir}")

        # 验证并创建缓存目录
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        self.model = None
        self.tokenizer = None
        # 性能统计
        self.generation_stats = {"count": 0, "total_time": 0}        
     
        # 初始化设备信息
        self.device_info = get_device_info()
        self.default_device = self.device_info["recommended_device"]
        
        print(f"检测到的设备: {self.device_info['device_type']} | "
              f"Detected device: {self.device_info['device_type']}")
        print(f"自动选择的运行设备: {self.default_device} | "
              f"Automatically selected device: {self.default_device}")
        
        if not self.device_info["memory_sufficient"]:
            print(f"警告: {self.device_info['warning_message']}")
        
        # 初始化内存优化选项
        self.optimize_for_low_memory = not self.device_info["memory_sufficient"]
        
    def clear_model_resources(self):
        """释放当前模型占用的资源
        Release resources occupied by the current model"""
        if self.model is not None:
            print("释放当前模型占用的资源... | "
                  "Releasing resources occupied by current model...")
            del self.model, self.tokenizer
            self.model = None
            self.tokenizer = None
            torch.cuda.empty_cache()  # 清理GPU缓存 | Clear GPU cache

    def check_memory_requirements(self, model_name, quantization):
        """检查当前设备内存是否满足模型要求，必要时调整量化级别
        Check if current device memory meets model requirements, adjust quantization level if necessary"""
        # 使用自动选择的设备
        device = self.default_device
        use_cpu = device == "cpu"
        use_mps = device == "mps"
        
        # 计算所需内存
        required_memory = calculate_required_memory(model_name, quantization, use_cpu, use_mps)
        
        if use_cpu or use_mps:
            # 检查系统内存
            available_memory = self.device_info["system_memory"]["available"]
            memory_type = "系统内存 | System memory"
        else:
            # 检查GPU内存
            available_memory = self.device_info["gpu"]["free_memory"]
            memory_type = "GPU显存 | GPU memory"
        
        # 添加20%的安全余量
        safety_margin = 1.2
        required_memory_with_margin = required_memory * safety_margin
        
        print(f"模型 {model_name} (量化: {quantization}) 需要 {required_memory:.2f} GB {memory_type} | "
              f"Model {model_name} (quantization: {quantization}) requires {required_memory:.2f} GB {memory_type}")
        print(f"考虑安全余量后，需要 {required_memory_with_margin:.2f} GB {memory_type} | "
              f"With safety margin, requires {required_memory_with_margin:.2f} GB {memory_type}")
        print(f"当前可用 {memory_type}: {available_memory:.2f} GB | "
              f"Current available {memory_type}: {available_memory:.2f} GB")
        
        # 如果内存不足，自动调整量化级别
        if required_memory_with_margin > available_memory:
            print(f"警告: 所选量化级别需要的{memory_type}超过可用内存，自动调整量化级别 | "
                  f"Warning: Selected quantization level requires more {memory_type} than available, automatically adjusting")
            
            # 降级策略
            if quantization == "🚫 None (Original Precision)":
                print("将量化级别从'无量化'调整为'8-bit' | "
                      "Adjusting quantization from 'None' to '8-bit'")
                return "⚖️ 8-bit (Balanced Precision)"
            elif quantization == "⚖️ 8-bit (Balanced Precision)":
                print("将量化级别从'8-bit'调整为'4-bit' | "
                      "Adjusting quantization from '8-bit' to '4-bit'")
                return "👍 4-bit (VRAM-friendly)"
            else:
                # 已经是4-bit，无法再降级
                print(f"错误: 即使使用4-bit量化，模型仍然需要更多{memory_type} | "
                      f"Error: Even with 4-bit quantization, model requires more {memory_type}")
                raise RuntimeError(f"错误: 可用{memory_type}不足，需要至少 {required_memory_with_margin:.2f} GB，但只有 {available_memory:.2f} GB | "
                                   f"Error: Insufficient {memory_type}, requires at least {required_memory_with_margin:.2f} GB, but only {available_memory:.2f} GB available")
        
        return quantization

    def load_model(self, model_name, quantization, enable_thinking=True):
        """加载指定模型和量化配置，支持思考模式
        Load specified model and quantization configuration, supporting thinking mode"""
        # 检查内存需求并可能调整量化级别
        adjusted_quantization = self.check_memory_requirements(model_name, quantization)
        
        # 使用自动选择的设备
        device = self.default_device
        print(f"使用设备: {device} | "
              f"Using device: {device}")

        # 检查是否需要重新加载模型
        if (self.model is not None and 
            self.current_model_name == model_name and 
            self.current_quantization == adjusted_quantization):
            print(f"使用已加载的模型: {model_name}，量化: {adjusted_quantization} | "
                  f"Using already loaded model: {model_name}, quantization: {adjusted_quantization}")
            return
        
        # 需要重新加载，先释放现有资源
        self.clear_model_resources()
        
        # 更新当前模型名称和量化配置
        self.current_model_name = model_name
        self.model_path = init_qwen_paths(self.current_model_name)  # 修改：获取模型目录路径
        self.current_quantization = adjusted_quantization

        # 检查模型文件是否存在且完整
        if not validate_model_path(self.model_path, self.current_model_name):
            print(f"检测到模型文件缺失，正在为你下载 {model_name} 模型，请稍候... | "
                  f"Model files detected missing, downloading {model_name} model, please wait...")
            print(f"下载将保存在: {self.model_path} | "
                  f"Download will be saved to: {self.model_path}")
            
            # 开始下载逻辑
            try:
                # 从注册表获取模型信息
                model_info = MODEL_REGISTRY[model_name]
                
                # 测试下载速度
                huggingface_test_url = f"https://huggingface.co/{model_info['repo_id']['huggingface']}/resolve/main/{model_info['test_file']}"
                modelscope_test_url = f"https://modelscope.cn/api/v1/models/{model_info['repo_id']['modelscope']}/repo?Revision=master&FilePath={model_info['test_file']}"
                huggingface_speed = test_download_speed(huggingface_test_url)
                modelscope_speed = test_download_speed(modelscope_test_url)

                print(f"Hugging Face下载速度: {huggingface_speed:.2f} KB/s | "
                      f"Hugging Face download speed: {huggingface_speed:.2f} KB/s")
                print(f"ModelScope下载速度: {modelscope_speed:.2f} KB/s | "
                      f"ModelScope download speed: {modelscope_speed:.2f} KB/s")

                # 根据下载速度选择优先下载源
                if huggingface_speed > modelscope_speed * 1.5:
                    download_sources = [
                        (snapshot_download, model_info['repo_id']['huggingface'], "Hugging Face"),
                        (modelscope_snapshot_download, model_info['repo_id']['modelscope'], "ModelScope")
                    ]
                    print("基于下载速度分析，优先尝试从Hugging Face下载 | "
                          "Based on download speed analysis, attempting download from Hugging Face first")
                else:
                    download_sources = [
                        (modelscope_snapshot_download, model_info['repo_id']['modelscope'], "ModelScope"),
                        (snapshot_download, model_info['repo_id']['huggingface'], "Hugging Face")
                    ]
                    print("基于下载速度分析，优先尝试从ModelScope下载 | "
                          "Based on download speed analysis, attempting download from ModelScope first")

                max_retries = 3
                success = False
                final_error = None
                used_cache_path = None

                for download_func, repo_id, source in download_sources:
                    for retry in range(max_retries):
                        try:
                            print(f"开始从 {source} 下载模型（第 {retry + 1} 次尝试）... | "
                                  f"Starting model download from {source} (attempt {retry + 1})...")
                            if download_func == snapshot_download:
                                cached_path = download_func(
                                    repo_id,
                                    cache_dir=self.cache_dir,
                                    ignore_patterns=["*.msgpack", "*.h5"],
                                    resume_download=True,
                                    local_files_only=False
                                )
                            else:
                                cached_path = download_func(
                                    repo_id,
                                    cache_dir=self.cache_dir,
                                    revision="master"
                                )

                            used_cache_path = cached_path  # 记录使用的缓存路径
                            
                            # 将下载的模型复制到模型目录
                            self.copy_cached_model_to_local(cached_path, self.model_path)
                            
                            print(f"成功从 {source} 下载模型到 {self.model_path} | "
                                  f"Successfully downloaded model from {source} to {self.model_path}")
                            success = True
                            break

                        except Exception as e:
                            final_error = e  # 保存最后一个错误
                            if retry < max_retries - 1:
                                print(f"从 {source} 下载模型失败（第 {retry + 1} 次尝试）: {e}，即将进行下一次尝试... | "
                                      f"Failed to download model from {source} (attempt {retry + 1}): {e}, trying again...")
                            else:
                                print(f"从 {source} 下载模型失败（第 {retry + 1} 次尝试）: {e}，尝试其他源... | "
                                      f"Failed to download model from {source} (attempt {retry + 1}): {e}, trying next source...")
                    if success:
                        break
                else:
                    raise RuntimeError("从所有源下载模型均失败。 | "
                                      "Failed to download model from all sources.")
                
                # 下载完成后再次验证
                if not validate_model_path(self.model_path, self.current_model_name):
                    raise RuntimeError(f"下载后模型文件仍不完整: {self.model_path} | "
                                      f"Model files still incomplete after download: {self.model_path}")
                
                print(f"模型 {model_name} 已准备就绪 | "
                      f"Model {model_name} is ready")
                
            except Exception as e:
                print(f"下载模型时发生错误: {e} | "
                      f"Error downloading model: {e}")
                
                # 下载失败提示
                if used_cache_path:
                    print("\n⚠️ 注意：下载过程中创建了缓存文件 | "
                          "\n⚠️ Attention: Cache files were created during download")
                    print(f"缓存路径: {used_cache_path} | "
                          f"Cache path: {used_cache_path}")
                    print("你可以前往此路径删除缓存文件以释放硬盘空间 | "
                          "You can delete these files to free up disk space")
                
                raise RuntimeError(f"无法下载模型 {model_name}，请手动下载并放置到 {self.model_path} | "
                                  f"Unable to download model {model_name}, please download manually and place in {self.model_path}")

        # 模型文件完整，正常加载
        print(f"加载模型: {self.model_path}，量化: {quantization} | "
              f"Loading model: {self.model_path}, quantization: {quantization}")

        # 检查模型是否已经量化
        is_quantized_model = MODEL_REGISTRY.get(model_name, {}).get("quantized", False)

        # 处理 FP8 跨 GPU 问题
        if "FP8" in model_name:
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        print(f"加载模型: {model_name}，量化: {adjusted_quantization}，思考模式: {'启用' if enable_thinking else '禁用'} | "
              f"Loading model: {model_name}, quantization: {adjusted_quantization}, thinking mode: {'enabled' if enable_thinking else 'disabled'}")

        # 配置量化参数
        if is_quantized_model:
            print(f"模型 {model_name} 已经是量化模型，将忽略用户的量化设置 | "
                  f"Model {model_name} is already quantized, ignoring user quantization settings")
            # 对于已经量化的模型，使用原始精度加载
            load_dtype = torch.float16
            quant_config = None
        else:
            # 配置量化参数
            if adjusted_quantization == "👍 4-bit (VRAM-friendly)":
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                load_dtype = torch.float16  # 让量化配置决定数据类型
            elif adjusted_quantization == "⚖️ 8-bit (Balanced Precision)":
                quant_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                load_dtype = torch.float16  # 让量化配置决定数据类型
            else:
                # 不使用量化，使用原始精度
                load_dtype = torch.float16
                quant_config = None

        # 配置device_map
        if device == "cuda":
            if torch.cuda.device_count() > 0:
                device_map = {"": 0}  # 使用第一个GPU
                print(f"使用GPU: {torch.cuda.get_device_name(0)} | "
                      f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                device_map = "auto"
                print("未检测到可用GPU，将尝试使用auto设备映射 | "
                      "No GPU detected, attempting to use auto device mapping")
        elif device == "mps":
            device_map = "auto"  # MPS不支持device_map，加载后需手动移到设备
        else:
            device_map = "auto"  # CPU加载

        # 准备加载参数
        load_kwargs = {
            "device_map": device_map,
            "torch_dtype": load_dtype,

        }

        # 如果有量化配置，添加到加载参数中
        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config

        # 创建文本生成pipeline - 使用self.model_path而不是model_name
        self.model = pipeline(
            "text-generation",
            model=self.model_path,  # 使用本地路径而不是模型名称
            **load_kwargs
        )
        
        # 获取tokenizer
        self.tokenizer = self.model.tokenizer

    def copy_cached_model_to_local(self, cached_path, target_path):
        """将缓存的模型文件复制到目标路径
        Copy cached model files to target path"""
        print(f"正在将模型从缓存复制到: {target_path} | "
              f"Copying model from cache to: {target_path}")
        target_path = Path(target_path)
        target_path.mkdir(parents=True, exist_ok=True)
        
        # 使用shutil进行递归复制
        import shutil
        for item in Path(cached_path).iterdir():
            if item.is_dir():
                shutil.copytree(item, target_path / item.name, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target_path / item.name)
        
        # 验证复制是否成功
        if validate_model_path(target_path, self.current_model_name):
            print(f"模型已成功复制到 {target_path} | "
                  f"Model successfully copied to {target_path}")
        else:
            raise RuntimeError(f"复制后模型文件仍不完整: {target_path} | "
                              f"Model files still incomplete after copying: {target_path}")

    @torch.no_grad()
    def generate_text(self, model_name, quantization, prompt, max_tokens,  
                    messages=None, enable_thinking=False, unload_after_generation=True):
        """生成文本，支持多轮对话、流式输出和思考模式
        Generate text, supporting multi-turn conversations, streaming output, and thinking mode
        
        Args:
            unload_after_generation: 生成后是否卸载模型以释放资源
        """
        start_time = time.time()
        
        # 只在必要时加载模型（模型名称或量化方式改变时）
        if (self.model is None or 
            self.current_model_name != model_name or 
            self.current_quantization != quantization):
            self.load_model(model_name, quantization)  # 注意这里不再传递enable_thinking
        
        # 构建输入
        if messages is not None:
            # 多轮对话模式
            input_data = messages
        else:
            # 单轮生成模式
            input_data = [{"role": "user", "content": prompt}]
        
        # 根据思考模式设置不同的采样参数
        if enable_thinking:
            # 思考模式推荐参数
            temperature = 0.6 
            top_p = 0.95 
        else:
            # 非思考模式推荐参数
            temperature = 0.7 
            top_p = 0.8 
        
        # 准备生成参数
        generate_kwargs = {
            "max_new_tokens": max(max_tokens, 10),
            "temperature": temperature,
            "do_sample": True,
            "top_p": top_p,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        
        # 应用思考模式（使用官方方法）
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                # 使用官方提供的模板方法，明确控制思考模式
                input_text = self.tokenizer.apply_chat_template(
                    input_data,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking  # 直接控制模型是否启用思考模式
                )
                
                # 确保input_text是字符串类型
                if isinstance(input_text, list):
                    input_text = " ".join(map(str, input_text))
                    
                input_data = [{"role": "user", "content": input_text}]
            except Exception as e:
                print(f"应用聊天模板时出错: {e}")
                # 如果出错，回退到原始输入
                input_data = [{"role": "user", "content": prompt}]
        else:
            # 如果没有apply_chat_template方法，使用原始输入
            input_data = [{"role": "user", "content": prompt}]
        
        # 记录GPU内存使用情况
        if torch.cuda.is_available():
            pre_forward_memory = torch.cuda.memory_allocated() / 1024**2
            print(f"生成前GPU内存使用: {pre_forward_memory:.2f} MB | "
                f"GPU memory usage before generation: {pre_forward_memory:.2f} MB")
        
        # 非流式输出
        invalid_kwargs = ['low_cpu_mem_usage', 'use_safetensors']
        for key in invalid_kwargs:
            if key in generate_kwargs:
                del generate_kwargs[key]
        
        # 确保输入数据格式正确
        if isinstance(input_data, list) and all(isinstance(item, dict) for item in input_data):
            result = self.model(input_data, **generate_kwargs)
        else:
            # 如果输入格式不正确，使用字符串形式
            input_str = input_data[0]["content"] if isinstance(input_data, list) else str(input_data)
            result = self.model(input_str, **generate_kwargs)
        
        generated_text = result[0]["generated_text"]

        # 记录GPU内存使用情况
        if torch.cuda.is_available():
            post_forward_memory = torch.cuda.memory_allocated() / 1024**2
            print(f"生成后GPU内存使用: {post_forward_memory:.2f} MB | "
                f"GPU memory usage after generation: {post_forward_memory:.2f} MB")
            print(f"生成过程中GPU内存增加: {post_forward_memory - pre_forward_memory:.2f} MB | "
                f"GPU memory increase during generation: {post_forward_memory - pre_forward_memory:.2f} MB")
        
        # 计算处理时间
        process_time = time.time() - start_time
        self.generation_stats["count"] += 1
        self.generation_stats["total_time"] += process_time
        
        # 打印性能统计
        print(f"生成完成，耗时: {process_time:.2f} 秒 | "
            f"Generation completed, time taken: {process_time:.2f} seconds")
        if self.generation_stats["count"] > 0:
            avg_time = self.generation_stats["total_time"] / self.generation_stats["count"]
            print(f"平均生成时间: {avg_time:.2f} 秒/次 | "
                f"Average generation time: {avg_time:.2f} seconds per generation")
        
        # 如果启用了卸载选项，释放模型资源
        if unload_after_generation:
            self.clear_model_resources()
            print("模型已卸载以释放资源。下次使用时将重新加载。 | "
                  "Model unloaded to free up resources. Will reload on next use.")
        
        # 直接返回生成的内容，不再手动过滤思考内容
        return generated_text


class QwenMultiTurnConversation:
    # 类级别变量，用于缓存处理器实例
    processor = None

    def __init__(self):
        # 初始化对话历史
        self.chat_history = []
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                   list(MODEL_REGISTRY.keys()),  # 动态生成模型选项 | Dynamically generate model options
                    {
                        "default": next((name for name, info in MODEL_REGISTRY.items() if info.get("default", False)), 
                                       list(MODEL_REGISTRY.keys())[0]),
                        "tooltip": "选择可用的模型版本。 | Select the available model version."
                    }
                ),
                "quantization": (
                    [
                        "👍 4-bit (VRAM-friendly)",
                        "⚖️ 8-bit (Balanced Precision)",
                        "🚫 None (Original Precision)"
                    ],
                    {
                        "default": "👍 4-bit (VRAM-friendly)",
                        "tooltip": "选择量化级别:\n✅ 4-bit: 显著减少显存使用。\n⚖️ 8-bit: 平衡精度和性能。\n🚫 None: 使用原始精度（需要高端GPU）。 \n "
                                   "Select the quantization level:\n✅ 4-bit: Significantly reduces VRAM usage.\n⚖️ 8-bit: Balances precision and performance.\n🚫 None: Uses original precision (requires high-end GPU)."
                    }
                ),
                "enable_thinking": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "启用或禁用思考模式。思考模式适用于复杂推理任务，非思考模式适用于高效对话。 | "
                                   "Enable or disable thinking mode. Thinking mode is suitable for complex reasoning tasks, while non-thinking mode is optimized for efficient conversations."
                    }
                ),
                "prompt": ("STRING", {
                    "default": "This is the prompt text used for generating images with Fulx: \"In the style of GHIBSKY, a cyberpunk panda holding a neon sign that reads: 'Designed by SXQBW'\". Please optimize, supplement and improve the prompt text according to its content, and make the generated image effect the best.",
                    "multiline": True,
                    "tooltip": "输入提示文本 | Enter the prompt text"
                }),
                "max_new_tokens": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 2048,
                    "step": 16,
                    "display": "slider",
                    "tooltip": "控制生成的最大token数 | Control the maximum number of tokens to generate"
                }),
                "clear_history": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "是否清除对话历史 | Whether to clear the conversation history"
                }),
                "unload_after_generation": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "生成后是否卸载模型以释放资源。启用此选项可减少内存占用，但会增加下次使用时的加载时间。 | "
                               "Whether to unload the model after generation to free up resources. Enabling this option reduces memory usage but increases load time for subsequent uses."
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "process"
    CATEGORY = "🐼Qwen"

    def process(self, model_name, quantization, enable_thinking, prompt, max_new_tokens, clear_history, unload_after_generation):
        # 检查是否需要清除历史
        if clear_history:
            self.chat_history = []
            print("对话历史已清除 | Conversation history cleared")
            
        # 使用类级别的处理器实例，避免重复加载模型
        if QwenMultiTurnConversation.processor is None:
            QwenMultiTurnConversation.processor = QwenTextProcessor()

        # 确保处理器已加载模型
        if QwenMultiTurnConversation.processor.model is None:
            QwenMultiTurnConversation.processor.load_model(model_name, quantization, enable_thinking)   
        
        # 添加用户输入到对话历史
        self.chat_history.append({"role": "user", "content": prompt})
        
        # 生成回复，传递完整的对话历史
        generated_text = QwenMultiTurnConversation.processor.generate_text(
            model_name=model_name,
            quantization=quantization,
            prompt=prompt,
            max_tokens=max_new_tokens,
            messages=self.chat_history,
            enable_thinking=enable_thinking,
            unload_after_generation=unload_after_generation
        )
        
        if isinstance(generated_text, list):
            # 如果返回的是消息列表，提取内容
            generated_text = "".join([msg.get("content", "") for msg in generated_text if msg.get("role") == "assistant"])
        
        # 替换思考标签
        generated_text = replace_thinking_tags(generated_text)

        if not enable_thinking:
            generated_text = remove_thinking_tags(generated_text)
        
        # 将Markdown转换为纯文本
        plain_text = markdown_to_plaintext(generated_text)
        
        # 添加AI回复到对话历史
        self.chat_history.append({"role": "assistant", "content": plain_text})
        
        # 格式化并返回完整的对话历史
        formatted_history = self.format_chat_history()
        return (formatted_history,)
    
    def format_chat_history(self):
        """格式化对话历史以便显示"""
        formatted_history = []
        for message in self.chat_history:
            role = message["role"].upper()
            content = message["content"]
            
            # 确保content是字符串类型
            if isinstance(content, list):
                content = " ".join(map(str, content))
            elif not isinstance(content, str):
                content = str(content)
            
            # 添加分隔符和角色标识
            formatted_history.append(f"[{role}]")
            formatted_history.append(content)
            formatted_history.append("-" * 60)  # 分隔线
        
        return "\n".join(formatted_history)


class QwenSingleTurnGeneration:
    """Qwen3单轮生成节点 | Qwen3 Single-turn Generation Node"""
    # 类级别变量，用于缓存处理器实例
    processor = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                   list(MODEL_REGISTRY.keys()),  # 动态生成模型选项 | Dynamically generate model options
                    {
                        "default": next((name for name, info in MODEL_REGISTRY.items() if info.get("default", False)), 
                                       list(MODEL_REGISTRY.keys())[0]),
                        "tooltip": "选择可用的模型版本。 | Select the available model version."
                    }
                ),
                "quantization": (
                    [
                        "👍 4-bit (VRAM-friendly)",
                        "⚖️ 8-bit (Balanced Precision)",
                        "🚫 None (Original Precision)"
                    ],
                    {
                        "default": "👍 4-bit (VRAM-friendly)",
                        "tooltip": "选择量化级别:\n✅ 4-bit: 显著减少显存使用。\n⚖️ 8-bit: 平衡精度和性能。\n🚫 None: 使用原始精度（需要高端GPU）。 \n "
                                   "Select the quantization level:\n✅ 4-bit: Significantly reduces VRAM usage.\n⚖️ 8-bit: Balances precision and performance.\n🚫 None: Uses original precision (requires high-end GPU)."
                    }
                ),
                "enable_thinking": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "启用或禁用思考模式。思考模式适用于复杂推理任务，非思考模式适用于高效对话。 | "
                                   "Enable or disable thinking mode. Thinking mode is suitable for complex reasoning tasks, while non-thinking mode is optimized for efficient conversations."
                    }
                ),
                "prompt": ("STRING", {
                    "default": "This is the prompt text used for generating images with Fulx: \"In the style of GHIBSKY, a cyberpunk panda holding a neon sign that reads: 'Designed by SXQBW'\". Please optimize, supplement and improve the prompt text according to its content, and make the generated image effect the best.Just provide the best answer content.",
                    "multiline": True,
                    "tooltip": "输入提示文本 | Enter the prompt text"
                }),
                "max_new_tokens": ("INT", {
                    "default": 2048,
                    "min": 64,
                    "max": 6144,
                    "step": 32,
                    "display": "slider",
                    "tooltip": "控制生成的最大token数 | Control the maximum number of tokens to generate"
                }),
                "unload_after_generation": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "生成后是否卸载模型以释放资源。启用此选项可减少内存占用，但会增加下次使用时的加载时间。 | "
                               "Whether to unload the model after generation to free up resources. Enabling this option reduces memory usage but increases load time for subsequent uses."
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "process"
    CATEGORY = "🐼Qwen"

    def process(self, model_name, quantization, enable_thinking, prompt, max_new_tokens, unload_after_generation):
        # 使用类级别的处理器实例，避免重复加载模型
        if QwenSingleTurnGeneration.processor is None:
            QwenSingleTurnGeneration.processor = QwenTextProcessor()

        # 确保处理器已加载模型
        if QwenSingleTurnGeneration.processor.model is None:
            QwenSingleTurnGeneration.processor.load_model(model_name, quantization, enable_thinking)              
        
        # 生成回复
        generated_text = QwenSingleTurnGeneration.processor.generate_text(
            model_name=model_name,
            quantization=quantization,
            prompt=prompt,
            max_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
            unload_after_generation=unload_after_generation
        )        

        if isinstance(generated_text, list):
            # 如果返回的是消息列表，提取内容
            generated_text = "".join([msg.get("content", "") for msg in generated_text if msg.get("role") == "assistant"])
        
        # 替换思考标签
        generated_text = replace_thinking_tags(generated_text)

        if not enable_thinking:
            generated_text = remove_thinking_tags(generated_text)
        
        # 将Markdown转换为纯文本
        plain_text = markdown_to_plaintext(generated_text)
        
        return (plain_text,)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "QwenMultiTurnConversation": QwenMultiTurnConversation,
    "QwenSingleTurnGeneration": QwenSingleTurnGeneration,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenMultiTurnConversation": "Qwen Conversation 🐼",
    "QwenSingleTurnGeneration": "Qwen Generation 🐼",
}