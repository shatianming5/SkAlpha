"""
Unicode字符处理器
"""

import sys
import re


def replace_unicode_chars(text):
    """
    替换Unicode字符为ASCII安全的标记
    
    Args:
        text: 输入文本
        
    Returns:
        str: 处理后的文本
    """
    if not isinstance(text, str):
        text = str(text)
    
    # 简单处理：保留中文和常见字符，移除emoji等特殊字符
    try:
        # 移除emoji和其他特殊Unicode字符，但保留中文
        cleaned_text = re.sub(r'[^\u4e00-\u9fff\u0020-\u007E\u00A0-\u00FF]', '', text)
        return cleaned_text
    except Exception:
        # 如果处理失败，返回ASCII安全版本
        return text.encode('ascii', 'ignore').decode('ascii')


def safe_print(message, msg_type="INFO"):
    """
    安全打印函数，处理Unicode字符
    
    Args:
        message: 要打印的消息
        msg_type: 消息类型，用于添加前缀
    """
    try:
        # 替换emoji和特殊字符为ASCII安全的标记
        safe_message = replace_unicode_chars(str(message))
        
        # 根据消息类型添加前缀
        type_prefixes = {
            "START": "[START]",
            "STEP1": "[STEP1]", 
            "STEP2": "[STEP2]",
            "STEP3": "[STEP3]",
            "STEP4": "[STEP4]",
            "STEP5": "[STEP5]",
            "OK": "[OK]",
            "ERROR": "[ERROR]",
            "WARNING": "[WARNING]",
            "INFO": "[INFO]",
            "SUCCESS": "[SUCCESS]",
            "END": "[END]",
            "PARAMS": "[PARAMS]",
            "CONFIG": "[CONFIG]",
            "INIT": "[INIT]",
            "LOAD": "[LOAD]",
            "RUN": "[RUN]",
            "LOOP": "[LOOP]",
            "SUMMARY": "[SUMMARY]"
        }
        
        prefix = type_prefixes.get(msg_type, f"[{msg_type}]")
        
        # 为SUMMARY类型使用特殊格式
        if msg_type == "SUMMARY":
            if "=" in safe_message:
                # 分隔线保持原样
                formatted_message = safe_message
            elif safe_message.startswith(("📊", "💡", "🔧", "📈", "🎯", "💭", "⚠️")):
                # 主要标题，加粗显示
                formatted_message = f"{prefix} {safe_message}"
            elif safe_message.startswith("   "):
                # 缩进内容，保持缩进
                formatted_message = f"         {safe_message}"
            else:
                formatted_message = f"{prefix} {safe_message}"
        else:
            formatted_message = f"{prefix} {safe_message}"
        
        print(formatted_message)
        sys.stdout.flush()
        
    except Exception as e:
        # 如果出现任何错误，使用最基本的打印方式
        try:
            print(f"[{msg_type}] {str(message).encode('ascii', 'ignore').decode('ascii')}")
            sys.stdout.flush()
        except:
            print(f"[{msg_type}] <message encoding error>")
            sys.stdout.flush()


def format_status(status, message):
    """
    格式化状态消息
    
    Args:
        status: 状态类型 (OK, WARN, ERROR, etc.)
        message: 消息内容
        
    Returns:
        str: 格式化后的消息
    """
    return f"[{status}] {message}"


def print_step(step_num, description):
    """
    打印步骤信息
    
    Args:
        step_num: 步骤编号
        description: 步骤描述
    """
    safe_print(f"步骤{step_num}: {description}", "STEP")


def print_progress(current, total, description=""):
    """
    打印进度信息
    
    Args:
        current: 当前进度
        total: 总数
        description: 描述
    """
    percentage = (current / total) * 100 if total > 0 else 0
    safe_print(f"进度: {current}/{total} ({percentage:.1f}%) {description}", "PROGRESS") 