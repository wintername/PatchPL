# ========== 5. 答案提取工具函数 ==========

def extract_answer_math(text):
    """
    从模型回复中提取最终数学答案
    
    提取策略优先级:
      1. \boxed{...}        ← LaTeX数学盒子(最标准)
      2. answer is/:= X     ← 自然语言声明
      3. 返回None           ← 无法提取(后续由normalize处理)
    
    注意: 这是v9的原始版本! 模型实际输出<<9*2=18>>18格式,
          但此函数不识别,导致大量正确答案被误判。
          v10中已修复此问题。
    """
    m = re.search(r'\\boxed\{([^}]*)\}', text)
    # ↑ 正则匹配 \boxed{内容}
    #   r'\\boxed\{' 匹配字面的"\boxed{"
    #   ([^}]*)      捕获任意非}字符
    if m: return m.group(1).strip()
    # ↑ 捕获组1是{}内的内容, strip去掉首尾空白
    
    m = re.search(
        r'(?:answer\s*(?:is|:|=))\s*(\d[\d,.\/]*)\b',
        text, re.IGNORECASE
    )
    # ↑ 匹配 "answer is 18" 或 "answer: 18" 等模式
    #   (?:...) 非捕获组,只分组不保存
    #   \s*      0或多个空白
    #   \d[\d,.\/]* 数字开头,后可跟数字/逗号/点/斜杠
    #   \b       单词边界
    if m: return m.group(1).strip()
    
    return None
    # ↑ 无法提取→返回None
    #   后续normalize(None)→None, 与GT比较→判为错误


def normalize(s):
    """
    答案标准化: 去除格式差异,统一比较
    
    标准化操作:
      去空格, 去逗号(1,000→1000)
      去百分号, 转小写
      去末尾句号
      分数转小数(1/2→0.5)
    
    None安全: 输入None→返回None
    """
    if s is None: return None
    # ↑ 空值保护
    
    s = s.replace(' ', '').replace(',', '').replace('%', '').lower().rstrip('.')
    # ↑ 链式替换: 空格→无, 逗号→无, %→无, 大写→小写, 末尾.→无
    #   例: "1,000.50" → "100050"
    
    if '/' in s:
        # ↑ 检测分数: 如 "1/2", "3/4"
        try:
            parts = s.split('/')
            return str(float(parts[0]) / float(parts[1]))
            # ↑ 分子÷分母, 转回字符串
            #   例: "1/2" → 0.5 → "0.5"
        except:
            pass  # 解析失败→保持原样
    
    return s


