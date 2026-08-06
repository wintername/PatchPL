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
    if m: return m.group(1).strip()
    m = re.search(
        r'(?:answer\s*(?:is|:|=))\s*(\d[\d,.\/]*)\b',
        text, re.IGNORECASE
    )
    if m: return m.group(1).strip()
    return None
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
    s = s.replace(' ', '').replace(',', '').replace('%', '').lower().rstrip('.')
    if '/' in s:
        try:
            parts = s.split('/')
            return str(float(parts[0]) / float(parts[1]))
        except:
            pass  # 解析失败→保持原样
    return s
