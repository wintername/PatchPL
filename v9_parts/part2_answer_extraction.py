def extract_answer_math(text):
    m = re.search(r'\\boxed\{([^}]*)\}', text)
    if m: return m.group(1).strip()
    m = re.search(
        r'(?:answer\s*(?:is|:|=))\s*(\d[\d,.\/]*)\b',
        text, re.IGNORECASE
    )
    if m: return m.group(1).strip()
    return None
def normalize(s):
    if s is None: return None
    s = s.replace(' ', '').replace(',', '').replace('%', '').lower().rstrip('.')
    if '/' in s:
        try:
            parts = s.split('/')
            return str(float(parts[0]) / float(parts[1]))
        except:
            pass  # 解析失败→保持原样
    return s