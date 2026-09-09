#!/usr/bin/env python3
"""Build chunked SFT data using DeepSeek labels (v2, ASCII only).

1. drop assistant steps labeled 0 (assistant msg + its tool outputs)
2. chunk each trajectory at assistant boundaries so every chunk fits
   MAX_CHUNK_TOKENS (each chunk = system+issue prefix + a segment)
Output: data/nemotron/deepseek_sft_chunks.jsonl
"""
import gzip
import glob
import json
import os

from transformers import AutoTokenizer

LABELS = "/home/wcx/swe/data/nemotron/deepseek_labels.jsonl"
BEST = "/home/wcx/swe/data/nemotron/best_sft.jsonl"
OUT = "/home/wcx/swe/data/nemotron/deepseek_sft_chunks.jsonl"
MAX_TOOL_CHARS = 8000
MAX_CHUNK_TOKENS = 32000
PAD = 64


def render_block(m):
    role = m.get('role')
    if role == 'system':
        return '<|im_start|>system\n' + str(m.get('content', '')) + '<|im_end|>'
    if role == 'user':
        return '<|im_start|>user\n' + str(m.get('content', '')) + '<|im_end|>'
    if role == 'assistant':
        text = ''
        rc = m.get('reasoning_content') or ''
        if rc:
            text += str(rc)
        for tc in m.get('tool_calls') or []:
            fn = tc.get('function', {})
            text += '\n<tool_call>\n%s\n%s\n</tool_call>' % (
                fn.get('name', ''), fn.get('arguments', ''))
        if m.get('content'):
            text += str(m['content'])
        return '<|im_start|>assistant\n' + text + '<|im_end|>'
    if role == 'tool':
        c = str(m.get('content') or '')
        if len(c) > MAX_TOOL_CHARS:
            c = c[:MAX_TOOL_CHARS] + '\n... (truncated)'
        return '<|im_start|>user\n<tool_output>\n' + c + \
            '\n</tool_output><|im_end|>'
    return ''


def main():
    tok = AutoTokenizer.from_pretrained(
        "/home/wcx/swe/models/Qwen2.5-Coder-3B", trust_remote_code=True)

    labels = {}
    for l in open(LABELS, encoding='utf-8'):
        r = json.loads(l)
        if r.get('status') == 'ok':
            labels[r['uuid']] = r['verdicts']
    print('labeled uuids:', len(labels))

    uids = [json.loads(l)['uuid'] for l in open(BEST, encoding='utf-8')]
    msgs_by = {}
    files = sorted(glob.glob("/home/wcx/swe/data/nemotron/train-*.jsonl.gz"))
    for fp in files:
        with gzip.open(fp, 'rt', encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                if r.get('uuid') in uids:
                    msgs_by[r['uuid']] = r.get('messages') or []

    total_dropped = 0
    n_changed = 0
    n_chunks = 0
    chunk_dist = {}
    with open(OUT, 'w', encoding='utf-8') as fo:
        for u in uids:
            msgs = msgs_by[u]
            v = labels.get(u, {})
            kept = []
            dropped = 0
            i = 0
            while i < len(msgs):
                m = msgs[i]
                if m.get('role') == 'assistant':
                    if v.get('msg%02d' % i, 1) == 0:
                        dropped += 1
                        i += 1
                        while i < len(msgs) and msgs[i].get('role') == 'tool':
                            i += 1
                        continue
                kept.append(m)
                i += 1
            if dropped:
                n_changed += 1
                total_dropped += dropped

            # prefix = system + issue (first two messages)
            prefix = kept[:2]
            rest = kept[2:]
            prefix_txt = '\n'.join(render_block(m) for m in prefix)
            prefix_tok = len(tok(prefix_txt)['input_ids'])

            chunks = []
            cur = [prefix_txt]
            cur_tok = prefix_tok
            for m in rest:
                blk = render_block(m)
                bt = len(tok(blk)['input_ids'])
                if cur_tok + bt > MAX_CHUNK_TOKENS - PAD and len(cur) > 1:
                    chunks.append('\n'.join(cur))
                    cur = [prefix_txt]
                    cur_tok = prefix_tok
                cur.append(blk)
                cur_tok += bt
            if len(cur) > 1:
                chunks.append('\n'.join(cur))

            for ci, txt in enumerate(chunks):
                fo.write(json.dumps({
                    'uuid': u, 'chunk': ci, 'label': 'pass',
                    'rendered': txt, 'est_chars': len(txt),
                    'n_chunks': len(chunks), 'n_dropped': dropped},
                    ensure_ascii=False) + '\n')
                n_chunks += 1
            chunk_dist[len(chunks)] = chunk_dist.get(len(chunks), 0) + 1

    print('chunks written:', n_chunks)
    print('trajectories changed:', n_changed, '/', len(uids))
    print('total steps dropped:', total_dropped)
    print('chunks-per-trajectory distribution:', sorted(chunk_dist.items()))


if __name__ == '__main__':
    main()
