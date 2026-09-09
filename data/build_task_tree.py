#!/usr/bin/env python3
"""Build per-task prefix TREES from Nemotron trajectories (v3).

- nodes = message states, edges = assistant actions with label 1/0
- NO trajectory stitching: training sample per task = the SHORTEST single
  PASS trajectory (all its edges are 1 by construction)
- edge label = 1 iff the action appears on some PASS trajectory (same task,
  same prefix); = 0 iff fail/none-only

Outputs:
- data/nemotron/task_trees.jsonl : nested tree + per-trajectory edge labels
- data/nemotron/best_sft.jsonl   : rendered shortest single pass trajectory per task
"""
import gzip, glob, json, os, re, hashlib, collections

DATA_DIR = '/home/wcx/swe/data/nemotron'
OUT_TREE = os.path.join(DATA_DIR, 'task_trees.jsonl')
OUT_SFT = os.path.join(DATA_DIR, 'best_sft.jsonl')
MAX_TOOL_CHARS = 8000

TEST_PASSED = re.compile(r"(\d+)\s+passed", re.I)
TEST_FAILED = re.compile(r"(\d+)\s+failed", re.I)
TEST_ERRORS = re.compile(r"(\d+)\s+errors?", re.I)
TEST_OK = re.compile(r"test result: ok", re.I)
ALL_PASS = re.compile(r"All tests passed", re.I)


def verdict(msgs):
    for m in reversed(msgs):
        if m.get("role") != "tool":
            continue
        c = m.get("content") or ""
        if not isinstance(c, str) or len(c) > 200000:
            continue
        if not re.search(r"passed|failed|test result|Ran \d+ tests", c, re.I):
            continue
        passed = sum(int(x) for x in TEST_PASSED.findall(c))
        failed = sum(int(x) for x in TEST_FAILED.findall(c)) + \
                 sum(int(x) for x in TEST_ERRORS.findall(c))
        if TEST_OK.search(c) or ALL_PASS.search(c):
            passed = max(passed, 1)
        if passed or failed:
            return "pass" if (passed > 0 and failed == 0) else "fail"
    return "none"


def first_user(msgs):
    for m in msgs:
        if m.get('role') == 'user':
            return str(m.get('content') or '').strip()
    return ''


def sig_of(m):
    out = []
    for tc in (m.get('tool_calls') or []):
        fn = (tc.get('function') or {}).get('name') or ''
        a = (tc.get('function') or {}).get('arguments') or ''
        a = ' '.join(str(a).split())
        h = hashlib.sha1(a[:300].encode('utf-8', 'replace')).hexdigest()[:12]
        out.append(fn + ':' + h)
    return tuple(out)


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
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'train-*.jsonl.gz')))
    print('shards:', len(files))

    # ---- pass 1: per-task prefix tree ----
    task_state = {}
    traj_meta = {}
    for fp in files:
        with gzip.open(fp, 'rt', encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                msgs = r.get('messages') or []
                uuid = r.get('uuid')
                task = first_user(msgs)
                out = verdict(msgs)
                st = task_state.get(task)
                if st is None:
                    st = {'children': {}, 'edges': [], 'term': {},
                          'next_node': 1}
                    task_state[task] = st
                cur = 0
                eids, sigs = [], []
                for i, m in enumerate(msgs):
                    if m.get('role') != 'assistant':
                        continue
                    sig = sig_of(m)
                    if not sig:
                        st['term'].setdefault(cur, set()).add(out)
                        continue
                    e = st['children'].setdefault(cur, {}).get(sig)
                    if e is None:
                        v = st['next_node']; st['next_node'] += 1
                        e = {'u': cur, 'v': v, 'sig': sig, 'trajs': [],
                             'eid': len(st['edges'])}
                        st['children'][cur][sig] = e
                        st['edges'].append(e)
                    e['trajs'].append((uuid, i))
                    eids.append(e['eid']); sigs.append(sig)
                    cur = e['v']
                st['term'].setdefault(cur, set()).add(out)
                traj_meta[uuid] = {'task': task, 'outcome': out,
                                   'eids': eids, 'sigs': sigs}

    groups = collections.defaultdict(list)
    for u, m in traj_meta.items():
        groups[m['task']].append(u)
    n_edges = sum(len(st['edges']) for st in task_state.values())
    print('trajectories:', len(traj_meta), 'tasks:', len(task_state),
          'edges:', n_edges)

    # ---- labels ----
    for st in task_state.values():
        for e in st['edges']:
            outs = {traj_meta[t[0]]['outcome'] for t in e['trajs']}
            e['label'] = 1 if 'pass' in outs else 0
            e['reason'] = 'pass' if 'pass' in outs else 'fail_only'
    n_zero = sum(1 for st in task_state.values()
                 for e in st['edges'] if e['label'] == 0)
    print('edges label=1:', n_edges - n_zero, 'label=0:', n_zero)

    # ---- best single pass trajectory per task (NO stitching) ----
    best = {}
    for task, st in task_state.items():
        passes = [u for u in groups[task] if traj_meta[u]['outcome'] == 'pass']
        best[task] = (min(passes, key=lambda u: len(traj_meta[u]['eids']))
                      if passes else None)
    n_best = sum(1 for u in best.values() if u)
    print('tasks with a best pass trajectory:', n_best, '/', len(task_state))

    # ---- serialize as nested tree ----
    def node_to_dict(nid, st):
        d = {'n': nid}
        terms = sorted(st['term'].get(nid, set()))
        if terms:
            d['term'] = terms
        kids = sorted(st['children'].get(nid, {}).items(),
                      key=lambda kv: (-kv[1]['label'], kv[1]['v']))
        if kids:
            cl = []
            for sig, e in kids:
                t = e['trajs'][0]
                cl.append({'tool': list(sig), 'label': e['label'],
                           'reason': e['reason'],
                           'msg': [t[0][:18], t[1]],
                           'node': node_to_dict(e['v'], st)})
            d['children'] = cl
        return d

    max_depth = [0]
    max_branch = [0]
    with open(OUT_TREE, 'w', encoding='utf-8') as ft:
        for task, st in task_state.items():
            uuids = groups[task]
            outs = collections.Counter(traj_meta[u]['outcome'] for u in uuids)
            tree = node_to_dict(0, st)

            def depth(d, cur=0):
                max_depth[0] = max(max_depth[0], cur)
                for c in d.get('children', []):
                    depth(c['node'], cur + 1)
                return d
            max_branch[0] = max(max_branch[0], len(tree.get('children', [])))
            depth(tree)
            rec = {
                'task_id': hashlib.sha1(task[:120].encode('utf-8')).hexdigest()[:16],
                'task_head': task[:80].replace('\n', ' '),
                'n_traj': len(uuids), 'n_pass': outs['pass'],
                'n_fail': outs['fail'], 'n_none': outs['none'],
                'trajectories': {
                    u: {'outcome': traj_meta[u]['outcome'],
                        'n_edges': len(traj_meta[u]['eids']),
                        'labels': [st['edges'][eid]['label']
                                   for eid in traj_meta[u]['eids']]}
                    for u in uuids},
                'best_traj': best[task],
                'best_len': (len(traj_meta[best[task]]['eids'])
                             if best[task] else None),
                'tree': tree,
            }
            ft.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print('max tree depth:', max_depth[0], 'max branching:', max_branch[0])

    # ---- render best single pass trajectories ----
    needed = {u for u in best.values() if u}
    msgs_by_uuid = {}
    for fp in files:
        with gzip.open(fp, 'rt', encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                if r.get('uuid') in needed:
                    msgs_by_uuid[r['uuid']] = r.get('messages') or []
    print('loaded msgs for', len(msgs_by_uuid), 'uuids')

    lens = []
    with open(OUT_SFT, 'w', encoding='utf-8') as fo:
        for task, u in best.items():
            if not u:
                continue
            msgs = msgs_by_uuid[u]
            rendered = '\n'.join(render_block(m) for m in msgs)
            lens.append(len(traj_meta[u]['eids']))
            fo.write(json.dumps({
                'uuid': u, 'task_id': hashlib.sha1(
                    task[:120].encode('utf-8')).hexdigest()[:16],
                'label': 'pass', 'rendered': rendered,
                'est_chars': len(rendered), 'n_msgs': len(msgs)},
                ensure_ascii=False) + '\n')
    print('best_sft.jsonl lines:', len(lens))
    if lens:
        print('best_len: avg=%.1f min=%d max=%d' % (
            sum(lens) / len(lens), min(lens), max(lens)))

    # ---- sample tree for the imbalanced task ----
    with open(OUT_TREE, encoding='utf-8') as ft:
        for line in ft:
            rec = json.loads(line)
            if any(u.startswith('4fa00e43') for u in rec['trajectories']):
                print('\n=== imbalanced task tree (first 2 levels) ===')
                print('best_traj:', rec['best_traj'])
                print('best_len:', rec['best_len'])
                root = rec['tree']
                for c in root.get('children', [])[:4]:
                    print('  label=%d tool=%s msg=%s' % (
                        c['label'], c['tool'], c['msg']))
                    for c2 in c['node'].get('children', [])[:3]:
                        print('    label=%d tool=%s msg=%s' % (
                            c2['label'], c2['tool'], c2['msg']))
                break


if __name__ == '__main__':
    main()
