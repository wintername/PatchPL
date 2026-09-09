#!/usr/bin/env python3
"""Batch DeepSeek annotation of best_sft trajectories (ASCII only).

For every best trajectory ask deepseek-chat to label EVERY assistant message
0/1 (strict criteria, full-transcript context). Checkpointed + resumable.
Output: data/nemotron/deepseek_labels.jsonl (one line per uuid)
"""
import concurrent.futures
import gzip
import glob
import json
import os
import re
import sys
import threading
import time
import urllib.request

KEY = re.sub(r"[^\x20-\x7e]", "",
             open(os.path.expanduser("~/.deepseek_key"),
                  encoding="utf-8", errors="replace").read()).strip()
API = "https://api.deepseek.com/chat/completions"
BEST = "/home/wcx/swe/data/nemotron/best_sft.jsonl"
OUT = "/home/wcx/swe/data/nemotron/deepseek_labels.jsonl"
LOG = "/home/wcx/swe/logs/deepseek_annot.log"
MAX_WORKERS = 8
TRANSCRIPT_CAP = 50000
LOCK = threading.Lock()


def log(*a):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(" ".join(str(x) for x in a) + "\n")
    print(*a, flush=True)


def transcript_of(msgs):
    parts = []
    total = 0
    for i, m in enumerate(msgs):
        role = m.get("role")
        seg = "### msg%02d [%s]\n" % (i, role)
        if m.get("reasoning_content"):
            seg += "[reasoning] " + str(m["reasoning_content"]) + "\n"
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {})
            seg += "[tool_call] %s %s\n" % (fn.get("name"),
                                            str(fn.get("arguments", "")))
        c = m.get("content")
        if c:
            cs = str(c)
            if len(cs) > 6000:
                cs = cs[:6000] + "\n...(output truncated)"
            seg += "[content] " + cs + "\n"
        parts.append((i, seg))
        total += len(seg)
    if total <= TRANSCRIPT_CAP:
        return "\n".join(s for _, s in parts)
    # head + tail with marker
    head, tail, hc, tc2 = [], [], 0, 0
    for i, s in parts:
        if hc + len(s) < TRANSCRIPT_CAP * 0.6:
            head.append(s); hc += len(s)
    for i, s in reversed(parts):
        if tc2 + len(s) < TRANSCRIPT_CAP * 0.4:
            tail.append(s); tc2 += len(s)
    return "\n".join(head) + "\n### (middle messages omitted) ###\n" + \
        "\n".join(reversed(tail))


SYSTEM = ("You are a trajectory graph-labeling assistant. "
          "Below is a COMPLETE coding-agent trajectory that eventually "
          "fixed the issue (final tests passed). "
          "For EVERY assistant message, judge whether that step's action "
          "(tool call / reasoning) was NECESSARY for the final fix. "
          "Strict criteria - label 0 if ANY of these hold: "
          "(a) it reads/explores something never used by later steps; "
          "(b) it repeats an action whose result was already obtained; "
          "(c) it was a failed/retried attempt superseded by a later "
          "successful one; "
          "(d) its edits were later reverted or overwritten. "
          "Label 1 only if the step's output was actually used by "
          "subsequent steps or appears in the final fix. "
          "Do NOT give everything 1 just because the trajectory succeeded. "
          "Output ONLY a JSON object mapping assistant msg ids to 0/1, "
          'e.g. {"msg02":1,"msg04":0}.')


def judge(uuid, msgs):
    transcript = transcript_of(msgs)
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": transcript}],
        "temperature": 0.0,
        "max_tokens": 2000,
    }
    req = urllib.request.Request(API, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + KEY})
    with urllib.request.urlopen(req, timeout=600) as resp:
        d = json.loads(resp.read().decode())
    return d["choices"][0]["message"]["content"]


def parse_verdicts(txt, msgs):
    mm = re.search(r"\{.*\}", txt, flags=re.DOTALL)
    verdicts = {}
    if mm:
        try:
            verdicts = json.loads(mm.group(0))
        except Exception:
            pass
    out = {}
    n_assistant = 0
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        n_assistant += 1
        k = "msg%02d" % i
        v = verdicts.get(k, 1)
        if v not in (0, 1):
            v = 1
        out[k] = v
    # force final summary (last assistant, no tools) to 1
    last_a = max((i for i, m in enumerate(msgs)
                  if m.get("role") == "assistant"), default=-1)
    if last_a >= 0:
        out["msg%02d" % last_a] = 1
    return out


def one(uuid, msgs):
    last_err = None
    for attempt in range(3):
        try:
            txt = judge(uuid, msgs)
            v = parse_verdicts(txt, msgs)
            nz = sum(1 for x in v.values() if x == 0)
            rec = {"uuid": uuid, "verdicts": v, "n_zero": nz,
                   "status": "ok"}
            with LOCK:
                with open(OUT, "a", encoding="utf-8") as fo:
                    fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return rec
        except Exception as e:
            last_err = e
            time.sleep(5 * (attempt + 1))
    log("FAIL", uuid, repr(last_err))
    with LOCK:
        with open(OUT, "a", encoding="utf-8") as fo:
            fo.write(json.dumps({"uuid": uuid, "verdicts": {},
                                 "n_zero": 0, "status": "error"},
                                ensure_ascii=False) + "\n")
    return None


def main():
    uids = [json.loads(l)["uuid"]
            for l in open(BEST, encoding="utf-8")]
    log("best uids:", len(uids))
    done = {}
    if os.path.exists(OUT):
        for l in open(OUT, encoding="utf-8"):
            r = json.loads(l)
            done[r["uuid"]] = r
    log("already done:", len(done))

    needed = [u for u in uids if u not in done]
    need_set = set(needed)
    log("to annotate:", len(needed))
    if not needed:
        log("ALL-DONE-ALREADY")
        return

    msgs_by = {}
    files = sorted(glob.glob("/home/wcx/swe/data/nemotron/train-*.jsonl.gz"))
    for fp in files:
        with gzip.open(fp, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("uuid") in need_set:
                    msgs_by[r["uuid"]] = r.get("messages") or []
    log("loaded msgs:", len(msgs_by))

    t0 = time.time()
    n = len(done)
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(one, u, msgs_by.get(u, [])): u
                for u in needed if u in msgs_by}
        for fut in concurrent.futures.as_completed(futs):
            u = futs[fut]
            try:
                fut.result()
            except Exception as e:
                log("EXC", u, repr(e))
            n += 1
            if n % 25 == 0:
                el = time.time() - t0
                log("progress %d/%d elapsed=%.0fs rate=%.1f/traj" %
                    (n, len(uids), el, el / max(1, n - len(done))))
    log("ANNOT-DONE total=%d" % n)


if __name__ == "__main__":
    main()
