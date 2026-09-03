#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MRCS STEP 5 — AI 분석 문단 생성 (로컬 모델)

화면의 aiParagraphs() 자리를 대신합니다.
지도에서 계산이 끝난 값을 넣으면 4문단을 만들어 돌려줍니다.

핵심 규칙
  ① 숫자는 모델이 쓰지 않습니다. {{gap}} 같은 슬롯을 남기게 하고 우리가 채웁니다.
  ② 나온 문장은 검사기를 통과해야 씁니다.
  ③ 실패하면 규칙 기반 문장으로 폴백합니다. 화면은 어떤 경우에도 돕니다.

사용
  python3 tools/ai_brief.py --model qwen3:8b
  python3 tools/ai_brief.py --model exaone3.5:7.8b --compare qwen3:8b
"""
import argparse, json, os, re, sys, time, urllib.request

OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "mrcs_alt_data.json")

# ── 지킬 선 — 정의서 §2.2 ────────────────────────────────────────
SYSTEM = """당신은 보험 보장분석 서비스 MRCS의 문장을 쓰는 작성자입니다.
계산이 끝난 결과를 소비자가 읽을 문장으로 옮기는 일만 합니다.

반드시 지킬 것
- 숫자를 직접 쓰지 마십시오. 반드시 {{슬롯이름}} 형태로 두십시오. 중괄호는 두 개입니다.
- 주어진 슬롯만 쓰고, 없는 이름을 만들지 마십시오.
- 슬롯이 무엇을 뜻하는지 정확히 읽으십시오. 뜻을 바꿔 쓰면 사실이 틀어집니다.
- 특정 보험회사나 상품 이름을 쓰지 마십시오.
- 겁을 주지 마십시오. '치명적', '생명을 위협' 같은 표현을 쓰지 마십시오.
- 보험이 보장하지 않는 영역을 '부족하다'고 쓰지 마십시오. 보험 밖이라고만 쓰십시오.
- 주어진 사실에 없는 내용을 덧붙이지 마십시오.

문체
- 반드시 '~습니다' 로 끝나는 존댓말 평서문입니다. 반말을 쓰면 안 됩니다.
- 아래 '쓸 내용' 설명을 그대로 옮겨 쓰지 마십시오. 완성된 문장으로 다시 쓰십시오.
- 한 문단은 두 문장 이내, 공백 포함 150자 이내. 짧게 끊으십시오.
- 슬롯 뒤에 단위를 붙이지 마십시오. 개월·개·%·만원이 값 안에 이미 들어 있습니다.
  {{restMonths}}개월 (X) → {{restMonths}} (O)
- 문단마다 지정된 슬롯을 모두 쓰십시오. 사실이 빠진 문장은 쓸모가 없습니다."""

USER_TMPL = """한 사람의 보장 준비 상태를 계산했습니다. 이 값으로 문단 4개를 써 주십시오.

■ 슬롯과 그 뜻 — 이 뜻대로만 쓰십시오
  {{fieldCount}}  고른 질병 분야 수 (예: 16개)
  {{indemRate}}    필요한 병원비 중에서 실손보험이 메워 주는 비율
  {{cashRate}}    정액 보장이 감당해야 할 몫 중에서 지금 준비된 비율
  {{restMonths}}   1순위 분야에서 일을 쉬게 되는 기간 (예: 36개월)
  {{needAmount}}   1순위 분야에서 필요한 총 금액 (예: 15,174만)
  {{emptyAmount}}      그중 아직 준비되지 않아 비어 있는 금액 (예: 12,989만)
  {{gap1Fields}}   '{ben1}' 보장이 비어 있는 분야 수 (예: 15개)
  {{gap2Fields}}   '{ben2}' 보장이 비어 있는 분야 수 (예: 15개)
  {{emergencyMonths}}   비상자금으로 생활비를 감당할 수 있는 기간 (예: 5.2개월)

■ 사실
  1순위 분야: {topName}
  그 분야에서 실제로 많이 겪는 병: {topRep}
  가장 많이 비어 있는 보장: {ben1}, 그다음 {ben2}
  간병비는 실손이 보장하지 않고, 장기요양보험은 등급 판정을 받아야 씁니다.

■ 문단별로 쓸 내용과 반드시 넣을 슬롯

1문단 — 전체 상태
   병원비는 실손이 상당 부분 메우지만, 정작 비어 있는 것은 치료하는 동안
   끊기는 수입이라는 점. 슬롯: {{fieldCount}} {{indemRate}} {{cashRate}}

2문단 — 가장 먼저 볼 분야
   {topName}을 먼저 보는 이유. {topRep}처럼 실제로 많이 겪는 병이 있고,
   일을 오래 쉬게 되어 필요 금액이 크다는 점. 슬롯: {{restMonths}} {{needAmount}} {{emptyAmount}}

3문단 — 보장 종류별 공백
   {ben1}과 {ben2}이 여러 분야에서 비어 있다는 점과,
   간병비가 제도와 보험 사이에 남는 자리라는 점. 슬롯: {{gap1Fields}} {{gap2Fields}}

4문단 — 감당과 전가의 경계
   비상자금으로 어디까지 버틸 수 있는지, 그 선이 보험으로 넘길 것과
   스스로 감당할 것을 나누는 기준이 된다는 점. 슬롯: {{emergencyMonths}}"""

# ── 검사기 ──────────────────────────────────────────────────────
# 지시문을 그대로 베껴 쓰거나 반말로 끝내면 걸러 냅니다
PARROT = ["라는 점", "는 이유", "다는 점", "쓸 내용", "슬롯:"]
BANMAL = re.compile(r"(?:이다|한다|된다|있다|없다|아니다|크다|많다|적다|필요하다|"
                    r"부족하다|감당한다|메운다)\s*[.。]?\s*$")
BANNED = ["치명적", "생명을 위협", "생명이 위험", "사망률", "위독",
          "삼성", "현대해상", "메리츠", "DB손해", "KB손해", "한화", "교보", "라이나",
          "가입하세요", "추천드립니다", "권해드립니다", "가입을 권"]
SLOT_RE = re.compile(r"\{\{?(\w+)\}\}?")   # {{slot}} 과 {slot} 을 모두 받는다
DIGIT_RE = re.compile(r"\d")


def check(paragraphs, allowed_slots):
    """통과하면 [], 아니면 사유 목록"""
    errs = []
    if not isinstance(paragraphs, list) or not (3 <= len(paragraphs) <= 5):
        return ["문단 수가 3~5개가 아닙니다"]
    for i, p in enumerate(paragraphs, 1):
        t = (p or {}).get("text", "") if isinstance(p, dict) else str(p)
        if not t.strip():
            errs.append(f"{i}문단이 비었습니다"); continue
        # 숫자를 직접 쓰면 탈락 (슬롯을 뺀 나머지에 숫자가 있으면 안 됨)
        bare = SLOT_RE.sub("", t)
        if DIGIT_RE.search(bare):
            errs.append(f"{i}문단에 숫자를 직접 썼습니다: {DIGIT_RE.findall(bare)[:3]}")
        for s in SLOT_RE.findall(t):
            if s not in allowed_slots:
                errs.append(f"{i}문단에 없는 슬롯 {{{{{s}}}}} 을 썼습니다")
        for w in BANNED:
            if w in t:
                errs.append(f"{i}문단에 금지 표현 '{w}' 이 있습니다")
        for w in PARROT:
            if w in t:
                errs.append(f"{i}문단이 지시문을 그대로 베꼈습니다 ('{w}'). 완성된 문장으로 쓰십시오")
                break
        for sent in re.split(r"(?<=[.!?])\s+", t):
            if sent.strip() and BANMAL.search(sent.strip()):
                errs.append(f"{i}문단이 반말로 끝납니다. '~습니다'로 끝내십시오")
                break
        if len(t) > 165:
            errs.append(f"{i}문단이 너무 깁니다 ({len(t)}자). 한 문장을 통째로 지우십시오")
    used = {s for p in paragraphs
            for s in SLOT_RE.findall((p or {}).get("text", "") if isinstance(p, dict) else str(p))}
    if len(used) < 5:
        errs.append(f"슬롯을 {len(used)}개만 썼습니다. 사실이 빠진 문장입니다 (5개 이상 필요)")
    return errs


DUP_RE = [(re.compile(r"(%)\s*%"), r"\1"),
          (re.compile(r"(만)\s*만원?"), r"\1"),
          (re.compile(r"(개월)\s*개월"), r"\1"),
          (re.compile(r"(개)\s*개(?![월])"), r"\1")]


def fill(text, slots):
    """슬롯을 채우고, 모델이 뒤에 또 붙인 단위를 정리한다"""
    t = SLOT_RE.sub(lambda m: str(slots.get(m.group(1), m.group(0))), text)
    for rx, rep in DUP_RE:
        t = rx.sub(rep, t)
    return t


# ── 모델 호출 ───────────────────────────────────────────────────
# format 에 스키마를 주면 모델이 그 모양으로만 답합니다.
# "JSON으로 답해줘"라고 부탁하는 것과 달리, 형식이 어긋날 수가 없습니다.
SCHEMA = {
    "type": "object",
    "properties": {
        "paragraphs": {
            "type": "array", "minItems": 4, "maxItems": 4,
            "items": {"type": "object",
                      "properties": {"text": {"type": "string"}},
                      "required": ["text"]},
        }
    },
    "required": ["paragraphs"],
}


def ask(model, system, user, timeout=240):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "stream": False, "format": SCHEMA, "think": False,
        "options": {"temperature": 0.3, "num_predict": 900},
    }).encode()
    req = urllib.request.Request(OLLAMA + "/api/chat", body,
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read().decode())
    return (out.get("message") or {}).get("content", ""), round(time.time() - t0, 1)


def parse(raw):
    """모델이 배열로도, {"paragraphs": [...]} 로도 줄 수 있어 둘 다 받는다"""
    try:
        v = json.loads(raw)
    except Exception:
        m = re.search(r"\[.*\]", raw, re.S)
        if not m:
            return None
        try:
            v = json.loads(m.group(0))
        except Exception:
            return None
    if isinstance(v, dict):
        for k in ("paragraphs", "items", "result", "data"):
            if isinstance(v.get(k), list):
                v = v[k]; break
    if not isinstance(v, list):
        return None
    return [x if isinstance(x, dict) else {"text": str(x)} for x in v]


# ── 지도 값 → 슬롯 ──────────────────────────────────────────────
def build_slots(combo, meta, labels, order):
    s, rows, top = combo["summary"], combo["rows"], combo["rows"][0]
    keys = [k for k in order if k != "indemnity"]
    rank = sorted(keys, key=lambda k: -combo["agg"][k]["score"])
    b1, b2 = rank[0], rank[1]
    pct = lambda v: "충분" if v >= 1 else f"{round(v*100)}%"
    won = lambda v: f"{round(v):,}만"
    return {
        "fieldCount": f"{s['fieldCount']}개",
        "indemRate": pct(s["medRatio"]), "cashRate": pct(s["livRatio"]),
        # 단위를 값 안에 모두 넣습니다. 모델이 뒤에 또 붙여 '74%%' '15,174만만'
        # 같은 중복이 나던 문제를 이 방식으로 없앱니다.
        "restMonths": f"{top['months']}개월",
        "needAmount": won(top["center"]), "emptyAmount": won(top["gap"]),
        "gap1Fields": f"{combo['agg'][b1]['none']}개",
        "gap2Fields": f"{combo['agg'][b2]['none']}개",
        "emergencyMonths": f"{meta['emergency']/meta['expense']:.1f}개월",
    }, {"topName": top["name"], "topRep": top["repName"],
        "ben1": labels[b1], "ben2": labels[b2]}


def run(model, combo, meta, labels, order, retries=2, verbose=True):
    slots, names = build_slots(combo, meta, labels, order)
    user = USER_TMPL.format(**names)
    for n in range(1, retries + 2):
        raw, sec = ask(model, SYSTEM, user)
        ps = parse(raw)
        if ps is None:
            if verbose: print(f"  [{n}회] JSON 파싱 실패 ({sec}초)")
            continue
        errs = check(ps, slots.keys())
        if not errs:
            return {"ok": True, "model": model, "sec": sec, "tries": n,
                    "paragraphs": [fill(p["text"], slots) for p in ps]}
        if verbose:
            print(f"  [{n}회] 검사 불통과 ({sec}초) — " + " / ".join(errs[:3]))
        user += "\n\n앞선 답에 문제가 있었습니다: " + "; ".join(errs[:4]) + "\n다시 써 주십시오."
    return {"ok": False, "model": model, "paragraphs": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--compare", nargs="*", default=[])
    ap.add_argument("--sex", default="m", choices=["m", "f"])
    ap.add_argument("--att", default="balanced",
                    choices=["stable", "balanced", "tolerant"])
    a = ap.parse_args()

    D = json.load(open(DATA, encoding="utf-8"))
    combo = D["combos"][f"{a.sex}:{a.att}"]
    args = (combo, D["meta"], D["benefitLabel"], D["order"])

    for model in [a.model] + a.compare:
        print(f"\n{'='*66}\n  {model}\n{'='*66}")
        try:
            r = run(model, *args)
        except Exception as e:
            print(f"  호출 실패 — {e}"); continue
        if not r["ok"]:
            print("  검사를 통과하지 못했습니다. 규칙 기반 문장으로 폴백합니다.")
            continue
        print(f"  통과 ({r['tries']}회 시도 · {r['sec']}초)\n")
        for i, t in enumerate(r["paragraphs"], 1):
            print(f"  {i}. {t}\n")


if __name__ == "__main__":
    main()
