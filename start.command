#!/bin/bash
# MRCS 생애위험준비지도 — 로컬 실행 (macOS)
# 이 파일을 더블클릭하면 실행됩니다.

cd "$(dirname "$0")" || exit 1
MODEL="${MRCS_MODEL:-qwen3:8b}"

echo ""
echo "  MRCS 생애위험준비지도 — 로컬 실행"
echo "  ────────────────────────────────────────────"

# ── 파이썬 확인 ──────────────────────────────────────────────
if ! command -v python3 >/dev/null 2>&1; then
  echo ""
  echo "  파이썬이 없습니다."
  echo "  터미널에서 아래를 실행한 뒤 다시 시도해 주세요."
  echo ""
  echo "      xcode-select --install"
  echo ""
  read -r -p "  엔터를 누르면 닫힙니다 " _; exit 1
fi

# ── AI 문장 기능은 선택 사항 ─────────────────────────────────
AI="off"
if command -v ollama >/dev/null 2>&1; then
  curl -sS http://localhost:11434/api/tags --max-time 3 >/dev/null 2>&1 || {
    echo "  · 모델 서버를 켭니다"
    (nohup ollama serve >/tmp/mrcs-ollama.log 2>&1 &)
    sleep 3
  }
  if ollama list 2>/dev/null | grep -q "^${MODEL%%:*}"; then
    AI="on"
  else
    echo ""
    echo "  AI 문장 기능을 쓰려면 모델을 한 번 받아야 합니다 (약 5GB)."
    read -r -p "  지금 받으시겠습니까? [y/N] " YN
    if [ "$YN" = "y" ] || [ "$YN" = "Y" ]; then
      ollama pull "$MODEL" && AI="on"
    fi
  fi
fi

if [ "$AI" = "on" ]; then
  echo "  · AI 문장 생성  사용 가능 ($MODEL)"
else
  echo "  · AI 문장 생성  꺼짐 — 화면은 기본 문장으로 그대로 동작합니다"
fi
echo ""

python3 tools/serve.py --model "$MODEL"
