"""Live latency smoke — mint a real Appwrite JWT for the test user and time
/api/module-chat wardrobe-board responses against prod. Throwaway diagnostic.

Usage:
  python scripts/live_latency_smoke.py
"""
from __future__ import annotations

import os
import sys
import time
import json
import urllib.request
import urllib.error

APPWRITE = "https://fra.cloud.appwrite.io/v1"
PROJECT = "69958f25003190519213"
USER_ID = "69e6096616923db26940"
PROD = os.getenv("SMOKE_BASE_URL", "https://ahvi-backend-lz3aebcusq-el.a.run.app")

PROMPTS = [
    "office outfit using my wardrobe",
    "client meeting outfit from my wardrobe",
    "smart casual look using my wardrobe",
]


def _api_key() -> str:
    for line in open(".env", encoding="utf-8"):
        if line.startswith("APPWRITE_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("APPWRITE_API_KEY not in .env")


def _post(url, data, headers, timeout):
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", **headers}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode())


def mint_jwt(key: str) -> str:
    status, body = _post(
        f"{APPWRITE}/users/{USER_ID}/jwts",
        {"duration": 900},
        {"X-Appwrite-Project": PROJECT, "X-Appwrite-Key": key},
        timeout=20,
    )
    jwt = body.get("jwt")
    if not jwt:
        raise SystemExit(f"JWT mint failed status={status} body={body}")
    return jwt


def hit(jwt: str, prompt: str):
    payload = {
        "message": prompt,
        "domain": "style",
        "module": "style",
        "user_profile": {"gender": "male"},
        "context": {"gender": "male"},
    }
    t0 = time.monotonic()
    try:
        status, body = _post(
            f"{PROD}/api/module-chat",
            payload,
            {"Authorization": f"Bearer {jwt}"},
            timeout=120,
        )
    except urllib.error.HTTPError as e:
        return time.monotonic() - t0, e.code, {"error": e.read().decode()[:200]}
    return time.monotonic() - t0, status, body


def summarize(body):
    rtype = body.get("type")
    cards = body.get("cards") or body.get("boards") or []
    titles = []
    for c in cards if isinstance(cards, list) else []:
        t = c.get("title") if isinstance(c, dict) else None
        items = []
        if isinstance(c, dict):
            bi = c.get("board_items") or c.get("items") or c.get("pieces") or []
            for it in bi if isinstance(bi, list) else []:
                items.append(it.get("name") if isinstance(it, dict) else str(it))
        titles.append((t, items[:5]))
    msg = body.get("message_text") or body.get("response") or ""
    return rtype, len(cards) if isinstance(cards, list) else 0, titles, str(msg)[:120]


def main() -> int:
    key = _api_key()
    print("Minting JWT...")
    jwt = mint_jwt(key)
    print(f"JWT ok ({len(jwt)} chars)\n" + "=" * 80)
    for i, p in enumerate(PROMPTS, 1):
        dt, status, body = hit(jwt, p)
        rtype, n, titles, msg = summarize(body)
        tag = "COLD" if i == 1 else "warm"
        print(f"\n[{i}/{len(PROMPTS)}] ({tag}) {dt:5.1f}s  http={status}  type={rtype}  boards={n}")
        print(f"     prompt: {p}")
        if titles:
            for t, items in titles:
                print(f"       - {t} -> {items}")
        elif msg:
            print(f"       msg: {msg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
