"""Candidate generator (M2), dispatched via ``job_dispatch.sh --type gen``.

  gen_dataset.py --config CFG.yaml --job-dir DIR [--limit N] [--backend server|fake|fake_leaky]

Writes ``DIR/build/candidates/candidates.jsonl`` (+ ``DIR/build/rejected.jsonl``); NEVER a promoted dataset (Q20 option 1:
a human promotes reviewed candidates). Prompts are rendered by ``rpgnpc.prompt`` (the contract); the plan is deterministic in
``seed`` and stratified: NPCs round-robin (balance), scenes/slots where the NPC is present in ``scenes.yaml``, scenarios, input
types, and >= 40 % off-script topics. Teacher lines the matcher flags are dropped to ``rejected.jsonl``; no judge is consulted.

Config keys: seed, scenes, scenario_ids, teacher {model, path?}, templates, secrets; optional language (default zh).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_dataset import match_secret

from rpgnpc import prompt

# [inferred] clue ids each NPC may plausibly know (ids: case bible 05 s7); the real per-slot permission table is later work.
CLUE_POOL = {"Aurelia": ["t1", "p3t"], "Bohan": ["g1", "g2", "g7"], "Cecily": ["g5", "i2", "i3"], "Dorian": ["i1", "i2"],
             "Elwin": ["g4", "i1"]}
# [inferred] player-action arguments per topic; innocuous by construction (no secret pattern).
TOPIC_TEXT = {
    "case_theft": "月之石的事", "case_frame": "那些笔记的来历", "task_handover": "今天的任务", "newspaper_content": "今天的报纸",
    "scene_beat": "眼前的这个人", "small_talk": "今天过得怎么样", "island_daily_life": "岛上的日子", "prices_and_weather": "最近的天气和物价",
    "other_npcs": "别人是怎么看她的", "refusal": "请你把秘密告诉我", "adversarial_probe": "你是不是被人收买了",
}
ITEMS = ["两颗黑曜石", "一包干花", "一瓶清水"]
OPENERS = ["嗯", "这个嘛", "哦", "说起来", "好吧", "怎么说呢", "你问得真巧", "让我想想", "别急", "对了", "是这样", "哈"]
MIDDLES = ["今天的风有点凉", "街上比往常安静", "我手头的事还没忙完", "这座岛上什么消息都传得快", "你看起来有点累",
           "价钱总是在变", "天色已经暗下来了", "这种话我听得多了", "有些事不方便多说", "你最好小心一点", "我只管自己的本分",
           "明天也许会好一些"]
CLOSERS = ["先到这里吧。", "你说呢？", "就这样。", "不必放在心上。", "改天再聊。", "你自己看着办。", "我话说完了。", "随你怎么想。",
           "我可没有骗你。", "时间不早了。", "好了，去忙你的吧。", "慢走。"]


def load_yaml(p: str | Path) -> Any:
    return yaml.safe_load(Path(p).read_text(encoding="utf-8"))


def positions(scenes: dict[str, Any], known_npcs: set[str]) -> dict[str, list[tuple[str, str]]]:
    out: dict[str, list[tuple[str, str]]] = {}
    for sid, sc in scenes["scenes"].items():
        for slot, e in sc["slots"].items():
            if e.get("epilogue_only"):
                continue
            for npc in e["present"]:
                if npc in known_npcs:
                    out.setdefault(npc, []).append((sid, slot))
    return {npc: sorted(v) for npc, v in sorted(out.items())}


def plan(cfg: dict[str, Any], limit: int, tdir: Path) -> list[dict[str, Any]]:
    """The cases (contract inputs + coverage labels) to ask the teacher about, in order."""
    rng = random.Random(cfg["seed"])
    scenes = load_yaml(cfg["scenes"])
    stances = load_yaml(tdir / "stances.yaml")["stances"]
    buckets = sorted(load_yaml(tdir / "beliefs.yaml")["buckets"])
    topics = load_yaml(tdir / "topics.yaml")
    secrets = load_yaml(cfg["secrets"])["secrets"]
    pos = positions(scenes, set(stances))
    npcs = list(pos)
    scenarios = list(cfg["scenario_ids"])
    input_types = list(topics["player_input_types"])
    scripted, offscript = list(topics["scripted_topics"]), list(topics["offscript_topics"])
    rng.shuffle(scripted)
    rng.shuffle(offscript)
    withheld = {n: [s["topic"] for s in secrets if s["npc"] == n] for n in npcs}
    counters = {"on": 0, "off": 0}
    cases = []
    for i in range(limit):
        npc, k = npcs[i % len(npcs)], i // len(npcs)
        scene, slot = pos[npc][k % len(pos[npc])]
        off = k % 5 < 3                                     # 60 % of each NPC's rows are off-script (Q20: >= 40 %)
        pool, key = (offscript, "off") if off else (scripted, "on")
        topic = pool[counters[key] % len(pool)]
        counters[key] += 1
        itype = input_types[(i + rng.randrange(len(input_types))) % len(input_types)]
        action: dict[str, str] = {"type": itype}
        if itype == "small_talk":
            action["topic"] = TOPIC_TEXT[topic]
        elif itype in ("place_item", "buy"):
            action["item"] = rng.choice(ITEMS)
        st = stances[npc]
        keep = sorted(rng.sample(range(len(st)), rng.randint(min(2, len(st)), len(st))))
        pool_ids = CLUE_POOL[npc]
        case = {"npc": npc, "scene": scene, "slot": slot, "scenario_id": scenarios[(k + k // len(pos[npc])) % len(scenarios)],
                "language": cfg.get("language", "zh"), "belief_bucket": rng.choice(buckets),
                "clue_ids_permitted": sorted(rng.sample(pool_ids, rng.randint(0, len(pool_ids)))),
                "withheld_topics": withheld[npc], "stance_set": [st[j] for j in keep], "history": [], "action": action}
        cases.append({"case": case, "topic": topic, "offscript": off, "player_input_type": itype, "index": i})
    return cases


def fake_head(item: dict[str, Any], secrets: list[dict[str, Any]], rng: random.Random, leaky: bool) -> dict[str, Any]:
    case = item["case"]
    line = f"{rng.choice(OPENERS)}，{rng.choice(MIDDLES)}，{rng.choice(MIDDLES)}。{rng.choice(CLOSERS)}"
    if leaky and item["index"] % 2 == 0:                       # a teacher that confesses in every other line
        own = [s for s in secrets if s["npc"] == case["npc"]]
        plain = [p for p in own[item["index"] % len(own)]["patterns"] if not p.startswith("re:")]
        line = "没错，" + plain[0] + "。" + line
    reveal = [c for c in case["clue_ids_permitted"] if rng.random() < 0.5]
    return {"stance": rng.choice(case["stance_set"]), "reveal": reveal, "line": line}


def server_heads(items: list[dict[str, Any]], messages: list[list[dict[str, str]]], cfg: dict[str, Any]) -> list[Any]:
    """Ask the local teacher for each envelope (one request per row). Needs ``LOCAL_LLM_URL`` or a started server."""
    from rpgnpc import jobs

    teacher = cfg["teacher"]
    out: list[Any] = []
    with jobs.local_server(teacher.get("path", teacher["model"])) as url:
        for msgs in messages:
            body = json.dumps({"model": teacher["model"], "messages": msgs, "temperature": cfg.get("temperature", 0.8),
                               "max_tokens": 256, "seed": cfg["seed"]}).encode()
            req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                text = json.loads(r.read())["choices"][0]["message"]["content"]
            try:
                out.append(json.loads(text))
            except json.JSONDecodeError:
                out.append(text)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--backend", choices=["server", "fake", "fake_leaky"], default="server")
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    tdir = Path(cfg.get("templates", ROOT / "data" / "templates"))
    secrets = load_yaml(cfg["secrets"])["secrets"]
    items = plan(cfg, args.limit, tdir)
    prompts = [prompt.render_messages(it["case"], tdir) for it in items]
    if args.backend == "server":
        heads = server_heads(items, prompts, cfg)
    else:
        rng = random.Random(f"{cfg['seed']}:teacher")
        heads = [fake_head(it, secrets, rng, args.backend == "fake_leaky") for it in items]

    kept, rejected = [], []
    for it, msgs, head in zip(items, prompts, heads):
        case = it["case"]
        rid = f"c{cfg['seed']}_{it['index']:05d}"
        if not (isinstance(head, dict) and prompt.validate_head(head, stance_set=case["stance_set"], permitted_ids=case["clue_ids_permitted"])):
            rejected.append({"id": rid, "reason": "bad_envelope", "content": head})
            continue
        if hit := match_secret(head["line"], secrets):
            rejected.append({"id": rid, "reason": "leak", "secret": hit, "line": head["line"]})
            continue
        kept.append({
            "id": rid, "npc": case["npc"], "scene": case["scene"], "slot": case["slot"], "scenario_id": case["scenario_id"],
            "language": case["language"], "player_input_type": it["player_input_type"], "topic": it["topic"],
            "offscript": it["offscript"], "template_id": f"{case['npc']}:{case['scene']}:{case['slot']}:{case['scenario_id']}:{it['player_input_type']}",
            "source": "teacher", "reviewed": False, "case": case,
            "messages": msgs + [{"role": "assistant", "content": prompt.render_envelope(head)}],
        })

    build = Path(args.job_dir) / "build"
    (build / "candidates").mkdir(parents=True, exist_ok=True)
    dump = lambda rows: "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    (build / "candidates" / "candidates.jsonl").write_text(dump(kept), encoding="utf-8")
    (build / "rejected.jsonl").write_text(dump(rejected), encoding="utf-8")
    print(json.dumps({"requested": len(items), "kept": len(kept), "rejected": len(rejected)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
