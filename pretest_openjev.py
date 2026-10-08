#!/usr/bin/env python3
"""Pre-test 'describe-the-concept-first' questions for OpenJev on 6 error docs.

V0  baseline sharp question
V10 concept framed with few details (broader + altLabels)
V11 concept framed with many details (broader + narrower + alts + def/scope)
"""

import requests
import rdflib

BASE = "http://localhost:8080/v1/systemone"
MODEL = "openjev"
TIMEOUT = 300
SKOS = rdflib.Namespace("http://www.w3.org/2004/02/skos/core#")

BASE_RULES = (
    "You are a librarian performing topical subject indexing. "
    "A document can have several central subjects. "
    "Do NOT assign the document's own format or genre as a subject "
    "(a book of recipes is not about 'cookbooks', a pattern book is not "
    "about 'handicraft patterns', a travel book is not about "
    "'travelogues').\n\nDocument to be indexed:\n"
)

DOCS = {
    "9789513979294": {
        "fps": ["literary criticism", "literature", "criticism"],
        "golds": ["textual analysis", "novels", "Finland-Swedish literature"],
    },
    "9789527258132": {
        "fps": ["art", "visual arts", "contemporary art"],
        "golds": ["photographers", "living environment", "residential environment"],
    },
    "9789518775754": {
        "fps": ["espionage", "intelligence and reconnaissance", "military"],
        "golds": ["spies", "history"],
    },
    "9789527381212": {
        "fps": ["food recipes", "food preparation", "vegetarian food"],
        "golds": ["vegetarian dishes"],
    },
    "9789528024897": {
        "fps": ["tourism", "foreign travel", "tourist destinations"],
        "golds": ["voyages and travels", "sights (attractions)"],
    },
    "9789528024552": {
        "fps": ["language learning", "English language", "glossaries"],
        "golds": ["Finnish language"],
    },
}


class Vocab:
    def __init__(self, labels: set[str]) -> None:
        self.g = rdflib.Graph()
        self.g.parse("yso-subjects.ttl", format="turtle")
        self.en_label: dict[str, str] = {}
        for c in self.g.subjects(rdflib.RDF.type, SKOS.Concept):
            for l in self.g.objects(c, SKOS.prefLabel):
                if l.language == "en":
                    self.en_label[str(c)] = str(l)
        self.info: dict[str, dict] = {}
        for c in self.g.subjects(rdflib.RDF.type, SKOS.Concept):
            for l in self.g.objects(c, SKOS.prefLabel):
                if str(l) in labels and str(l) not in self.info:
                    self.info[str(l)] = self._concept_info(c)

    def _lbl(self, uri: str) -> str:
        return self.en_label.get(uri, "")

    def _concept_info(self, c) -> dict:
        broader = sorted({self._lbl(str(b)) for b in self.g.objects(c, SKOS.broader)})
        narrower = sorted({self._lbl(str(n)) for n in self.g.objects(c, SKOS.narrower)})
        alts = [str(a) for a in self.g.objects(c, SKOS.altLabel)
                if a.language in ("en", None)][:3]
        defs = [str(d) for d in self.g.objects(c, SKOS.definition)]
        scope = [str(d) for d in self.g.objects(c, SKOS.scopeNote)]
        return {"broader": broader, "narrower": narrower[:4], "alts": alts,
                "def": defs[0] if defs else "", "scope": scope[0] if scope else ""}


def instr_for(variant: str, label: str, info: dict) -> str:
    i = info.get(label, {}) if "broader" not in info else info
    info = i
    core = (
        "Is '{label}' a central subject of this document - one a "
        "librarian would assign as a primary heading - not merely a "
        "passing or incidental mention?"
    )
    if variant == "V0":
        return core.format(label=label)
    parts = []
    if variant == "V10":
        if info["broader"]:
            parts.append(f"part of the field '{info['broader'][0]}'")
        if info["alts"]:
            parts.append(f"also known as {', '.join(info['alts'])}")
    else:  # V11
        if info["broader"]:
            parts.append(f"part of the field '{', '.join(info['broader'][:2])}'")
        if info["narrower"]:
            parts.append(f"more specific subjects under it include "
                         f"{', '.join(info['narrower'])}")
        if info["alts"]:
            parts.append(f"also known as {', '.join(info['alts'])}")
        if info["def"]:
            parts.append(f"defined as: {info['def']}")
        elif info["scope"]:
            parts.append(f"scope: {info['scope']}")
    desc = ", ".join(parts) if parts else "a library subject heading"
    return (
        f"'{label}' is a library subject heading, {desc}. "
        + core.format(label=label)
    )


def ask(doc: str, variant: str, info: dict) -> dict[str, float]:
    text = open(f"jev-test-sets/en/{doc}.txt", encoding="utf-8").read()
    labels = DOCS[doc]["fps"] + DOCS[doc]["golds"]
    questions = {
        str(idx): {"type": "noul", "instructions": instr_for(variant, lab, info)}
        for idx, lab in enumerate(labels)
    }
    payload = {"model": MODEL, "state": BASE_RULES + text, "questions": questions}
    resp = requests.post(BASE, json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    ans = resp.json()["answers"]
    return {lab: ans[str(idx)]["noul"] for idx, lab in enumerate(labels)}


def score(scores: dict[str, float], doc: str) -> tuple[float, float]:
    golds = DOCS[doc]["golds"]
    fps = DOCS[doc]["fps"]
    g = [scores[l] for l in golds if l in scores]
    f = [scores[l] for l in fps if l in scores]
    return sum(g) / len(g), sum(f) / len(f)


def main() -> None:
    labels = {l for d in DOCS.values() for l in d["fps"] + d["golds"]}
    print("loading YSO vocabulary...", flush=True)
    vocab = Vocab(labels)
    info = vocab.info
    missing = labels - set(info)
    if missing:
        print(f"WARNING: no vocab info for: {missing}")
    variants = ["V0", "V10", "V11"]
    all_scores: dict[str, dict[str, dict[str, float]]] = {}
    for doc, d in DOCS.items():
        print(f"\n== {doc}", flush=True)
        for variant in variants:
            scores = ask(doc, variant, info)
            all_scores.setdefault(variant, {})[doc] = scores
            mg, mf = score(scores, doc)
            print(f"  {variant}: gold={mg:.3f} fp={mf:.3f} margin={mg - mf:+.3f}",
                  flush=True)
            for lab in d["golds"] + d["fps"]:
                tag = "G" if lab in d["golds"] else "f"
                print(f"     [{tag}] {lab:35s} {scores[lab]:.3f}")
    print("\n== aggregate (mean over 6 docs) ==")
    for variant in variants:
        mg = [score(s, doc)[0] for doc, s in all_scores[variant].items()]
        mf = [score(s, doc)[1] for doc, s in all_scores[variant].items()]
        print(f"  {variant}: gold={sum(mg)/len(mg):.3f} fp={sum(mf)/len(mf):.3f} "
              f"margin={sum(mg)/len(mg) - sum(mf)/len(mf):+.3f}")
    print("\n== sample question texts (V10/V11) ==")
    for lab in ["espionage", "spies", "food recipes", "voyages and travels"]:
        if lab in info:
            print(f"\n--- {lab}")
            print("V10:", instr_for("V10", lab, info[lab]))
            print("V11:", instr_for("V11", lab, info[lab]))


if __name__ == "__main__":
    main()
