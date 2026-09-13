"""Banned-lexicon and leaked-identity scanner.

Every CICIDS2017 attack is a documented, named tool — an LLM that names
one is reciting a textbook entry rather than reasoning from evidence,
and a real zero-day has no textbook entry to recite. This module scans
every agent output (never sees the ground-truth label; scans the
model's own free-text fields only) and reports what leaked.

This is deliberately also a *result*, not just a guard: per-agent hit
rates go straight into the paper (see eval/agent_eval.py). Contamination
found here is never silently stripped or retried — that would hide the
finding, not fix it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

#: named attack tools/families/malware from CICIDS2017 and common
#: adjacent tooling, plus the class-label words themselves.
ATTACK_NAME_TERMS = [
    "nmap", "masscan", "zmap", "zenmap",
    "slowloris", "slowhttptest", "slow http test",
    "hulk", "goldeneye", "rudy", "r-u-dead-yet",
    "mirai", "zeus", "ares",
    "patator", "ftp-patator", "ssh-patator", "hydra", "medusa",
    "loic", "hoic",
    "heartbleed", "eternalblue", "shellshock",
    "metasploit", "nikto", "sqlmap", "burp suite", "burpsuite",
    "cicids", "cicflowmeter",
]

#: category / framing words the prompts must never use and the model
#: must never echo back.
FRAMING_TERMS = [
    "attack", "attacker", "malicious", "malware", "intrusion",
    "threat", "threat actor", "exploit", "exploitation", "adversary",
    "adversarial", "hacker", "hacking", "cyberattack", "cyber attack",
    "command and control", "botnet", "bot herder",
    "ddos", "dos attack", "denial of service",
    "port scan", "portscan", "network scan", "reconnaissance scan",
    "brute force", "brute-force", "credential stuffing",
    "sql injection", "sqli", "cross-site scripting", "xss",
    "infiltration", "web attack", "zero-day", "zero day", "0day",
]

BANNED_TERMS: List[str] = ATTACK_NAME_TERMS + FRAMING_TERMS

_IPV4_RE = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
_MAC_RE = re.compile(r"\b[0-9A-Fa-f]{2}(?::[0-9A-Fa-f]{2}){5}\b")
_ABSOLUTE_TIMESTAMP_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\b|\b\d{1,2}/\d{1,2}/\d{4}\b"
)


def _word_pattern(term: str) -> re.Pattern:
    escaped = re.escape(term)
    return re.compile(rf"(?<![a-zA-Z0-9]){escaped}(?![a-zA-Z0-9])", re.IGNORECASE)


_TERM_PATTERNS = {term: _word_pattern(term) for term in BANNED_TERMS}


@dataclass
class ContaminationReport:
    lexicon_hits: List[str] = field(default_factory=list)
    ip_hits: List[str] = field(default_factory=list)
    mac_hits: List[str] = field(default_factory=list)
    timestamp_hits: List[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not (self.lexicon_hits or self.ip_hits or self.mac_hits or self.timestamp_hits)

    def to_dict(self) -> dict:
        return {
            "lexicon_hits": self.lexicon_hits,
            "ip_hits": self.ip_hits,
            "mac_hits": self.mac_hits,
            "timestamp_hits": self.timestamp_hits,
        }


def scan_text(text: str) -> ContaminationReport:
    if not text:
        return ContaminationReport()
    lexicon = sorted({term for term, pat in _TERM_PATTERNS.items() if pat.search(text)})
    ips = sorted(set(_IPV4_RE.findall(text)))
    macs = sorted(set(_MAC_RE.findall(text)))
    timestamps = sorted(set(_ABSOLUTE_TIMESTAMP_RE.findall(text)))
    return ContaminationReport(
        lexicon_hits=lexicon, ip_hits=ips, mac_hits=macs, timestamp_hits=timestamps
    )


def _texts_from_agent_response(response) -> List[str]:
    """Pull every free-text field out of a ClaimsResponse /
    HypothesisResponse / A5Response-shaped object. Feature names
    and enum-valued fields are excluded deliberately — they're drawn
    from closed, pre-approved vocabularies, not model-generated prose."""
    texts: List[str] = []
    for claim in getattr(response, "claims", []) or []:
        texts.append(claim.statement)
    for hyp in getattr(response, "hypotheses", []) or []:
        texts.append(hyp.description)
        if getattr(hyp, "prediction", None):
            texts.append(hyp.prediction)
    rationale = getattr(response, "rationale", None)
    if rationale:
        texts.append(rationale)
    return texts


def scan_agent_response(response) -> ContaminationReport:
    """Scan every free-text field of one agent's parsed response.
    Returns the union of hits across all its fields."""
    combined = ContaminationReport()
    seen_lexicon, seen_ip, seen_mac, seen_ts = set(), set(), set(), set()
    for text in _texts_from_agent_response(response):
        r = scan_text(text)
        seen_lexicon.update(r.lexicon_hits)
        seen_ip.update(r.ip_hits)
        seen_mac.update(r.mac_hits)
        seen_ts.update(r.timestamp_hits)
    combined.lexicon_hits = sorted(seen_lexicon)
    combined.ip_hits = sorted(seen_ip)
    combined.mac_hits = sorted(seen_mac)
    combined.timestamp_hits = sorted(seen_ts)
    return combined


def scan_prompt(prompt_text: str) -> ContaminationReport:
    """Same scan applied to a prompt template itself — used by tests to
    assert prompt files never leak a banned term into the framing."""
    return scan_text(prompt_text)


@dataclass
class ContaminationSummary:
    """Per-agent aggregate across many records — what eval/agent_eval.py
    reports: this is the result, not just the guard."""

    per_agent_responses: int = 0
    per_agent_contaminated: int = 0
    hit_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def contamination_rate(self) -> float:
        if self.per_agent_responses == 0:
            return 0.0
        return self.per_agent_contaminated / self.per_agent_responses

    def add(self, report: ContaminationReport) -> None:
        self.per_agent_responses += 1
        if not report.is_clean:
            self.per_agent_contaminated += 1
        for term in report.lexicon_hits:
            self.hit_counts[term] = self.hit_counts.get(term, 0) + 1


def summarize(reports_by_agent: Dict[str, List[ContaminationReport]]) -> Dict[str, ContaminationSummary]:
    out: Dict[str, ContaminationSummary] = {}
    for agent, reports in reports_by_agent.items():
        summary = ContaminationSummary()
        for r in reports:
            summary.add(r)
        out[agent] = summary
    return out
