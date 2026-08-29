"""hermetic tests for TinyFish free-web grounding: a fake Search API server,
no network access, no real key required (tests/test_tinyfish.py)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from simurg import veritas_dashboard as vd
from simurg.veritas.tinyfish import search, snippets

RESULTS = [
    {"position": 1, "site_name": "example.com",
     "title": "Y2K bug overview",
     "snippet": "The Y2K bug was a class of bugs in 1999 software dating logic.",
     "url": "https://example.com/y2k"},
    {"position": 2, "site_name": "history.org",
     "title": "Y2K timeline",
     "snippet": "The Y2K bug was fixed before the year 2000 deadline.",
     "url": "https://history.org/y2k"},
]


class FakeTinyFish(BaseHTTPRequestHandler):
    results = RESULTS
    fail = False
    seen_keys = []
    seen_paths = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        FakeTinyFish.seen_keys.append(self.headers.get("X-API-Key"))
        FakeTinyFish.seen_paths.append(self.path)
        if self.fail:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"boom")
            return
        results = self.results
        body = json.dumps({"query": "q", "results": results,
                           "total_results": len(results), "page": 0}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def tf(monkeypatch):
    FakeTinyFish.results = RESULTS
    FakeTinyFish.fail = False
    FakeTinyFish.seen_keys = []
    FakeTinyFish.seen_paths = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeTinyFish)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    monkeypatch.setenv("TINYFISH_SEARCH_URL",
                       "http://127.0.0.1:{}".format(httpd.server_address[1]))
    yield FakeTinyFish
    httpd.shutdown()


# ── client unit behaviour ──────────────────────────────────────────────────────

def test_no_key_returns_empty_without_network(monkeypatch):
    monkeypatch.delenv("TINYFISH_API_KEY", raising=False)
    monkeypatch.setenv("TINYFISH_SEARCH_URL", "http://127.0.0.1:1")  # unroutable
    assert search("y2k bug") == []
    assert snippets("y2k bug") == []


def test_search_returns_ranked_results(tf, monkeypatch):
    monkeypatch.setenv("TINYFISH_API_KEY", "tf-test-key")
    got = search("y2k bug", k=5)
    assert len(got) == 2
    assert got[0]["title"] == "Y2K bug overview"
    assert got[0]["url"] == "https://example.com/y2k"
    assert got[1]["site_name"] == "history.org"
    assert FakeTinyFish.seen_keys == ["tf-test-key"]
    assert "query=y2k+bug" in FakeTinyFish.seen_paths[0]


def test_explicit_key_beats_environment(tf, monkeypatch):
    monkeypatch.setenv("TINYFISH_API_KEY", "env-key")
    got = search("y2k bug", api_key="explicit-key")
    assert got
    assert FakeTinyFish.seen_keys == ["explicit-key"]


def test_snippets_merge_title_and_text(tf, monkeypatch):
    monkeypatch.setenv("TINYFISH_API_KEY", "tf-test-key")
    ev = snippets("y2k bug")
    assert len(ev) == 2
    assert ev[0].startswith("Y2K bug overview")
    assert "1999" in ev[0]


def test_k_limits_results(tf, monkeypatch):
    monkeypatch.setenv("TINYFISH_API_KEY", "tf-test-key")
    assert len(search("y2k bug", k=1)) == 1


def test_server_error_degrades_to_empty(tf, monkeypatch):
    monkeypatch.setenv("TINYFISH_API_KEY", "tf-test-key")
    FakeTinyFish.fail = True
    assert search("y2k bug") == []
    assert snippets("y2k bug") == []


def test_empty_result_set(tf, monkeypatch):
    monkeypatch.setenv("TINYFISH_API_KEY", "tf-test-key")
    FakeTinyFish.results = []
    assert search("y2k bug") == []


# ── grounding wiring in the veritas dashboard ──────────────────────────────────

@pytest.fixture()
def tf_env(monkeypatch, tf):
    """TinyFish wired into _ground_check; Wikipedia stubbed for hermeticity."""
    monkeypatch.setenv("TINYFISH_API_KEY", "tf-test-key")
    monkeypatch.setattr(vd, "_wiki_hits", lambda q: (0, ""))
    return tf


def test_web_snippets_prefers_tinyfish_over_ddg(tf_env, monkeypatch):
    monkeypatch.setattr(vd, "_ddg_snippets", lambda q, k=6: ["should-not-be-used"])
    snips, src = vd._web_snippets("y2k bug")
    assert src == "tinyfish"
    assert len(snips) == 2
    assert snips[0].startswith("Y2K bug overview")


def test_web_snippets_keyless_falls_back_to_ddg(monkeypatch):
    monkeypatch.delenv("TINYFISH_API_KEY", raising=False)
    monkeypatch.setattr(vd, "_ddg_snippets", lambda q, k=6: ["ddg-snippet"])
    snips, src = vd._web_snippets("y2k bug")
    assert (snips, src) == (["ddg-snippet"], "ddg")


def test_web_snippets_tinyfish_failure_falls_back_to_ddg(tf_env, monkeypatch):
    FakeTinyFish.fail = True
    monkeypatch.setattr(vd, "_ddg_snippets", lambda q, k=6: ["ddg-snippet"])
    snips, src = vd._web_snippets("y2k bug")
    assert (snips, src) == (["ddg-snippet"], "ddg")


def test_ground_check_fabricated_subject_abstains(tf_env, monkeypatch):
    FakeTinyFish.results = []                     # nothing on the free web
    monkeypatch.setattr(vd, "_ddg_snippets", lambda q, k=6: [])  # no fallback evidence
    out = vd._ground_check(
        "When did the Zorbachian treaty of 1874 collapse?",
        "The Zorbachian treaty collapsed in March 1874.")
    assert out["decision"] == "abstain"
    assert "no knowledge-base or web record" in out["reason"]


def test_ground_check_tinyfish_date_contradiction(tf_env):
    FakeTinyFish.results = [
        {"position": 1, "site_name": "example.com",
         "title": "Launch date",
         "snippet": "The program's first public launch happened in March 2024.",
         "url": "https://example.com/launch"},
        {"position": 2, "site_name": "news.org",
         "title": "Launch coverage",
         "snippet": "Reports confirm the March 2024 launch date.",
         "url": "https://news.org/launch"},
    ]
    out = vd._ground_check("When did the program first launch?",
                           "The program first launched in July 2023.")
    assert out["decision"] == "abstain"
    assert out["reason"] == "date CONTRADICTS the evidence"
    assert out["source"] == "tinyfish"
    assert "March 2024" in out["evidence_date"]


def test_ground_check_tinyfish_attested_subject(tf_env):
    out = vd._ground_check("What was the Y2K bug?",
                           "The Y2K bug was a class of software dating bugs.")
    assert out["decision"] == "confident"
    assert out["source"] == "tinyfish+wiki"
    assert out["hits"] == 2
