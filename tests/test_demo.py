from __future__ import annotations

import http.client
import json
import threading
from pathlib import Path

import pytest

from rlcompare.config import RunConfig
from rlcompare.demo import DemoServer, TrainingSession


def run_session(method, *, evaluate=False, batch=200):
    session = TrainingSession(RunConfig(method=method, seed=7, total_steps=1200,
                                       eval_episodes=2))
    try:
        for _ in range(600 // batch):
            session.advance(batch)
        if evaluate:
            before = session.state()
            result = session.evaluate()
            after = session.state()
            assert after["steps"] == before["steps"]
            assert after["updates"] == before["updates"]
            assert after["history"] == before["history"]
            assert after["frame"] == before["frame"]
            assert len(result["evaluation"]["scores"]) == 2
            assert result["frames"][-1]["score"] == result["evaluation"]["scores"][0]
            assert all(len(frame["observation"]) == 4 for frame in result["frames"])
        for _ in range(600 // batch):
            session.advance(batch)
        return session.state()
    finally:
        session.close()


@pytest.mark.parametrize("method", ["tabular", "dqn", "reinforce"])
def test_batch_size_changes_only_display_sampling(method):
    batched = run_session(method)
    individual = run_session(method, batch=1)
    assert batched["history"] == individual["history"]
    assert batched["frame"] == individual["frame"]
    assert batched["diagnostics"] == individual["diagnostics"]
    assert batched["updates"] > 0


@pytest.mark.parametrize("method", ["tabular", "dqn", "reinforce"])
def test_evaluation_does_not_change_subsequent_training(method):
    evaluated = run_session(method, evaluate=True)
    uninterrupted = run_session(method)
    assert evaluated["history"] == uninterrupted["history"]
    assert evaluated["frame"] == uninterrupted["frame"]
    assert evaluated["diagnostics"] == uninterrupted["diagnostics"]
    assert evaluated["updates"] == uninterrupted["updates"]


def test_budget_stops_learning_and_evaluation_remains_available():
    session = TrainingSession(RunConfig(method="tabular", seed=0, total_steps=101,
                                       eval_episodes=2))
    try:
        state = session.advance(200)
        assert state["complete"]
        assert state["steps"] == 101
        assert session.advance(1) == state
        assert session.evaluate()["state"]["steps"] == 101
        assert session.evaluate()["state"]["evaluations"][-1]["step"] == 101
        assert len(session.evaluations) == 1
    finally:
        session.close()


@pytest.fixture
def demo_server():
    root = Path(__file__).resolve().parents[1]
    with DemoServer(("127.0.0.1", 0), root) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server
        finally:
            server.shutdown()
            thread.join(timeout=5)


def call(server, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
    try:
        connection.request("POST" if body is not None else "GET", path,
                           body=json.dumps(body) if body is not None else None,
                           headers=headers or {"Content-Type": "application/json"})
        response = connection.getresponse()
        data = response.read()
        return response.status, json.loads(data) if "json" in response.getheader("Content-Type", "") else data
    finally:
        connection.close()


def test_http_session_reset_and_stale_request_rejection(demo_server):
    status, first = call(demo_server, "/api/reset", {"method": "tabular", "seed": 3, "budget": 100})
    assert status == 200
    status, advanced = call(demo_server, "/api/advance", {"generation": first["generation"], "count": 20})
    assert status == 200 and advanced["steps"] == 20
    status, second = call(demo_server, "/api/reset", {"method": "tabular", "seed": 3, "budget": 100})
    assert status == 200 and second["steps"] == 0
    assert second["generation"] != first["generation"]
    assert second["frame"] == first["frame"]
    status, _ = call(demo_server, "/api/advance", {"generation": first["generation"], "count": 1})
    assert status == 409


@pytest.mark.parametrize("body", [
    {"method": "invalid"}, {"seed": -1}, {"seed": True}, {"seed": 1.5},
    {"budget": 0}, {"budget": 100001}, {"budget": "1000"},
])
def test_http_rejects_invalid_settings(demo_server, body):
    status, result = call(demo_server, "/api/reset", body)
    assert status == 400
    assert "error" in result
    assert demo_server.session is None


def test_http_blocks_foreign_origin_and_serves_only_demo_assets(demo_server):
    status, _ = call(demo_server, "/api/reset", {},
                     {"Content-Type": "application/json", "Origin": "https://example.com"})
    assert status == 403
    assert call(demo_server, "/README.md")[0] == 404
    assert call(demo_server, "/../configs/tuned.yaml")[0] == 404
    for path in ["/", "/demo.js", "/demo.css"]:
        status, data = call(demo_server, path)
        assert status == 200 and len(data) > 100
