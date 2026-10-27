"""Local, request-driven CartPole training with the study's agents."""

from __future__ import annotations

import copy
import json
import random
import uuid
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from .agents import make_agent
from .config import RunConfig, load_yaml
from .experiments.runner import EVALUATION_SEEDS
from .utils import seed_everything


class TrainingSession:
    def __init__(self, config: RunConfig):
        self.config = config
        self.generation = uuid.uuid4().hex
        torch.set_num_threads(1)
        seed_everything(config.seed)
        self.env = gym.make(config.env_id)
        self.eval_env = gym.make(config.env_id)
        self.agent = make_agent(config.method, config.typed_params, 4, 2, config.seed)
        self.observation, _ = self.env.reset(seed=config.seed)
        self.steps = 0
        self.episode = 1
        self.score = 0.0
        self.history: deque[dict[str, Any]] = deque(maxlen=1000)
        self.evaluations: list[dict[str, Any]] = []
        self.frame = self._frame(self.observation, None, False, False)

    def _frame(self, observation, action, terminated, truncated) -> dict[str, Any]:
        return dict(observation=observation.tolist(), action=action, step=self.steps,
                    episode=self.episode, score=self.score,
                    terminated=bool(terminated), truncated=bool(truncated))

    def state(self) -> dict[str, Any]:
        return dict(generation=self.generation, method=self.config.method,
                    seed=self.config.seed, budget=self.config.total_steps,
                    steps=self.steps, episode=self.episode, score=self.score,
                    complete=self.steps >= self.config.total_steps,
                    updates=int(self.agent.update_count), frame=self.frame,
                    history=list(self.history), evaluations=self.evaluations,
                    diagnostics=self.agent.diagnostics())

    def advance(self, count: int) -> dict[str, Any]:
        if type(count) is not int or not 1 <= count <= 200:
            raise ValueError("Batch size must be an integer from 1 to 200.")
        for _ in range(min(count, self.config.total_steps - self.steps)):
            action = self.agent.act(self.observation)
            next_observation, reward, terminated, truncated, _ = self.env.step(action)
            self.agent.observe(self.observation, action, float(reward), next_observation,
                               terminated, self.steps)
            self.steps += 1
            self.score += float(reward)
            self.frame = self._frame(next_observation, action, terminated, truncated)
            if terminated or truncated:
                self.agent.end_episode()
                self.history.append(dict(step=self.steps, episode=self.episode, score=self.score))
                self.episode += 1
                self.score = 0.0
                self.observation, _ = self.env.reset()
            else:
                self.observation = next_observation
        return self.state()

    def evaluate(self) -> dict[str, Any]:
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        torch_rng = torch.get_rng_state()
        agent_rng = getattr(self.agent, "rng", None)
        agent_rng_state = copy.deepcopy(agent_rng.bit_generator.state) if agent_rng else None
        generator = getattr(self.agent, "generator", None)
        generator_state = generator.get_state() if generator is not None else None
        scores = []
        frames = []
        try:
            for episode in range(self.config.eval_episodes):
                observation, _ = self.eval_env.reset(seed=EVALUATION_SEEDS[episode])
                score = 0.0
                if episode == 0:
                    frames.append(self._frame(observation, None, False, False) | {"score": 0})
                while True:
                    action = self.agent.act(observation, greedy=True)
                    observation, reward, terminated, truncated, _ = self.eval_env.step(action)
                    score += float(reward)
                    if episode == 0:
                        frames.append(self._frame(observation, action, terminated, truncated)
                                      | {"score": score})
                    if terminated or truncated:
                        break
                scores.append(score)
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_rng_state(torch_rng)
            if agent_rng is not None:
                agent_rng.bit_generator.state = agent_rng_state
            if generator is not None:
                generator.set_state(generator_state)
        evaluation = dict(step=self.steps, mean=float(np.mean(scores)), scores=scores)
        if self.evaluations and self.evaluations[-1]["step"] == self.steps:
            self.evaluations[-1] = evaluation
        else:
            self.evaluations.append(evaluation)
        return dict(state=self.state(), evaluation=evaluation, frames=frames)

    def close(self) -> None:
        self.env.close()
        self.eval_env.close()


class DemoServer(HTTPServer):
    def __init__(self, address, root: Path):
        self.root = root
        self.session: TrainingSession | None = None
        super().__init__(address, DemoHandler)

    def reset_session(self, body: dict[str, Any]) -> dict[str, Any]:
        method = body.get("method", "reinforce")
        seed = body.get("seed", 0)
        budget = body.get("budget", 100_000)
        if method not in ("tabular", "dqn", "reinforce"):
            raise ValueError("Choose Tabular Q-learning, DQN, or REINFORCE.")
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("Seed must be an integer from 0 to 4294967295.")
        if type(budget) is not int or not 100 <= budget <= 100_000:
            raise ValueError("Training budget must be an integer from 100 to 100000.")
        tuned = load_yaml(self.root / "configs" / "tuned.yaml")
        settings = load_yaml(self.root / "configs" / "study.yaml")["run"]
        config = RunConfig(method=method, seed=seed, group="demo",
                           params=tuned[method], **(settings | {"total_steps": budget}))
        new_session = TrainingSession(config)
        if self.session is not None:
            self.session.close()
        self.session = new_session
        return self.session.state()

    def server_close(self):
        if self.session is not None:
            self.session.close()
        super().server_close()


class DemoHandler(BaseHTTPRequestHandler):
    server: DemoServer

    def log_message(self, format, *args):
        pass

    def _send(self, status: int, data: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; object-src 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status: int, data):
        self._send(status, json.dumps(data, allow_nan=False).encode(), "application/json")

    def _local_request(self) -> bool:
        host = self.headers.get("Host", "")
        expected = f"127.0.0.1:{self.server.server_port}"
        origin = self.headers.get("Origin")
        return (host in (expected, f"localhost:{self.server.server_port}")
                and (origin is None or origin == f"http://{host}")
                and self.headers.get("Sec-Fetch-Site") != "cross-site")

    def do_GET(self):
        if not self._local_request():
            self._json(403, {"error": "Open the demo through its local URL."})
            return
        assets = {"/": ("demo.html", "text/html; charset=utf-8"),
                  "/demo.js": ("demo.js", "text/javascript; charset=utf-8"),
                  "/demo.css": ("demo.css", "text/css; charset=utf-8")}
        if self.path in assets:
            filename, content_type = assets[self.path]
            self._send(200, (Path(__file__).parent / "demo_assets" / filename).read_bytes(), content_type)
        elif self.path == "/api/state":
            self._json(200, self.server.session.state() if self.server.session else None)
        elif self.path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
        else:
            self._json(404, {"error": "Page not found."})

    def do_POST(self):
        if not self._local_request():
            self._json(403, {"error": "Requests must come from this local demo."})
            return
        if self.path not in ("/api/reset", "/api/advance", "/api/evaluate"):
            self._json(404, {"error": "Action not found."})
            return
        try:
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                raise ValueError("Send a JSON request.")
            length = int(self.headers.get("Content-Length", "0"))
            if not 1 <= length <= 2048:
                raise ValueError("Request must contain 1 to 2048 bytes.")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("Request must be a JSON object.")
            if self.path == "/api/reset":
                result = self.server.reset_session(body)
            else:
                session = self.server.session
                if session is None or body.get("generation") != session.generation:
                    self._json(409, {"error": "This session changed. Reload the page to reconnect."})
                    return
                result = (session.advance(body.get("count")) if self.path == "/api/advance"
                          else session.evaluate())
            self._json(200, result)
        except (ValueError, TypeError, OverflowError) as error:
            self._json(400, {"error": str(error)})


def serve_demo(root: Path, port: int = 8765, open_browser: bool = True) -> None:
    if not 0 <= port <= 65535:
        raise ValueError("Port must be from 0 to 65535.")
    with DemoServer(("127.0.0.1", port), root) as server:
        url = f"http://127.0.0.1:{server.server_port}"
        print(f"CartPole live training demo at {url}", flush=True)
        print("Press Ctrl+C to stop. One presentation session is shared by local tabs.", flush=True)
        if open_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
