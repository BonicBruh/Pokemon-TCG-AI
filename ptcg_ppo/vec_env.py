from __future__ import annotations
import multiprocessing as mp
import traceback
from .env import CabtMatchEnv


def _worker(conn, kwargs):
    env = None
    try:
        env = CabtMatchEnv(**kwargs)
        while True:
            command, payload = conn.recv()
            if command == "reset":
                conn.send(("ok", env.reset(**(payload or {}))))
            elif command == "step":
                conn.send(("ok", env.step(payload)))
            elif command == "close":
                env.close(); conn.send(("ok", None)); break
            else:
                raise KeyError(command)
    except BaseException:
        conn.send(("error", traceback.format_exc()))
    finally:
        if env is not None: env.close()
        conn.close()


class SubprocCabtVecEnv:
    def __init__(self, num_envs: int, env_kwargs: dict, *, start_method="spawn"):
        if num_envs <= 0: raise ValueError("num_envs must be positive")
        ctx = mp.get_context(start_method)
        self.parents, self.processes = [], []
        for index in range(num_envs):
            parent, child = ctx.Pipe()
            kwargs = dict(env_kwargs); kwargs["seed"] = int(kwargs.get("seed", 0)) + index * 1009
            process = ctx.Process(target=_worker, args=(child, kwargs), daemon=True)
            process.start(); child.close()
            self.parents.append(parent); self.processes.append(process)
        self.num_envs = num_envs

    @staticmethod
    def _recv(conn):
        status, payload = conn.recv()
        if status == "error": raise RuntimeError(payload)
        return payload

    def reset(self):
        for conn in self.parents: conn.send(("reset", None))
        results = [self._recv(conn) for conn in self.parents]
        return [x[0] for x in results], [x[1] for x in results]

    def reset_at(self, indices):
        indices = list(indices)
        for i in indices: self.parents[i].send(("reset", None))
        return {i: self._recv(self.parents[i]) for i in indices}

    def step(self, actions):
        if len(actions) != self.num_envs: raise ValueError("Action batch size mismatch")
        for conn, action in zip(self.parents, actions, strict=True): conn.send(("step", action))
        results = [self._recv(conn) for conn in self.parents]
        obs, rewards, terminated, truncated, infos = zip(*results, strict=True)
        return list(obs), list(rewards), list(terminated), list(truncated), list(infos)

    def close(self):
        for conn in self.parents:
            try: conn.send(("close", None))
            except Exception: pass
        for conn in self.parents:
            try: self._recv(conn)
            except Exception: pass
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive(): process.terminate()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()
