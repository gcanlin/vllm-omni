# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Task-local full1088 harness: full warmup, two measurements, keep server."""

import argparse
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/moss_shared_kv_c128_20260909"
MODEL = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"


def record(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--result-dir", type=Path, default=OUT)
    parser.add_argument("--deploy-config", type=Path, default=ROOT / "vllm_omni/deploy/moss_tts_local.yaml")
    parser.add_argument("--shared-kv", choices=("0", "1"), default="1")
    args = parser.parse_args()
    OUT = args.result_dir.resolve()
    OUT.mkdir(exist_ok=False)
    config = args.deploy_config.resolve()
    (OUT / "deploy.yaml").write_text(config.read_text())
    common = os.environ.copy()
    common.update(HF_HOME="/mnt/huggingface", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    server_env = common.copy()
    server_env.pop("VLLM_OMNI_MOSS_REF_ENCODER_DEVICE", None)
    server_env.update(CUDA_VISIBLE_DEVICES="0", VLLM_OMNI_MOSS_CODEC_SHARED_KV=args.shared_kv)
    server_cmd = [
        "vllm",
        "serve",
        MODEL,
        "--omni",
        "--host",
        "0.0.0.0",
        "--port",
        "8124",
        "--trust-remote-code",
        "--deploy-config",
        str(config),
        "--allowed-local-media-path",
        str(ROOT.parent),
    ]
    record(
        "server_command.json",
        dict(
            command=server_cmd,
            env={
                k: server_env[k]
                for k in (
                    "HF_HOME",
                    "HF_HUB_OFFLINE",
                    "TRANSFORMERS_OFFLINE",
                    "CUDA_VISIBLE_DEVICES",
                    "VLLM_OMNI_MOSS_CODEC_SHARED_KV",
                )
            },
        ),
    )
    log = (OUT / "server.log").open("w")
    server = subprocess.Popen(server_cmd, cwd=ROOT, env=server_env, stdout=log, stderr=subprocess.STDOUT)
    record("progress.json", dict(phase="startup", server_pid=server.pid))
    print(f"server_pid={server.pid} results={OUT}", flush=True)
    deadline = time.monotonic() + 1800
    while True:
        if server.poll() is not None:
            raise RuntimeError(f"Server exited {server.returncode}; see {OUT / 'server.log'}")
        try:
            with urllib.request.urlopen("http://127.0.0.1:8124/health", timeout=5) as response:
                if response.status == 200:
                    break
        except (urllib.error.URLError, TimeoutError):
            pass
        if time.monotonic() > deadline:
            raise TimeoutError("Server startup exceeded 30 minutes; inspect server.log")
        time.sleep(5)
    print("server ready", flush=True)
    client_env = common.copy()
    client_env.pop("CUDA_VISIBLE_DEVICES", None)
    client_env.update(
        SEED_TTS_EVAL_DEVICE="cuda:6",
        SEED_TTS_WER_SAVE_AUDIO_DIR=str(ROOT / "results"),
        VLLM_OMNI_BENCH_AUDIO_SAMPLE_RATE="48000",
        VLLM_OMNI_BENCH_AUDIO_CHANNELS="2",
    )
    for phase in ("warmup", "round2", "round3"):
        dest = OUT / phase
        dest.mkdir()
        command = [
            "python",
            "benchmarks/tts/bench_tts.py",
            "--host",
            "127.0.0.1",
            "--port",
            "8124",
            "--model",
            MODEL,
            "--task",
            "voice_clone",
            "--locale",
            "en",
            "--dataset-path",
            str(ROOT.parent / "seedtts_testset"),
            "--num-prompts",
            "1088",
            "--concurrency",
            str(args.concurrency),
            "--output-len",
            "256",
            "--output-dir",
            str(dest),
            "--",
            "--save-detailed",
        ]
        record(f"{phase}_command.json", command)
        record("progress.json", dict(phase=phase, server_pid=server.pid))
        print(f"starting {phase}", flush=True)
        with (OUT / f"{phase}.log").open("w") as stream:
            result = subprocess.run(command, cwd=ROOT, env=client_env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{phase} failed: {result.returncode}")
        print(f"finished {phase}", flush=True)
    record("progress.json", dict(phase="complete", server_pid=server.pid))
    print("measurements complete; keeping server alive", flush=True)
    server.wait()


if __name__ == "__main__":
    main()
