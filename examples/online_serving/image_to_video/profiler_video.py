#!/usr/bin/env python3
"""Script to profile a single video generation request.

Usage:
    python profiler_script.py --host localhost --port 8091
"""

import argparse
import base64
import json
import requests


def start_profile(base_url: str, stages: list[int] | None = None) -> dict:
    """Start profiling for specified stages."""
    url = f"{base_url}/start_profile"
    payload = {"stages": stages} if stages else {}
    response = requests.post(url, json=payload if payload else None)
    response.raise_for_status()
    print(f"[+] Started profiling for stages: {stages if stages else 'all'}")
    return response.json() if response.text else {}


def stop_profile(base_url: str, stages: list[int] | None = None) -> dict:
    """Stop profiling for specified stages."""
    url = f"{base_url}/stop_profile"
    payload = {"stages": stages} if stages else {}
    response = requests.post(url, json=payload if payload else None)
    response.raise_for_status()
    print(f"[+] Stopped profiling for stages: {stages if stages else 'all'}")
    return response.json() if response.text else {}


def send_video_request(
    base_url: str,
    prompt: str,
    input_reference: str,
    negative_prompt: str = " ",
    size: str = "48x64",
    seconds: int = 1,
    fps: int = 16,
    num_frames: int = 2,
    guidance_scale: float = 3.5,
    guidance_scale_2: float = 3.5,
    flow_shift: float = 5.0,
    num_inference_steps: int = 2,
    seed: int = 42,
) -> dict:
    """Send a video generation request with multipart form data."""
    url = f"{base_url}/v1/videos"

    files = {}
    data = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "size": size,
        "seconds": str(seconds),
        "fps": str(fps),
        "num_frames": str(num_frames),
        "guidance_scale": str(guidance_scale),
        "guidance_scale_2": str(guidance_scale_2),
        "flow_shift": str(flow_shift),
        "num_inference_steps": str(num_inference_steps),
        "seed": str(seed),
    }

    if input_reference:
        files["input_reference"] = open(input_reference, "rb")

    print(f"[*] Sending video request: prompt={prompt[:50]}...")
    print(f"    size={size}, seconds={seconds}, fps={fps}, num_frames={num_frames}")
    print(f"    guidance_scale={guidance_scale}, guidance_scale_2={guidance_scale_2}")
    print(f"    flow_shift={flow_shift}, steps={num_inference_steps}, seed={seed}")

    try:
        response = requests.post(url, data=data, files=files if files else None)
        response.raise_for_status()
        result = response.json()
        print("[+] Response received")
        return result
    finally:
        for f in files.values():
            f.close()


def main():
    parser = argparse.ArgumentParser(
        description="Profile a single video generation request"
    )
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8091, help="Server port")
    parser.add_argument("--stages", type=int, nargs="*", default=[0, 1],
                        help="Stage IDs to profile (default: [0, 1])")
    parser.add_argument("--prompt", default="一只棕色野兔的正面特写镜头",
                        help="Prompt for video generation")
    parser.add_argument("--negative-prompt", default=" ",
                        help="Negative prompt")
    parser.add_argument("--input-reference", default="test1.jpeg",
                        help="Path to reference image file")
    parser.add_argument("--size", default="48x64",
                        help="Video size (default: 48x64)")
    parser.add_argument("--seconds", type=int, default=1,
                        help="Video duration in seconds")
    parser.add_argument("--fps", type=int, default=16, help="Frames per second")
    parser.add_argument("--num-frames", type=int, default=2,
                        help="Number of frames")
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--guidance-scale-2", type=float, default=3.5)
    parser.add_argument("--flow-shift", type=float, default=5.0)
    parser.add_argument("--num-inference-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="wan22_i2v_output.mp4",
                        help="Output file path")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    stages = args.stages if args.stages else None

    print(f"[*] Connecting to {base_url}")
    print(f"[*] Profiling stages: {stages if stages else 'all'}")
    print("-" * 50)

    try:
        # 1. Start profiling
        start_result = start_profile(base_url, stages)
        if start_result:
            print(f"    Result: {json.dumps(start_result, indent=2)}")

        # 2. Send video generation request
        print("-" * 50)
        result = send_video_request(
            base_url,
            prompt=args.prompt,
            input_reference=args.input_reference,
            negative_prompt=args.negative_prompt,
            size=args.size,
            seconds=args.seconds,
            fps=args.fps,
            num_frames=args.num_frames,
            guidance_scale=args.guidance_scale,
            guidance_scale_2=args.guidance_scale_2,
            flow_shift=args.flow_shift,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
        )

        # Save video output
        if "data" in result and result["data"]:
            b64_data = result["data"][0].get("b64_json", "")
            if b64_data:
                video_bytes = base64.b64decode(b64_data)
                with open(args.output, "wb") as f:
                    f.write(video_bytes)
                print(f"[+] Video saved to {args.output}")
            else:
                print("[!] No b64_json in response data")
        else:
            print(f"[!] Unexpected response: {json.dumps(result, indent=2)[:500]}")

        # 3. Stop profiling
        print("-" * 50)
        stop_result = stop_profile(base_url, stages)
        if stop_result:
            print(f"    Result: {json.dumps(stop_result, indent=2)}")

        print("-" * 50)
        print("[+] Profiling complete! Check ./test-omni for trace files.")

    except requests.exceptions.ConnectionError:
        print(f"[!] Error: Could not connect to {base_url}")
        print("    Make sure the server is running.")
    except requests.exceptions.HTTPError as e:
        print(f"[!] HTTP Error: {e}")
        print(f"    Response: {e.response.text if e.response else 'N/A'}")


if __name__ == "__main__":
    main()
