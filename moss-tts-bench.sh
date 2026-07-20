CUDA_VISIBLE_DEVICES=7  vllm serve OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
  --omni \
  --host 0.0.0.0     --port 8123     --trust-remote-code     --deploy-config vllm_omni/deploy/moss_tts_local.yaml --allowed-local-media-path /root/vllm-omni-workspace

vllm bench serve --omni \
    --host 127.0.0.1 --port 8123 \
    --model OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
    --backend openai-audio-speech \
    --endpoint /v1/audio/speech \
    --dataset-name seed-tts \
    --dataset-path /root/vllm-omni-workspace/seedtts_testset \
    --seed-tts-locale en \
    --hf-output-len 128 \
    --num-prompts 50 \
    --num-warmups 3 \
    --max-concurrency 4 \
    --request-rate inf \
    --percentile-metrics ttft,e2el,audio_rtf,audio_ttfp,audio_duration,audio_underrun \
    --save-result \
    --result-dir /root/vllm-omni-workspace/vllm-omni/results/moss-local-v15


curl  -N http://127.0.0.1:8123/v1/audio/speech     -H 'Content-Type: application/json'     -o /root/vllm-omni-workspace/moss_local_stream.wav     -d '{
      "model": "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
      "input": "The rain had been falling since late afternoon, soft at first, then steadily heavier, until the entire city seemed to blur behind a silver curtain. By the time Daniel reached the old train station, the streets were shining like black glass, reflecting the yellow glow of streetlamps and the restless movement of passing cars. He pulled his coat tighter around his shoulders and stepped under the wide iron roof of the platform, where the air smelled of wet stone, engine smoke, and coffee from a small kiosk near the entrance.",
      "ref_audio": "file:///root/vllm-omni-workspace/vllm-omni/local_v15_test.wav",
      "stream": true,
      "stream_format": "audio",
      "response_format": "wav"
    }'

curl  -N http://127.0.0.1:8123/v1/audio/speech     -H 'Content-Type: application/json'     -o /root/vllm-omni-workspace/moss_local_stream.wav     -d '{
      "model": "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
      "input": "The rain had been falling since late afternoon, soft at first, then steadily heavier, until the entire city seemed to blur behind a silver curtain. By the time Daniel reached the old train station, the streets were shining like black glass, reflecting the yellow glow of streetlamps and the restless movement of passing cars. He pulled his coat tighter around his shoulders and stepped under the wide iron roof of the platform, where the air smelled of wet stone, engine smoke, and coffee from a small kiosk near the entrance.",
      "ref_audio": "file:///root/vllm-omni-workspace/vllm-omni/local_v15_test.wav",
      "stream": false,
      "response_format": "wav"
    }'

# bad case
curl  -N http://127.0.0.1:8123/v1/audio/speech     -H 'Content-Type: application/json'     -o /root/vllm-omni-workspace/moss_local_stream.wav     -d '{
      "model": "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
      "input": "Hello, I am canlin guo. How are you?",
      "ref_audio": "file:///root/vllm-omni-workspace/vllm-omni/local_v15_test.wav",
      "stream": true
    }'

curl  -N http://127.0.0.1:8123/v1/audio/speech     -H 'Content-Type: application/json'     -o /root/vllm-omni-workspace/moss_local_stream.wav     -d '{
      "model": "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
      "input": "Later, I found out he was a eunuch, and I felt the sky fall?",
      "ref_audio": "file:///root/vllm-omni-workspace/vllm-omni/local_v15_test.wav",
      "stream": true
    }'

vllm bench serve --omni \
    --host 127.0.0.1 --port 8123 \
    --model OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
    --backend openai-audio-speech \
    --endpoint /v1/audio/speech \
    --dataset-name seed-tts \
    --dataset-path /root/vllm-omni-workspace/seedtts_testset \
    --seed-tts-locale en \
    --num-prompts 50 \
    --num-warmups 3 \
    --max-concurrency 4 \
    --request-rate inf \
    --percentile-metrics ttft,e2el,audio_rtf,audio_ttfp,audio_duration,audio_underrun \
    --save-result \
    --result-dir /root/vllm-omni-workspace/vllm-omni/results/moss-local-v15

# SEED_TTS_EVAL_DEVICE=cuda:6 \
# PYTHONPATH=/root/vllm-omni-workspace/vllm-omni \
# python benchmarks/tts/bench_tts.py \
#     --model OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
#     --task voice_clone \
#     --locale en \
#     --concurrency 1 \
#     --num-prompts 200 \
#     --dataset-path /root/vllm-omni-workspace/seedtts_testset \
#     --wer-eval \
#     --host 127.0.0.1 \
#     --port 8123 \
#     --output-dir /root/vllm-omni-workspace/vllm-omni/results/moss_local_seedtts_acc

SEED_TTS_EVAL_DEVICE=cuda:6 \
SEED_TTS_WER_SAVE_AUDIO_DIR=/root/vllm-omni-workspace/vllm-omni/results/moss_local_debug/refactor \
VLLM_OMNI_BENCH_AUDIO_SAMPLE_RATE=48000 \
VLLM_OMNI_BENCH_AUDIO_CHANNELS=2 \
PYTHONPATH=/root/vllm-omni-workspace/vllm-omni \
python benchmarks/tts/bench_tts.py \
  --host 127.0.0.1 \
  --port 8123 \
  --model OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
  --task voice_clone \
  --locale en \
  --dataset-path /root/vllm-omni-workspace/seedtts_testset \
  --num-prompts 50 \
  --concurrency 4 \
  --wer-eval \
  --output-dir /root/vllm-omni-workspace/vllm-omni/results/moss_local_debug/kernel

SEED_TTS_EVAL_DEVICE=cuda:6 \
  SEED_TTS_WER_SAVE_AUDIO_DIR=/root/vllm-omni-workspace/vllm-omni/results/moss_local_debug/with_graph \
  VLLM_OMNI_BENCH_AUDIO_SAMPLE_RATE=48000 \
  VLLM_OMNI_BENCH_AUDIO_CHANNELS=2 \
  PYTHONPATH=/root/vllm-omni-workspace/vllm-omni \
  python benchmarks/tts/bench_tts.py \
    --host 127.0.0.1 \
    --port 8123 \
    --model OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
    --task voice_clone \
    --locale en \
    --dataset-path /root/vllm-omni-workspace/seedtts_testset \
    --num-prompts 50 \
    --concurrency 4 \
    --wer-eval \
    --output-dir /root/vllm-omni-workspace/vllm-omni/results/moss_local_debug/with_graph


 VLLM_OMNI_BENCH_AUDIO_SAMPLE_RATE=48000 \
  VLLM_OMNI_BENCH_AUDIO_CHANNELS=2 \
  vllm bench serve --omni \
    --host 127.0.0.1 \
    --port 8123 \
    --model OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
    --backend openai-audio-speech \
    --endpoint /v1/audio/speech \
    --dataset-name seed-tts \
    --dataset-path /root/vllm-omni-workspace/seedtts_testset \
    --seed-tts-locale en \
    --num-prompts 8 \
    --num-warmups 8 \
    --max-concurrency 8 \
    --request-rate inf \
    --percentile-metrics ttft,e2el,audio_rtf,audio_ttfp,audio_duration,audio_underrun \
    --profile \
    --save-result \
    --result-dir /root/vllm-omni-workspace/vllm-omni/results/moss_local_stage1_profile_bench


# local graph

# batch 1

SEED_TTS_EVAL_DEVICE=cuda:6 \
  SEED_TTS_WER_SAVE_AUDIO_DIR=/root/vllm-omni-workspace/vllm-omni/results/moss_local_debug/with_graph \
  VLLM_OMNI_BENCH_AUDIO_SAMPLE_RATE=48000 \
  VLLM_OMNI_BENCH_AUDIO_CHANNELS=2 \
  PYTHONPATH=/root/vllm-omni-workspace/vllm-omni \
  python benchmarks/tts/bench_tts.py \
    --host 127.0.0.1 \
    --port 8123 \
    --model OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
    --task voice_clone \
    --locale en \
    --dataset-path /root/vllm-omni-workspace/seedtts_testset \
    --num-prompts 50 \
    --concurrency 1 \
    --wer-eval \
    --output-dir /root/vllm-omni-workspace/vllm-omni/results/moss_local_debug/with_graph

===== Seed-TTS eval (seed-tts-eval protocol) =====
Evaluated (WER, lower is better):        50
Mean WER:                                0.0298
Median WER:                              0.0000
Request failed:                          0
No PCM captured:                         0
ASR / WER failed:                        0
==================================================

=======================================================================================
BENCHMARK SUMMARY
=======================================================================================
Task             Concurrency   RTF mean  TTFP (ms)   Throughput     WER     SIM   UTMOS
---------------------------------------------------------------------------------------
voice_clone                1      0.213         92        4.749     n/a     n/a     n/a
=======================================================================================

============ Serving Benchmark Result ============
Successful requests:                     50
Failed requests:                         0
Maximum request concurrency:             1
Benchmark duration (s):                  44.32
Request throughput (req/s):              1.13
Peak concurrent requests:                3.00
----------------End-to-end Latency----------------
Mean E2EL (ms):                          879.57
Median E2EL (ms):                        873.32
P99 E2EL (ms):                           1404.15
================== Text Result ===================
Total input tokens:                      6600
Total generated tokens:                  0
Output token throughput (tok/s):         0.00
Peak output token throughput (tok/s):    2.00
Peak concurrent requests:                3.00
Total Token throughput (tok/s):          148.92
================== Audio Result ==================
Total audio duration generated(s):       210.48
Total audio frames generated:            10103040
Audio throughput(audio duration/s):      4.75
Streaming continuity OK rate:            0.00%
-----------------Real Time Factor-----------------
Mean AUDIO_RTF:                          0.21
Median AUDIO_RTF:                        0.21
P99 AUDIO_RTF:                           0.29
---------------Time to First Packet---------------
Mean AUDIO_TTFP (ms):                    92.43
Median AUDIO_TTFP (ms):                  91.89
P99 AUDIO_TTFP (ms):                     98.02
------------------Audio Duration------------------
Mean AUDIO_DURATION (s):                 4.21
Median AUDIO_DURATION (s):               4.12
P99 AUDIO_DURATION (s):                  7.09
-------------Streaming Audio Underrun-------------
Mean AUDIO_UNDERRUN (s):                 0.20
Median AUDIO_UNDERRUN (s):               0.20
P99 AUDIO_UNDERRUN (s):                  0.20
==================================================

# batch 8

SEED_TTS_EVAL_DEVICE=cuda:6 \
  VLLM_OMNI_BENCH_AUDIO_SAMPLE_RATE=48000 \
  VLLM_OMNI_BENCH_AUDIO_CHANNELS=2 \
  PYTHONPATH=/root/vllm-omni-workspace/vllm-omni \
  python benchmarks/tts/bench_tts.py \
    --host 127.0.0.1 \
    --port 8123 \
    --model OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
    --task voice_clone \
    --locale en \
    --dataset-path /root/vllm-omni-workspace/seedtts_testset \
    --num-prompts 100 \
    --concurrency 8 \
    --wer-eval \
    --output-dir /root/vllm-omni-workspace/vllm-omni/results/moss_local_debug/with_graph

===== Seed-TTS eval (seed-tts-eval protocol) =====
Evaluated (WER, lower is better):        100
Mean WER:                                0.0233
Median WER:                              0.0000
Request failed:                          0
No PCM captured:                         0
ASR / WER failed:                        0
==================================================

=======================================================================================
BENCHMARK SUMMARY
=======================================================================================
Task             Concurrency   RTF mean  TTFP (ms)   Throughput     WER     SIM   UTMOS
---------------------------------------------------------------------------------------
voice_clone                8      0.632        693       12.786     n/a     n/a     n/a
=======================================================================================


============ Serving Benchmark Result ============
Successful requests:                     100
Failed requests:                         0
Maximum request concurrency:             8
Benchmark duration (s):                  33.37
Request throughput (req/s):              3.00
Peak concurrent requests:                14.00
----------------End-to-end Latency----------------
Mean E2EL (ms):                          2614.79
Median E2EL (ms):                        2661.91
P99 E2EL (ms):                           3832.26
================== Text Result ===================
Total input tokens:                      13230
Total generated tokens:                  0
Output token throughput (tok/s):         0.00
Peak output token throughput (tok/s):    9.00
Peak concurrent requests:                14.00
Total Token throughput (tok/s):          396.50
================== Audio Result ==================
Total audio duration generated(s):       426.64
Total audio frames generated:            20478720
Audio throughput(audio duration/s):      12.79
Streaming continuity OK rate:            2.00%
-----------------Real Time Factor-----------------
Mean AUDIO_RTF:                          0.63
Median AUDIO_RTF:                        0.62
P99 AUDIO_RTF:                           0.97
---------------Time to First Packet---------------
Mean AUDIO_TTFP (ms):                    693.45
Median AUDIO_TTFP (ms):                  717.67
P99 AUDIO_TTFP (ms):                     1177.35
------------------Audio Duration------------------
Mean AUDIO_DURATION (s):                 4.27
Median AUDIO_DURATION (s):               4.44
P99 AUDIO_DURATION (s):                  6.64
-------------Streaming Audio Underrun-------------
Mean AUDIO_UNDERRUN (s):                 0.41
Median AUDIO_UNDERRUN (s):               0.39
P99 AUDIO_UNDERRUN (s):                  0.97
==================================================

# batch 16

SEED_TTS_EVAL_DEVICE=cuda:6 \
  SEED_TTS_WER_SAVE_AUDIO_DIR=/root/vllm-omni-workspace/vllm-omni/results/moss_local_debug/with_graph \
  VLLM_OMNI_BENCH_AUDIO_SAMPLE_RATE=48000 \
  VLLM_OMNI_BENCH_AUDIO_CHANNELS=2 \
  PYTHONPATH=/root/vllm-omni-workspace/vllm-omni \
  python benchmarks/tts/bench_tts.py \
    --host 127.0.0.1 \
    --port 8123 \
    --model OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
    --task voice_clone \
    --locale en \
    --dataset-path /root/vllm-omni-workspace/seedtts_testset \
    --num-prompts 100 \
    --concurrency 16 \
    --wer-eval \
    --output-dir /root/vllm-omni-workspace/vllm-omni/results/moss_local_debug/with_graph


===== Seed-TTS eval (seed-tts-eval protocol) =====
Evaluated (WER, lower is better):        100
Mean WER:                                0.0455
Median WER:                              0.0000
Request failed:                          0
No PCM captured:                         0
ASR / WER failed:                        0
==================================================

=======================================================================================
BENCHMARK SUMMARY
=======================================================================================
Task             Concurrency   RTF mean  TTFP (ms)   Throughput     WER     SIM   UTMOS
---------------------------------------------------------------------------------------
voice_clone               16      2.311       2978        6.357     n/a     n/a     n/a
=======================================================================================


============ Serving Benchmark Result ============
Successful requests:                     100
Failed requests:                         0
Maximum request concurrency:             16
Benchmark duration (s):                  77.84
Request throughput (req/s):              1.28
Peak concurrent requests:                22.00
----------------End-to-end Latency----------------
Mean E2EL (ms):                          10205.36
Median E2EL (ms):                        9927.01
P99 E2EL (ms):                           15594.02
================== Text Result ===================
Total input tokens:                      13230
Total generated tokens:                  0
Output token throughput (tok/s):         0.00
Peak output token throughput (tok/s):    16.00
Peak concurrent requests:                22.00
Total Token throughput (tok/s):          169.96
================== Audio Result ==================
Total audio duration generated(s):       494.88
Total audio frames generated:            23754240
Audio throughput(audio duration/s):      6.36
Streaming continuity OK rate:            0.00%
-----------------Real Time Factor-----------------
Mean AUDIO_RTF:                          2.31
Median AUDIO_RTF:                        2.31
P99 AUDIO_RTF:                           3.51
---------------Time to First Packet---------------
Mean AUDIO_TTFP (ms):                    2978.31
Median AUDIO_TTFP (ms):                  3367.84
P99 AUDIO_TTFP (ms):                     4541.28
------------------Audio Duration------------------
Mean AUDIO_DURATION (s):                 4.95
Median AUDIO_DURATION (s):               4.40
P99 AUDIO_DURATION (s):                  8.00
-------------Streaming Audio Underrun-------------
Mean AUDIO_UNDERRUN (s):                 3.20
Median AUDIO_UNDERRUN (s):               3.06
P99 AUDIO_UNDERRUN (s):                  5.36
==================================================


# batch 32

SEED_TTS_EVAL_DEVICE=cuda:6 \
  SEED_TTS_WER_SAVE_AUDIO_DIR=/root/vllm-omni-workspace/vllm-omni/results/moss_local_debug/with_graph \
  VLLM_OMNI_BENCH_AUDIO_SAMPLE_RATE=48000 \
  VLLM_OMNI_BENCH_AUDIO_CHANNELS=2 \
  PYTHONPATH=/root/vllm-omni-workspace/vllm-omni \
  python benchmarks/tts/bench_tts.py \
    --host 127.0.0.1 \
    --port 8123 \
    --model OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
    --task voice_clone \
    --locale en \
    --dataset-path /root/vllm-omni-workspace/seedtts_testset \
    --num-prompts 200 \
    --concurrency 32 \
    --wer-eval \
    --output-dir /root/vllm-omni-workspace/vllm-omni/results/moss_local_debug/with_graph

===== Seed-TTS eval (seed-tts-eval protocol) =====
Evaluated (WER, lower is better):        200
Mean WER:                                0.0364
Median WER:                              0.0000
Request failed:                          0
No PCM captured:                         0
ASR / WER failed:                        0
==================================================

=======================================================================================
BENCHMARK SUMMARY
=======================================================================================
Task             Concurrency   RTF mean  TTFP (ms)   Throughput     WER     SIM   UTMOS
---------------------------------------------------------------------------------------
voice_clone               32      3.069       3864        8.735     n/a     n/a     n/a
=======================================================================================

============ Serving Benchmark Result ============
Successful requests:                     200
Failed requests:                         0
Maximum request concurrency:             32
Benchmark duration (s):                  118.87
Request throughput (req/s):              1.68
Peak concurrent requests:                40.00
----------------End-to-end Latency----------------
Mean E2EL (ms):                          13529.47
Median E2EL (ms):                        12870.61
P99 E2EL (ms):                           24283.26
================== Text Result ===================
Total input tokens:                      26488
Total generated tokens:                  0
Output token throughput (tok/s):         0.00
Peak output token throughput (tok/s):    32.00
Peak concurrent requests:                40.00
Total Token throughput (tok/s):          222.83
================== Audio Result ==================
Total audio duration generated(s):       1038.32
Total audio frames generated:            49839360
Audio throughput(audio duration/s):      8.73
Streaming continuity OK rate:            0.00%
-----------------Real Time Factor-----------------
Mean AUDIO_RTF:                          3.07
Median AUDIO_RTF:                        2.95
P99 AUDIO_RTF:                           4.92
---------------Time to First Packet---------------
Mean AUDIO_TTFP (ms):                    3864.10
Median AUDIO_TTFP (ms):                  4145.73
P99 AUDIO_TTFP (ms):                     6596.39
------------------Audio Duration------------------
Mean AUDIO_DURATION (s):                 5.19
Median AUDIO_DURATION (s):               4.24
P99 AUDIO_DURATION (s):                  9.30
-------------Streaming Audio Underrun-------------
Mean AUDIO_UNDERRUN (s):                 5.39
Median AUDIO_UNDERRUN (s):               5.11
P99 AUDIO_UNDERRUN (s):                  12.22
==================================================





# multi-replica 

vllm serve OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
  --omni \
  --host 0.0.0.0     --port 8123     --trust-remote-code     --deploy-config vllm_omni/deploy/moss_tts_local.yaml     --allowed-local-media-path /root/vllm-omni-workspace --stage-overrides '{"0":{"num_replicas":8,"devices":"0"},"1":{"num_replicas":8,"devices":"0"}}' --init-timeout 1800

SEED_TTS_EVAL_DEVICE=cuda:6 \
  VLLM_OMNI_BENCH_AUDIO_SAMPLE_RATE=48000 \
  VLLM_OMNI_BENCH_AUDIO_CHANNELS=2 \
  PYTHONPATH=/root/vllm-omni-workspace/vllm-omni \
  python benchmarks/tts/bench_tts.py \
    --host 127.0.0.1 \
    --port 8123 \
    --model OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
    --task voice_clone \
    --locale en \
    --dataset-path /root/vllm-omni-workspace/seedtts_testset \
    --num-prompts 100 \
    --concurrency 16 \
    --wer-eval 

## batch 16

===== Seed-TTS eval (seed-tts-eval protocol) =====
Evaluated (WER, lower is better):        100
Mean WER:                                0.0291
Median WER:                              0.0000
Request failed:                          0
No PCM captured:                         0
ASR / WER failed:                        0
==================================================

=======================================================================================
BENCHMARK SUMMARY
=======================================================================================
Task             Concurrency   RTF mean  TTFP (ms)   Throughput     WER     SIM   UTMOS
---------------------------------------------------------------------------------------
voice_clone               16      0.352        306       14.867     n/a     n/a     n/a
=======================================================================================

============ Serving Benchmark Result ============
Successful requests:                     100
Failed requests:                         0
Maximum request concurrency:             16
Benchmark duration (s):                  39.56
Request throughput (req/s):              2.53
Peak concurrent requests:                30.00
----------------End-to-end Latency----------------
Mean E2EL (ms):                          1791.48
Median E2EL (ms):                        1443.32
P99 E2EL (ms):                           3127.53
================== Text Result ===================
Total input tokens:                      13230
Total generated tokens:                  0
Output token throughput (tok/s):         0.00
Peak output token throughput (tok/s):    16.00
Peak concurrent requests:                30.00
Total Token throughput (tok/s):          334.45
================== Audio Result ==================
Total audio duration generated(s):       588.08
Total audio frames generated:            28227840
Audio throughput(audio duration/s):      14.87
Streaming continuity OK rate:            4.00%
-----------------Real Time Factor-----------------
Mean AUDIO_RTF:                          0.35
Median AUDIO_RTF:                        0.35
P99 AUDIO_RTF:                           0.55
---------------Time to First Packet---------------
Mean AUDIO_TTFP (ms):                    306.02
Median AUDIO_TTFP (ms):                  297.45
P99 AUDIO_TTFP (ms):                     594.28
------------------Audio Duration------------------
Mean AUDIO_DURATION (s):                 5.88
Median AUDIO_DURATION (s):               4.32
P99 AUDIO_DURATION (s):                  8.85
-------------Streaming Audio Underrun-------------
Mean AUDIO_UNDERRUN (s):                 0.29
Median AUDIO_UNDERRUN (s):               0.27
P99 AUDIO_UNDERRUN (s):                  0.58
==================================================


## batch 32


===== Seed-TTS eval (seed-tts-eval protocol) =====
Evaluated (WER, lower is better):        200
Mean WER:                                0.0198
Median WER:                              0.0000
Request failed:                          0
No PCM captured:                         0
ASR / WER failed:                        0
==================================================

=======================================================================================
BENCHMARK SUMMARY
=======================================================================================
Task             Concurrency   RTF mean  TTFP (ms)   Throughput     WER     SIM   UTMOS
---------------------------------------------------------------------------------------
voice_clone               32      0.502        514       24.656     n/a     n/a     n/a
=======================================================================================

============ Serving Benchmark Result ============
Successful requests:                     200
Failed requests:                         0
Maximum request concurrency:             32
Benchmark duration (s):                  41.17
Request throughput (req/s):              4.86
Peak concurrent requests:                54.00
----------------End-to-end Latency----------------
Mean E2EL (ms):                          2221.27
Median E2EL (ms):                        2022.01
P99 E2EL (ms):                           3417.82
================== Text Result ===================
Total input tokens:                      26488
Total generated tokens:                  0
Output token throughput (tok/s):         0.00
Peak output token throughput (tok/s):    32.00
Peak concurrent requests:                54.00
Total Token throughput (tok/s):          643.31
================== Audio Result ==================
Total audio duration generated(s):       1015.20
Total audio frames generated:            48729600
Audio throughput(audio duration/s):      24.66
Streaming continuity OK rate:            3.50%
-----------------Real Time Factor-----------------
Mean AUDIO_RTF:                          0.50
Median AUDIO_RTF:                        0.48
P99 AUDIO_RTF:                           0.94
---------------Time to First Packet---------------
Mean AUDIO_TTFP (ms):                    514.21
Median AUDIO_TTFP (ms):                  487.68
P99 AUDIO_TTFP (ms):                     913.16
------------------Audio Duration------------------
Mean AUDIO_DURATION (s):                 5.08
Median AUDIO_DURATION (s):               4.32
P99 AUDIO_DURATION (s):                  7.71
-------------Streaming Audio Underrun-------------
Mean AUDIO_UNDERRUN (s):                 0.33
Median AUDIO_UNDERRUN (s):               0.31
P99 AUDIO_UNDERRUN (s):                  0.83
==================================================

  


# fp32 -> bf 16

===== Seed-TTS eval (seed-tts-eval protocol) =====
Evaluated (WER, lower is better):        100
Mean WER:                                0.0239
Median WER:                              0.0000
Request failed:                          0
No PCM captured:                         0
ASR / WER failed:                        0
==================================================

=======================================================================================
BENCHMARK SUMMARY
=======================================================================================
Task             Concurrency   RTF mean  TTFP (ms)   Throughput     WER     SIM   UTMOS
---------------------------------------------------------------------------------------
voice_clone                8      0.433        563       18.818     n/a     n/a     n/a
=======================================================================================

============ Serving Benchmark Result ============
Successful requests:                     100
Failed requests:                         0
Maximum request concurrency:             8
Benchmark duration (s):                  22.53
Request throughput (req/s):              4.44
Peak concurrent requests:                15.00
----------------End-to-end Latency----------------
Mean E2EL (ms):                          1781.36
Median E2EL (ms):                        1814.04
P99 E2EL (ms):                           2699.75
================== Text Result ===================
Total input tokens:                      13230
Total generated tokens:                  0
Output token throughput (tok/s):         0.00
Peak output token throughput (tok/s):    9.00
Peak concurrent requests:                15.00
Total Token throughput (tok/s):          587.29
================== Audio Result ==================
Total audio duration generated(s):       423.92
Total audio frames generated:            20348160
Audio throughput(audio duration/s):      18.82
Streaming continuity OK rate:            7.00%
-----------------Real Time Factor-----------------
Mean AUDIO_RTF:                          0.43
Median AUDIO_RTF:                        0.42
P99 AUDIO_RTF:                           0.61
---------------Time to First Packet---------------
Mean AUDIO_TTFP (ms):                    562.86
Median AUDIO_TTFP (ms):                  580.34
P99 AUDIO_TTFP (ms):                     746.55
------------------Audio Duration------------------
Mean AUDIO_DURATION (s):                 4.24
Median AUDIO_DURATION (s):               4.24
P99 AUDIO_DURATION (s):                  7.44
-------------Streaming Audio Underrun-------------
Mean AUDIO_UNDERRUN (s):                 0.22
Median AUDIO_UNDERRUN (s):               0.22
P99 AUDIO_UNDERRUN (s):                  0.44
==================================================


===== Seed-TTS eval (seed-tts-eval protocol) =====
Evaluated (WER, lower is better):        100
Mean WER:                                0.0360
Median WER:                              0.0000
Request failed:                          0
No PCM captured:                         0
ASR / WER failed:                        0
==================================================

=======================================================================================
BENCHMARK SUMMARY
=======================================================================================
Task             Concurrency   RTF mean  TTFP (ms)   Throughput     WER     SIM   UTMOS
---------------------------------------------------------------------------------------
voice_clone               16      0.775        982       11.807     n/a     n/a     n/a
=======================================================================================

============ Serving Benchmark Result ============
Successful requests:                     100
Failed requests:                         0
Maximum request concurrency:             16
Benchmark duration (s):                  62.19
Request throughput (req/s):              1.61
Peak concurrent requests:                26.00
----------------End-to-end Latency----------------
Mean E2EL (ms):                          3662.04
Median E2EL (ms):                        3229.29
P99 E2EL (ms):                           5336.75
================== Text Result ===================
Total input tokens:                      13230
Total generated tokens:                  0
Output token throughput (tok/s):         0.00
Peak output token throughput (tok/s):    16.00
Peak concurrent requests:                26.00
Total Token throughput (tok/s):          212.74
================== Audio Result ==================
Total audio duration generated(s):       734.24
Total audio frames generated:            35243520
Audio throughput(audio duration/s):      11.81
Streaming continuity OK rate:            0.00%
-----------------Real Time Factor-----------------
Mean AUDIO_RTF:                          0.78
Median AUDIO_RTF:                        0.77
P99 AUDIO_RTF:                           1.10
---------------Time to First Packet---------------
Mean AUDIO_TTFP (ms):                    981.67
Median AUDIO_TTFP (ms):                  1033.59
P99 AUDIO_TTFP (ms):                     1383.96
------------------Audio Duration------------------
Mean AUDIO_DURATION (s):                 7.34
Median AUDIO_DURATION (s):               4.20
P99 AUDIO_DURATION (s):                  10.91
-------------Streaming Audio Underrun-------------
Mean AUDIO_UNDERRUN (s):                 0.43
Median AUDIO_UNDERRUN (s):               0.40
P99 AUDIO_UNDERRUN (s):                  0.76
==================================================


# streaming exec mask D2D

===== Seed-TTS eval (seed-tts-eval protocol) =====
Evaluated (WER, lower is better):        100
Mean WER:                                0.0348
Median WER:                              0.0000
Request failed:                          0
No PCM captured:                         0
ASR / WER failed:                        0
==================================================

=======================================================================================
BENCHMARK SUMMARY
=======================================================================================
Task             Concurrency   RTF mean  TTFP (ms)   Throughput     WER     SIM   UTMOS
---------------------------------------------------------------------------------------
voice_clone                8      0.371        431       21.780     n/a     n/a     n/a
=======================================================================================

============ Serving Benchmark Result ============
Successful requests:                     100
Failed requests:                         0
Maximum request concurrency:             8
Benchmark duration (s):                  19.88
Request throughput (req/s):              5.03
Peak concurrent requests:                16.00
----------------End-to-end Latency----------------
Mean E2EL (ms):                          1565.24
Median E2EL (ms):                        1574.11
P99 E2EL (ms):                           2398.98
================== Text Result ===================
Total input tokens:                      13230
Total generated tokens:                  0
Output token throughput (tok/s):         0.00
Peak output token throughput (tok/s):    12.00
Peak concurrent requests:                16.00
Total Token throughput (tok/s):          665.42
================== Audio Result ==================
Total audio duration generated(s):       433.04
Total audio frames generated:            20785920
Audio throughput(audio duration/s):      21.78
Streaming continuity OK rate:            24.00%
-----------------Real Time Factor-----------------
Mean AUDIO_RTF:                          0.37
Median AUDIO_RTF:                        0.36
P99 AUDIO_RTF:                           0.52
---------------Time to First Packet---------------
Mean AUDIO_TTFP (ms):                    431.10
Median AUDIO_TTFP (ms):                  429.11
P99 AUDIO_TTFP (ms):                     683.39
------------------Audio Duration------------------
Mean AUDIO_DURATION (s):                 4.33
Median AUDIO_DURATION (s):               4.24
P99 AUDIO_DURATION (s):                  7.06
-------------Streaming Audio Underrun-------------
Mean AUDIO_UNDERRUN (s):                 0.18
Median AUDIO_UNDERRUN (s):               0.15
P99 AUDIO_UNDERRUN (s):                  0.39
==================================================


===== Seed-TTS eval (seed-tts-eval protocol) =====
Evaluated (WER, lower is better):        100
Mean WER:                                0.0261
Median WER:                              0.0000
Request failed:                          0
No PCM captured:                         0
ASR / WER failed:                        0
==================================================

=======================================================================================
BENCHMARK SUMMARY
=======================================================================================
Task             Concurrency   RTF mean  TTFP (ms)   Throughput     WER     SIM   UTMOS
---------------------------------------------------------------------------------------
voice_clone               16      0.717        937       22.317     n/a     n/a     n/a
=======================================================================================

============ Serving Benchmark Result ============
Successful requests:                     100
Failed requests:                         0
Maximum request concurrency:             16
Benchmark duration (s):                  18.97
Request throughput (req/s):              5.27
Peak concurrent requests:                23.00
----------------End-to-end Latency----------------
Mean E2EL (ms):                          2938.94
Median E2EL (ms):                        2979.89
P99 E2EL (ms):                           4285.57
================== Text Result ===================
Total input tokens:                      13230
Total generated tokens:                  0
Output token throughput (tok/s):         0.00
Peak output token throughput (tok/s):    16.00
Peak concurrent requests:                23.00
Total Token throughput (tok/s):          697.54
================== Audio Result ==================
Total audio duration generated(s):       423.28
Total audio frames generated:            20317440
Audio throughput(audio duration/s):      22.32
Streaming continuity OK rate:            4.00%
-----------------Real Time Factor-----------------
Mean AUDIO_RTF:                          0.72
Median AUDIO_RTF:                        0.71
P99 AUDIO_RTF:                           1.14
---------------Time to First Packet---------------
Mean AUDIO_TTFP (ms):                    937.35
Median AUDIO_TTFP (ms):                  998.36
P99 AUDIO_TTFP (ms):                     1314.86
------------------Audio Duration------------------
Mean AUDIO_DURATION (s):                 4.23
Median AUDIO_DURATION (s):               4.12
P99 AUDIO_DURATION (s):                  6.72
-------------Streaming Audio Underrun-------------
Mean AUDIO_UNDERRUN (s):                 0.41
Median AUDIO_UNDERRUN (s):               0.39
P99 AUDIO_UNDERRUN (s):                  0.71
==================================================



# D2H

===== Seed-TTS eval (seed-tts-eval protocol) =====
Evaluated (WER, lower is better):        100
Mean WER:                                0.0250
Median WER:                              0.0000
Request failed:                          0
No PCM captured:                         0
ASR / WER failed:                        0
==================================================

=======================================================================================
BENCHMARK SUMMARY
=======================================================================================
Task             Concurrency   RTF mean  TTFP (ms)   Throughput     WER     SIM   UTMOS
---------------------------------------------------------------------------------------
voice_clone               16      0.638        815       24.862     n/a     n/a     n/a
=======================================================================================

============ Serving Benchmark Result ============
Successful requests:                     100
Failed requests:                         0
Maximum request concurrency:             16
Benchmark duration (s):                  17.24
Request throughput (req/s):              5.80
Peak concurrent requests:                27.00
----------------End-to-end Latency----------------
Mean E2EL (ms):                          2655.99
Median E2EL (ms):                        2690.90
P99 E2EL (ms):                           3767.17
================== Text Result ===================
Total input tokens:                      13230
Total generated tokens:                  0
Output token throughput (tok/s):         0.00
Peak output token throughput (tok/s):    16.00
Peak concurrent requests:                27.00
Total Token throughput (tok/s):          767.23
================== Audio Result ==================
Total audio duration generated(s):       428.72
Total audio frames generated:            20578560
Audio throughput(audio duration/s):      24.86
Streaming continuity OK rate:            0.00%
-----------------Real Time Factor-----------------
Mean AUDIO_RTF:                          0.64
Median AUDIO_RTF:                        0.63
P99 AUDIO_RTF:                           0.93
---------------Time to First Packet---------------
Mean AUDIO_TTFP (ms):                    815.40
Median AUDIO_TTFP (ms):                  850.01
P99 AUDIO_TTFP (ms):                     1182.43
------------------Audio Duration------------------
Mean AUDIO_DURATION (s):                 4.29
Median AUDIO_DURATION (s):               4.36
P99 AUDIO_DURATION (s):                  6.73
-------------Streaming Audio Underrun-------------
Mean AUDIO_UNDERRUN (s):                 0.36
Median AUDIO_UNDERRUN (s):               0.35
P99 AUDIO_UNDERRUN (s):                  0.62
=================================================