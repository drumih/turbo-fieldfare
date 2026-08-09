swift build -c release --product TurboFieldfareServer

.build/release/TurboFieldfareServer \
  --model scratch/kimi-k3.gturbo \
  --port 8080 \
  --max-context 262144 \
  --prompt-cache-mode single-prefix \
  --queue-limit 1 \
  --model-verification trusted-install
