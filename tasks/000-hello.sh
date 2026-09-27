# kind: cpu
# time: 00:15:00
# cpus: 1
# What does a Burst CPU node look like? (no venv needed)
{
  echo "host: $(hostname)  date: $(date -u +%FT%TZ)"
  echo "scratch: $(df -h "/scratch/$USER" | tail -1)"
  echo "python3: $(python3 --version 2>&1)  bash: $BASH_VERSION"
  echo "cpus: $(nproc)"; free -g | head -2
  curl -sS -o /dev/null -w "huggingface.co: HTTP %{http_code}\n" https://huggingface.co || echo "huggingface.co: unreachable"
  echo "partitions:"; sinfo -h -o "  %P %G %l" 2>/dev/null | sort -u
} > "$OUT/hello.txt" 2>&1
cat "$OUT/hello.txt"
