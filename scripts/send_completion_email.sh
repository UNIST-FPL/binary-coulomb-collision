#!/usr/bin/env bash
set -eu

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <subject> <summary>" >&2
  exit 2
fi

subject=$1
summary=$2

sender="sungpilyum@pikachu.unist.ac.kr"
recipient="sungpil.yum@unist.ac.kr"

repo_root=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
head_commit=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
timestamp=$(date "+%Y-%m-%d %H:%M:%S%z")

body=$(cat <<EOF
Completion report

Summary: $summary
Repository: $repo_root
Branch: $branch
HEAD: $head_commit
Time: $timestamp
EOF
)

printf '%s\n' "$body" | mail -r "$sender" -s "$subject" "$recipient"
