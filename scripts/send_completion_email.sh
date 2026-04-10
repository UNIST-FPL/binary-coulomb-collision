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
project_name=$(basename "$repo_root")
branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
head_commit=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
timestamp=$(date "+%Y-%m-%d %H:%M:%S%z")

case "$subject" in
  *"$project_name"*)
    subject_with_project=$subject
    ;;
  *)
    subject_with_project="[$project_name] $subject"
    ;;
esac

body=$(cat <<EOF
Completion report

Summary: $summary
Repository: $repo_root
Branch: $branch
HEAD: $head_commit
Time: $timestamp
EOF
)

printf '%s\n' "$body" | mail -r "$sender" -s "$subject_with_project" "$recipient"
