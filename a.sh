#!/bin/bash

N=50232

factors=()

for (( i=2; i<=N; i++ )); do
    if (( N % i == 0 )); then
        factors+=($i)
    fi
done

factors=("${factors[@]:8}")

factors=("${factors[@]::${#factors[@]}-8}")

echo "Factors: ${factors[@]}"

for i in "${!factors[@]}"; do
    python3 -m scripts.run_simple_optim ${factors[$i]}
done
