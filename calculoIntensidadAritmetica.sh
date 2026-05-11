#!/bin/bash

set -e

SRC="secuencial.c"
ASM="secuencial.s"

WIDTH=1000
HEIGHT=1000
ITERATION=1000

gcc "$SRC" -S -O3 -fopenmp -DBENCHMARK -o "$ASM"

CELDAS=$((WIDTH * HEIGHT))
ITER_UPDATE=$((CELDAS / 4))
BYTES_MEMCPY=$((CELDAS * 4 * 2))

awk -v iter_update="$ITER_UPDATE" \
    -v total_iter="$ITERATION" \
    -v bytes_memcpy="$BYTES_MEMCPY" '
BEGIN {
    current="GLOBAL"
}

/^[A-Za-z_][A-Za-z0-9_]*:$/ {
    current=$1
    gsub(":", "", current)
    next
}

current != "" && /^[ \t]*[a-zA-Z]/ {
    instr=$1

    op=0
    if (instr ~ /^(add|addl|addq|sub|subl|subq|imul|imull|imulq|mul|div|idiv|and|andl|or|orl|xor|xorl|cmp|cmpl|test|testl|shl|shr|sal|sar)$/) op=1
    if (instr ~ /^(paddd|psubd|pmuludq|pcmpgtd|pcmpeqd|pand|pandn|por|pslld|psrld|psllq|psrlq|psrad)$/) op=4

    b=0
    if ($0 ~ /\(.*\)/) {
        if (instr ~ /^(movdqa|movaps)$/) b=16
        else if (instr=="movd") b=4
        else if (instr ~ /q$/) b=8
        else if (instr ~ /l$/) b=4
        else if (instr ~ /^(paddd|psubd|pmuludq|pcmpgtd|pcmpeqd|pand|pandn|por)$/) b=16
        else b=4
    }

    ops[current]+=op
    bytes[current]+=b
}

END {
    printf "\nIA estatica por funcion\n"
    printf "---------------------------------------------------------------\n"
    printf "%-30s %12s %12s %15s\n", "Funcion", "Ops", "Bytes", "AI ops/byte"
    printf "---------------------------------------------------------------\n"

    for (f in ops) {
        ai_f = bytes[f] > 0 ? ops[f] / bytes[f] : 0
        printf "%-30s %12d %12d %15.6f\n", f, ops[f], bytes[f], ai_f
    }

    ai_update_static = bytes["updateGrid"] > 0 ? ops["updateGrid"] / bytes["updateGrid"] : 0

    ops_update_real = ops["updateGrid"] * iter_update * total_iter
    bytes_update_real = bytes["updateGrid"] * iter_update * total_iter
    bytes_update_real += bytes_memcpy * total_iter

    ai_real = bytes_update_real > 0 ? ops_update_real / bytes_update_real : 0

    printf "\nIA desde ensamblador ponderada por ejecucion real\n"
    printf "-------------------------------------------------\n"
    printf "Ops estaticas updateGrid:      %d\n", ops["updateGrid"]
    printf "Bytes estaticos updateGrid:    %d\n", bytes["updateGrid"]
    printf "AI estatica updateGrid:        %.6f OPS/byte\n", ai_update_static
    printf "\n"
    printf "Iteraciones SIMD por update:   %d\n", iter_update
    printf "Generaciones:                 %d\n", total_iter
    printf "Bytes memcpy por generacion:   %d\n", bytes_memcpy
    printf "\n"
    printf "Ops reales aproximadas:        %.0f\n", ops_update_real
    printf "Bytes reales aproximados:      %.0f\n", bytes_update_real
    printf "AI real aproximada:            %.6f OPS/byte\n", ai_real
}
' "$ASM"