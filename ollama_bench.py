#!/usr/bin/env python3
"""
Ollama LLM Benchmark Script
Testa più modelli con domande predefinite e riporta metriche di performance.
"""

import json
import time
import requests
from dataclasses import dataclass, field
from typing import Optional
from statistics import mean

# ─────────────────────────────────────────────
#  CONFIGURAZIONE — modifica qui
# ─────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434"  # URL base di Ollama

MODELS = [
    "llama3.2",
    "mistral",
    "gemma3:4b",
]

# Thinking (ragionamento esteso): supportato da alcuni modelli come deepseek-r1
ENABLE_THINKING = False

# Header HTTP aggiuntivi (es. per proxy autenticati)
EXTRA_HEADERS: dict = {
    # "Authorization": "Bearer token123",
    # "X-Custom-Header": "valore",
    # "X-API-Token: {token}",
}

QUESTIONS = [
    "Cos'è il machine learning? Rispondi in 3 frasi.",
    "Quali sono i principali vantaggi dell'energia solare?",
    "Spiega brevemente come funziona internet.",
    "Qual è la differenza tra Python e JavaScript?",
    "Cosa si intende per intelligenza artificiale generativa?",
]

# ─────────────────────────────────────────────
#  STRUTTURE DATI
# ─────────────────────────────────────────────

@dataclass
class QuestionResult:
    question: str
    output_tokens: int
    elapsed_seconds: float
    tokens_per_second: float
    error: Optional[str] = None

@dataclass
class ModelResult:
    model: str
    results: list[QuestionResult] = field(default_factory=list)

    def avg_output_tokens(self) -> float:
        vals = [r.output_tokens for r in self.results if not r.error]
        return mean(vals) if vals else 0.0

    def avg_elapsed(self) -> float:
        vals = [r.elapsed_seconds for r in self.results if not r.error]
        return mean(vals) if vals else 0.0

    def avg_tokens_per_second(self) -> float:
        vals = [r.tokens_per_second for r in self.results if not r.error]
        return mean(vals) if vals else 0.0

    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.results if not r.error)

# ─────────────────────────────────────────────
#  LOGICA DI CHIAMATA
# ─────────────────────────────────────────────

def build_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    headers.update(EXTRA_HEADERS)
    return headers


def call_ollama(model: str, question: str) -> QuestionResult:
    """Chiama l'API /api/generate di Ollama e raccoglie le metriche."""
    url = f"{OLLAMA_URL.rstrip('/')}/api/generate"

    payload: dict = {
        "model": model,
        "prompt": question,
        "stream": False,
    }

    if ENABLE_THINKING:
        # Alcuni modelli (es. deepseek-r1) supportano thinking tramite options
        payload["options"] = {"think": True}

    start = time.perf_counter()
    try:
        response = requests.post(
            url,
            headers=build_headers(),
            json=payload,
            timeout=300,
        )
        elapsed = time.perf_counter() - start
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as exc:
        elapsed = time.perf_counter() - start
        return QuestionResult(
            question=question,
            output_tokens=0,
            elapsed_seconds=round(elapsed, 3),
            tokens_per_second=0.0,
            error=str(exc),
        )

    # Ollama restituisce eval_count = token generati
    output_tokens: int = data.get("eval_count", 0)
    # eval_duration è in nanosecondi
    eval_ns: int = data.get("eval_duration", 0)
    eval_sec: float = eval_ns / 1_000_000_000 if eval_ns else elapsed

    tps = output_tokens / eval_sec if eval_sec > 0 else 0.0

    return QuestionResult(
        question=question,
        output_tokens=output_tokens,
        elapsed_seconds=round(elapsed, 3),
        tokens_per_second=round(tps, 2),
    )


# ─────────────────────────────────────────────
#  STAMPA RISULTATI
# ─────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
DIM    = "\033[2m"


def print_separator(char: str = "─", width: int = 70) -> None:
    print(char * width)


def print_model_table(model_results: list[ModelResult]) -> None:
    """Stampa la tabella riassuntiva finale."""
    col_w = [22, 14, 16, 18, 18]
    headers = ["Modello", "Tot. Tokens", "Avg Tokens/s", "Avg Tempo (s)", "Tot. Domande OK"]

    print()
    print_separator("═")
    print(f"{BOLD}{CYAN}  RIEPILOGO BENCHMARK{RESET}")
    print_separator("═")

    # Header tabella
    row = "  ".join(h.ljust(col_w[i]) for i, h in enumerate(headers))
    print(f"{BOLD}{row}{RESET}")
    print_separator()

    for mr in model_results:
        ok = sum(1 for r in mr.results if not r.error)
        row_data = [
            mr.model,
            str(mr.total_output_tokens()),
            f"{mr.avg_tokens_per_second():.1f}",
            f"{mr.avg_elapsed():.2f}",
            f"{ok}/{len(mr.results)}",
        ]
        colored_row = "  ".join(v.ljust(col_w[i]) for i, v in enumerate(row_data))
        print(colored_row)

    print_separator("═")


def print_question_detail(mr: ModelResult) -> None:
    print()
    print(f"{BOLD}{CYAN}► {mr.model}{RESET}")
    print_separator()
    for i, r in enumerate(mr.results, 1):
        q_short = r.question[:60] + ("…" if len(r.question) > 60 else "")
        print(f"  {BOLD}Q{i}{RESET}: {q_short}")
        if r.error:
            print(f"      {RED}ERRORE: {r.error}{RESET}")
        else:
            print(
                f"      Tokens: {GREEN}{r.output_tokens:>5}{RESET}  |  "
                f"Tok/s: {YELLOW}{r.tokens_per_second:>7.2f}{RESET}  |  "
                f"Tempo: {r.elapsed_seconds:.3f}s"
            )


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main() -> None:
    print()
    print(f"{BOLD}{'═'*70}{RESET}")
    print(f"{BOLD}{CYAN}  OLLAMA LLM BENCHMARK{RESET}")
    print(f"{DIM}  URL: {OLLAMA_URL}  |  Thinking: {'ON' if ENABLE_THINKING else 'OFF'}{RESET}")
    print(f"{BOLD}{'═'*70}{RESET}")
    print(f"  Modelli da testare : {', '.join(MODELS)}")
    print(f"  Domande per modello: {len(QUESTIONS)}")
    print()

    all_results: list[ModelResult] = []

    for model in MODELS:
        mr = ModelResult(model=model)
        print(f"{BOLD}Testing → {CYAN}{model}{RESET}")

        for idx, question in enumerate(QUESTIONS, 1):
            print(f"  [{idx}/{len(QUESTIONS)}] {DIM}{question[:55]}…{RESET}", end=" ", flush=True)
            result = call_ollama(model, question)
            mr.results.append(result)

            if result.error:
                print(f"{RED}ERRORE{RESET}")
            else:
                print(
                    f"{GREEN}✓{RESET}  "
                    f"{result.output_tokens} tok  "
                    f"{result.tokens_per_second:.1f} tok/s  "
                    f"{result.elapsed_seconds:.2f}s"
                )

        all_results.append(mr)
        print()

    # Dettaglio per modello
    print_separator("═")
    print(f"{BOLD}{CYAN}  DETTAGLIO PER MODELLO{RESET}")
    for mr in all_results:
        print_question_detail(mr)

    # Riepilogo finale
    print_model_table(all_results)

    # Export JSON opzionale
    export_json(all_results)


def export_json(all_results: list[ModelResult]) -> None:
    """Salva i risultati in un file JSON."""
    output = []
    for mr in all_results:
        output.append({
            "model": mr.model,
            "avg_output_tokens": mr.avg_output_tokens(),
            "avg_tokens_per_second": mr.avg_tokens_per_second(),
            "avg_elapsed_seconds": mr.avg_elapsed(),
            "total_output_tokens": mr.total_output_tokens(),
            "questions": [
                {
                    "question": r.question,
                    "output_tokens": r.output_tokens,
                    "elapsed_seconds": r.elapsed_seconds,
                    "tokens_per_second": r.tokens_per_second,
                    "error": r.error,
                }
                for r in mr.results
            ],
        })

    filename = "benchmark_results.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n{DIM}  Risultati salvati in → {filename}{RESET}\n")


if __name__ == "__main__":
    main()
