# ollama_bench
A Python script to benchmark multiple models on ollama

The script only requires the requests package:
```
pip install requests
python ollama_bench.py
```

You can customize the script by changing:
```
OLLAMA_URL = "http://localhost:11434"  # URL base di Ollama
```
``` 
MODELS = [
    "llama3.2",
    "mistral",
    "gemma3:4b",
]
```
``` 
# Thinking (ragionamento esteso): supportato da alcuni modelli come deepseek-r1
ENABLE_THINKING = False
```
```
# Header HTTP aggiuntivi (es. per proxy autenticati)
EXTRA_HEADERS: dict = {
    # "Authorization": "Bearer token123",
    # "X-Custom-Header": "valore",
}
```
``` 
QUESTIONS = [
    "Cos'è il machine learning? Rispondi in 3 frasi.",
    "Quali sono i principali vantaggi dell'energia solare?",
    "Spiega brevemente come funziona internet.",
    "Qual è la differenza tra Python e JavaScript?",
    "Cosa si intende per intelligenza artificiale generativa?",
]
```
