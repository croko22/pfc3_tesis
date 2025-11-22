# 🚀 Quick Start - LLM Adapters

## 🌟 Opción Recomendada: OpenRouter

**Una API key, todos los modelos** (GPT-4, Claude, Gemini, Llama, DeepSeek, etc.)

```bash
# 1. Regístrate en https://openrouter.ai/
# 2. Obtén tu API key: https://openrouter.ai/keys
# 3. Configura
export OPENROUTER_API_KEY="sk-or-v1-..."

# 4. ¡Listo! Usa cualquier modelo
python scripts/pipeline/phase2_llm_refinement.py \
  --adapter openrouter \
  --model "deepseek/deepseek-coder"  # GRATIS
```

---

## Instalación de Dependencies LLM

Dependiendo del adapter que quieras usar:

```bash
# OpenRouter (solo requests, ya incluido en requirements.txt)
# No necesitas instalar nada extra!

# OpenAI (si prefieres directo sin OpenRouter)
pip install openai

# Anthropic (si prefieres directo sin OpenRouter)
pip install anthropic

# HuggingFace (CodeGemma, CodeLlama, etc.)
pip install transformers torch accelerate

# Ollama (ya tiene requests en requirements.txt)
```

O instala todos:
```bash
pip install -r requirements_llm.txt
```

---

## Uso Rápido

### 1. Test del Adapter

Verifica que tu adapter funcione:

```bash
# OpenRouter (RECOMENDADO)
python scripts/testing/test_llm_adapters.py \
  --adapter openrouter \
  --model "deepseek/deepseek-coder"

# HuggingFace (CodeGemma)
python scripts/testing/test_llm_adapters.py \
  --adapter huggingface \
  --model "google/codegemma-7b-it"

# Ollama
python scripts/testing/test_llm_adapters.py --adapter ollama
```

### 2. Ejecutar Pipeline con LLM

```bash
# Fase 1: Generar baseline
python scripts/pipeline/phase1_generate_baseline.py --limit 10

# Fase 2: Refinar con LLM (OpenRouter)
python scripts/pipeline/phase2_llm_refinement.py \
  --adapter openrouter \
  --model "deepseek/deepseek-coder"

# Continuar con fases 3-5...
```

---

## Ejemplos Comunes

### OpenRouter - DeepSeek Coder (GRATIS, especializado en código)
```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
python scripts/pipeline/phase2_llm_refinement.py \
  --adapter openrouter \
  --model "deepseek/deepseek-coder"
```

### OpenRouter - Llama 3 70B (GRATIS, alta calidad)
```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
python scripts/pipeline/phase2_llm_refinement.py \
  --adapter openrouter \
  --model "meta-llama/llama-3-70b-instruct"
```

### OpenRouter - Claude Sonnet (PAGO, máxima calidad)
```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
python scripts/pipeline/phase2_llm_refinement.py \
  --adapter openrouter \
  --model "anthropic/claude-3-sonnet"
```

### HuggingFace (Tu modelo finetuneado)
```bash
python scripts/pipeline/phase2_llm_refinement.py \
  --adapter huggingface \
  --model "croko22/codegemma-test-refiner" \
  --temperature 0.2 \
  --max-tokens 2048
```

### Ollama (Local)
```bash
ollama pull codellama:13b
python scripts/pipeline/phase2_llm_refinement.py \
  --adapter ollama \
  --model codellama:13b
```

---

## Comparación Rápida

| Opción | Costo | Calidad | Setup |
|--------|-------|---------|-------|
| **OpenRouter + DeepSeek** | **GRATIS** | ⭐⭐⭐⭐ | Fácil |
| **OpenRouter + Llama 3** | **GRATIS** | ⭐⭐⭐⭐ | Fácil |
| OpenRouter + Claude | $1.50/100 tests | ⭐⭐⭐⭐⭐ | Fácil |
| HuggingFace | GPU propia | ⭐⭐⭐⭐ | Complejo |
| Ollama | GPU propia | ⭐⭐⭐ | Medio |

---

## Más Información

📖 **Guía Completa**: `docs/guides/LLM_ADAPTERS.md`

Esta guía incluye:
- Comparación detallada de adapters
- Todos los modelos disponibles en OpenRouter
- Requisitos hardware para HuggingFace
- Estimaciones de costo
- Troubleshooting
