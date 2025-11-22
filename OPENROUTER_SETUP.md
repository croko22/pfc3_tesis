# 🚀 Setup Rápido con OpenRouter

## ¿Por qué OpenRouter?

✅ **Una API key para todos** - GPT-4, Claude, Gemini, Llama, DeepSeek  
✅ **Modelos GRATIS** - DeepSeek, Llama 3, Mistral  
✅ **Precios competitivos** - A menudo más barato que directo  
✅ **Sin límites estrictos** - Mejor que OpenAI/Anthropic directo  
✅ **Facturación transparente** - Ves cuánto gastas en tiempo real  

---

## Setup en 3 Pasos

### 1. Obtén tu API Key (2 minutos)

```bash
# Visita: https://openrouter.ai/
# Click en "Sign In" → "Continue with Google/GitHub"
# Entra a: https://openrouter.ai/keys
# Click "Create Key" → Copia tu sk-or-v1-...
```

### 2. Configura en tu terminal

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."

# Para que sea permanente (añade a ~/.bashrc o ~/.config/fish/config.fish):
echo 'export OPENROUTER_API_KEY="sk-or-v1-..."' >> ~/.bashrc
```

### 3. ¡Úsalo!

```bash
# Test rápido
python scripts/testing/test_llm_adapters.py \
  --adapter openrouter \
  --model "deepseek/deepseek-coder"

# Pipeline completo
python scripts/pipeline/phase2_llm_refinement.py \
  --adapter openrouter \
  --model "deepseek/deepseek-coder"
```

---

## 🆓 Modelos Gratuitos

### Llama 3 8B (Rápido y GRATIS)
```bash
python scripts/pipeline/phase2_llm_refinement.py \
  --adapter openrouter \
  --model "meta-llama/llama-3-8b-instruct"
```

**Por qué es bueno:**
- Muy rápido
- GRATIS
- Bueno para exploración rápida
- Modelo oficial de Meta

### Llama 3 70B (Alta calidad GRATIS)
```bash
python scripts/pipeline/phase2_llm_refinement.py \
  --adapter openrouter \
  --model "meta-llama/llama-3-70b-instruct"
```

**Por qué es bueno:**
- Modelo grande (70B params)
- Muy buena calidad general
- GRATIS
- Buen seguimiento de instrucciones

### Gemini Flash (Rápido, casi gratis)
```bash
python scripts/pipeline/phase2_llm_refinement.py \
  --adapter openrouter \
  --model "google/gemini-flash-1.5"
```

**Por qué es bueno:**
- Extremadamente rápido
- Casi gratis ($0.01 por 100 tests)
- Buena calidad
- De Google

---

## 💰 Modelos de Pago (Alta Calidad)

### Claude 3 Sonnet (Equilibrio)
```bash
python scripts/pipeline/phase2_llm_refinement.py \
  --adapter openrouter \
  --model "anthropic/claude-3-sonnet"
```

**Costo:** ~$1.50 por 100 tests  
**Por qué vale la pena:**
- Excelente para código complejo
- Muy buen seguimiento de instrucciones
- Menos errores que modelos gratuitos

### GPT-4 Turbo (Máxima calidad)
```bash
python scripts/pipeline/phase2_llm_refinement.py \
  --adapter openrouter \
  --model "openai/gpt-4-turbo"
```

**Costo:** ~$3.00 por 100 tests  
**Por qué vale la pena:**
- Mejor calidad absoluta
- Refinamientos más sofisticados
- Para paper final o producción

### Gemini Pro 1.5 (Barato + contexto largo)
```bash
python scripts/pipeline/phase2_llm_refinement.py \
  --adapter openrouter \
  --model "google/gemini-pro-1.5"
```

**Costo:** ~$0.12 por 100 tests  
**Por qué vale la pena:**
- Muy barato
- Contexto de 1M tokens
- Bueno para tests grandes

---

## 📊 Comparación Rápida

| Modelo | Costo/100 tests | Calidad | Velocidad | Uso Recomendado |
|--------|-----------------|---------|-----------|-----------------|
| **DeepSeek Coder** | **GRATIS** | ⭐⭐⭐⭐ | ⚡⚡⚡⚡ | **Exploración, desarrollo** |
| **Llama 3 70B** | **GRATIS** | ⭐⭐⭐⭐ | ⚡⚡⚡ | **Investigación académica** |
| Llama 3 8B | GRATIS | ⭐⭐⭐ | ⚡⚡⚡⚡⚡ | Prototipado rápido |
| Gemini Flash | $0.01 | ⭐⭐⭐ | ⚡⚡⚡⚡⚡ | Exploración masiva |
| Gemini Pro 1.5 | $0.12 | ⭐⭐⭐⭐ | ⚡⚡⚡⚡ | Tests complejos/largos |
| GPT-3.5 | $0.05 | ⭐⭐⭐ | ⚡⚡⚡⚡ | Baseline OpenAI |
| Claude Haiku | $0.10 | ⭐⭐⭐⭐ | ⚡⚡⚡⚡ | Rápido + calidad |
| **Claude Sonnet** | **$1.50** | **⭐⭐⭐⭐⭐** | **⚡⚡⚡** | **Paper final** |
| GPT-4 Turbo | $3.00 | ⭐⭐⭐⭐⭐ | ⚡⚡ | Máxima calidad |
| Claude Opus | $7.50 | ⭐⭐⭐⭐⭐ | ⚡⚡ | Casos críticos |

---

## 🎯 Workflow Recomendado

### Para Tesis/Paper

```bash
# FASE 1: Exploración (GRATIS)
# Prueba con DeepSeek o Llama 3 para ver si funciona
python scripts/pipeline/phase2_llm_refinement.py \
  --adapter openrouter \
  --model "deepseek/deepseek-coder" \
  --limit 10

# FASE 2: Desarrollo (~GRATIS)
# Una vez que funciona, procesa más datos
python scripts/pipeline/phase2_llm_refinement.py \
  --adapter openrouter \
  --model "meta-llama/llama-3-70b-instruct" \
  --limit 100

# FASE 3: Paper Final ($1.50)
# Para resultados finales del paper, usa Claude Sonnet
python scripts/pipeline/phase2_llm_refinement.py \
  --adapter openrouter \
  --model "anthropic/claude-3-sonnet"
```

**Costo total:** ~$1.50 (solo fase 3)  
**Tiempo ahorrado:** Semanas de desarrollo

---

## 🔍 Ver Uso y Gastos

OpenRouter te muestra en tiempo real cuánto gastas:

1. Visita: https://openrouter.ai/activity
2. Ve tu uso por modelo
3. Exporta para tu paper (transparency)

---

## 💡 Tips Pro

### Tip 1: Compara modelos fácilmente
```bash
# Prueba 3 modelos diferentes
for model in "deepseek/deepseek-coder" "meta-llama/llama-3-70b-instruct" "anthropic/claude-3-sonnet"
do
  echo "Testing $model..."
  python scripts/testing/test_llm_adapters.py --adapter openrouter --model "$model"
done
```

### Tip 2: Empieza siempre con modelos gratis
```bash
# Primero valida que funcione (GRATIS)
python scripts/pipeline/phase2_llm_refinement.py \
  --adapter openrouter \
  --model "deepseek/deepseek-coder" \
  --limit 5

# Luego escala con mejor modelo si es necesario
```

### Tip 3: Usa temperatura baja para consistencia
```bash
# Temperatura 0.1 = más determinista (mejor para tests)
python scripts/pipeline/phase2_llm_refinement.py \
  --adapter openrouter \
  --model "anthropic/claude-3-sonnet" \
  --temperature 0.1
```

### Tip 4: Para paper, documenta el modelo exacto
```bash
# En tu paper:
"We used DeepSeek Coder (deepseek/deepseek-coder) via OpenRouter API
for test refinement, with temperature=0.2 and max_tokens=2048"
```

---

## ❓ Troubleshooting

### Error: API key not set
```bash
# Asegúrate que la variable esté configurada
echo $OPENROUTER_API_KEY

# Si no sale nada:
export OPENROUTER_API_KEY="sk-or-v1-..."
```

### Error: Rate limit
```bash
# OpenRouter tiene límites más generosos que otros
# Pero si llegas al límite, añade un delay:
python scripts/pipeline/phase2_llm_refinement.py \
  --adapter openrouter \
  --model "deepseek/deepseek-coder" \
  --delay 1  # 1 segundo entre requests
```

### Quiero probar sin gastar
```bash
# Usa SOLO modelos gratuitos:
# - deepseek/deepseek-coder
# - meta-llama/llama-3-70b-instruct
# - meta-llama/llama-3-8b-instruct
# - mistralai/mistral-7b-instruct
```

---

## 📚 Más Info

- **Todos los modelos:** https://openrouter.ai/models
- **Precios:** https://openrouter.ai/models (click en cada modelo)
- **Docs:** https://openrouter.ai/docs
- **Activity:** https://openrouter.ai/activity (ver tu uso)

---

## 🎓 Para tu Tesis

OpenRouter es IDEAL para tesis porque:

1. **Reproducibilidad:** Puedes documentar el modelo exacto usado
2. **Transparencia:** Puedes mostrar costos y uso
3. **Flexibilidad:** Puedes comparar múltiples modelos fácilmente
4. **Gratis/Barato:** Modelos gratuitos excelentes + opciones premium

En tu metodología puedes escribir:

> "We evaluated our approach using multiple LLMs accessed via OpenRouter API:
> - DeepSeek Coder (free tier) for exploratory development
> - Llama 3 70B (free tier) for baseline results  
> - Claude 3 Sonnet (paid tier, $15/1M tokens) for final results
> 
> Total cost for 500 test refinements: $7.50"

Esto muestra profesionalismo y transparencia 🚀
