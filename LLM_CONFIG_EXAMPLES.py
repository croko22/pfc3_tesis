#!/usr/bin/env python3
"""
Configuraciones recomendadas para diferentes escenarios de uso.
Copia y modifica según tu caso de uso.
"""

# ============================================================================
# ESCENARIO 1: Investigación Académica con HuggingFace
# ============================================================================
# Recomendado para: Tesis, papers, reproducibilidad
# Hardware: GPU con 8GB+ VRAM (RTX 3070, 4060 Ti, etc.)

ACADEMIC_CONFIG = {
    "adapter": "huggingface",
    "model": "google/codegemma-7b-it",  # O tu modelo finetuneado
    "temperature": 0.2,
    "max_tokens": 2048,
    "notes": """
    Ventajas:
    - Gratis (solo necesitas GPU)
    - Reproducible (puedes compartir el modelo exacto)
    - Sin límites de rate
    - Datos no salen de tu máquina
    
    Comando:
    python scripts/pipeline/phase2_llm_refinement.py \
      --adapter huggingface \
      --model "google/codegemma-7b-it" \
      --temperature 0.2 \
      --max-tokens 2048
    """
}

# ============================================================================
# ESCENARIO 2: Prototipado Rápido con GPT-3.5
# ============================================================================
# Recomendado para: Exploración inicial, pruebas rápidas
# Hardware: Cualquiera (API cloud)

PROTOTYPE_CONFIG = {
    "adapter": "openai",
    "model": "gpt-3.5-turbo",
    "temperature": 0.2,
    "max_tokens": 2000,
    "notes": """
    Ventajas:
    - Setup inmediato (solo API key)
    - Rápido
    - Económico ($0.002 por test aprox.)
    
    Setup:
    export OPENAI_API_KEY="sk-..."
    
    Comando:
    python scripts/pipeline/phase2_llm_refinement.py \
      --adapter openai \
      --model gpt-3.5-turbo \
      --temperature 0.2
    
    Costo estimado (100 tests): ~$0.20
    """
}

# ============================================================================
# ESCENARIO 3: Máxima Calidad con GPT-4
# ============================================================================
# Recomendado para: Tests críticos, producción, paper final
# Hardware: Cualquiera (API cloud)

PRODUCTION_CONFIG = {
    "adapter": "openai",
    "model": "gpt-4-turbo-preview",
    "temperature": 0.1,  # Más bajo para consistencia
    "max_tokens": 4000,
    "notes": """
    Ventajas:
    - Máxima calidad de refinamiento
    - Mejor seguimiento de instrucciones complejas
    - Menos errores de compilación
    
    Desventajas:
    - Más caro ($0.03 por test aprox.)
    - Rate limits más estrictos
    
    Setup:
    export OPENAI_API_KEY="sk-..."
    
    Comando:
    python scripts/pipeline/phase2_llm_refinement.py \
      --adapter openai \
      --model gpt-4-turbo-preview \
      --temperature 0.1 \
      --max-tokens 4000
    
    Costo estimado (100 tests): ~$3.00
    """
}

# ============================================================================
# ESCENARIO 4: Local/Privado con Ollama
# ============================================================================
# Recomendado para: Datos sensibles, sin internet, exploración
# Hardware: GPU con 6GB+ VRAM (para modelo 13B)

LOCAL_CONFIG = {
    "adapter": "ollama",
    "model": "codellama:13b",
    "temperature": 0.2,
    "max_tokens": 2048,
    "notes": """
    Ventajas:
    - Totalmente local (privacidad)
    - Sin costos recurrentes
    - Sin límites de rate
    - Setup simple
    
    Setup:
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull codellama:13b
    
    Comando:
    python scripts/pipeline/phase2_llm_refinement.py \
      --adapter ollama \
      --model codellama:13b \
      --temperature 0.2
    
    Alternativas:
    - codellama:7b (más rápido, menos VRAM)
    - deepseek-coder:6.7b (buen equilibrio)
    - codellama:34b (mejor calidad, 20GB VRAM)
    """
}

# ============================================================================
# ESCENARIO 5: Claude 3 (Alternativa a GPT-4)
# ============================================================================
# Recomendado para: Prompts complejos, contexto largo
# Hardware: Cualquiera (API cloud)

CLAUDE_CONFIG = {
    "adapter": "anthropic",
    "model": "claude-3-sonnet-20240229",
    "temperature": 0.2,
    "max_tokens": 4000,
    "notes": """
    Ventajas:
    - Excelente seguimiento de instrucciones
    - Contexto muy largo (200K tokens)
    - Precio competitivo vs GPT-4
    - Bueno para explicar razonamiento
    
    Setup:
    export ANTHROPIC_API_KEY="sk-ant-..."
    
    Comando:
    python scripts/pipeline/phase2_llm_refinement.py \
      --adapter anthropic \
      --model claude-3-sonnet-20240229 \
      --temperature 0.2
    
    Modelos disponibles:
    - claude-3-haiku-20240307 (rápido, económico)
    - claude-3-sonnet-20240229 (equilibrio)
    - claude-3-opus-20240229 (máxima calidad)
    """
}

# ============================================================================
# ESCENARIO 6: Modelo Finetuneado Custom
# ============================================================================
# Recomendado para: Máxima personalización, paper con contribución
# Hardware: GPU con 14GB+ VRAM (RTX 3090, 4090, A100)

FINETUNED_CONFIG = {
    "adapter": "huggingface",
    "model": "croko22/codegemma-test-refiner",  # TU modelo
    "temperature": 0.2,
    "max_tokens": 2048,
    "notes": """
    Si tienes un modelo finetuneado en HuggingFace:
    
    1. Sube tu modelo:
       huggingface-cli login
       model.push_to_hub("tu-usuario/codegemma-test-refiner")
    
    2. Usa en el pipeline:
       python scripts/pipeline/phase2_llm_refinement.py \
         --adapter huggingface \
         --model "tu-usuario/codegemma-test-refiner" \
         --temperature 0.2
    
    3. Documenta en tu paper:
       - Arquitectura base (CodeGemma 7B)
       - Dataset de finetuning
       - Hiperparámetros
       - Modelo disponible en HuggingFace
    
    Ventajas:
    - Especializado en tu dominio
    - Contribución metodológica al paper
    - Mejor rendimiento que modelos base
    """
}

# ============================================================================
# ESCENARIO 7: GPU Limitada (Cuantización)
# ============================================================================
# Recomendado para: GPUs con <8GB VRAM
# Hardware: RTX 3060 (6GB), GTX 1660 Ti, etc.

LOW_VRAM_CONFIG = {
    "adapter": "huggingface",
    "model": "google/codegemma-2b",  # Modelo más pequeño
    "temperature": 0.2,
    "max_tokens": 1500,
    "notes": """
    Para GPUs con poca VRAM, hay varias opciones:
    
    OPCIÓN 1: Modelo más pequeño
    python scripts/pipeline/phase2_llm_refinement.py \
      --adapter huggingface \
      --model "google/codegemma-2b" \
      --temperature 0.2
    
    OPCIÓN 2: Cuantización 4-bit (modifica adapter en código)
    # En HuggingFaceAdapter.__init__():
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    OPCIÓN 3: Ollama (gestión automática)
    ollama pull codellama:7b
    python scripts/pipeline/phase2_llm_refinement.py \
      --adapter ollama \
      --model codellama:7b
    
    Comparación VRAM:
    - CodeGemma 7B (float16): 14GB
    - CodeGemma 7B (4-bit): 5GB
    - CodeGemma 2B (float16): 4GB
    - Ollama CodeLlama 7B: 6GB
    """
}


# ============================================================================
# COMPARACIÓN RÁPIDA
# ============================================================================

COMPARISON_TABLE = """
┌─────────────────┬──────────┬──────────┬──────────┬───────────┐
│ Escenario       │ Costo    │ Velocidad│ Calidad  │ Setup     │
├─────────────────┼──────────┼──────────┼──────────┼───────────┤
│ Academic (HF)   │ Gratis*  │ Media    │ Alta     │ Complejo  │
│ Prototype (3.5) │ Bajo     │ Rápida   │ Media    │ Simple    │
│ Production (4)  │ Alto     │ Media    │ Muy Alta │ Simple    │
│ Local (Ollama)  │ Gratis*  │ Rápida   │ Media    │ Simple    │
│ Claude          │ Medio    │ Media    │ Alta     │ Simple    │
│ Finetuned       │ Gratis*  │ Media    │ Muy Alta │ Complejo  │
│ Low VRAM        │ Gratis*  │ Lenta    │ Media    │ Medio     │
└─────────────────┴──────────┴──────────┴──────────┴───────────┘

* Requiere GPU propia
"""


if __name__ == "__main__":
    print("="*80)
    print("CONFIGURACIONES RECOMENDADAS - LLM ADAPTERS")
    print("="*80)
    print("\nEste archivo contiene configuraciones pre-definidas para diferentes")
    print("escenarios de uso. Elige el que mejor se adapte a tu caso:\n")
    
    scenarios = [
        ("1. Investigación Académica", ACADEMIC_CONFIG),
        ("2. Prototipado Rápido", PROTOTYPE_CONFIG),
        ("3. Máxima Calidad", PRODUCTION_CONFIG),
        ("4. Local/Privado", LOCAL_CONFIG),
        ("5. Claude (Alternativa GPT-4)", CLAUDE_CONFIG),
        ("6. Modelo Finetuneado", FINETUNED_CONFIG),
        ("7. GPU Limitada", LOW_VRAM_CONFIG),
    ]
    
    for title, config in scenarios:
        print(f"\n{title}")
        print("-" * 70)
        print(f"Adapter: {config['adapter']}")
        print(f"Model: {config['model']}")
        print(f"Temperature: {config['temperature']}")
        print(f"Max Tokens: {config['max_tokens']}")
        print(config['notes'])
    
    print("\n" + "="*80)
    print("COMPARACIÓN")
    print("="*80)
    print(COMPARISON_TABLE)
    
    print("\n💡 Para más detalles: docs/guides/LLM_ADAPTERS.md")
    print("="*80)
