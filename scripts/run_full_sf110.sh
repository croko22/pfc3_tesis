#!/bin/bash
#
# Script para ejecutar el pipeline completo sobre TODO el SF110
#
# ADVERTENCIA: Esto puede tomar MUCHAS HORAS
# SF110 tiene ~1000 clases, con 60s por clase = ~16 horas solo Phase 1
#
# Uso:
#   ./scripts/run_full_sf110.sh              # Pipeline completo
#   ./scripts/run_full_sf110.sh --phase 1    # Solo fase 1 (baseline)
#

set -e  # Exit on error

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}    EJECUCIÓN COMPLETA SF110${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

# Verificar que estamos en el directorio correcto
if [ ! -f "scripts/pipeline/phase1_generate_baseline.py" ]; then
    echo -e "${RED}❌ Error: Ejecuta este script desde la raíz del proyecto${NC}"
    exit 1
fi

# Contar clases en SF110
if [ -f "data/SF110-binary/classes.csv" ]; then
    TOTAL_CLASSES=$(tail -n +2 data/SF110-binary/classes.csv | wc -l)
    echo -e "${YELLOW}📊 Total de clases en SF110: ${TOTAL_CLASSES}${NC}"
    
    # Estimar tiempo (asumiendo 60s por clase por fase)
    ESTIMATE_HOURS=$((TOTAL_CLASSES * 60 / 3600))
    echo -e "${YELLOW}⏱️  Tiempo estimado Phase 1: ~${ESTIMATE_HOURS} horas${NC}"
    echo ""
fi

# Confirmar
echo -e "${RED}⚠️  ADVERTENCIA: Esto puede tomar MUCHAS HORAS${NC}"
echo -e "¿Deseas continuar? (escribe 'yes' para confirmar)"
read -r response

if [ "$response" != "yes" ]; then
    echo -e "${YELLOW}❌ Cancelado${NC}"
    exit 0
fi

echo ""
echo -e "${GREEN}✅ Iniciando ejecución completa...${NC}"
echo ""

# Timestamp de inicio
START_TIME=$(date +%s)
echo "Inicio: $(date)"

# Ejecutar pipeline
if [ "$1" = "--phase" ] && [ -n "$2" ]; then
    # Ejecutar solo una fase
    echo -e "${BLUE}Ejecutando solo Phase $2${NC}"
    python scripts/pipeline/run_pipeline.py --full --phase "$2"
else
    # Pipeline completo
    echo -e "${BLUE}Ejecutando pipeline completo${NC}"
    python scripts/pipeline/run_pipeline.py --full
fi

# Timestamp de fin
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_HOURS=$((ELAPSED / 3600))
ELAPSED_MINS=$(((ELAPSED % 3600) / 60))

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ COMPLETADO${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo "Fin: $(date)"
echo "Tiempo total: ${ELAPSED_HOURS}h ${ELAPSED_MINS}m"
echo ""
