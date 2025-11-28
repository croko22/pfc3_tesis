import json
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from ..core.config import cfg

class AnalysisPhase:
    """Phase 5: Analysis and Visualization."""
    
    def __init__(self):
        self.output_dir = cfg.base_dir / "figures"
        self.output_dir.mkdir(exist_ok=True)
        
        # Style
        sns.set_style("whitegrid")
        sns.set_palette("colorblind")
        plt.rcParams['figure.dpi'] = 300
        
    def run(self):
        print("Phase 5: Analysis & Visualization")
        
        data = self._load_data()
        if not data.get('baseline') or not data.get('valid'):
            print("⚠️  Missing data from previous phases")
            return
            
        df = self._create_dataframe(data)
        if df.empty:
            print("⚠️  No paired data for comparison")
            return
            
        print(f"📊 Analyzing {len(df)} classes")
        
        self._plot_coverage(df)
        self._plot_mutation(df)
        self._save_summary(df)
        
    def _load_data(self) -> Dict:
        data = {}
        base_file = cfg.base_dir / "generated_tests/baseline/T_base_results.json"
        valid_file = cfg.base_dir / "generated_tests/validated/T_valid_results.json"
        
        if base_file.exists():
            with open(base_file) as f: data['baseline'] = json.load(f)
        if valid_file.exists():
            with open(valid_file) as f: data['valid'] = json.load(f)
            
        return data
        
    def _create_dataframe(self, data: Dict) -> pd.DataFrame:
        rows = []
        base_map = {r['class']: r for r in data['baseline'] if r.get('success')}
        
        for valid in data['valid']:
            if not valid.get('verified'): continue
            
            cls = valid['class']
            base = base_map.get(cls)
            if not base: continue
            
            # Extract metrics (assuming structure from previous phases)
            # Note: Phase 4 (Evaluation) should have added metrics to T_valid or separate file
            # For now, we'll use what's available or placeholders if Phase 4 didn't run fully
            
            rows.append({
                'class': cls,
                'baseline_cov': base.get('coverage', {}).get('Line', 0),
                'valid_cov': valid.get('coverage', {}).get('Line', 0), # Placeholder if not in valid
                'baseline_mut': base.get('mutation_score', 0),
                'valid_mut': valid.get('mutation_score', 0)
            })
            
        return pd.DataFrame(rows)

    def _plot_coverage(self, df: pd.DataFrame):
        if 'baseline_cov' not in df.columns: return
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.boxplot([df['baseline_cov'], df['valid_cov']], labels=['Baseline', 'Valid'])
        plt.title('Coverage Distribution')
        
        plt.subplot(1, 2, 2)
        plt.scatter(df['baseline_cov'], df['valid_cov'])
        plt.plot([0, 100], [0, 100], 'r--')
        plt.xlabel('Baseline')
        plt.ylabel('Valid')
        plt.title('Coverage Improvement')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'coverage_comparison.png')
        plt.close()
        print(f"✅ Saved coverage plot to {self.output_dir}")

    def _plot_mutation(self, df: pd.DataFrame):
        if 'baseline_mut' not in df.columns: return
        
        plt.figure(figsize=(6, 4))
        plt.boxplot([df['baseline_mut'], df['valid_mut']], labels=['Baseline', 'Valid'])
        plt.title('Mutation Score')
        plt.savefig(self.output_dir / 'mutation_comparison.png')
        plt.close()

    def _save_summary(self, df: pd.DataFrame):
        summary = df.describe()
        summary.to_csv(self.output_dir / 'summary_stats.csv')
        print(f"✅ Saved summary stats to {self.output_dir}")
