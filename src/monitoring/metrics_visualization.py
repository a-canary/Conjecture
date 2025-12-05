#!/usr/bin/env python3
"""
Metrics Visualization Tools for Conjecture
Provides charts and graphs for metrics analysis and reporting
"""

import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

# Visualization imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    import pandas as pd
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Charts will be generated as text.")

logger = logging.getLogger(__name__)


@dataclass
class ChartConfig:
    """Configuration for chart generation"""
    title: str
    x_label: str
    y_label: str
    chart_type: str = "line"  # line, bar, scatter, histogram, box
    width: int = 10
    height: int = 6
    save_path: Optional[str] = None
    show_grid: bool = True
    color_palette: str = "viridis"


class MetricsVisualizer:
    """Advanced visualization tools for metrics analysis"""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("research/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color schemes
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'warning': '#ff7f0e',
            'error': '#d62728',
            'info': '#9467bd'
        }
        
        # Chart style configuration
        if VISUALIZATION_AVAILABLE:
            plt.style.use('classic')
            sns.set_palette("Set1")
    
    def create_performance_timeline(self, 
                              metrics_data: List[Dict[str, Any]], 
                              config: ChartConfig = None) -> str:
        """Create timeline chart of performance metrics"""
        
        if config is None:
            config = ChartConfig(
                title="Performance Timeline",
                x_label="Time",
                y_label="Response Time (seconds)",
                chart_type="line"
            )
        
        if not VISUALIZATION_AVAILABLE:
            return self._create_text_timeline(metrics_data, config)
        
        # Extract data
        timestamps = []
        values = []
        models = set()
        
        for entry in metrics_data:
            timestamp = datetime.fromisoformat(entry.get('timestamp', datetime.utcnow().isoformat()))
            timestamps.append(timestamp)
            values.append(entry.get('value', 0))
            models.add(entry.get('model', 'unknown'))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(config.width, config.height))
        
        # Plot by model if multiple models
        if len(models) > 1:
            for model in models:
                model_data = [e for e in metrics_data if e.get('model') == model]
                model_timestamps = [datetime.fromisoformat(e.get('timestamp')) for e in model_data]
                model_values = [e.get('value', 0) for e in model_data]
                
                ax.plot(model_timestamps, model_values, 
                       label=model, marker='o', linewidth=2)
        else:
            ax.plot(timestamps, values, marker='o', linewidth=2, color=self.colors['primary'])
        
        # Formatting
        ax.set_title(config.title, fontsize=14, fontweight='bold')
        ax.set_xlabel(config.x_label, fontsize=12)
        ax.set_ylabel(config.y_label, fontsize=12)
        
        if config.show_grid:
            ax.grid(True, alpha=0.3)
        
        if len(models) > 1:
            ax.legend()
        
        # Format x-axis for dates
        if timestamps and all(isinstance(t, datetime) for t in timestamps):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save chart
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_timeline_{timestamp}.png"
        filepath = self.output_dir / filename
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance timeline saved to {filepath}")
        return str(filepath)
    
    def create_model_comparison_chart(self, 
                                 comparison_data: Dict[str, Dict[str, float]], 
                                 config: ChartConfig = None) -> str:
        """Create comparison chart between models"""
        
        if config is None:
            config = ChartConfig(
                title="Model Performance Comparison",
                x_label="Model",
                y_label="Score",
                chart_type="bar"
            )
        
        if not VISUALIZATION_AVAILABLE:
            return self._create_text_comparison(comparison_data, config)
        
        # Prepare data
        models = list(comparison_data.keys())
        metrics = set()
        
        for model_data in comparison_data.values():
            metrics.update(model_data.keys())
        
        metrics = list(metrics)
        
        # Create figure with subplots for each metric
        fig, axes = plt.subplots(len(metrics), 1, 
                                figsize=(config.width, config.height * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            ax = axes[i] if len(metrics) > 1 else axes
            
            values = []
            for model in models:
                values.append(comparison_data[model].get(metric, 0))
            
            bars = ax.bar(models, values, color=self.colors['primary'])
            ax.set_title(f"{metric.replace('_', ' ').title()}", fontweight='bold')
            ax.set_ylabel(config.y_label)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.3f}', ha='center', va='bottom')
            
            if config.show_grid:
                ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(config.title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save chart
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"model_comparison_{timestamp}.png"
        filepath = self.output_dir / filename
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Model comparison chart saved to {filepath}")
        return str(filepath)
    
    def create_distribution_chart(self, 
                              data: List[float], 
                              config: ChartConfig = None) -> str:
        """Create distribution chart (histogram and box plot)"""
        
        if config is None:
            config = ChartConfig(
                title="Response Time Distribution",
                x_label="Response Time (seconds)",
                y_label="Frequency",
                chart_type="histogram"
            )
        
        if not VISUALIZATION_AVAILABLE:
            return self._create_text_distribution(data, config)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, 
                                        figsize=(config.width * 1.5, config.height))
        
        # Histogram
        ax1.hist(data, bins=30, color=self.colors['primary'], alpha=0.7, edgecolor='black')
        ax1.set_title('Distribution', fontweight='bold')
        ax1.set_xlabel(config.x_label)
        ax1.set_ylabel(config.y_label)
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(data, vert=True, patch_artist=True,
                   boxprops=dict(facecolor=self.colors['secondary'], alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        ax2.set_title('Box Plot', fontweight='bold')
        ax2.set_ylabel(config.y_label)
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(config.title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save chart
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"distribution_{timestamp}.png"
        filepath = self.output_dir / filename
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Distribution chart saved to {filepath}")
        return str(filepath)
    
    def create_statistical_significance_chart(self, 
                                       test_results: List[Dict[str, Any]], 
                                       config: ChartConfig = None) -> str:
        """Create chart showing statistical significance results"""
        
        if config is None:
            config = ChartConfig(
                title="Statistical Significance Results",
                x_label="Comparison",
                y_label="P-value",
                chart_type="scatter"
            )
        
        if not VISUALIZATION_AVAILABLE:
            return self._create_text_significance(test_results, config)
        
        # Prepare data
        comparisons = []
        p_values = []
        significant = []
        effect_sizes = []
        
        for result in test_results:
            comparisons.append(result.get('comparison', 'Unknown'))
            p_values.append(result.get('p_value', 1.0))
            significant.append(result.get('is_significant', False))
            effect_sizes.append(result.get('effect_size', 0))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(config.width, config.height))
        
        # Color points by significance
        colors = [self.colors['success'] if sig else self.colors['error'] for sig in significant]
        sizes = [abs(es) * 100 + 50 for es in effect_sizes]  # Size by effect size
        
        scatter = ax.scatter(range(len(comparisons)), p_values, 
                          c=colors, s=sizes, alpha=0.7, edgecolors='black')
        
        # Add significance threshold line
        ax.axhline(y=0.05, color=self.colors['warning'], linestyle='--', 
                   alpha=0.7, label='Significance Threshold (p=0.05)')
        
        # Formatting
        ax.set_title(config.title, fontsize=14, fontweight='bold')
        ax.set_xlabel(config.x_label, fontsize=12)
        ax.set_ylabel(config.y_label, fontsize=12)
        ax.set_yscale('log')  # Log scale for p-values
        
        # Set x-axis labels
        ax.set_xticks(range(len(comparisons)))
        ax.set_xticklabels(comparisons, rotation=45, ha='right')
        
        if config.show_grid:
            ax.grid(True, alpha=0.3)
        
        ax.legend()
        plt.tight_layout()
        
        # Save chart
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"statistical_significance_{timestamp}.png"
        filepath = self.output_dir / filename
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Statistical significance chart saved to {filepath}")
        return str(filepath)
    
    def create_pipeline_flow_chart(self, 
                               pipeline_data: Dict[str, Dict[str, Any]], 
                               config: ChartConfig = None) -> str:
        """Create pipeline flow visualization"""
        
        if config is None:
            config = ChartConfig(
                title="Pipeline Performance Flow",
                x_label="Pipeline Stage",
                y_label="Time (seconds)",
                chart_type="bar"
            )
        
        if not VISUALIZATION_AVAILABLE:
            return self._create_text_pipeline_flow(pipeline_data, config)
        
        # Prepare data
        stages = list(pipeline_data.keys())
        avg_times = [pipeline_data[stage].get('average_time', 0) for stage in stages]
        error_rates = [pipeline_data[stage].get('error_rate', 0) * 100 for stage in stages]
        
        # Create figure with secondary y-axis
        fig, ax1 = plt.subplots(figsize=(config.width, config.height))
        
        # Bar chart for times
        bars = ax1.bar(stages, avg_times, color=self.colors['primary'], alpha=0.7)
        ax1.set_title(config.title, fontsize=14, fontweight='bold')
        ax1.set_xlabel(config.x_label, fontsize=12)
        ax1.set_ylabel('Average Time (seconds)', fontsize=12, color=self.colors['primary'])
        ax1.tick_params(axis='y', labelcolor=self.colors['primary'])
        
        # Add value labels on bars
        for bar, time_val in zip(bars, avg_times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.3f}s', ha='center', va='bottom')
        
        # Secondary axis for error rates
        ax2 = ax1.twinx()
        line = ax2.plot(stages, error_rates, color=self.colors['error'], 
                       marker='o', linewidth=2, label='Error Rate')
        ax2.set_ylabel('Error Rate (%)', fontsize=12, color=self.colors['error'])
        ax2.tick_params(axis='y', labelcolor=self.colors['error'])
        
        if config.show_grid:
            ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save chart
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"pipeline_flow_{timestamp}.png"
        filepath = self.output_dir / filename
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Pipeline flow chart saved to {filepath}")
        return str(filepath)
    
    def create_heatmap(self, 
                     data: Dict[str, Dict[str, float]], 
                     config: ChartConfig = None) -> str:
        """Create heatmap for correlation or performance matrix"""
        
        if config is None:
            config = ChartConfig(
                title="Performance Heatmap",
                x_label="Metric",
                y_label="Model",
                chart_type="heatmap"
            )
        
        if not VISUALIZATION_AVAILABLE:
            return self._create_text_heatmap(data, config)
        
        # Convert to DataFrame for seaborn
        df = pd.DataFrame(data)
        
        # Create figure
        plt.figure(figsize=(config.width, config.height))
        
        # Create heatmap
        sns.heatmap(df, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, fmt='.3f', cbar_kws={'label': 'Value'})
        
        plt.title(config.title, fontsize=14, fontweight='bold')
        plt.xlabel(config.x_label, fontsize=12)
        plt.ylabel(config.y_label, fontsize=12)
        plt.tight_layout()
        
        # Save chart
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"heatmap_{timestamp}.png"
        filepath = self.output_dir / filename
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Heatmap saved to {filepath}")
        return str(filepath)
    
    def _create_text_timeline(self, metrics_data: List[Dict[str, Any]], 
                           config: ChartConfig) -> str:
        """Create text-based timeline chart"""
        
        lines = [
            f"# {config.title}",
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Timeline Data",
            ""
        ]
        
        # Group by model if multiple models
        models = set(entry.get('model', 'unknown') for entry in metrics_data)
        
        if len(models) > 1:
            for model in sorted(models):
                model_data = [e for e in metrics_data if e.get('model') == model]
                lines.extend([
                    f"### {model}",
                    "",
                    "| Time | Value |",
                    "|-------|-------|"
                ])
                
                for entry in model_data:
                    timestamp = entry.get('timestamp', '')
                    value = entry.get('value', 0)
                    lines.append(f"| {timestamp} | {value:.3f} |")
                
                lines.append("")
        else:
            lines.extend([
                "| Time | Value |",
                "|-------|-------|"
            ])
            
            for entry in metrics_data:
                timestamp = entry.get('timestamp', '')
                value = entry.get('value', 0)
                lines.append(f"| {timestamp} | {value:.3f} |")
        
        return "\n".join(lines)
    
    def _create_text_comparison(self, comparison_data: Dict[str, Dict[str, float]], 
                           config: ChartConfig) -> str:
        """Create text-based comparison chart"""
        
        lines = [
            f"# {config.title}",
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Model Comparison",
            ""
        ]
        
        models = list(comparison_data.keys())
        metrics = set()
        
        for model_data in comparison_data.values():
            metrics.update(model_data.keys())
        
        metrics = list(metrics)
        
        # Create table
        header = "| Model | " + " | ".join(m.replace('_', ' ').title() for m in metrics) + " |"
        separator = "|--------|" + "|".join(["--------"] * len(metrics)) + "|"
        
        lines.extend([header, separator])
        
        for model in models:
            row = f"| {model} |"
            for metric in metrics:
                value = comparison_data[model].get(metric, 0)
                row += f" {value:.3f} |"
            lines.append(row)
        
        return "\n".join(lines)
    
    def _create_text_distribution(self, data: List[float], config: ChartConfig) -> str:
        """Create text-based distribution chart"""
        
        if not data:
            return f"# {config.title}\n\nNo data available."
        
        lines = [
            f"# {config.title}",
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Distribution Statistics",
            "",
            f"- **Count**: {len(data)}",
            f"- **Mean**: {statistics.mean(data):.3f}",
            f"- **Median**: {statistics.median(data):.3f}",
            f"- **Std Dev**: {statistics.stdev(data):.3f}",
            f"- **Min**: {min(data):.3f}",
            f"- **Max**: {max(data):.3f}",
            "",
            "## Histogram (Text)",
            ""
        ]
        
        # Create simple text histogram
        bins = 10
        min_val, max_val = min(data), max(data)
        bin_width = (max_val - min_val) / bins
        
        bin_counts = [0] * bins
        for value in data:
            bin_index = min(int((value - min_val) / bin_width), bins - 1)
            bin_counts[bin_index] += 1
        
        for i, count in enumerate(bin_counts):
            bin_start = min_val + i * bin_width
            bin_end = bin_start + bin_width
            bar_length = int(count / max(bin_counts) * 50)  # Scale to 50 chars max
            
            lines.append(f"{bin_start:6.2f}-{bin_end:6.2f} | {'█' * bar_length} ({count})")
        
        return "\n".join(lines)
    
    def _create_text_significance(self, test_results: List[Dict[str, Any]], 
                              config: ChartConfig) -> str:
        """Create text-based significance chart"""
        
        lines = [
            f"# {config.title}",
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Statistical Significance Results",
            "",
            "| Comparison | P-value | Significant | Effect Size | Interpretation |",
            "|------------|----------|------------|-------------|----------------|"
        ]
        
        for result in test_results:
            comparison = result.get('comparison', 'Unknown')
            p_value = result.get('p_value', 1.0)
            is_sig = result.get('is_significant', False)
            effect_size = result.get('effect_size', 0)
            interpretation = result.get('interpretation', '')
            
            sig_marker = "✅" if is_sig else "❌"
            
            lines.append(f"| {comparison} | {p_value:.4f} | {sig_marker} | {effect_size:.3f} | {interpretation} |")
        
        return "\n".join(lines)
    
    def _create_text_pipeline_flow(self, pipeline_data: Dict[str, Dict[str, Any]], 
                               config: ChartConfig) -> str:
        """Create text-based pipeline flow chart"""
        
        lines = [
            f"# {config.title}",
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Pipeline Performance",
            "",
            "| Stage | Avg Time (s) | Error Rate (%) | Throughput (ops/s) |",
            "|--------|----------------|----------------|-------------------|"
        ]
        
        for stage, data in pipeline_data.items():
            avg_time = data.get('average_time', 0)
            error_rate = data.get('error_rate', 0) * 100
            throughput = data.get('throughput', 0)
            
            lines.append(f"| {stage} | {avg_time:.3f} | {error_rate:.1f} | {throughput:.2f} |")
        
        return "\n".join(lines)
    
    def _create_text_heatmap(self, data: Dict[str, Dict[str, float]], 
                          config: ChartConfig) -> str:
        """Create text-based heatmap"""
        
        lines = [
            f"# {config.title}",
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Performance Heatmap",
            ""
        ]
        
        rows = list(data.keys())
        cols = set()
        for row_data in data.values():
            cols.update(row_data.keys())
        cols = list(cols)
        
        # Create header
        header = "| Model | " + " | ".join(cols) + " |"
        separator = "|--------|" + "|".join(["--------"] * len(cols)) + "|"
        lines.extend([header, separator])
        
        # Create rows with color coding
        for row in rows:
            line = f"| {row} |"
            for col in cols:
                value = data[row].get(col, 0)
                
                # Simple color coding using symbols
                if value > 0.8:
                    symbol = "🟢"  # Green
                elif value > 0.6:
                    symbol = "🟡"  # Yellow  
                elif value > 0.4:
                    symbol = "🟠"  # Orange
                else:
                    symbol = "🔴"  # Red
                
                line += f" {symbol} {value:.3f} |"
            
            lines.append(line)
        
        return "\n".join(lines)
    
    def create_dashboard(self, 
                     metrics_data: Dict[str, Any], 
                     save_path: Optional[str] = None) -> str:
        """Create comprehensive dashboard with multiple charts"""
        
        dashboard_html = self._generate_html_dashboard(metrics_data)
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"dashboard_{timestamp}.html"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        
        logger.info(f"Dashboard saved to {filepath}")
        return str(filepath)
    
    def _generate_html_dashboard(self, metrics_data: Dict[str, Any]) -> str:
        """Generate HTML dashboard"""
        
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Conjecture Metrics Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .dashboard { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .chart-container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .chart-title { font-size: 18px; font-weight: bold; margin-bottom: 15px; color: #333; }
        .metric-card { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }
        .metric-value { font-size: 24px; font-weight: bold; color: #2ca02c; }
        .metric-label { font-size: 14px; color: #666; margin-top: 5px; }
        .summary-section { grid-column: 1 / -1; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
    </style>
</head>
<body>
    <h1>Conjecture Metrics Dashboard</h1>
    <p>Generated: {timestamp}</p>
    
    <div class="summary-section">
        <h2>Summary Metrics</h2>
        <div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
            {metric_cards}
        </div>
    </div>
    
    <div class="dashboard">
        {charts}
    </div>
    
    <script>
        {chart_scripts}
    </script>
</body>
</html>
        """
        
        # Generate metric cards
        metric_cards = ""
        if 'summary' in metrics_data:
            summary = metrics_data['summary']
            
            for key, value in summary.items():
                if isinstance(value, (int, float)):
                    display_value = f"{value:.3f}" if isinstance(value, float) else str(value)
                else:
                    display_value = str(value)
                
                metric_cards += f"""
                <div class="metric-card">
                    <div class="metric-value">{display_value}</div>
                    <div class="metric-label">{key.replace('_', ' ').title()}</div>
                </div>
                """
        
        # Generate chart placeholders and scripts
        charts = ""
        chart_scripts = ""
        
        # Add timeline chart if data available
        if 'timeline_data' in metrics_data:
            charts += f"""
            <div class="chart-container">
                <div class="chart-title">Performance Timeline</div>
                <canvas id="timelineChart"></canvas>
            </div>
            """
            
            chart_scripts += self._generate_timeline_script(metrics_data['timeline_data'])
        
        # Add model comparison if data available
        if 'model_comparison' in metrics_data:
            charts += f"""
            <div class="chart-container">
                <div class="chart-title">Model Comparison</div>
                <canvas id="comparisonChart"></canvas>
            </div>
            """
            
            chart_scripts += self._generate_comparison_script(metrics_data['model_comparison'])
        
        return html_template.format(
            timestamp=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            metric_cards=metric_cards,
            charts=charts,
            chart_scripts=chart_scripts
        )
    
    def _generate_timeline_script(self, timeline_data: List[Dict[str, Any]]) -> str:
        """Generate JavaScript for timeline chart"""
        
        # Extract data for JavaScript
        labels = [entry.get('timestamp', '') for entry in timeline_data]
        datasets = {}
        
        for entry in timeline_data:
            model = entry.get('model', 'default')
            if model not in datasets:
                datasets[model] = []
            datasets[model].append(entry.get('value', 0))
        
        # Create dataset configurations
        dataset_configs = []
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, (model, values) in enumerate(datasets.items()):
            dataset_configs.append(f"""
            {{
                label: '{model}',
                data: {values},
                borderColor: '{colors[i % len(colors)]}',
                backgroundColor: '{colors[i % len(colors)]}20',
                tension: 0.1
            }}
            """)
        
        return f"""
    // Timeline Chart
    const timelineCtx = document.getElementById('timelineChart').getContext('2d');
    new Chart(timelineCtx, {{
        type: 'line',
        data: {{
            labels: {labels},
            datasets: [{', '.join(dataset_configs)}]
        }},
        options: {{
            responsive: true,
            scales: {{
                y: {{
                    beginAtZero: true,
                    title: {{
                        display: true,
                        text: 'Response Time (seconds)'
                    }}
                }}
            }}
        }}
    }});
    """
    
    def _generate_comparison_script(self, comparison_data: Dict[str, Dict[str, float]]) -> str:
        """Generate JavaScript for model comparison chart"""
        
        models = list(comparison_data.keys())
        metrics = set()
        
        for model_data in comparison_data.values():
            metrics.update(model_data.keys())
        
        metrics = list(metrics)
        
        # Create datasets for each metric
        dataset_configs = []
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, metric in enumerate(metrics):
            values = [comparison_data[model].get(metric, 0) for model in models]
            
            dataset_configs.append(f"""
            {{
                label: '{metric.replace('_', ' ').title()}',
                data: {values},
                backgroundColor: '{colors[i % len(colors)]}'
            }}
            """)
        
        return f"""
    // Comparison Chart
    const comparisonCtx = document.getElementById('comparisonChart').getContext('2d');
    new Chart(comparisonCtx, {{
        type: 'bar',
        data: {{
            labels: {models},
            datasets: [{', '.join(dataset_configs)}]
        }},
        options: {{
            responsive: true,
            scales: {{
                y: {{
                    beginAtZero: true
                }}
            }}
        }}
    }});
    """


# Utility functions
def create_visualizer(output_dir: str = None) -> MetricsVisualizer:
    """Create and initialize a metrics visualizer"""
    output_path = Path(output_dir) if output_dir else None
    return MetricsVisualizer(output_path)


def generate_standard_charts(metrics_data: Dict[str, Any], 
                        output_dir: str = None) -> List[str]:
    """Generate standard set of charts for metrics analysis"""
    visualizer = create_visualizer(output_dir)
    chart_paths = []
    
    # Timeline chart
    if 'timeline_data' in metrics_data:
        timeline_config = ChartConfig(
            title="Performance Timeline",
            x_label="Time",
            y_label="Response Time (seconds)"
        )
        path = visualizer.create_performance_timeline(
            metrics_data['timeline_data'], timeline_config
        )
        chart_paths.append(path)
    
    # Model comparison
    if 'model_comparison' in metrics_data:
        comparison_config = ChartConfig(
            title="Model Performance Comparison",
            x_label="Model",
            y_label="Score"
        )
        path = visualizer.create_model_comparison_chart(
            metrics_data['model_comparison'], comparison_config
        )
        chart_paths.append(path)
    
    # Distribution
    if 'response_times' in metrics_data:
        dist_config = ChartConfig(
            title="Response Time Distribution",
            x_label="Response Time (seconds)",
            y_label="Frequency"
        )
        path = visualizer.create_distribution_chart(
            metrics_data['response_times'], dist_config
        )
        chart_paths.append(path)
    
    # Dashboard
    dashboard_path = visualizer.create_dashboard(metrics_data)
    chart_paths.append(dashboard_path)
    
    return chart_paths