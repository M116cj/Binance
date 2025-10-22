#!/usr/bin/env python3
"""
README Generator Script
Automatically generates README.md from project_manifest.yaml and template.
"""

import os
import sys
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

try:
    import yaml
    from jinja2 import Template
except ImportError:
    print("Installing required dependencies...")
    os.system("pip install pyyaml jinja2")
    import yaml
    from jinja2 import Template


def load_manifest() -> Dict[str, Any]:
    """Load project manifest from YAML file."""
    manifest_path = Path(__file__).parent.parent / "docs" / "project_manifest.yaml"
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def format_features_list(features: list) -> str:
    """Format features list as markdown."""
    result = []
    for feature in features:
        name = feature.get('name', '')
        description = feature.get('description', '')
        
        result.append(f"### {name}")
        if description:
            result.append(f"{description}\n")
        
        if 'components' in feature:
            for component in feature['components']:
                result.append(f"- {component}")
            result.append("")
        
        if 'features' in feature:
            for sub_feature in feature['features']:
                result.append(f"- {sub_feature}")
            result.append("")
    
    return "\n".join(result)


def format_architecture_modes(modes: list) -> str:
    """Format architecture modes as markdown."""
    result = []
    for mode in modes:
        name = mode.get('name', '')
        description = mode.get('description', '')
        result.append(f"**{name}**: {description}")
    return "\n\n".join(result)


def format_trading_strategies(strategies: list) -> str:
    """Format trading strategies as markdown table."""
    result = ["| Strategy | Icon | θ_up | θ_dn | τ | κ | Description |"]
    result.append("|----------|------|------|------|---|---|-------------|")
    
    for strategy in strategies:
        name = strategy.get('name', '')
        icon = strategy.get('icon', '')
        theta_up = strategy.get('theta_up', '')
        theta_dn = strategy.get('theta_dn', '')
        tau = strategy.get('tau', '')
        kappa = strategy.get('kappa', '')
        description = strategy.get('description', '')
        
        result.append(f"| {name} | {icon} | {theta_up} | {theta_dn} | {tau} | {kappa} | {description} |")
    
    return "\n".join(result)


def format_analytics_reports(features: list) -> str:
    """Extract and format analytics reports."""
    for feature in features:
        if feature.get('name') == '7 Analytics Reports':
            components = feature.get('components', [])
            return "\n".join([f"- {comp}" for comp in components])
    return ""


def format_prerequisites(prereqs: list) -> str:
    """Format prerequisites as markdown list."""
    return "\n".join([f"- {prereq}" for prereq in prereqs])


def format_env_vars(env_vars: list) -> str:
    """Format environment variables as markdown table."""
    if not env_vars:
        return "None"
    
    result = ["| Variable | Description |"]
    result.append("|----------|-------------|")
    
    for var in env_vars:
        name = var.get('name', '')
        description = var.get('description', '')
        result.append(f"| `{name}` | {description} |")
    
    return "\n".join(result)


def format_recent_updates(updates: list) -> str:
    """Format recent updates as markdown."""
    if not updates:
        return "No recent updates."
    
    result = []
    for update in updates:
        version = update.get('version', '')
        date = update.get('date', '')
        title = update.get('title', '')
        changes = update.get('changes', [])
        
        result.append(f"### Version {version} ({date}): {title}\n")
        for change in changes:
            result.append(f"- {change}")
        result.append("")
    
    return "\n".join(result)


def format_strategy_details(strategies: list) -> str:
    """Format detailed strategy information."""
    result = []
    for strategy in strategies:
        name = strategy.get('name', '')
        icon = strategy.get('icon', '')
        description = strategy.get('description', '')
        
        result.append(f"### {icon} {name}\n")
        result.append(f"{description}\n")
        result.append(f"**Parameters:**")
        result.append(f"- θ_up (bullish threshold): {strategy.get('theta_up', 'N/A')}")
        result.append(f"- θ_dn (bearish threshold): {strategy.get('theta_dn', 'N/A')}")
        result.append(f"- τ (tau - probability threshold): {strategy.get('tau', 'N/A')}")
        result.append(f"- κ (kappa - utility multiplier): {strategy.get('kappa', 'N/A')}\n")
    
    return "\n".join(result)


def calculate_checksum(content: str) -> str:
    """Calculate MD5 checksum of content."""
    return hashlib.md5(content.encode()).hexdigest()[:8]


def generate_readme(manifest: Dict[str, Any]) -> str:
    """Generate README content from manifest."""
    
    # Load template
    template_path = Path(__file__).parent.parent / "docs" / "README_template.md"
    with open(template_path, 'r', encoding='utf-8') as f:
        template_content = f.read()
    
    template = Template(template_content)
    
    # Prepare template variables
    project = manifest.get('project', {})
    performance = manifest.get('performance', {})
    architecture = manifest.get('architecture', {})
    tech_stack = manifest.get('tech_stack', {})
    features = manifest.get('features', [])
    installation = manifest.get('installation', {})
    env_variables = manifest.get('environment_variables', {})
    ui_theme = manifest.get('ui_theme', {})
    recent_updates = manifest.get('recent_updates', [])
    
    # Extract repository path
    repository = project.get('repository', '')
    repository_path = repository.replace('https://github.com/', '') if repository else 'user/repo'
    
    # Backend description
    backend = architecture.get('backend', {})
    backend_features = backend.get('features', [])
    backend_description = ', '.join(backend_features[:3])
    
    # Frontend description
    frontend = architecture.get('frontend', {})
    frontend_features = frontend.get('features', [])
    frontend_description = ', '.join(frontend_features[:3])
    
    # ML description
    ml = architecture.get('ml_pipeline', {})
    ml_features = ml.get('features', [])
    ml_description = ', '.join(ml_features[:2])
    
    # Storage stack
    storage = architecture.get('storage', {})
    storage_stack = f"{storage.get('hot_cache', 'N/A')}, {storage.get('cold_storage', 'N/A')}, {storage.get('relational_db', 'N/A')}"
    storage_description = "Hot cache, cold storage, relational data"
    
    # UI theme description
    colors = ui_theme.get('colors', {})
    ui_theme_description = f"""
The dashboard features a professional **TradingView-inspired dark theme** optimized for extended trading sessions:

- **Reduced Eye Strain**: Dark background ({colors.get('background', '#131722')}) minimizes fatigue during long monitoring sessions
- **High Contrast**: Critical data stands out clearly
- **Professional Aesthetics**: Similar visual language to leading trading platforms
- **Theme Toggle**: Switch between dark and light modes via sidebar button
""".strip()
    
    # Render template
    content = template.render(
        project=project,
        repository=repository,
        repository_path=repository_path,
        license=manifest.get('license', 'MIT'),
        performance=performance,
        architecture_modes=format_architecture_modes(architecture.get('modes', [])),
        trading_strategies=format_trading_strategies(architecture.get('strategies', [])),
        backend=backend,
        backend_description=backend_description,
        frontend=frontend,
        frontend_description=frontend_description,
        ml=ml,
        ml_description=ml_description,
        storage_stack=storage_stack,
        storage_description=storage_description,
        features_list=format_features_list(features),
        analytics_reports=format_analytics_reports(features),
        ui_theme_description=ui_theme_description,
        colors=colors,
        prerequisites_list=format_prerequisites(installation.get('prerequisites', [])),
        env_vars_required=format_env_vars(env_variables.get('required', [])),
        env_vars_optional=format_env_vars(env_variables.get('optional', [])),
        strategy_details=format_strategy_details(architecture.get('strategies', [])),
        recent_updates=format_recent_updates(recent_updates),
        author=manifest.get('author', 'Your Name'),
        contact=manifest.get('contact', 'your.email@example.com'),
        last_updated=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
        checksum=''  # Will be calculated after rendering
    )
    
    # Calculate checksum
    checksum = calculate_checksum(content)
    content = content.replace('{{ checksum }}', checksum)
    
    return content


def main():
    """Main execution function."""
    try:
        print("🔄 Loading project manifest...")
        manifest = load_manifest()
        
        print("📝 Generating README.md...")
        readme_content = generate_readme(manifest)
        
        # Write README.md
        readme_path = Path(__file__).parent.parent / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"✅ README.md successfully generated at: {readme_path}")
        print(f"📊 Content length: {len(readme_content)} characters")
        
        return 0
    
    except Exception as e:
        print(f"❌ Error generating README: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
