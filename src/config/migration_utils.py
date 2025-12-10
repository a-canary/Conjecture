"""
Configuration Migration Utilities
Provides tools for migrating between configuration formats
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from rich.console import Console

from .unified_validator import UnifiedConfigValidator, ConfigFormat

console = Console()

class ConfigurationMigrator:
    """
    Handles migration from old configuration formats to unified format
    """
    
    def __init__(self, env_file: str = ".env"):
        self.env_file = Path(env_file)
        self.validator = UnifiedConfigValidator()
        self.backup_dir = Path("config_backups")
        self.backup_dir.mkdir(exist_ok=True)

    def create_backup(self) -> bool:
        """
        Create backup of current .env file
        """
        try:
            if self.env_file.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_filename = f"{self.env_file.stem}.backup.{timestamp}{self.env_file.suffix}"
                backup_path = self.backup_dir / backup_filename
                
                # Copy file content
                backup_path.write_text(self.env_file.read_text())
                console.print(f"[green]✅ Backup created: {backup_path}[/green]")
                return True
            else:
                console.print(f"[yellow]⚠️  {self.env_file} does not exist, no backup needed[/yellow]")
                return True
                
        except Exception as e:
            console.print(f"[red]❌ Failed to create backup: {e}[/red]")
            return False

    def analyze_migration(self) -> Dict[str, Any]:
        """
        Analyze current configuration and recommend migration strategy
        """
        analysis = {
            "current_format": None,
            "detected_formats": [],
            "recommended_format": ConfigFormat.UNIFIED_PROVIDER,
            "complexity_score": 0,
            "migration_difficulty": "easy",
            "benefits": [],
            "risks": [],
            "steps": []
        }

        # Get current configuration analysis
        validation_result = self.validator.validate_configuration()
        analysis["current_format"] = validation_result.active_format
        analysis["detected_formats"] = validation_result.all_detected_formats

        # Calculate complexity
        num_formats = len(analysis["detected_formats"])
        num_providers = len(validation_result.providers)
        analysis["complexity_score"] = num_formats * 10 + num_providers * 5

        # Determine difficulty
        if analysis["current_format"] == ConfigFormat.UNIFIED_PROVIDER:
            analysis["migration_difficulty"] = "none"
        elif analysis["current_format"] == ConfigFormat.SIMPLE_PROVIDER:
            analysis["migration_difficulty"] = "easy"
        elif analysis["current_format"] == ConfigFormat.INDIVIDUAL_ENV:
            analysis["migration_difficulty"] = "medium"
        else:
            analysis["migration_difficulty"] = "hard"

        # Benefits of migration
        analysis["benefits"] = [
            "Simplified configuration management",
            "Consistent format across all providers",
            "Easier debugging and troubleshooting",
            "Better tooling and automation support",
            "Reduced cognitive overhead"
        ]

        # Risks to consider
        if len(analysis["detected_formats"]) > 1:
            analysis["risks"].append("Multiple formats detected - potential conflicts")

        if analysis["current_format"] != ConfigFormat.UNIFIED_PROVIDER:
            analysis["risks"].append("Need to update existing scripts/tools")
            analysis["risks"].append("Temporary disruption during migration")

        # Generate steps
        if analysis["current_format"] != ConfigFormat.UNIFIED_PROVIDER:
            analysis["steps"] = self._generate_migration_steps(analysis["current_format"])

        return analysis

    def _generate_migration_steps(self, current_format: ConfigFormat) -> List[str]:
        """
        Generate migration steps for a specific format
        """
        base_steps = [
            "1. Create backup of current configuration",
            "2. Review and understand the new unified format",
            "3. Apply migration changes to .env file",
            "4. Test new configuration",
            "5. Update any scripts or tools that reference old format"
        ]

        if current_format == ConfigFormat.SIMPLE_PROVIDER:
            base_steps.insert(3, "2a. Convert PROVIDER_[NAME] variables to PROVIDER_* format")
            base_steps.insert(4, "2b. Select the primary provider to migrate")
        elif current_format == ConfigFormat.INDIVIDUAL_ENV:
            base_steps.insert(3, "2a. Convert [PROVIDER]_API_URL format to PROVIDER_API_URL")
            base_steps.insert(4, "2b. Select one model from PROVIDER_MODELS list")
            base_steps.insert(5, "2c. Update API key variable name")
        elif current_format == ConfigFormat.SIMPLE_VALIDATOR:
            base_steps.insert(3, "2a. Convert provider-specific variables to unified format")
            base_steps.insert(4, "2b. Use the highest priority configured provider")

        return base_steps

    def generate_migration_script(self, target_format: ConfigFormat = ConfigFormat.JSON) -> Dict[str, Any]:
        """
        Generate shell script for migration between formats
        """
        current_format = self.validator.get_active_format()
        
        if current_format == target_format:
            return {
                "success": False,
                "message": "Already using target format",
                "script": []
            }

        analysis = self.analyze_migration()
        
        script = [
            "#!/bin/bash",
            "# Configuration Migration Script",
            f"# Generated on: {datetime.now().isoformat()}",
            f"# Source format: {current_format.value}",
            f"# Target format: {target_format.value}",
            "",
            "set -e  # Exit on any error",
            "",
            "# Color codes for output",
            "RED='\\033[0;31m'",
            "GREEN='\\033[0;32m'",
            "YELLOW='\\033[1;33m'",
            "NC='\\033[0m' # No Color", 
            "",
            "echo -e '${GREEN}Starting configuration migration...${NC}'",
            "",
            "# Step 1: Create backup"
        ]

        if self.env_file.exists():
            script.extend([
                "echo 'Creating backup...'",
                f"if [ -f '{self.env_file}' ]; then",
                f"    cp '{self.env_file}' '{self.env_file}.backup.$(date +%Y%m%d_%H%M%S)'",
                "    echo -e '${GREEN}✅ Backup created${NC}'",
                "else",
                "    echo -e '${YELLOW}⚠️  No .env file found${NC}'",
                "fi",
                ""
            ])

        # Add migration commands
        migration_commands = self._generate_migration_commands(current_format, target_format)
        
        if migration_commands.get("success"):
            script.extend([
                "# Step 2: Apply migration changes",
                "echo 'Applying migration changes...'",
                "# Remove old configuration variables",
                "# Add new configuration variables"
            ])

            for line in migration_commands["success"]:
                if line.startswith('#'):
                    script.append(f"echo '{line}'")
                else:
                    script.append(f"echo '{line}' >> {self.env_file}")

            script.extend([
                "",
                "echo -e '${GREEN}✅ Migration completed${NC}'",
                "",
                "echo 'Testing new configuration...'",
                "python -m conjecture.config.unified_validator test",
                "",
                "echo -e '${GREEN}🎉 Migration successful!${NC}'"
            ])

        if migration_commands.get("errors"):
            script.extend([
                "# Migration errors detected",
                "echo -e '${RED}❌ Migration errors:${NC}'"
            ])

            for error in migration_commands["errors"]:
                script.append(f"echo '  • {error}'")
            script.append("exit 1")

        return {
            "success": True,
            "script": script,
            "analysis": analysis,
            "migration_commands": migration_commands
        }

    def _generate_migration_commands(self, current_format: ConfigFormat, target_format: ConfigFormat) -> Dict[str, Any]:
        """
        Generate the actual migration commands
        """
        # This is a simplified version - in real implementation would
        # parse and transform configuration variables
        
        return {
            "success": ["# Migration commands would go here"],
            "errors": []
        }

    def execute_migration(self, dry_run: bool = True) -> Dict[str, Any]:
        """
        Execute migration (or dry run)
        """
        result = {
            "success": False,
            "backup_created": False,
            "changes_applied": False,
            "errors": [],
            "warnings": []
        }

        try:
            # Create backup
            backup_success = self.create_backup()
            result["backup_created"] = backup_success

            if not backup_success:
                result["errors"].append("Failed to create backup - aborting migration")
                return result

            # Get migration commands
            current_format = self.validator.get_active_format()
            migration_commands = self._generate_migration_commands(current_format, ConfigFormat.UNIFIED_PROVIDER)

            if migration_commands.get("errors"):
                result["errors"].extend(migration_commands["errors"])
                return result

            if dry_run:
                result["warnings"].append("DRY RUN MODE - No changes will be applied")
                result["changes"] = migration_commands.get("success", [])
                result["success"] = True
            else:
                # Apply changes
                if self.env_file.exists():
                    new_content = self.env_file.read_text()
                    for line in migration_commands["success"]:
                        if "=" in line:
                            var_name = line.split("=")[0]
                            new_content = new_content.replace(f"{var_name}=", f"# {var_name}=")

                    new_content += "\n# Unified Configuration (migrated)\n"
                    for line in migration_commands["success"]:
                        if not line.startswith('#') and "=" in line:
                            new_content += f"{line}\n"

                    # Write new content
                    self.env_file.write_text(new_content)
                    result["changes_applied"] = True

                    # Test new configuration
                    validation_result = self.validator.validate_configuration()
                    if validation_result.success:
                        result["success"] = True
                    else:
                        result["errors"].extend(validation_result.errors)
                        result["warnings"].extend(validation_result.warnings)

        except Exception as e:
            result["errors"].append(f"Failed to apply changes: {str(e)}")

        return result

    def show_migration_analysis(self):
        """
        Display migration analysis to console
        """
        analysis = self.analyze_migration()

        console.print("[bold blue]Configuration Migration Analysis[/bold blue]")
        console.print("=" * 50)

        current_format_name = analysis["current_format"].value.replace('_', ' ').title()
        console.print(f"[bold]Current Format:[/bold] {current_format_name}")
        console.print(f"[bold]Detected Formats:[/bold] {len(analysis['detected_formats'])}")
        console.print(f"[bold]Complexity Score:[/bold] {analysis['complexity_score']}")
        console.print(f"[bold]Migration Difficulty:[/bold] {analysis['migration_difficulty'].title()}")

        console.print(f"\n[bold green]Benefits of Migration:[/bold green]")
        for benefit in analysis["benefits"]:
            console.print(f"  • {benefit}")

        if analysis["risks"]:
            console.print(f"\n[bold yellow]Risks to Consider:[/bold yellow]")
            for risk in analysis["risks"]:
                console.print(f"  • {risk}")

        if analysis["steps"]:
            console.print(f"\n[bold blue]Recommended Steps:[/bold blue]")
            for step in analysis["steps"]:
                console.print(f"  {step}")

    def generate_migration_guide(self, migration_commands: Dict[str, Any]) -> str:
        """
        Generate detailed migration guide
        """
        current_format = self.validator.get_active_format()
        analysis = self.analyze_migration()

        guide = "# Configuration Migration Guide\n\n"
        guide += f"## Current Status\n"
        guide += f"- **Format**: {current_format.value.replace('_', ' ').title()}\n"
        guide += f"- **Complexity Score**: {analysis['complexity_score']}\n"
        guide += f"- **Migration Difficulty**: {analysis['migration_difficulty'].title()}\n\n"
        guide += f"## Why Migrate?\n"

        for benefit in analysis["benefits"]:
            guide += f"- {benefit}\n"

        guide += "\n## Migration Steps\n"
        if analysis["steps"]:
            for step in analysis["steps"]:
                guide += f"{step}\n"

        guide += "\n## Migration Commands\n"

        if migration_commands.get("success"):
            for command in migration_commands["success"]:
                guide += f"{command}\n"

        if migration_commands.get("errors"):
            guide += "\n## Potential Issues\n"
            for error in migration_commands["errors"]:
                guide += f"- {error}\n"

        guide += "\n## Testing\n\n"
        guide += "After migration, test your new configuration:\n\n"
        guide += "```bash\n"
        guide += "python -m conjecture.config.unified_validator test\n"
        guide += "```\n\n"

        guide += "## New CLI Usage\n\n"
        guide += "The old CLI files have been redirected to the new modular system:\n"
        guide += "```bash\n"
        guide += "# New entry point (recommended)\n"
        guide += "conjecture create \"test claim\"\n\n"
        guide += "# Alternative entry points\n"
        guide += "python conjecture create \"test claim\"\n"
        guide += "python -m src.cli.modular_cli create \"test claim\"\n\n"
        guide += "# Backend selection\n"
        guide += "conjecture --backend local create \"test claim\"\n"
        guide += "conjecture --backend cloud analyze c1234567\n"
        guide += "```\n\n"

        guide += f"Generated on: {datetime.now().isoformat()}\n"

        return guide

    def export_migration_analysis(self, format: str = "json") -> str:
        """
        Export migration analysis in specified format
        """
        analysis = self.analyze_migration()

        if format.lower() == "json":
            return json.dumps({
                "current_format": analysis["current_format"].value,
                "detected_formats": [f.value for f in analysis["detected_formats"]],
                "recommended_format": analysis["recommended_format"].value,
                "complexity_score": analysis["complexity_score"],
                "migration_difficulty": analysis["migration_difficulty"],
                "benefits": analysis["benefits"],
                "risks": analysis["risks"],
                "steps": analysis["steps"],
                "generated_at": datetime.now().isoformat()
            }, indent=2)
        elif format.lower() == "markdown":
            # Generate markdown version
            return self.generate_migration_guide({"success": ["Sample commands"]})
        else:
            raise ValueError(f"Unsupported export format: {format}")

def analyze_migration(env_file: str = ".env") -> Dict[str, Any]:
    """Analyze migration options"""
    migrator = ConfigurationMigrator(env_file)
    return migrator.analyze_migration()

def execute_migration(env_file: str = ".env", dry_run: bool = True) -> Dict[str, Any]:
    """Execute migration (or dry run)"""
    migrator = ConfigurationMigrator(env_file)
    return migrator.execute_migration(dry_run=dry_run)

def show_migration_analysis(env_file: str = ".env"):
    """Show migration analysis"""
    migrator = ConfigurationMigrator(env_file)
    migrator.show_migration_analysis()

if __name__ == "__main__":
    """Test migration utilities when run directly"""
    console.print("[bold green]Configuration Migration Utilities Test[/bold green]")
    console.print("=" * 60)
    
    migrator = ConfigurationMigrator()
    migrator.show_migration_analysis()
    
    console.print(f"\n[bold blue]Dry Run Migration:[/bold blue]")
    result = execute_migration(dry_run=True)
    
    if result["changes"]:
        console.print("[yellow]Changes that would be applied:[/yellow]")
        for change in result["changes"]:
            console.print(f"  {change}")