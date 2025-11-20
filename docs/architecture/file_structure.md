# Conjecture Codebase File Structure Reference

## Overview

This document provides a comprehensive reference of the entire Conjecture codebase structure, serving as a guide for developers to understand the organization, purpose, and relationships between different components of the system.

## Complete Directory Tree

```
conjecture/
├── archive/                          # Historical files and previous phase implementations
│   ├── discovery/                    # Discovery subsystem (archived)
│   │   ├── __init__.py
│   │   ├── config_updater.py
│   │   ├── provider_discovery.py
│   │   └── service_detector.py
│   ├── conjecture_archive/           # Previous architecture implementations
│   │   ├── ClaimSchema.md
│   │   ├── EMBEDDING_OPTIMIZATION_PLAN.md
│   │   └── EmbeddingMethods.py
│   ├── migration-log/                # Migration decision logs
│   └── [various archived files]
│
├── src/                             # Main source code directory
│   ├── __init__.py
│   ├── core.py                      # Core system functionality
│   ├── data.py                      # Data layer and models
│   ├── engine.py                    # Main processing engine
│   ├── simple_cli.py                # Simple CLI interface
│   ├── tools.py                     # Tool management system
│   ├── utils/                       # Utility modules
│   │   ├── __init__.py
│   │   ├── id_generator.py          # Unique ID generation
│   │   └── simple_yaml.py           # YAML handling utilities
│   └── [additional modules as needed]
│
├── test/                            # Primary test directory
│   ├── comprehensive_test_suite.py
│   ├── performance_quality_validation.py
│   ├── quick_discovery_test.py
│   ├── test_api_debug.py
│   ├── test_api_simple.py
│   ├── test_architecture.py
│   ├── test_architecture_simple.py
│   ├── test_chutes_api.py
│   ├── test_chutes_integration.py
│   ├── test_cli_functionality.py
│   ├── test_cli_rubric.py
│   ├── test_core_cli.py
│   ├── test_data_layer.py
│   ├── test_dirty_flag_standalone.py
│   ├── test_discovery_simple.py
│   ├── test_discovery_system.py
│   ├── test_full_stack_chutes.py
│   ├── test_local_integration.py
│   ├── test_output.py
│   ├── test_provider_standalone.py
│   ├── test_providers_integration.py
│   ├── test_providers_simple.py
│   ├── test_quick_validation.py
│   ├── test_sample.py
│   ├── test_setup_wizard_simple.py
│   ├── test_simple_config.py
│   ├── test_simple_config_no_unicode.py
│   ├── test_simple_error_handling.py
│   └── test_three_part_architecture.py
│
├── tests/                           # Extended test suite
│   ├── phase3/                      # Phase 3 specific tests
│   │   ├── test_phase3_core.py
│   │   ├── test_simple_validation.py
│   │   └── test_standalone_validation.py
│   ├── phase4/                      # Phase 4 integration tests
│   │   └── test_integration.py
│   ├── refined_architecture/        # Tests for refined architecture
│   │   ├── test_simple_validation.py
│   │   ├── test_weather_concepts.py
│   │   └── test_weather_example.py
│   ├── skill_agency/                # Skill agency subsystem tests
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   ├── pytest.ini
│   │   ├── requirements.txt
│   │   ├── run_tests.py
│   │   ├── test_edge_cases.py
│   │   ├── test_example_generator.py
│   │   ├── test_integration.py
│   │   ├── test_performance.py
│   │   ├── test_response_parser.py
│   │   ├── test_security.py
│   │   ├── test_skill_manager.py
│   │   ├── test_skill_models.py
│   │   ├── test_smoke.py
│   │   └── test_tool_executor.py
│   ├── conftest.py                  # Pytest configuration
│   ├── performance_benchmarks.py
│   ├── performance_benchmarks_final.py
│   ├── performance_benchmarks_fixed.py
│   ├── pytest.ini
│   ├── requirements.txt
│   ├── run_tests.py
│   ├── test_basic_functionality.py
│   ├── test_basic_workflows.py
│   ├── test_chroma_manager.py
│   ├── test_data_layer.py
│   ├── test_data_layer_complete.py
│   ├── test_data_manager_integration.py
│   ├── test_data_models.py
│   ├── test_dirty_flag.py
│   ├── test_dirty_flag_integration.py
│   ├── test_embedding_service.py
│   ├── test_enhanced_conjecture_comprehensive.py
│   ├── test_error_handling.py
│   ├── test_fallback_mechanisms.py
│   ├── test_granite_model_specific.py
│   ├── test_integration_end_to_end.py
│   ├── test_llm_providers_comprehensive.py
│   ├── test_llm_providers_mock.py
│   ├── test_lm_studio_e2e.py
│   ├── test_lm_studio_performance.py
│   ├── test_lm_studio_provider.py
│   ├── test_models.py
│   ├── test_modular_cli.py
│   ├── test_performance.py
│   ├── test_processing_layer.py
│   ├── test_setup_wizard.py
│   ├── test_simple_functionality.py
│   ├── test_simple_functionality_fixed.py
│   ├── test_simplified_architecture.py
│   ├── test_simplified_conjecture.py
│   ├── test_sqlite_manager.py
│   ├── test_sqlite_manager_comprehensive.py
│   ├── test_suite.py
│   ├── test_tool_validator_simple.py
│   ├── test_tool_validator_standalone.py
│   ├── test_unified_validator.py
│   ├── test_use_cases.py
│   └── test_utilities.py
│
├── docs/                            # Documentation directory
│   ├── architecture/                # Architecture documentation
│   │   ├── agent_subsystem.md
│   │   ├── implementation.md
│   │   ├── llm_subsystem.md
│   │   ├── local_subsystem.md
│   │   ├── main.md
│   │   └── processing_subsystem.md
│   ├── configuration/               # Configuration guides
│   │   └── setup.md
│   ├── reference/                   # Reference documentation
│   │   ├── llm_providers.md
│   │   └── prompts.md
│   ├── tutorials/                   # User tutorials
│   │   ├── advanced.md
│   │   └── basic_usage.md
│   ├── architecture_gap_analysis.md
│   ├── cli_rubric.md
│   ├── data_layer_architecture.md
│   ├── data_layer_rubric.md
│   ├── data_layer_summary.md
│   ├── data_layer_validation_report.md
│   ├── lm_studio_provider.md
│   ├── phase2_completion_summary.md
│   ├── phase2_skill_agency_plan.md
│   ├── phase3_architecture.md
│   ├── phase3_implementation_report.md
│   ├── phase3_rubric.md
│   ├── phase4_implementation_report.md
│   ├── phase4_integration_plan.md
│   ├── phase4_integration_test_plan.md
│   ├── security_audit_report.md
│   ├── strategic_next_steps.md
│   └── testing_report.md
│
├── demo/                            # Demo applications and examples
│   ├── demo_auto_configure.py
│   ├── demo_config_guidance.py
│   ├── demo_discovery.py
│   ├── demo_final_config.py
│   ├── demo_local_services.py
│   ├── demo_new_config.py
│   ├── demo_setup_wizard.py
│   ├── demo_simple_architecture.py
│   ├── demo_simple_auto_configure.py
│   ├── demo_simplified_architecture.py
│   ├── demo_unified_config.py
│   ├── setup_demo.py
│   ├── simple_conjecture.py
│   ├── simple_conjecture_cli.py
│   ├── simple_conjecture_cli_redirected.py
│   ├── simple_conjecture_standalone.py
│   ├── test_real_chroma.py
│   └── unified_api_demo.py
│
├── tools/                           # Utility tools
│   ├── apply_diff.py
│   ├── readFiles.py
│   ├── webSearch.py
│   └── writeFiles.py
│
├── skills/                          # Skill-related modules
│   ├── coding_principles.py
│   ├── research_coding_projects.py
│   ├── skill_creation.py
│   └── tool_creation.py
│
├── specs/                           # Specifications and requirements
│   ├── architecture/               # Architecture specifications
│   │   └── simple-universal-claim-architecture.md
│   ├── context/                    # Context specifications
│   │   └── complete-relationship-context-builder.md
│   ├── implementation/             # Implementation specs
│   │   └── unified-claim-system-implementation.md
│   ├── llm/                        # LLM-related specs
│   │   └── instruction-support-relationship-protocol.md
│   ├── migration/                  # Migration specifications
│   │   └── simplified-architecture-migration.md
│   ├── README.md
│   ├── interface_design.md
│   ├── old_req.md
│   ├── phase3_rubric.md
│   ├── phases.md
│   ├── project.md
│   └── requirements.md
│
├── .env.example                     # Environment configuration template
├── .gitignore                       # Git ignore rules
├── QWEN.md                          # QWEN model documentation
├── README.md                        # Project overview
├── SIMPLIFICATION_SUMMARY.md        # Architecture simplification summary
├── cli_test_results.json           # CLI test results
├── conjecture                      # Main executable
├── performance_quality_results.json # Performance test results
├── performance_report.txt          # Performance report
├── pre-commit-hook.sh             # Git pre-commit hook
├── query_chutes_models.py         # Model querying utility
├── requirements.txt               # Python dependencies
├── simple_local_cli.py            # Simple local CLI
├── test_results.json              # Test execution results
└── validation_report.py           # Validation reporting tool
```

## Directory Purpose and Responsibilities

### `/src/` - Main Source Code
**Purpose**: Contains the core application logic and primary modules.

**Key Components**:
- `core.py`: Central system functionality and main orchestration
- `data.py`: Data models, persistence, and data layer operations
- `engine.py`: Processing engine for handling requests and workflows
- `simple_cli.py`: Command-line interface implementation
- `tools.py`: Tool management and execution system
- `utils/`: Utility functions and helper modules

**Relationships**: This is the heart of the application where all subsystems converge. The core module orchestrates interactions between data, engine, and CLI components.

### `/test/` - Primary Test Suite
**Purpose**: Contains the main testing framework and comprehensive test cases.

**Key Test Categories**:
- **Architecture Tests**: `test_architecture.py`, `test_architecture_simple.py`
- **API Tests**: `test_api_debug.py`, `test_api_simple.py`
- **Integration Tests**: `test_chutes_integration.py`, `test_local_integration.py`
- **CLI Tests**: `test_cli_functionality.py`, `test_core_cli.py`
- **Data Layer Tests**: `test_data_layer.py`
- **Configuration Tests**: `test_simple_config.py`, `test_setup_wizard_simple.py`

**Relationships**: These tests validate the entire system integration and ensure all components work together correctly.

### `/tests/` - Extended Test Suite
**Purpose**: Provides comprehensive testing coverage with specialized test modules.

**Specialized Test Areas**:
- **Phase-Specific Tests**: `phase3/`, `phase4/` directories for development phase validation
- **Skill Agency Tests**: `skill_agency/` directory for skill management system testing
- **Performance Tests**: `performance_benchmarks*.py` for system performance validation
- **Integration Tests**: End-to-end testing across multiple subsystems

**Relationships**: Complements the primary test suite with more detailed and specialized testing scenarios.

### `/docs/` - Documentation
**Purpose**: Comprehensive documentation covering architecture, usage, and implementation details.

**Documentation Categories**:
- **Architecture Docs**: Detailed system architecture documentation
- **Configuration Guides**: Setup and configuration instructions
- **Reference Materials**: API references and technical specifications
- **Tutorials**: User guides and learning materials
- **Reports**: Analysis reports and validation results

**Relationships**: Serves as the primary knowledge base for developers, users, and stakeholders.

### `/demo/` - Demonstration Applications
**Purpose**: Provides working examples and demonstrations of system capabilities.

**Demo Types**:
- **Configuration Demos**: Show different configuration approaches
- **Architecture Demos**: Demonstrate various architectural patterns
- **Setup Demos**: Guide users through initial setup process
- **Integration Demos**: Show system integration capabilities

**Relationships**: Helps users understand system capabilities and provides starting points for custom implementations.

### `/tools/` - Utility Tools
**Purpose**: Contains utility scripts and tools for development and maintenance.

**Tool Categories**:
- **File Operations**: `readFiles.py`, `writeFiles.py`
- **Web Operations**: `webSearch.py`
- **System Operations**: `apply_diff.py`

**Relationships**: Supports development workflow and system maintenance tasks.

### `/skills/` - Skill Management
**Purpose**: Contains skill-related modules and implementations.

**Skill Types**:
- **Coding Skills**: Programming and development capabilities
- **Research Skills**: Information gathering and analysis
- **Creation Skills**: Content and tool generation
- **Tool Creation Skills**: Dynamic tool development

**Relationships**: Integrates with the main system to provide extensible skill capabilities.

### `/specs/` - Specifications
**Purpose**: Contains detailed specifications, requirements, and design documents.

**Specification Types**:
- **Architecture Specifications**: System design and structure
- **Implementation Specifications**: Detailed implementation guidelines
- **Migration Specifications**: Upgrade and migration procedures
- **Interface Specifications**: API and component interfaces

**Relationships**: Guides development and ensures consistency across implementations.

### `/archive/` - Historical Files
**Purpose**: Stores previous versions, deprecated components, and historical documentation.

**Archive Categories**:
- **Previous Implementations**: Older system versions
- **Migration Logs**: Historical migration decisions and processes
- **Research Documents**: Research findings and analysis
- **Deprecated Features**: Outdated but potentially reference-worthy code

**Relationships**: Maintains historical context and provides reference for past decisions.

## Configuration Files

### `.env.example`
**Purpose**: Template for environment configuration
**Contents**: Example environment variables and configuration settings
**Usage**: Copy to `.env` and customize for specific deployments

### `requirements.txt`
**Purpose**: Python dependency specifications
**Contents**: All required Python packages and versions
**Usage**: `pip install -r requirements.txt` to install dependencies

### `.gitignore`
**Purpose**: Git version control configuration
**Contents**: Patterns for files and directories to exclude from version control
**Usage**: Automatically applied by Git to prevent tracking of sensitive or generated files

## Testing Structure

### Test Organization
The testing structure is organized into multiple levels:

1. **Unit Tests**: Individual component testing in both `/test/` and `/tests/`
2. **Integration Tests**: Cross-component interaction testing
3. **End-to-End Tests**: Full system workflow validation
4. **Performance Tests**: System performance and scalability testing
5. **Specialized Tests**: Domain-specific testing (skills, CLI, data layer)

### Test Execution
- **Primary Suite**: Run from `/test/` directory for core functionality
- **Extended Suite**: Run from `/tests/` directory for comprehensive coverage
- **Skill Tests**: Specialized testing for skill agency functionality
- **Performance Tests**: Benchmarking and performance validation

### Test Configuration
- **Pytest Configuration**: `pytest.ini` files in test directories
- **Test Requirements**: Separate `requirements.txt` for test dependencies
- **Conftest Files**: Shared test fixtures and configuration

## Subsystem Relationships

### Core System Flow
```
CLI/Interface → Core Engine → Data Layer → Tools/Skills → Output
     ↓              ↓           ↓           ↓           ↓
  User Input → Processing → Storage → Execution → Results
```

### Integration Points
- **CLI Layer**: User interface and command processing
- **Core Layer**: Business logic and orchestration
- **Data Layer**: Persistence and data management
- **Tool Layer**: Extensible functionality and capabilities
- **Skill Layer**: Advanced AI-powered capabilities

### External Dependencies
- **LLM Providers**: Integration with various language model services
- **Local Services**: Local embedding and vector storage services
- **Configuration**: Environment-based configuration management

## Development Guidelines

### File Naming Conventions
- **Modules**: Lowercase with underscores (`simple_cli.py`)
- **Classes**: PascalCase (`SimpleCLI`)
- **Functions**: Lowercase with underscores (`get_config()`)
- **Constants**: UPPERCASE (`MAX_RETRIES`)

### Directory Structure Principles
- **Separation of Concerns**: Each directory has a specific responsibility
- **Modularity**: Components are self-contained where possible
- **Testability**: Clear separation between code and tests
- **Documentation**: Comprehensive documentation at multiple levels

### Import Organization
- **Standard Library**: Python built-in modules
- **Third Party**: External dependencies
- **Local**: Project-specific modules
- **Relative**: Intra-project imports using relative paths

This file structure reference serves as the definitive guide for understanding and navigating the Conjecture codebase, providing developers with the context needed to effectively contribute to and maintain the system.