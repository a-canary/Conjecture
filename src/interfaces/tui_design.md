# Conjecture Interface Design Specification

## Interface Layers

### Terminal User Interface (TUI)

The TUI provides an interactive, keyboard-driven interface for exploration and manipulation of claims with rich visualization capabilities.

**Key Features:**
- Multi-panel layout with collapsible sections
- Real-time claim exploration with dynamic filtering
- Interactive claim editing and validation
- Progress tracking on research goals
- Visual representation of claim relationships

**User Experience Goals:**
- Fast response times for exploration (<100ms)
- Intuitive navigation patterns
- Rich information density without overwhelming users
- Offline capability for research work

### Command Line Interface (CLI)

The CLI provides programmatic access for automation and scripting with a comprehensive command set.

**Core Commands:**
- `claim`: Create claims from statements
- `prompt`: Ask questions about knowledge base
- `inspect`: Retrieve claims relevant to query
- `goal`: Set or modify research goals
- `status`: Report system and progress status

**Integration Features:**
- JSON output for integration with other tools
- Batch processing capabilities
- Configuration management commands
- Shell completion support

### Model Context Protocol (MCP)

The MCP interface enables seamless integration with coding assistants and AI services through standardized actions.

**Core Actions:**
- `claim(statement)`: Build claim(s) from statement
- `prompt(question)`: Ask question(s) about knowledge base
- `inspect(query, count)`: Return N claims relevant to query

**Integration Benefits:**
- Bi-directional knowledge synchronization
- Real-time claim validation
- Context-aware task extraction
- Progressive disclosure based on importance

### Web User Interface (WebUI)

The WebUI provides a rich visual interface for collaborative research with advanced visualization capabilities.

**Collaboration Features:**
- Integration with OpenWebUI platform
- Specialized panels for claim visualization
- Interactive relationship graphs
- Collaborative annotation tools
- Multi-user support with real-time updates

**Accessibility Goals:**
- Mobile-responsive design
- Interface accessibility following WCAG standards
- Intuitive visual representation of complex relationships
