# Skills System Examples

This directory contains examples of how to use ClaimCreate to store "skills" - knowledge about how to do things well.

## What is a Skill in Conjecture?

A skill is simply a claim that describes knowledge or instruction about how to do something well. Instead of a separate skills system, we use the unified claim system to store and retrieve procedural knowledge.

## Example Skills

### Python Programming Skills
```bash
conjecture claim create "To write clean Python code, follow PEP 8 style guidelines, use descriptive variable names, keep functions under 20 lines, and include docstrings for all public functions."

conjecture claim create "For effective Python debugging, use print statements for quick checks, pdb.set_trace() for interactive debugging, and logging for production code. Always check variable types and values at breakpoints."

conjecture claim create "Python list comprehensions should be used for simple transformations, but switch to regular loops when the logic becomes complex or multiple operations are needed."
```

### Git Workflow Skills
```bash
conjecture claim create "For effective Git workflow, commit frequently with descriptive messages, use branches for features, and pull changes before pushing to avoid conflicts."

conjecture claim create "When resolving Git merge conflicts, first understand what changed on both sides, use 'git diff' to see conflicts clearly, and test the merged code before committing."
```

### Data Analysis Skills
```bash
conjecture claim create "For data analysis with pandas, always use .info() and .describe() first to understand data structure, handle missing values before analysis, and visualize distributions to identify outliers."

conjecture claim create "Effective data visualization requires choosing the right chart type for your data: bar charts for comparisons, line charts for trends, scatter plots for correlations, and histograms for distributions."
```

## How Skills Are Retrieved

Skills are automatically retrieved through the context collection system based on relevance to your current task. The system uses semantic similarity to find the most relevant "how-to" knowledge and includes it in the LLM context.

## Benefits of This Approach

1. **Unified System** - Skills are just claims, no separate system to maintain
2. **Automatic Retrieval** - Context system finds relevant skills based on your task
3. **Easy to Add** - Simply use `conjecture claim create` to add new skills
4. **Searchable** - All skills are searchable through the normal claim search system
5. **Version Controlled** - Skills evolve with your claim history

## Adding Your Own Skills

To add a skill, simply create a claim that describes how to do something well:

```bash
conjecture claim create "Your skill description here - be specific about the procedure, best practices, or key insights."
```

The system will automatically make this skill available when relevant to future tasks.