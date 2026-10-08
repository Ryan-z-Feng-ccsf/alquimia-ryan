---
name: comment
description: Enforces minimal/essential code comments and standard API docstrings. Prevents redundant comments and stale code.
user-invocable: true
---
# Code Commenting Rules

You MUST strictly adhere to the following strict commenting constraints for all code edits:

1. **No Redundant Comments:** Write comments EXCLUSIVELY for complex logic, algorithms, or non-obvious edge cases. NEVER write self-explanatory comments (e.g., `i++ // add 1`).
2. **Standard API Docs:** Document all public APIs, modules, and functions using the language's standard format (JSDoc, TypeDoc, Docstring, Rustdoc). Define parameters and return types.
3. **Strict Sync:** When modifying code, you MUST synchronously update all related comments. Stale or misleading comments are prohibited.
4. **Preserve Context:** NEVER arbitrarily delete existing comments or `TODO/FIXME` tags unless the referenced code block is entirely removed.