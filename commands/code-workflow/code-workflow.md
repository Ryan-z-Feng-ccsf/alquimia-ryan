---
name: run-flow
description: Executes the autonomous engineering loop (Planner -> Reviewer -> Coder -> Tester -> Loop).
user-invocable: true
---

# Autonomous Engineering Loop Execution

When the user calls `/run-flow`, you must immediately initiate and manage the following iterative engineering lifecycle:

1. **[PHASE: PLAN](invoke planer)** Analyze the user's requirement. Generate a comprehensive architecture and execution blueprint.
2. **[PHASE: REVIEW](invoke reviewer)** Critically evaluate your own blueprint. Act as a Staff Engineer to find edge cases, structural flaws, or missed requirements. Refine the plan until it is flawless.
3. **[PHASE: CODE](invoke coder)** Implement the finalized plan. Directly modify the local workspace. You MUST strictly adhere to the project's `/comment` rules (no redundant comments, keep docstrings synchronized).
4. **[PHASE: TEST](invoke tester)** Execute compilation and test suites (e.g., `make test` or custom testing binaries). Capture all stdout/stderr.
5. **[PHASE: EVALUATE & LOOP]** - **Success (Exit Code 0):** Present the summary of changes, test results, and finish.
   - **Failure (Exit Code != 0):** Do NOT give up. Treat the failure logs as fresh context. Loop back directly to **[PHASE: REVIEW]**. Diagnose why it failed, alter the implementation strategy, rewrite the code, and re-test. Repeat this cycle indefinitely until the tests completely pass.