---
name: init-memory
description: Initialize and maintain project auto memory according to structural index rules and maintenance protocols.
---

1. **Check Auto Memory Status:** Verify if the project's auto memory feature is enabled. If disabled or uncertain, pause execution and inform the user how to enable it. Do not proceed further.
2. **Initialize Auto Memory:** Inspect and initialize the project's Auto Memory. First, review the existing `MEMORY.md` and all memory subfiles without directly overwriting them.
3. **Establish and Enforce Maintenance Rules:**
   - **Rule 1:** `MEMORY.md` acts as an index only. Keep each topic to a single line; do not pile full text into the index.
   - **Rule 2:** Use 150 lines or 20KB as the daily soft limit. Never approach the hard loading limit of 200 lines or 25KB.
   - **Rule 3:** Each subfile must store only one clear topic.
   - **Rule 4:** Search existing memories before writing: update old records instead of redundantly adding new ones.
   - **Rule 5:** When a new conclusion overturns an old one, directly revise the old record rather than keeping two contradictory versions.
   - **Rule 6:** Do not duplicate content into memory that already exists clearly and can be easily retrieved from Git, code, or project documentation.
   - **Rule 7:** Passwords, API keys, cookies, tokens, private keys, and other secrets must never be written to memory.
   - **Rule 8:** If the current implementation supports memory categorization, classify them into `user`, `feedback`, `project`, and `reference`; do not force-create empty directories just for categorization.
4. **Save Rules:** Save this set of disciplines as a long-term memory maintenance rule.
5. **Report Upon Completion:** Provide a report detailing:
   - Current memory directory structure
   - Files created or modified
   - Current line count and file size of `MEMORY.md`
   - Any redundancies, conflicts, or potential risks discovered