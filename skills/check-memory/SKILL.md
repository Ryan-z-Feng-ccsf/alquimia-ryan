---
name: check-memory
description: Perform a complete maintenance and audit check on the project's Auto Memory, including `MEMORY.md` and all subfiles, following strict consolidation, cleanup, and security guidelines.
---

1. **Perform a Complete Audit:** Conduct a thorough maintenance check of the project's Auto Memory by reviewing `MEMORY.md` and all memory subfiles item by item:
   - **Rule 1:** Merge duplicate content.
   - **Rule 2:** Resolve conflicting content based on the latest evidence-backed conclusions and update old records accordingly.
   - **Rule 3:** Delete content that is confirmed to be incorrect.
   - **Rule 4:** Delete content that is outdated and has no remaining reference value.
   - **Rule 5:** Merge files whose topics are too fragmented or scattered.
   - **Rule 6:** Split files that contain multiple unrelated topics.
   - **Rule 7:** Check for and remove content duplicated from information already clearly present in Git, code, or project documentation.
   - **Rule 8:** Scan for and eliminate any passwords, API keys, tokens, cookies, private keys, or other sensitive information.
   - **Rule 9:** Verify that `MEMORY.md` strictly maintains a one-topic-per-line format.
   - **Rule 10:** Count the current line count and byte size of `MEMORY.md`, ensuring there is sufficient buffer distance from the 200 lines / 25KB limit.
2. **Handle Uncertainties:** If you are unsure whether certain content should be deleted, isolate and list those items to ask the user first; do not delete them on your own.
3. **Report Upon Completion:** Provide a summary report detailing:
   - What was merged
   - What was updated
   - What was deleted
   - Which items are pending user confirmation
   - The current line count and size of `MEMORY.md`