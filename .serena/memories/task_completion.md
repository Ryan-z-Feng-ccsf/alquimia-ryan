# Task Completion

To declare a coding task complete, perform these steps in order:

1. **Rebuild:** Run `make -j$(nproc)` in the build directory to ensure clean compilation without warnings.
2. **Test:** Run `ctest --output-on-failure` from `build/alquimia-build/` to verify all 14 tests pass successfully.
3. **Comment Compliance:** Verify any modified public signatures conform to Doxygen standards.
4. **Leak-free check:** Ensure all created allocations are released in the corresponding shutdown routines.