# Plan: Runtime In-Memory Patch for `chat_template.jinja`

## Objective
Implement a runtime, in-memory patch for the Gemma 4 chat template to prevent crashes/errors caused by calling the `upper` filter on non-string values. This ensures a fix is active even for freshly downloaded models without modifying the on-disk assets.

## Constraints
- **Persistence:** Purely in-memory. No changes to the `.gturbo` bundle on disk.
- **Scope:** Limited specifically to `chat_template.jinja` for the Gemma model.
- **Resilience:** Fall back to the unpatched template if patching fails, with a visible warning in the terminal.
- **Silence:** No output if the patch is successful or if no replacements were necessary.

---

## Execution Steps

### Phase 1: Discovery & Analysis
1. **Locate Template Loading:** Search the Swift source (primarily `Sources/TurboFieldfare/`) to identify the exact code path where `chat_template.jinja` is read from the model bundle into a `String`.
2. **Analyze Template Lifecycle:** Determine where the template string is stored and passed to the Jinja/Templating engine to identify the optimal injection point for the patch.

### Phase 2: Implementation
1. **Implement `ChatTemplatePatcher`:** Create a Swift utility to perform the following literal string replacements:
    - `value['type'] | upper` $\rightarrow$ `(value['type'] or '') | upper`
    - `params['type'] | upper` $\rightarrow$ `(params['type'] or '') | upper`
    - `response_declaration['type'] | upper` $\rightarrow$ `(response_declaration['type'] or '') | upper`
    - `item_value | upper` $\rightarrow$ `(item_value or '') | upper`
2. **Inject Patch Logic:**
    - Insert the call to the patcher immediately after the file content is loaded into a string.
    - Implement a safety wrapper:
        - **Success/No-op:** Continue silently.
        - **Failure:** Catch any string manipulation errors, log a warning to the terminal, and return the original unpatched string.

### Phase 3: Verification
1. **Runtime Inspection:** Use debug logs or breakpoints to verify that the string passed to the template engine contains the patched syntax.
2. **Functional Validation:** 
    - Test standard chat interactions to ensure no regressions.
    - Test tool-calling scenarios that specifically exercise the patched lines to verify the fix.
3. **Fallback Test:** (Optional) Simulate a patching failure to verify the warning is printed and the system remains operational via fallback.
