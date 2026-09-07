# Conversation history

The Mac app saves completed exchanges automatically. Use the sidebar to browse,
search, rename or delete chats. Browsing reads the transcript without loading it
into the model. Sending in another saved chat restores its recorded token IDs
and stored images before continuing.

Only one conversation's KV state is held in memory. Browsing away and back
preserves that state. Continuing a different chat replaces it. **New Chat**
starts a new lineage; its first message does not reuse the previous chat's
context. The composer is shared across sidebar selections, not saved separately
for every chat.

## Storage and privacy

History lives in `conversations/` beside the configured model directory and
`mac-app-settings.json`. With the default development model this is
`scratch/conversations/`. Each chat has a UUID directory containing:

- `transcript.jsonl`: the authoritative record, including text and token IDs.
- `conversation.json`: metadata used to list chats, rebuilt from the record.
- `images/`: stored model-input pictures and thumbnails.

These are local, unencrypted files. History is not uploaded by the app. Anyone
with access to the files can read the text and images; backup or sync software
may copy them. Delete removes the chat directory and its images permanently,
without an in-app undo. It does not remove copies made by other software.

## Recovery and compatibility

Reloading or quitting releases live KV. Saved chats remain readable and replay
on continuation. A different model, checkpoint, chat format, missing image or
newer history format may prevent continuation; the app shows the reason. A chat
that exceeds the current context requires a larger context before replay.

An interrupted final exchange is recovered to the last complete exchange.
Write failures show a warning: copy any unsaved answer before leaving the chat.
If the first save fails, the app will not later save only the continuation of
that unsaved context. Start a new chat after storage recovers to resume saving.

Only one app instance writes a store. A second instance shows a read-only
notice while the writer lock is held. Newer-format files are not rewritten or
cleaned up by this build.
