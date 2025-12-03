# Open WebUI Graphiti Memory Extensions

[Graphiti](https://github.com/getzep/graphiti)-based knowledge graph memory extensions for [Open WebUI](https://github.com/open-webui/open-webui).

## Overview

This extension provides **temporal knowledge graph-based memory** for Open WebUI, powered by [Graphiti](https://github.com/getzep/graphiti). The core implementation is a Filter that runs inside Open WebUI, and the same logic can optionally be hosted via the Pipelines framework (still operating in “filter” mode) when you prefer to run it in a separate process.

### Key Benefits

- **Temporal Memory**: Graphiti tracks when information was valid, allowing accurate historical queries
- **Transparent Integration**: Memory is automatically searched and injected into LLM context, and new information is automatically extracted and saved after each turn
- **Knowledge Graph Structure**: Entities, relationships, and episodes are extracted and interconnected
- **Multi-User Isolation**: Each user has their own isolated memory space
- **Rich Citations**: Interactive HTML citation cards with graph visualizations
- **Distributed Support**: Redis integration for multi-instance deployments
- **Flexible Deployment**: Choose between integrated Filter or standalone Pipeline based on your needs

## Components

### 📝 Filter: Graphiti Memory (Integrated - Simple Setup)

**Location**: `functions/filter/graphiti_memory.py`

The Filter version runs inside Open WebUI server for simple deployments:

1. **Before LLM Processing**: Automatically searches for relevant memories based on the current conversation
2. **Context Injection**: Injects retrieved memories into the LLM's context
3. **After Response**: Automatically stores new information as episodes in the knowledge graph

**Features:**

- Automatic memory search and injection
- RAG document integration
- Configurable search strategies (Fast/Balanced/Quality)
- Per-user memory isolation
- Optional automatic saving of user/assistant messages
- Rich HTML citations with interactive graph visualizations
- Episode deduplication to prevent duplicate saves in concurrent scenarios
- Redis support for distributed deployments

### 🔘 Action: Add Graphiti Memory

**Location**: `functions/action/add_graphiti_memory_action.py`

Provides an action button to manually save specific messages to memory.

**Use Case**: Save the assistant's final response when Filter's automatic saving is disabled. By default, the Filter only saves:

- The last user message
- Part of the content extracted from files attached to the chat
- The assistant message immediately before the user's message

The final assistant response is NOT saved by default. This design prevents incorrect assistant responses from being automatically stored during the regeneration process (when users modify their message and regenerate responses multiple times). Use this Action button to explicitly save important assistant responses when you're satisfied with the answer. (Not needed if you enable `save_assistant_response` in Filter's User Valves.)

### 🛠️ Tools: Graphiti Memory Manage

**Location**: `tools/graphiti_memory_manage.py`

AI-callable tools for memory management.

**Features:**

- **Add Memory**: Let AI add specific episodes to the knowledge graph on demand (commented out by default; uncomment if Filter is disabled)
- **Migration**: Migrate Open WebUI's built-in memories to Graphiti knowledge graph
- **Precise Search**: Search for specific Entities, Facts (relationships), or Episodes separately
- **Confirmation Dialogs**: Well-designed confirmation dialogs before deletion operations
- **Safe Deletion**: Search and delete specific memories with preview
- **Batch Operations**: Manage multiple memory items at once
- **UUID-based Operations**: Direct manipulation via UUIDs when needed

**Memory Types:**

- **Entities**: People, places, concepts with summaries
- **Facts (Edges)**: Relationships between entities with temporal validity
- **Episodes**: Conversation history and source documents

#### Built-in Memory Migration

The `migrate_builtin_memories` tool migrates your existing Open WebUI built-in memories to the Graphiti knowledge graph.

**How to use:**

1. Ask the AI: "Migrate my built-in memories to Graphiti" or "ビルトインメモリをGraphitiに移行して"
2. The AI will show a confirmation dialog with migration details
3. Confirm to proceed with the migration

**Migration features:**

- **Idempotent**: Safe to run multiple times - already migrated memories are skipped
- **Chronological order**: Memories are migrated from oldest to newest
- **Timestamp preservation**: Original creation time is preserved in UTC
- **Progress display**: Shows real-time progress with extracted entities and facts
- **Graceful interruption**: Can be stopped anytime; already migrated memories are preserved
- **Dry-run mode**: Preview what would be migrated without making changes

**Note**: Built-in memories are NOT automatically deleted after migration. You can manually delete them from Open WebUI's Settings → Personalization → Memory if needed.

## Requirements

- Python 3.10+
- Open WebUI
- Neo4j database (recommended)
- OpenAI-compatible LLM endpoint with JSON structured output support

## Installation

Choose one of the following installation methods:

### Default: Filter Installation (Recommended)

**Best for:** Most deployments, including production. If you need to run the same logic in the Pipelines service, copy `graphiti/pipelines/graphiti_memory_pipeline.py` to your pipelines folder (runs in the same filter mode).

#### 1. Install Graph Database

#### Neo4j (Recommended)

```bash
docker run -d -p 7687:7687 -p 7474:7474 -e NEO4J_AUTH=neo4j/password -v ./data:/data neo4j:latest
```

#### FalkorDB (Alternative, not recently tested)

> **⚠️ Warning: Redis Version Conflict**
>
> The `falkordb` package requires `redis<6.0`, which conflicts with Open WebUI's redis dependency (latest version, currently 7.x). If you use Open WebUI's Redis features (e.g., `WEBSOCKET_MANAGER=redis`), this downgrade can cause issues.
>
> **If you still want to use FalkorDB:**
>
> Install the FalkorDB extra in your environment (recommended - persists across extension updates):
>
> ```bash
> pip install graphiti-core[falkordb]
> ```
>
> Or add it to your Dockerfile based on the Open WebUI image.
>
> Alternatively, you can edit the `requirements` line in each extension file to `graphiti-core[falkordb]`, but this must be repeated after every update.
>
> **Neo4j backend is recommended** to avoid this conflict.

```bash
docker run -d -p 6379:6379 -p 3000:3000 -v ./data:/var/lib/falkordb/data falkordb/falkordb
```

#### 2. Add Filter to Open WebUI

Copy the raw GitHub URLs and paste them into Open WebUI's import dialog:

- Filter: `https://raw.githubusercontent.com/Skyzi000/open-webui-graphiti-memory/main/functions/filter/graphiti_memory.py`
- Action: `https://raw.githubusercontent.com/Skyzi000/open-webui-graphiti-memory/main/functions/action/add_graphiti_memory_action.py`
- Tools: `https://raw.githubusercontent.com/Skyzi000/open-webui-graphiti-memory/main/tools/graphiti_memory_manage.py`

For detailed instructions, refer to the [Open WebUI official documentation](https://docs.openwebui.com/).

Open WebUI will automatically install dependencies (`graphiti-core`) when you activate these extensions.

## Configuration

### Valves Settings

#### Graph Database

- `graph_db_backend`: `'neo4j'` (recommended) or `'falkordb'`
- For Neo4j: Configure `neo4j_uri`, `neo4j_user`, `neo4j_password`
- For FalkorDB: Configure `falkordb_host`, `falkordb_port`

#### LLM Settings

**Recommended**: Use OpenAI's official API for best compatibility, especially for JSON structured output.

- `llm_client_type`: `'openai'` (recommended) or `'generic'`
- `openai_api_url`: Your OpenAI-compatible endpoint
- `model`: Memory processing model (default recommended)
- `embedding_model`: Embedding model (default recommended)
- `api_key`: Your API key

**Important**: The LLM endpoint must support JSON structured output properly. Endpoints that don't handle structured output correctly will cause ingestion failures.

#### Search Strategy (Filter only)

- `search_strategy`:
  - `'fast'`: BM25 full-text search only (no embedding calls)
  - `'balanced'`: BM25 + Cosine Similarity (DEFAULT)
  - `'quality'`: + Cross-Encoder reranking (may not work correctly in current version)

Note: The 'quality' strategy may have compatibility issues in the current version.

#### Memory Isolation

- `group_id_format`: Format for user memory isolation (default: `{user_id}`)
  - Use `{user_id}` for per-user isolation
  - Available placeholders: `{user_id}`, `{user_name}`, `{user_email}`
  - Example: `{user_email}` converts `user@example.com` to `user_example_com`
  - Using `{user_email}` makes it easier to share memory across different applications
  - **Warning**: Email/name may be changed, which would change the group_id. Use `{user_id}` for stable isolation.
  - Set to `'none'` for shared memory space

#### Redis Settings (Optional)

For distributed deployments with multiple Open WebUI instances:

- `use_redis`: Enable Redis for distributed features (episode dedup, citations). Automatically enabled when Open WebUI is configured with `WEBSOCKET_MANAGER=redis`
- `redis_url`: Redis connection URL. Uses Open WebUI's `REDIS_URL` environment variable by default
- `redis_key_prefix`: Custom prefix for Redis keys

#### Advanced Settings

- `semaphore_limit`: Maximum concurrent LLM operations (default: 10). Adjust based on your LLM provider's rate limits
- `add_episode_timeout`: Timeout in seconds for adding episodes (default: 240)
- `max_search_message_length`: Maximum length of user message for Graphiti search (default: 5000)
- `memory_message_role`: Role for injected memory messages (`'system'` or `'user'`)
- `enable_entity_delete_button`: Show delete button in entity citation cards (experimental)

### User Valves

Users can customize their experience:

**Note**: To change the default values for all users, administrators should edit the script files directly.

**Filter:**

- `enabled`: Enable/disable automatic memory
- `show_status`: Show status messages during memory operations
- `show_citation`: Emit retrieval results as citation events (affects fact/entity previews)
- `rich_html_citations`: Render citation results as rich HTML with interactive graph visualizations
- `show_citation_parameters`: Include detailed metadata parameters in citation events
- `save_user_message`: Auto-save user messages as episodes
- `save_assistant_response`: Auto-save the latest assistant response as episodes
- `save_previous_assistant_message`: Auto-save the assistant message before the user's message
- `merge_retrieved_context`: Include part of the content from files attached by the user
- `allowed_rag_source_types`: Comma-separated list of retrieval source types to merge (e.g., `'file,text'`)
- `inject_facts`: Inject relationship facts from memory search results
- `inject_entities`: Inject entity summaries from memory search results
- `search_history_turns`: Number of recent user/assistant messages to include in search query
- `episode_dedup_enabled`: Enable duplicate episode detection for concurrent scenarios
- `skip_save_regex`: Regex pattern to skip saving certain messages (e.g., admin commands)

**Action:**

- `show_status`: Show status messages during save operations
- `save_user_message`: Save user messages when action is triggered
- `save_assistant_response`: Save assistant responses when action is triggered

**Tools:**

- `message_language`: UI language for confirmation dialogs (`'en'` or `'ja'`)
- `show_extracted_details`: Show extracted entities and facts in status after adding memory or during migration (default: `true`)

## Optional: Pipeline Hosting

`pipelines/graphiti_memory_pipeline.py` packages the exact same filter logic so it can run under the Pipelines service. Open WebUI treats it as a "filter"-type pipeline, and its `inlet`/`outlet` behavior is identical to the in-app filter.  
To use this pipeline version, enter the following GitHub Raw URL in **Admin Settings → Pipelines**.

```text
https://raw.githubusercontent.com/Skyzi000/open-webui-graphiti-memory/main/pipelines/graphiti_memory_pipeline.py
```

**Important pipeline limitations:** Open WebUI currently does not pass per-user `UserValves` or `__event_emitter__` callbacks into pipelines. As a result, unlike the Filter version, the Pipeline version always falls back to the default user settings defined in the script, and the live status display is also unavailable. If you need per-user customization or status display, deploy the filter variant instead of the pipeline.

## How It Works

### Transparent Memory Integration

1. **User sends a message** → Search for relevant memories
2. **Memories are injected** into the LLM's context
3. **LLM processes** the message with memory context
4. **Response is generated** with awareness of past information
5. **New information is extracted** and stored as episodes/entities/facts

### Architecture Comparison

**Filter/Pipeline Architecture:**

```text
OpenWebUI → Filter (inlet) → LLM → Filter (outlet) → Response
```

### Request Headers to LLM Provider

- Context variables are used to pass user-specific headers to the LLM provider for each request
- Designed for complete request isolation with no shared state between concurrent requests
- User information headers (User-Name, User-Id, User-Email, User-Role, Chat-Id) follow Open WebUI's `ENABLE_FORWARD_USER_INFO_HEADERS` environment variable by default, but can be overridden in Valves settings
- **Note**: This feature has not been extensively tested in all environments. Please report any issues you encounter.

## Related Projects

- [Open WebUI](https://github.com/open-webui/open-webui) - Main web interface
- [Graphiti](https://github.com/getzep/graphiti) - Temporal knowledge graph framework
- [Neo4j](https://neo4j.com/) - Graph database (recommended)
- [FalkorDB](https://www.falkordb.com/) - Alternative graph database

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions welcome! Please feel free to submit issues or pull requests.
