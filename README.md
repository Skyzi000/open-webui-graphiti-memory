# Open WebUI Graphiti Memory Extensions

[Graphiti](https://github.com/getzep/graphiti)-based knowledge graph memory extensions for [Open WebUI](https://github.com/open-webui/open-webui).

## Overview

This extension provides **temporal knowledge graph-based memory** for Open WebUI, powered by [Graphiti](https://github.com/getzep/graphiti). The core implementation is a Filter that runs inside Open WebUI, and the same logic can optionally be hosted via the Pipelines framework (still operating in “filter” mode) when you prefer to run it in a separate process.

### Key Benefits

- **Temporal Memory**: Graphiti tracks when information was valid, allowing accurate historical queries
- **Transparent Integration**: Memory is automatically searched and injected into LLM context, and new information is automatically extracted and saved after each turn
- **Knowledge Graph Structure**: Entities, relationships, and episodes are extracted and interconnected
- **Multi-User Isolation**: Each user has their own isolated memory space
- **Flexible Deployment**: Choose between integrated Filter or standalone Pipeline based on your needs

## Components

### 📝 Filter: Graphiti Memory (Integrated - Simple Setup)

**Location**: `functions/filter/graphiti_memory.py`

The Filter version runs inside OpenWebUI server for simple deployments:

1. **Before LLM Processing**: Automatically searches for relevant memories based on the current conversation
2. **Context Injection**: Injects retrieved memories into the LLM's context
3. **After Response**: Automatically stores new information as episodes in the knowledge graph

**Features:**

- Automatic memory search and injection
- RAG document integration
- Configurable search strategies (Fast/Balanced/Quality)
- Per-user memory isolation
- Optional automatic saving of user/assistant messages

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

- **Precise Search**: Search for specific Entities, Facts (relationships), or Episodes separately
- **Confirmation Dialogs**: Well-designed confirmation dialogs before deletion operations
- **Safe Deletion**: Search and delete specific memories with preview
- **Batch Operations**: Manage multiple memory items at once
- **UUID-based Operations**: Direct manipulation via UUIDs when needed

**Memory Types:**

- **Entities**: People, places, concepts with summaries
- **Facts (Edges)**: Relationships between entities with temporal validity
- **Episodes**: Conversation history and source documents

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
docker run -p 7687:7687 -p 7474:7474 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

#### FalkorDB (Alternative, not recently tested)

```bash
docker run -p 6379:6379 falkordb/falkordb:edge
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

### User Valves

Users can customize their experience:

**Note**: To change the default values for all users, administrators should edit the script files directly.

**Filter:**

- `enabled`: Enable/disable automatic memory
- `show_status`: Show status messages during memory operations
- `save_user_message`: Auto-save user messages as episodes
- `save_assistant_response`: Auto-save the latest assistant response as episodes
- `save_previous_assistant_message`: Auto-save the assistant message before the user's message
- `merge_retrieved_context`: Include part of the content from files attached by the user
- `allowed_rag_source_types`: Comma-separated list of retrieval source types to merge (e.g., `'file,text'`)
- `inject_facts`: Inject relationship facts from memory search results
- `inject_entities`: Inject entity summaries from memory search results

**Tools:**

- `message_language`: UI language (`'en'` or `'ja'`)

## Optional: Pipeline Hosting

`pipelines/graphiti_memory_pipeline.py` packages the exact same filter logic so it can run under the Pipelines service. Open WebUI treats it as a "filter"-type pipeline, and its `inlet`/`outlet` behavior is identical to the in-app filter. Copy it to the Pipelines server only when you want to run Graphiti memory on a separate host—no extra configuration or LLM proxying is required.

## How It Works

### Transparent Memory Integration

1. **User sends a message** → Search for relevant memories
2. **Memories are injected** into the LLM's context
3. **LLM processes** the message with memory context
4. **Response is generated** with awareness of past information
5. **New information is extracted** and stored as episodes/entities/facts

### Architecture Comparison

**Filter/Pipeline Architecture:**
```
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
