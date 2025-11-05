# Open WebUI Graphiti Memory Extensions

[Graphiti](https://github.com/getzep/graphiti-core)-based knowledge graph memory extensions for [Open WebUI](https://github.com/open-webui/open-webui).

## Features

### 📝 Filter: Graphiti Memory (`functions/filter/graphiti_memory.py`)
Automatically identifies and stores valuable information from chats as memories in a knowledge graph.

**Features:**
- Automatic memory search before chat processing
- Memory injection into conversation context
- Automatic memory storage after chat completion
- RAG context integration
- Multi-user support with isolated memory spaces
- Configurable search strategies (Fast/Balanced/Quality)

### 🔘 Action: Add Graphiti Memory (`functions/action/add_graphiti_memory_action.py`)
Action button to manually save clicked messages to Graphiti knowledge graph memory.

**Features:**
- Manual memory save via button click
- User/Assistant message selection
- Episode metadata tracking
- Group-based memory isolation

### 🛠️ Tools: Graphiti Memory Manage (`tools/graphiti_memory_manage.py`)
Comprehensive memory management tools for AI-driven operations.

**Features:**
- Add new memories (text/message/json)
- Search entities, facts, and episodes
- Delete specific memories with confirmation
- Clear all memory (with double confirmation)
- UUID-based precise deletion
- Batch operations

## Requirements

- Python 3.10+
- Open WebUI
- Graph database (Neo4j or FalkorDB)
- OpenAI-compatible LLM endpoint

## Installation

### 1. Install Graph Database

**Option A: FalkorDB (Recommended for Docker)**
```bash
docker run -p 6379:6379 falkordb/falkordb:edge
```

**Option B: Neo4j**
```bash
docker run -p 7687:7687 -p 7474:7474 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

### 2. Add to Open WebUI

**Method 1: As Submodule (Recommended)**

Add this repository as a submodule to your Open WebUI extensions:

```bash
cd /path/to/your/open-webui-extensions
git submodule add https://github.com/Skyzi000/open-webui-graphiti-memory.git graphiti
```

Then create symbolic links:
```bash
ln -s graphiti/functions/filter/graphiti_memory.py functions/filter/
ln -s graphiti/functions/action/add_graphiti_memory_action.py functions/action/
ln -s graphiti/tools/graphiti_memory_manage.py tools/
```

**Method 2: Direct Installation**

Copy the files directly to your Open WebUI functions/tools directories.

### 3. Install Dependencies

```bash
pip install graphiti-core[falkordb]  # For FalkorDB
# or
pip install graphiti-core[neo4j]  # For Neo4j
```

## Configuration

### Valves Settings

Configure the extensions via Open WebUI's admin panel:

**Graph Database:**
- `graph_db_backend`: Choose 'neo4j' or 'falkordb'
- Neo4j: Configure `neo4j_uri`, `neo4j_user`, `neo4j_password`
- FalkorDB: Configure `falkordb_host`, `falkordb_port`

**LLM Settings:**
- `llm_client_type`: 'openai' or 'generic' (try both to see which works better)
- `openai_api_url`: Your OpenAI-compatible endpoint
- `model`: Model for memory processing (e.g., 'gpt-4')
- `embedding_model`: Embedding model (e.g., 'text-embedding-3-small')
- `api_key`: Your API key

**Search Strategy (Filter only):**
- `search_strategy`: 
  - 'fast': BM25 only (~0.1s, no embedding calls)
  - 'balanced': BM25 + Cosine Similarity (~0.5s) - DEFAULT
  - 'quality': + Cross-Encoder reranking (~1-5s)

**Memory Isolation:**
- `group_id_format`: Format for user memory isolation (default: `{user_id}`)
  - Use `{user_id}` for stable, per-user isolation
  - Set to 'none' for shared memory space

### User Valves

Users can customize their experience:

**Filter:**
- `enabled`: Enable/disable automatic memory
- `save_user_message`: Auto-save user messages
- `save_assistant_response`: Auto-save assistant responses
- `merge_retrieved_context`: Merge RAG context into memories

**Tools:**
- `message_language`: UI language ('en' or 'ja')

## Architecture

### Multi-User Support

The extensions use context variables for complete request isolation:
- Custom OpenAI client classes inject per-user headers dynamically
- No shared state between concurrent requests
- Headers forwarded: User-Name, User-Id, User-Email, User-Role, Chat-Id

### Search Strategies

Three performance/quality tradeoffs:
1. **Fast** (~100ms): BM25 full-text search only
2. **Balanced** (~500ms): BM25 + Cosine Similarity (DEFAULT)
3. **Quality** (~1-5s): + Cross-Encoder reranking

### Memory Types

- **Entities (Nodes)**: People, places, concepts with summaries
- **Facts (Edges)**: Relationships with validity periods
- **Episodes**: Conversation history with metadata

## Usage Examples

### Filter: Automatic Memory

Just chat normally - the filter automatically:
1. Searches for relevant memories before processing
2. Injects memories into conversation context
3. Stores new information after completion

### Action: Manual Save

Click the action button on any message to manually save it to memory.

### Tools: Memory Management

```
# Add new memory
add_memory(name="Meeting Notes", content="Discussed Q1 targets with John", source="text")

# Search entities
search_entities(query="John", limit=10)

# Delete specific memories
search_and_delete_entities(query="John", limit=5)

# Clear all memory (requires double confirmation)
clear_all_memory()
```

## Related Projects

- [Open WebUI](https://github.com/open-webui/open-webui) - Main web interface
- [Graphiti](https://github.com/getzep/graphiti-core) - Knowledge graph memory system
- [FalkorDB](https://github.com/FalkorDB/FalkorDB) - Graph database

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions welcome! Please feel free to submit issues or pull requests.

## Author

- **Skyzi000**
- GitHub: [@Skyzi000](https://github.com/Skyzi000)
- Website: [skyzi000.hatenablog.com](https://skyzi000.hatenablog.com/)
