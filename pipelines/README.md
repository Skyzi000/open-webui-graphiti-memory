# Graphiti Memory Pipeline

**Standalone Pipeline version** of the Graphiti Memory system for Open WebUI.

## Overview

This Pipeline version runs as an independent service, separate from the OpenWebUI server. It acts as a middleware/proxy that:

1. **Intercepts** requests from OpenWebUI
2. **Searches** for relevant memories in the knowledge graph
3. **Injects** memories into the conversation context
4. **Forwards** the modified request to your configured LLM
5. **Streams** the LLM response back to OpenWebUI
6. **Stores** new memories asynchronously (non-blocking)

## Key Differences from Filter Version

| Aspect | Filter | Pipeline |
|--------|--------|----------|
| **Deployment** | Runs inside OpenWebUI server | Runs as standalone service |
| **Load** | Adds load to OpenWebUI server | Offloads processing to separate service |
| **Architecture** | Uses `inlet()` and `outlet()` | Uses `pipe()` method |
| **LLM Forwarding** | Direct OpenWebUI → LLM | OpenWebUI → Pipeline → LLM |
| **Scalability** | Limited by server resources | Can scale independently |
| **Dependencies** | Installed in OpenWebUI | Self-contained service |

## Benefits

- **Reduced OpenWebUI Server Load**: Memory processing runs on a separate service
- **Independent Scaling**: Scale the memory service independently from OpenWebUI
- **Fault Isolation**: Pipeline issues don't affect OpenWebUI stability
- **Flexible Deployment**: Run on different hardware or cloud services
- **Better Resource Management**: Dedicated resources for memory operations

## Installation

### 1. Prerequisites

- Python 3.10+
- Neo4j or FalkorDB graph database
- OpenAI-compatible LLM endpoint
- Docker (recommended for deployment)

### 2. Deploy the Pipeline Server

#### Using Docker (Recommended)

1. Create a directory for your pipeline:
```bash
mkdir -p ~/graphiti-pipeline
cd ~/graphiti-pipeline
```

2. Copy `graphiti_memory_pipeline.py` to this directory

3. Create a `docker-compose.yml`:
```yaml
version: '3.8'

services:
  graphiti-pipeline:
    image: ghcr.io/open-webui/pipelines:main
    ports:
      - "9099:9099"
    volumes:
      - ./graphiti_memory_pipeline.py:/app/pipelines/graphiti_memory_pipeline.py
      - pipeline-data:/app/pipelines
    environment:
      - PIPELINES_API_KEY=your-secret-key-here
    restart: unless-stopped

  neo4j:
    image: neo4j:latest
    ports:
      - "7687:7687"
      - "7474:7474"
    environment:
      - NEO4J_AUTH=neo4j/your-neo4j-password
    volumes:
      - neo4j-data:/data
    restart: unless-stopped

volumes:
  pipeline-data:
  neo4j-data:
```

4. Start the services:
```bash
docker-compose up -d
```

#### Manual Installation

1. Install the Pipelines framework:
```bash
pip install open-webui-pipelines
```

2. Install dependencies:
```bash
pip install graphiti-core[falkordb] httpx
```

3. Create pipelines directory and copy the file:
```bash
mkdir -p ~/pipelines
cp graphiti_memory_pipeline.py ~/pipelines/
```

4. Run the pipeline server:
```bash
cd ~/pipelines
python -m open_webui_pipelines
```

### 3. Configure OpenWebUI to Use the Pipeline

1. Open OpenWebUI Admin Panel
2. Go to **Settings** → **Connections**
3. Add OpenAI API connection:
   - **API URL**: `http://localhost:9099` (or your pipeline server URL)
   - **API Key**: Your `PIPELINES_API_KEY` from docker-compose.yml
4. The Graphiti Memory Pipeline should appear as a model option

### 4. Configure the Pipeline

In OpenWebUI, go to **Workspace** → **Models** → Select the Graphiti Memory Pipeline model → **Settings**:

#### Essential Configuration

**Memory Processing LLM** (for extracting entities/facts):
- `openai_api_url`: Your LLM endpoint for memory processing
- `api_key`: API key for memory processing
- `model`: Model for memory extraction (e.g., `gpt-4o-mini`)
- `embedding_model`: Model for embeddings (e.g., `text-embedding-3-small`)

**Target LLM** (where responses come from):
- `target_llm_url`: Your main LLM endpoint (e.g., `https://api.openai.com/v1`)
- `target_llm_api_key`: API key for the main LLM

**Graph Database**:
- `graph_db_backend`: `neo4j` or `falkordb`
- For Neo4j: Configure `neo4j_uri`, `neo4j_user`, `neo4j_password`
- For FalkorDB: Configure `falkordb_host`, `falkordb_port`

#### User Configuration

Users can customize their experience in **Settings** → **Account** → **User Valves**:

- `enabled`: Enable/disable memory features
- `show_status`: Show status messages during operations
- `save_user_message`: Auto-save user messages
- `save_assistant_response`: Auto-save assistant responses
- `inject_facts`: Inject relationship facts from memory
- `inject_entities`: Inject entity summaries from memory

## Usage

Once configured, the pipeline works transparently:

1. Start a chat in OpenWebUI
2. Select the Graphiti Memory Pipeline as your model
3. Send messages normally
4. The pipeline will:
   - Search for relevant memories
   - Inject them into context
   - Forward to your configured LLM
   - Store new memories
   - Return responses

## Configuration Options

### Valves (Admin Configuration)

#### LLM Configuration
- `llm_client_type`: `'openai'` or `'generic'` for memory processing
- `openai_api_url`: Endpoint for memory processing LLM
- `model`: Model for memory extraction
- `embedding_model`: Embedding model
- `api_key`: API key for memory processing

#### Target LLM Configuration
- `target_llm_url`: Endpoint for main LLM responses
- `target_llm_api_key`: API key for main LLM

#### Graph Database
- `graph_db_backend`: `'neo4j'` or `'falkordb'`
- Neo4j: `neo4j_uri`, `neo4j_user`, `neo4j_password`
- FalkorDB: `falkordb_host`, `falkordb_port`

#### Search Strategy
- `search_strategy`:
  - `'fast'`: BM25 only (~0.1s, no embedding calls)
  - `'balanced'`: BM25 + Cosine Similarity (~0.5s) - DEFAULT
  - `'quality'`: + Cross-Encoder reranking (~1-5s)

#### Memory Management
- `group_id_format`: User isolation format (default: `{user_id}`)
  - `{user_id}`: Per-user isolation (recommended)
  - `{user_email}`: Email-based isolation
  - `'none'`: Shared memory space
- `memory_message_role`: `'system'` or `'user'` for injected memories
- `use_user_name_in_episode`: Use actual names in saved episodes

#### Performance
- `semaphore_limit`: Concurrent LLM operations (default: 10)
- `add_episode_timeout`: Timeout for saving episodes (default: 240s)
- `max_search_message_length`: Max search query length (default: 5000)

### UserValves (Per-User Configuration)

#### Feature Control
- `enabled`: Enable/disable memory features
- `show_status`: Show status messages

#### Message Saving
- `save_user_message`: Auto-save user messages
- `save_assistant_response`: Auto-save assistant responses
- `save_previous_assistant_message`: Save previous assistant message

#### Memory Injection
- `inject_facts`: Inject relationship facts
- `inject_entities`: Inject entity summaries

#### RAG Integration
- `merge_retrieved_context`: Merge RAG context into messages
- `allowed_rag_source_types`: Allowed source types (e.g., `'file,text'`)

## Architecture

```
┌─────────────┐
│  OpenWebUI  │
│   Client    │
└──────┬──────┘
       │ 1. Send request
       ▼
┌─────────────────────────────────┐
│  Graphiti Memory Pipeline       │
│  ┌──────────────────────────┐  │
│  │ 2. Search memories       │  │
│  │ 3. Inject context        │  │
│  └──────────────────────────┘  │
│  ┌──────────────────────────┐  │
│  │ 4. Forward to LLM        │──┼──► Target LLM
│  │ 5. Stream response       │◄─┼─── (OpenAI, etc.)
│  └──────────────────────────┘  │
│  ┌──────────────────────────┐  │
│  │ 6. Store memories (async)│  │
│  └──────────────────────────┘  │
└────────────┬────────────────────┘
             │
             ▼
    ┌────────────────┐
    │ Neo4j/FalkorDB │
    │  Graph Database│
    └────────────────┘
```

## Monitoring and Debugging

Enable debug output:
```python
# In Valves settings
debug_print: True
```

This will print:
- Initialization status
- Search queries and results
- Memory injection details
- Storage operations
- Error messages

Check logs:
```bash
# Docker
docker-compose logs -f graphiti-pipeline

# Manual
# Check your terminal where the pipeline server is running
```

## Troubleshooting

### Pipeline not appearing in OpenWebUI

1. Check pipeline server is running:
```bash
curl http://localhost:9099/health
```

2. Verify OpenWebUI connection settings
3. Check API key matches

### Memory search failing

1. Verify graph database is running:
```bash
# Neo4j
curl http://localhost:7474

# FalkorDB
redis-cli -p 6379 ping
```

2. Check database credentials in Valves
3. Enable `debug_print` to see detailed errors

### LLM forwarding issues

1. Verify `target_llm_url` is correct
2. Check `target_llm_api_key` is valid
3. Test target LLM endpoint directly
4. Check network connectivity

### Performance issues

1. Adjust `search_strategy`:
   - Use `'fast'` for quicker responses
   - Use `'balanced'` (default) for good tradeoff
   - Use `'quality'` for best results (slower)

2. Tune `semaphore_limit`:
   - Decrease if hitting rate limits
   - Increase if provider allows higher throughput

3. Consider scaling:
   - Run multiple pipeline instances
   - Use load balancer
   - Scale graph database

## Comparison with Filter Version

**When to use Filter**:
- Simple deployments
- Low traffic
- Minimal resource concerns
- Prefer integrated solution

**When to use Pipeline**:
- High traffic scenarios
- Need to scale independently
- Want fault isolation
- Have separate infrastructure
- Need flexible deployment options

## Migration from Filter

If you're currently using the Filter version:

1. Deploy the Pipeline as described above
2. Configure it with the same database settings
3. Switch your OpenWebUI model selection to the Pipeline
4. Existing memories will work seamlessly (same database)
5. Optionally remove the Filter if no longer needed

## Support

For issues, questions, or contributions:
- **Repository**: https://github.com/Skyzi000/open-webui-graphiti-memory
- **Open WebUI**: https://github.com/open-webui/open-webui
- **Graphiti**: https://github.com/getzep/graphiti

## License

MIT License - see [LICENSE](../LICENSE) file for details.
