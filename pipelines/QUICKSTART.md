# Quick Start Guide: Graphiti Memory Pipeline

This guide will help you deploy the Graphiti Memory Pipeline in under 10 minutes.

## Prerequisites

- Docker and Docker Compose installed
- 4GB+ free RAM
- OpenWebUI instance (running separately)

## Step 1: Download Files

```bash
# Create directory
mkdir ~/graphiti-pipeline && cd ~/graphiti-pipeline

# Download the pipeline file
curl -O https://raw.githubusercontent.com/Skyzi000/open-webui-graphiti-memory/main/pipelines/graphiti_memory_pipeline.py

# Download docker-compose file
curl -o docker-compose.yml https://raw.githubusercontent.com/Skyzi000/open-webui-graphiti-memory/main/pipelines/docker-compose.example.yml
```

## Step 2: Configure

Edit `docker-compose.yml` and change:

```yaml
# Change this line (required):
- PIPELINES_API_KEY=your-secret-pipeline-key-here

# And this line (required):
- NEO4J_AUTH=neo4j/your-neo4j-password-here
```

**Important**: Use strong passwords for production!

## Step 3: Start Services

```bash
docker-compose up -d
```

Wait 30 seconds for services to start, then verify:

```bash
# Check pipeline is running
curl http://localhost:9099/health

# Check Neo4j is running (should show login page)
curl http://localhost:7474
```

## Step 4: Configure OpenWebUI

1. Open OpenWebUI Admin Panel
2. Go to **Settings** → **Connections**
3. Add OpenAI API connection:
   - **Name**: Graphiti Memory Pipeline
   - **API Base URL**: `http://localhost:9099` (or your server IP)
   - **API Key**: Your `PIPELINES_API_KEY` from docker-compose.yml

4. The Graphiti Memory Pipeline should appear as a model

## Step 5: Configure Pipeline Settings

In OpenWebUI:
1. Go to **Workspace** → **Models**
2. Find "Graphiti Memory Pipeline"
3. Click **Settings** (gear icon)

### Essential Settings:

**Memory Processing (for extracting entities/facts):**
- `openai_api_url`: Your LLM endpoint (e.g., `https://api.openai.com/v1`)
- `api_key`: Your OpenAI API key (or compatible service)
- `model`: Model name (e.g., `gpt-4o-mini`)
- `embedding_model`: `text-embedding-3-small`

**Target LLM (where responses come from):**
- `target_llm_url`: Your main LLM endpoint (same or different from above)
- `target_llm_api_key`: API key for main LLM

**Database (pre-filled if using docker-compose):**
- `graph_db_backend`: `neo4j`
- `neo4j_uri`: `bolt://neo4j:7687` (or `bolt://localhost:7687` if pipeline is not in Docker)
- `neo4j_user`: `neo4j`
- `neo4j_password`: Your Neo4j password from docker-compose.yml

## Step 6: Test It

1. Start a new chat in OpenWebUI
2. Select "Graphiti Memory Pipeline" as your model
3. Send a message like: "My name is John and I work at Acme Corp"
4. Send another: "What's my name?"
5. The model should remember your name!

Check status messages to see:
- 🔍 Memory search operations
- 🧠 Found facts and entities
- ✍️ Memory storage
- ✅ Completion status

## Troubleshooting

### Pipeline not appearing in OpenWebUI

```bash
# Check if pipeline is running
docker-compose ps

# Check logs
docker-compose logs graphiti-pipeline

# Verify health
curl http://localhost:9099/health
```

### Memory not working

```bash
# Check Neo4j is running
docker-compose logs neo4j

# Test Neo4j connection
docker exec -it graphiti-neo4j cypher-shell -u neo4j -p your-password

# Inside cypher-shell, run:
MATCH (n) RETURN count(n);
```

### Connection refused errors

- If running OpenWebUI in Docker, use `host.docker.internal:9099` instead of `localhost:9099`
- Or add both services to the same Docker network

## Next Steps

### User Configuration

Users can customize their experience in **Settings** → **Account** → **User Valves**:

- `enabled`: Toggle memory on/off
- `show_status`: Show/hide status messages
- `save_user_message`: Auto-save user messages
- `save_assistant_response`: Auto-save AI responses
- `inject_facts`: Include relationship facts in context
- `inject_entities`: Include entity information in context

### Performance Tuning

In Pipeline Settings (**Valves**):

- `search_strategy`: 
  - `'fast'`: Fastest (~0.1s)
  - `'balanced'`: Default (~0.5s) 
  - `'quality'`: Best results (~1-5s)

- `semaphore_limit`: Concurrent operations (default: 10)
  - Increase if your API allows higher rate limits
  - Decrease if you see rate limit errors

### Viewing Memories

Access Neo4j Browser at http://localhost:7474

Example queries:

```cypher
// View all entities
MATCH (n:Entity) 
RETURN n.name, n.summary 
LIMIT 10;

// View relationships
MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
RETURN a.name, type(r), b.name, r.fact
LIMIT 10;

// View episodes (conversations)
MATCH (e:Episode)
RETURN e.name, e.content
ORDER BY e.created_at DESC
LIMIT 5;
```

## Advanced Configuration

See [README.md](README.md) for:
- Advanced configuration options
- Scaling strategies
- Security best practices
- Monitoring and debugging
- Migration from Filter version

## Support

- Issues: https://github.com/Skyzi000/open-webui-graphiti-memory/issues
- OpenWebUI: https://github.com/open-webui/open-webui
- Graphiti: https://github.com/getzep/graphiti

## Clean Up

To stop and remove everything:

```bash
# Stop services
docker-compose down

# Remove all data (WARNING: This deletes all memories!)
docker-compose down -v
```
