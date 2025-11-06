"""
title: Graphiti Memory Pipeline
author: Skyzi000
description: Temporal knowledge graph-based memory system using Graphiti as a standalone Pipeline. Automatically extracts entities, facts, and their relationships from conversations, stores them with timestamps in a graph database, and retrieves relevant context for future conversations. This Pipeline version runs as an independent service to reduce OpenWebUI server load.
author_url: https://github.com/Skyzi000
repository_url: https://github.com/Skyzi000/open-webui-graphiti-memory
version: 1.0
requirements: graphiti-core[falkordb], httpx

Design:
- Main class: Pipeline (not Filter)
- Standalone service: Runs independently from OpenWebUI server
- Acts as middleware/proxy between OpenWebUI and LLM
- Related components:
  - Graphiti: Knowledge graph memory system
  - FalkorDriver: FalkorDB backend driver for graph storage
  - OpenAIClient: OpenAI client with JSON structured output support
  - OpenAIGenericClient: Generic OpenAI-compatible client
  - OpenAIEmbedder: Embedding model for semantic search
  - OpenAIRerankerClient: Cross-encoder for result reranking

Architecture:
- Initialization: _initialize_graphiti() sets up the graph database connection
- LLM Client Selection: Configurable client type selection
  - OpenAI client: Better for some providers/models
  - Generic client: Better for others
  - Try both to see which works better for your setup
- Search Strategy: Three performance/quality tradeoffs
  - Fast: BM25 only (~100ms, no embedding calls)
  - Balanced: BM25 + Cosine Similarity (~500ms, 1 embedding call) - DEFAULT
  - Quality: + Cross-Encoder reranking (~1-5s, multiple LLM calls)
- Lazy initialization: _ensure_graphiti_initialized() provides automatic retry
- Memory search: Performed before forwarding request to LLM
- Memory storage: Performed after receiving response from LLM
- RAG context ingestion: Captures retrieval-augmented context returned by LLM so referenced material is persisted in memory alongside the conversation.
- Request forwarding: Proxies modified requests to configured LLM endpoint and returns streaming/non-streaming responses
"""

import asyncio
import contextvars
import hashlib
import json
import os
import re
import time
import traceback
from datetime import datetime
from typing import Optional, AsyncGenerator, Any
from urllib.parse import quote

from pydantic import BaseModel, Field
import httpx

from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.nodes import EpisodeType
from openai import AsyncOpenAI
from graphiti_core.search.search_config_recipes import (
    COMBINED_HYBRID_SEARCH_RRF,
    COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
)
from graphiti_core.search.search_config import (
    SearchConfig,
    EdgeSearchConfig,
    NodeSearchConfig,
    EpisodeSearchConfig,
    EdgeSearchMethod,
    NodeSearchMethod,
    EpisodeSearchMethod,
    EdgeReranker,
    NodeReranker,
    EpisodeReranker,
)
from graphiti_core.driver.falkordb_driver import FalkorDriver

# Context variable to store user-specific headers for each async request
# This ensures complete isolation between concurrent requests without locks
user_headers_context = contextvars.ContextVar('user_headers', default={})


class MultiUserOpenAIClient(OpenAIClient):
    """
    Custom OpenAI LLM client that retrieves user-specific headers from context variables.
    This allows a single Graphiti instance to safely handle concurrent requests from multiple users.
    """
    
    def __init__(self, config: LLMConfig | None = None, cache: bool = False, **kwargs):
        if config is None:
            config = LLMConfig()
        
        # Store base client for dynamic header injection
        self._base_client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        
        # Initialize parent with our base client and any additional kwargs
        super().__init__(config, cache, self._base_client, **kwargs)
    
    @property
    def client(self) -> AsyncOpenAI:
        """Dynamically return client with user-specific headers from context"""
        headers = user_headers_context.get()
        if headers:
            return self._base_client.with_options(default_headers=headers)
        return self._base_client
    
    @client.setter
    def client(self, value: AsyncOpenAI):
        """Store base client for future header injection"""
        self._base_client = value


class MultiUserOpenAIGenericClient(OpenAIGenericClient):
    """
    Custom OpenAI-compatible generic LLM client that retrieves user-specific headers from context variables.
    """
    
    def __init__(self, config: LLMConfig | None = None, cache: bool = False):
        if config is None:
            config = LLMConfig()
        
        # Store base client for dynamic header injection
        self._base_client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        
        # Initialize parent with our base client
        super().__init__(config, cache, self._base_client)
    
    @property
    def client(self) -> AsyncOpenAI:
        """Dynamically return client with user-specific headers from context"""
        headers = user_headers_context.get()
        if headers:
            return self._base_client.with_options(default_headers=headers)
        return self._base_client
    
    @client.setter
    def client(self, value: AsyncOpenAI):
        """Store base client for future header injection"""
        self._base_client = value


class MultiUserOpenAIEmbedder(OpenAIEmbedder):
    """
    Custom OpenAI embedder that retrieves user-specific headers from context variables.
    """
    
    def __init__(
        self,
        config: OpenAIEmbedderConfig | None = None,
        client: AsyncOpenAI | None = None,
    ):
        if config is None:
            config = OpenAIEmbedderConfig()
        
        # Store base client for dynamic header injection
        if client is not None:
            self._base_client = client
        else:
            self._base_client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        
        # Initialize parent with our base client
        super().__init__(config, self._base_client)
    
    @property
    def client(self) -> AsyncOpenAI:
        """Dynamically return client with user-specific headers from context"""
        headers = user_headers_context.get()
        if headers:
            return self._base_client.with_options(default_headers=headers)
        return self._base_client
    
    @client.setter
    def client(self, value: AsyncOpenAI):
        """Store base client for future header injection"""
        self._base_client = value


class Pipeline:
    """
    Open WebUI Pipeline for Graphiti-based memory management.
    
    This Pipeline version runs as a standalone service, reducing OpenWebUI server load.
    It acts as a middleware/proxy that:
    1. Intercepts requests from OpenWebUI
    2. Searches for relevant memories and injects them into the context
    3. Forwards the modified request to the actual LLM
    4. Processes the LLM response
    5. Extracts and stores new memories
    6. Returns the response to OpenWebUI
    
    Design References:
    - See module docstring for overall architecture
    - Graphiti documentation: https://github.com/getzep/graphiti-core
    
    Related Classes:
    - Valves: Configuration settings for the pipeline
    - UserValves: Per-user configuration settings
    
    Key Methods:
    - _initialize_graphiti(): Initialize the graph database connection
    - _ensure_graphiti_initialized(): Lazy initialization with retry logic
    - pipe(): Main method that handles request processing, memory injection, and LLM forwarding
    
    Flow:
    1. User sends message → pipe() is called with request body
    2. Search for relevant memories in graph database
    3. Inject found memories into conversation context
    4. Forward modified request to configured LLM endpoint
    5. Stream/return LLM response to user
    6. Extract and store new memories in graph database (async, non-blocking)
    """
    
    class Valves(BaseModel):
        # LLM Configuration (for memory processing)
        llm_client_type: str = Field(
            default="openai",
            description="Type of LLM client to use for memory processing: 'openai' for OpenAI client, 'generic' for OpenAI-compatible generic client. Try both to see which works better with your LLM provider.",
        )
        openai_api_url: str = Field(
            default="https://api.openai.com/v1",
            description="OpenAI compatible endpoint for memory processing",
        )
        model: str = Field(
            default="gpt-4o-mini",
            description="Model to use for memory processing.",
        )
        small_model: str = Field(
            default="gpt-4o-mini",
            description="Smaller model to use for memory processing in legacy mode.",
        )
        embedding_model: str = Field(
            default="text-embedding-3-small",
            description="Model to use for embedding memories.",
        )
        embedding_dim: int = Field(
            default=1536, description="Dimension of the embedding model."
        )
        api_key: str = Field(
            default="", description="API key for memory processing OpenAI compatible endpoint"
        )

        # Target LLM Configuration (where to forward requests)
        target_llm_url: str = Field(
            default="https://api.openai.com/v1",
            description="OpenAI compatible endpoint to forward chat requests to (the actual LLM that will generate responses)",
        )
        target_llm_api_key: str = Field(
            default="", description="API key for target LLM endpoint"
        )

        # Graph Database Configuration
        graph_db_backend: str = Field(
            default="neo4j",
            description="Graph database backend to use (e.g., 'neo4j', 'falkordb')",
        )

        neo4j_uri: str = Field(
            default="bolt://localhost:7687",
            description="Neo4j database connection URI",
        )
        neo4j_user: str = Field(
            default="neo4j",
            description="Neo4j database username",
        )
        neo4j_password: str = Field(
            default="password",
            description="Neo4j database password",
        )

        falkordb_host: str = Field(
            default="localhost",
            description="FalkorDB host address",
        )
        falkordb_port: int = Field(
            default=6379,
            description="FalkorDB port number",
        )
        falkordb_username: Optional[str] = Field(
            default=None,
            description="FalkorDB username (if applicable)",
        )
        falkordb_password: Optional[str] = Field(
            default=None,
            description="FalkorDB password (if applicable)",
        )

        # Graphiti Configuration
        graphiti_telemetry_enabled: bool = Field(
            default=False,
            description="Enable Graphiti telemetry",
        )
        
        update_communities: bool = Field(
            default=False,
            description="Update community detection when adding episodes using label propagation. EXPERIMENTAL: May cause errors with some Graphiti versions. Set to True to enable community updates.",
        )
        
        add_episode_timeout: int = Field(
            default=240,
            description="Timeout in seconds for adding episodes to memory. Set to 0 to disable timeout.",
        )
        
        semaphore_limit: int = Field(
            default=10,
            description="Maximum number of concurrent LLM operations in Graphiti. Default is 10 to prevent 429 rate limit errors. Increase for faster processing if your LLM provider allows higher throughput. Decrease if you encounter rate limit errors.",
        )
        
        max_search_message_length: int = Field(
            default=5000,
            description="Maximum length of user message to send to Graphiti search. Messages longer than this will be truncated (keeping first and last parts, dropping middle). Set to 0 to disable truncation. Note: This should be set to a size that the embedding model can handle to avoid errors.",
        )
        
        sanitize_search_query: bool = Field(
            default=True,
            description="Sanitize search queries to avoid FalkorDB/RediSearch syntax errors by removing special characters like @, :, \", (, ). Disable if you want to use raw queries or if using a different backend.",
        )
        
        search_strategy: str = Field(
            default="balanced",
            description="Search strategy: 'fast' (BM25 only, ~0.1s), 'balanced' (BM25+Cosine, ~0.5s), 'quality' (Cross-Encoder, ~1-5s)",
        )

        group_id_format: str = Field(
            default="{user_id}",
            description="Format string for group_id. Available placeholders: {user_id}, {user_email}, {user_name}. Email addresses are automatically sanitized (@ becomes _at_, . becomes _). Examples: '{user_id}', '{user_id}_chat', 'user_{user_id}'. Set to 'none' to disable group filtering (all users share the same memory space). Recommended: Use {user_id} (default) as it's stable; email/name changes could cause memory access issues.",
        )
        
        memory_message_role: str = Field(
            default="system",
            description="Role to use when injecting memory search results into the conversation. Options: 'system' (system message, more authoritative), 'user' (user message, more conversational). Default is 'system'.",
        )
        
        forward_user_info_headers: str = Field(
            default="default",
            description="Forward user information headers (User-Name, User-Id, User-Email, User-Role, Chat-Id) to OpenAI API for memory processing. Options: 'default' (follow environment variable ENABLE_FORWARD_USER_INFO_HEADERS, defaults to false if not set), 'true' (always forward), 'false' (never forward).",
        )
        
        use_user_name_in_episode: bool = Field(
            default=True,
            description="Use actual user name instead of 'User' label when saving conversations to memory. When enabled, episodes will be saved as '{user_name}: {message}' instead of 'User: {message}'.",
        )
        
        debug_print: bool = Field(
            default=False,
            description="Enable debug printing to console. When enabled, prints detailed information about search results, memory injection, and processing steps. Useful for troubleshooting.",
        )

    class UserValves(BaseModel):
        enabled: bool = Field(
            default=True,
            description="Enable or disable Graphiti Memory feature for this user. When disabled, no memory search or storage will be performed.",
        )
        show_status: bool = Field(
            default=True, description="Show status messages during memory operations."
        )
        save_user_message: bool = Field(
            default=True,
            description="Automatically save user messages as memories.",
        )
        save_assistant_response: bool = Field(
            default=False,
            description="Automatically save assistant responses (latest) as memories.",
        )
        
        save_previous_assistant_message: bool = Field(
            default=True,
            description="Save the assistant message that the user is responding to (the one before the latest user message).",
        )
        merge_retrieved_context: bool = Field(
            default=True,
            description="Merge RAG retrieved context (e.g., file attachments, knowledge base hits) into the user message before saving to memory.",
        )
        allowed_rag_source_types: str | None = Field(
            default="file,text",
            description="Comma-separated list of retrieval source types to merge (e.g., 'file, web_search'). Leave blank to disable merging.",
        )
        
        inject_facts: bool = Field(
            default=True,
            description="Inject facts (EntityEdge/relationships) from memory search results.",
        )
        
        inject_entities: bool = Field(
            default=True,
            description="Inject entities (EntityNode summaries) from memory search results.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.graphiti = None
        self._indices_built = False  # Track if indices have been built
        self._last_config = None  # Track configuration for change detection
        # Try to initialize, but it's okay if it fails - will retry later
        try:
            self._initialize_graphiti()
        except Exception as e:
            if self.valves.debug_print:
                print(f"Initial Graphiti initialization skipped (will retry on first use): {e}")
    
    def _get_config_hash(self) -> str:
        """
        Generate a hash of current configuration to detect changes.
        
        Returns:
            str: Hash of relevant configuration values
        """
        # Get all valve values as dict, excluding non-config fields
        valve_dict = self.valves.model_dump(
            exclude={
                'debug_print',  # Debugging settings don't affect initialization
                'group_id_format',  # Group ID format doesn't affect Graphiti init
                'search_strategy',  # Search strategy doesn't affect Graphiti init
                'save_assistant_response',  # Message saving behavior doesn't affect Graphiti init
                'inject_facts',  # Memory injection settings don't affect Graphiti init
                'inject_entities',  # Memory injection settings don't affect Graphiti init
                'update_communities',  # Community update setting doesn't affect Graphiti init
                'add_episode_timeout',  # Timeout settings don't affect Graphiti init
                'max_search_message_length',  # Message truncation doesn't affect Graphiti init
                'sanitize_search_query',  # Query sanitization doesn't affect Graphiti init
                'memory_message_role',  # Message role doesn't affect Graphiti init
                'forward_user_info_headers',  # Header forwarding doesn't affect Graphiti init
                'use_user_name_in_episode',  # Episode formatting doesn't affect Graphiti init
                'target_llm_url',  # Target LLM settings don't affect Graphiti init
                'target_llm_api_key',  # Target LLM settings don't affect Graphiti init
            }
        )
        # Sort keys for consistent hashing
        config_str = '|'.join(f"{k}={v}" for k, v in sorted(valve_dict.items()))
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _config_changed(self) -> bool:
        """
        Check if configuration has changed since last initialization.
        
        Returns:
            bool: True if configuration changed, False otherwise
        """
        current_hash = self._get_config_hash()
        if self._last_config != current_hash:
            if self._last_config is not None:
                if self.valves.debug_print:
                    print(f"Configuration change detected, will reinitialize Graphiti")
            return True
        return False
    
    def _get_user_info_headers(self, user: Optional[dict] = None, chat_id: Optional[str] = None) -> dict:
        """
        Build user information headers dictionary.
        
        Args:
            user: User dictionary containing 'id', 'email', 'name', 'role'
            chat_id: Current chat ID
            
        Returns:
            Dictionary of headers to send to OpenAI API
        """
        # Check Valves setting first
        valves_setting = self.valves.forward_user_info_headers.lower()
        
        if valves_setting == 'true':
            enable_forward = True
        elif valves_setting == 'false':
            enable_forward = False
        elif valves_setting == 'default':
            # Use environment variable (defaults to false if not set)
            env_setting = os.environ.get('ENABLE_FORWARD_USER_INFO_HEADERS', 'false').lower()
            enable_forward = env_setting == 'true'
        else:
            # Invalid value, default to false
            enable_forward = False
        
        if not enable_forward:
            return {}
        
        headers = {}
        if user:
            if user.get('name'):
                headers['X-OpenWebUI-User-Name'] = quote(str(user['name']), safe=" ")
            if user.get('id'):
                headers['X-OpenWebUI-User-Id'] = str(user['id'])
            if user.get('email'):
                headers['X-OpenWebUI-User-Email'] = str(user['email'])
            if user.get('role'):
                headers['X-OpenWebUI-User-Role'] = str(user['role'])
        
        if chat_id:
            headers['X-OpenWebUI-Chat-Id'] = str(chat_id)
        
        return headers
    
    def _initialize_graphiti(self) -> bool:
        """
        Initialize Graphiti instance with configured backend.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            os.environ['GRAPHITI_TELEMETRY_ENABLED'] = 'true' if self.valves.graphiti_telemetry_enabled else 'false'
            os.environ['SEMAPHORE_LIMIT'] = str(self.valves.semaphore_limit)
            
            # Configure LLM client
            llm_config = LLMConfig(
                api_key=self.valves.api_key,
                model=self.valves.model,
                small_model=self.valves.small_model,
                base_url=self.valves.openai_api_url,
            )

            # Select LLM client based on configuration - use multi-user versions
            if self.valves.llm_client_type.lower() == "openai":
                llm_client = MultiUserOpenAIClient(config=llm_config)
                if self.valves.debug_print:
                    print("Using Multi-User OpenAI client")
            elif self.valves.llm_client_type.lower() == "generic":
                llm_client = MultiUserOpenAIGenericClient(config=llm_config)
                if self.valves.debug_print:
                    print("Using Multi-User OpenAI-compatible generic client")
            else:
                # Default to OpenAI client for unknown values
                llm_client = MultiUserOpenAIClient(config=llm_config)
                if self.valves.debug_print:
                    print(f"Unknown client type '{self.valves.llm_client_type}', defaulting to Multi-User OpenAI client")

            falkor_driver = None
            if self.valves.graph_db_backend.lower() == "falkordb":
                falkor_driver = FalkorDriver(
                    host=self.valves.falkordb_host,
                    port=self.valves.falkordb_port,
                    username=self.valves.falkordb_username,
                    password=self.valves.falkordb_password,
                )

            # Initialize Graphiti
            if falkor_driver:
                self.graphiti = Graphiti(
                    graph_driver=falkor_driver,
                    llm_client=llm_client,
                    embedder=MultiUserOpenAIEmbedder(
                        config=OpenAIEmbedderConfig(
                            api_key=self.valves.api_key,
                            embedding_model=self.valves.embedding_model,
                            embedding_dim=self.valves.embedding_dim,
                            base_url=self.valves.openai_api_url,
                        )
                    ),
                    # OpenAIRerankerClient requires AsyncOpenAI client
                    # Use base_client from our custom multi-user client
                    cross_encoder=OpenAIRerankerClient(client=llm_client._base_client, config=llm_config),
                )
            elif self.valves.graph_db_backend.lower() == "neo4j":
                self.graphiti = Graphiti(
                    self.valves.neo4j_uri,
                    self.valves.neo4j_user,
                    self.valves.neo4j_password,
                    llm_client=llm_client,
                    embedder=MultiUserOpenAIEmbedder(
                        config=OpenAIEmbedderConfig(
                            api_key=self.valves.api_key,
                            embedding_model=self.valves.embedding_model,
                            embedding_dim=self.valves.embedding_dim,
                            base_url=self.valves.openai_api_url,
                        )
                    ),
                    # OpenAIRerankerClient requires AsyncOpenAI client
                    # Use base_client from our custom multi-user client
                    cross_encoder=OpenAIRerankerClient(client=llm_client._base_client, config=llm_config),
                )
            else:
                print(f"Unsupported graph database backend: {self.valves.graph_db_backend}. Supported backends are 'neo4j' and 'falkordb'.")
                return False
            
            # Save current configuration hash after successful initialization
            self._last_config = self._get_config_hash()
            if self.valves.debug_print:
                print("Graphiti initialized successfully.")
            return True
            
        except Exception as e:
            print(f"Graphiti initialization failed (will retry later if needed): {e}")
            # Only print traceback in debug scenarios
            # import traceback
            # traceback.print_exc()
            self.graphiti = None
            return False
    
    async def _build_indices(self) -> bool:
        """
        Build database indices and constraints for Graphiti.
        This should be called once after initialization and before the first query.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.graphiti is None:
            return False
            
        if self._indices_built:
            return True
            
        try:
            if self.valves.debug_print:
                print("Building Graphiti database indices and constraints...")
            await self.graphiti.build_indices_and_constraints()
            self._indices_built = True
            if self.valves.debug_print:
                print("Graphiti indices and constraints built successfully.")
            return True
        except Exception as e:
            print(f"Failed to build Graphiti indices: {e}")
            return False
    
    async def _ensure_graphiti_initialized(self) -> bool:
        """
        Ensure Graphiti is initialized and indices are built, attempting re-initialization if necessary.
        Automatically reinitializes if configuration changes are detected.
        
        Returns:
            bool: True if Graphiti is ready to use, False otherwise
        """
        # Check if configuration changed - if so, force reinitialization
        if self._config_changed():
            if self.valves.debug_print:
                print("Configuration changed, reinitializing Graphiti...")
            self.graphiti = None
            self._indices_built = False
        
        if self.graphiti is None:
            if self.valves.debug_print:
                print("Graphiti not initialized. Attempting to initialize...")
            if not self._initialize_graphiti():
                return False
        
        # Build indices if not already built
        if not self._indices_built:
            if not await self._build_indices():
                return False
        
        return True
    
    def _get_group_id(self, user: dict) -> Optional[str]:
        """
        Generate group_id for the user based on format string configuration.
        
        Args:
            user: User dictionary containing 'id' and optionally 'email', 'name'
            
        Returns:
            Sanitized group_id safe for Graphiti (alphanumeric, dashes, underscores only),
            or None if group_id_format is 'none' (to disable group filtering)
        """
        # Return None if format is 'none' (disable group filtering for shared memory space)
        if self.valves.group_id_format.lower().strip() == "none":
            return None
        
        # Prepare replacement values
        user_id = user.get('id', 'unknown')
        user_email = user.get('email', user_id)
        user_name = user.get('name', user_id)
        
        # Sanitize email to meet Graphiti's group_id requirements
        sanitized_email = user_email.replace('@', '_at_').replace('.', '_')
        
        # Sanitize name (replace spaces and special characters)
        sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '_', user_name)
        
        # Format the group_id using the template
        group_id = self.valves.group_id_format.format(
            user_id=user_id,
            user_email=sanitized_email,
            user_name=sanitized_name,
        )
        
        # Final sanitization to ensure only alphanumeric, dashes, underscores
        group_id = re.sub(r'[^a-zA-Z0-9_-]', '_', group_id)
        
        return group_id
    
    def _get_search_config(self):
        """
        Get search configuration based on the configured search strategy.
        
        Returns:
            SearchConfig: Configured search strategy
        """
        strategy = self.valves.search_strategy.lower()
        
        if strategy == "fast":
            # BM25 only - fastest, no embedding calls
            return SearchConfig(
                edge_config=EdgeSearchConfig(
                    search_methods=[EdgeSearchMethod.bm25],
                    reranker=EdgeReranker.rrf,
                ),
                node_config=NodeSearchConfig(
                    search_methods=[NodeSearchMethod.bm25],
                    reranker=NodeReranker.rrf,
                ),
                episode_config=EpisodeSearchConfig(
                    search_methods=[EpisodeSearchMethod.bm25],
                    reranker=EpisodeReranker.rrf,
                ),
            )
        elif strategy == "quality":
            # Cross-encoder - highest quality, slowest
            return COMBINED_HYBRID_SEARCH_CROSS_ENCODER
        else:
            # Default: balanced (BM25 + Cosine Similarity + RRF)
            # Best speed/quality tradeoff for most use cases
            return COMBINED_HYBRID_SEARCH_RRF
    
    def _get_content_from_message(self, message: dict) -> Optional[str]:
        """
        Extract text content from a message, handling both string and list formats.
        
        Open WebUI messages can have content as:
        - Simple string: "Hello"
        - List with text and images: [{"type": "text", "text": "Hello"}, {"type": "image_url", ...}]
        
        Args:
            message: Message dictionary with 'content' field
            
        Returns:
            Extracted text content, or None if no text content found
        """
        content = message.get("content")
        
        # Handle list format (with images)
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    return item.get("text", "")
            return ""  # No text found in list
        
        # Handle string format
        return content if isinstance(content, str) else ""
    
    def _extract_rag_sources_text(
        self,
        message: Optional[dict],
        allowed_types: Optional[set[str]] = None,
    ) -> str:
        """
        Build a readable text block from RAG retrieval sources attached to a message.
        """
        if not message:
            return ""
        
        sources = (
            message.get("sources")
            or message.get("citations")
        )
        if not sources:
            return ""

        sections: list[str] = []
        for idx, source in enumerate(sources, 1):
            source_info = source.get("source") or {}
            source_type = str(source_info.get("type", "")).lower().strip()
            if allowed_types is not None:
                if source_type == "":
                    if "" not in allowed_types:
                        continue
                elif source_type not in allowed_types:
                    continue
            base_label = (
                source_info.get("name")
                or source_info.get("id")
                or f"Source {idx}"
            )
            documents = source.get("document") or []
            metadatas = source.get("metadata") or []
            
            for doc_index, document_text in enumerate(documents, 1):
                if not isinstance(document_text, str):
                    continue
                
                metadata = (
                    metadatas[doc_index - 1]
                    if doc_index - 1 < len(metadatas)
                    else {}
                )
                title = (
                    metadata.get("name")
                    or metadata.get("source")
                    or metadata.get("file_id")
                    or f"{base_label}#{doc_index}"
                )
                
                heading = f"[{base_label} #{doc_index}] {title}".strip()
                sections.append(f"{heading}\n{document_text.strip()}")
        
        return "\n\n".join(section for section in sections if section.strip())
    
    @staticmethod
    def _parse_allowed_source_types(value: Any) -> Optional[set[str]]:
        """
        Normalize user-configured RAG source type filters to a lowercase set.
        Returns:
            set[str]: Allowed types; empty set disables merging; None permits all.
        """
        if value is None:
            return None
        
        if isinstance(value, (bytes, bytearray)):
            value = value.decode("utf-8", errors="ignore")

        if isinstance(value, str):
            candidates = value.split(",")
        elif isinstance(value, list):
            # Backward compatibility for older list-based valves
            candidates = value
        else:
            candidates = [value]

        normalized: set[str] = set()
        for item in candidates:
            if isinstance(item, (bytes, bytearray)):
                item = item.decode("utf-8", errors="ignore")
            elif not isinstance(item, str):
                item = str(item)

            item = item.strip()
            if item:
                normalized.add(item.lower())

        return normalized
    
    def _sanitize_search_query(self, query: str) -> str:
        """
        Sanitize search query to avoid FalkorDB/RediSearch syntax errors.
        
        Only removes the most problematic characters that cause RediSearch errors.
        Keeps most punctuation to preserve query meaning.
        
        Args:
            query: The original search query
            
        Returns:
            Sanitized query safe for FalkorDB search
        """
        # Only remove the most problematic RediSearch operators:
        # ( ) - parentheses cause syntax errors with AND operator
        # @ - field selector
        # : - field separator  
        # " - quote operator
        # Keep: !, ?, ., ,, and other common punctuation
        sanitized = re.sub(r'[@:"()]', ' ', query)
        
        # Replace multiple spaces with single space
        sanitized = re.sub(r'\s+', ' ', sanitized)
        
        # Trim whitespace
        sanitized = sanitized.strip()
        
        return sanitized

    async def _search_and_inject_memories(
        self,
        body: dict,
        user: Optional[dict] = None,
        user_valves: Optional[UserValves] = None,
    ) -> tuple[dict, Optional[dict]]:
        """
        Search for relevant memories and inject them into the message context.
        
        Args:
            body: Request body containing messages
            user: User information
            user_valves: User-specific settings
            
        Returns:
            Tuple of (modified_body, search_results_info) where search_results_info contains
            metadata about the search (for status reporting)
        """
        if user_valves is None:
            user_valves = self.UserValves()
        
        # Check if this is a "Continue Response" action
        messages = body.get("messages", [])
        if messages and messages[-1].get("role") == "assistant":
            if self.valves.debug_print:
                print("Detected 'Continue Response' action (last message is assistant). Skipping memory search.")
            return body, None
        
        # Find the last user message
        user_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = self._get_content_from_message(msg)
                break
        
        if not user_message:
            if self.valves.debug_print:
                print("No user message found. Skipping memory search.")
            return body, None
        
        # Sanitize query for FalkorDB/RediSearch compatibility
        sanitized_query = user_message
        if self.valves.sanitize_search_query:
            sanitized_query = self._sanitize_search_query(user_message)
            if not sanitized_query:
                if self.valves.debug_print:
                    print("Search query is empty after sanitization. Skipping memory search.")
                return body, None
        
        # Truncate message if too long
        original_length = len(sanitized_query)
        max_length = self.valves.max_search_message_length
        if max_length > 0 and len(sanitized_query) > max_length:
            keep_length = max_length // 2 - 25
            sanitized_query = (
                sanitized_query[:keep_length] 
                + "\n\n[...]\n\n" 
                + sanitized_query[-keep_length:]
            )
            if self.valves.debug_print:
                print(f"User message truncated from {original_length} to {len(sanitized_query)} characters")
        
        # Get search configuration
        search_config = self._get_search_config()
        
        if self.valves.debug_print:
            print(f"Using search strategy: {self.valves.search_strategy}")
        
        # Generate group_id
        group_id = self._get_group_id(user) if user else None
        
        # Perform search
        search_start_time = time.time()
        
        try:
            if group_id is not None:
                results = await self.graphiti.search_(
                    query=sanitized_query,
                    group_ids=[group_id],
                    config=search_config,
                )
            else:
                results = await self.graphiti.search_(
                    query=sanitized_query,
                    config=search_config,
                )
            
            search_duration = time.time() - search_start_time
            
            if self.valves.debug_print:
                print(f"Search completed in {search_duration:.2f}s")
        except Exception as e:
            search_duration = time.time() - search_start_time
            error_msg = str(e)
            print(f"Error during Graphiti search (after {search_duration:.2f}s): {e}")
            return body, {"error": error_msg, "duration": search_duration}
        
        # Check if any results were found
        if len(results.edges) == 0 and len(results.nodes) == 0:
            return body, {"found": False, "duration": search_duration}
        
        # Process results
        facts = []
        entities = {}
        
        # Process edges (facts) if enabled
        if user_valves.inject_facts:
            for result in results.edges:
                if self.valves.debug_print:
                    print(f'Fact({result.name}): {result.fact}')
                facts.append((result.fact, result.valid_at, result.invalid_at, result.name))
        
        # Process nodes (entities) if enabled
        if user_valves.inject_entities:
            for result in results.nodes:
                if self.valves.debug_print:
                    print(f'Entity({result.name}): {result.summary}')
                if result.name and result.summary:
                    entities[result.name] = result.summary
        
        # Inject memory message if we have facts OR entities
        if len(facts) > 0 or len(entities) > 0:
            # Find the index of the last user message
            last_user_msg_index = None
            for i in range(len(body['messages']) - 1, -1, -1):
                if body['messages'][i].get("role") == "user":
                    last_user_msg_index = i
                    break
            
            # Determine the role to use for memory message
            memory_role = self.valves.memory_message_role.lower()
            if memory_role not in ["system", "user"]:
                if self.valves.debug_print:
                    print(f"Invalid memory_message_role '{memory_role}', using 'system'")
                memory_role = "system"
            
            # Format memory content
            memory_content = "FACTS and ENTITIES represent relevant context to the current conversation.  \n"
            
            if len(facts) > 0:
                memory_content += "# These are the most relevant facts and their valid date ranges  \n"
                memory_content += "# format: FACT (Date range: from - to)  \n"
                memory_content += "<FACTS>  \n"
                
                for fact, valid_at, invalid_at, name in facts:
                    valid_str = str(valid_at) if valid_at else "unknown"
                    invalid_str = str(invalid_at) if invalid_at else "present"
                    memory_content += f"  - {fact} ({valid_str} - {invalid_str})  \n"
                
                memory_content += "</FACTS>"
            
            if len(entities) > 0:
                if len(facts) > 0:
                    memory_content += "  \n\n"
                memory_content += "# These are the most relevant entities  \n"
                memory_content += "# ENTITY_NAME: entity summary  \n"
                memory_content += "<ENTITIES>  \n"
                
                for entity_name, entity_summary in entities.items():
                    memory_content += f"  - {entity_name}: {entity_summary}  \n"
                
                memory_content += "</ENTITIES>"
            
            # Insert memory before the last user message
            memory_message = {
                "role": memory_role,
                "content": memory_content
            }
            
            if last_user_msg_index is not None:
                body['messages'].insert(last_user_msg_index, memory_message)
            else:
                body['messages'].append(memory_message)
            
            return body, {
                "found": True,
                "facts_count": len(facts),
                "entities_count": len(entities),
                "duration": search_duration,
                "facts": facts,
                "entities": entities,
            }
        
        return body, {"found": False, "duration": search_duration}

    async def _store_memories(
        self,
        body: dict,
        user: Optional[dict] = None,
        user_valves: Optional[UserValves] = None,
        metadata: Optional[dict] = None,
    ) -> Optional[dict]:
        """
        Extract and store memories from the conversation.
        This runs asynchronously and does not block the response.
        
        Args:
            body: Request/response body containing messages
            user: User information
            user_valves: User-specific settings
            metadata: Metadata containing chat_id, message_id, etc.
            
        Returns:
            Dictionary with storage results or None if nothing to store
        """
        if user_valves is None:
            user_valves = self.UserValves()
        
        messages = body.get("messages", [])
        if len(messages) == 0:
            return None
        
        # Determine which messages to save
        messages_to_save = []
        
        last_user_message = None
        last_assistant_message = None
        previous_assistant_message = None
        
        for msg in reversed(messages):
            if msg.get("role") == "user" and last_user_message is None:
                last_user_message = msg
            elif msg.get("role") == "assistant":
                if last_user_message is None:
                    if last_assistant_message is None:
                        last_assistant_message = msg
                else:
                    if previous_assistant_message is None:
                        previous_assistant_message = msg
                        break
        
        # Build messages_to_save list
        if user_valves.save_previous_assistant_message and previous_assistant_message:
            previous_assistant_content = self._get_content_from_message(previous_assistant_message)
            if previous_assistant_content:
                messages_to_save.append(("previous_assistant", previous_assistant_content))
        
        user_content_block = ""
        if user_valves.save_user_message and last_user_message:
            user_content_block = self._get_content_from_message(last_user_message) or ""
            if user_valves.merge_retrieved_context:
                allowed_types_set = self._parse_allowed_source_types(
                    getattr(user_valves, "allowed_rag_source_types", None)
                )
                rag_context_block = self._extract_rag_sources_text(
                    last_assistant_message,
                    allowed_types_set,
                )
                if rag_context_block:
                    if user_content_block.strip():
                        user_content_block = (
                            f"{user_content_block.strip()}\n\nRetrieved Context:\n{rag_context_block}"
                        )
                    else:
                        user_content_block = f"Retrieved Context:\n{rag_context_block}"
        
        if user_valves.save_user_message and last_user_message:
            if user_content_block:
                messages_to_save.append(("user", user_content_block))
        
        if user_valves.save_assistant_response and last_assistant_message:
            assistant_content = self._get_content_from_message(last_assistant_message)
            if assistant_content:
                messages_to_save.append(("assistant", assistant_content))
        
        if len(messages_to_save) == 0:
            return None
        
        # Sort messages chronologically
        role_order = {"previous_assistant": 0, "user": 1, "assistant": 2}
        messages_to_save.sort(key=lambda x: role_order.get(x[0], 99))
        
        episode_parts = []
        for role, content in messages_to_save:
            if role == "user":
                if self.valves.use_user_name_in_episode and user and user.get('name'):
                    role_label = user['name']
                else:
                    role_label = "User"
            elif role in ("assistant", "previous_assistant"):
                role_label = "Assistant"
            else:
                role_label = role.capitalize()
            episode_parts.append(f"{role_label}: {content}")
        
        episode_body = "\n".join(episode_parts)
        
        # Generate group_id
        group_id = self._get_group_id(user) if user else None
        
        # Get chat_id and message_id from metadata
        chat_id = metadata.get('chat_id', 'unknown') if metadata else 'unknown'
        message_id = metadata.get('message_id', 'unknown') if metadata else 'unknown'
        
        try:
            # Add episode to Graphiti
            if self.valves.add_episode_timeout > 0:
                if group_id is not None:
                    add_results = await asyncio.wait_for(
                        self.graphiti.add_episode(
                            name=f"Chat_Interaction_{chat_id}_{message_id}",
                            episode_body=episode_body,
                            source=EpisodeType.message,
                            source_description="Chat conversation",
                            reference_time=datetime.now(),
                            group_id=group_id,
                            update_communities=self.valves.update_communities,
                        ),
                        timeout=self.valves.add_episode_timeout
                    )
                else:
                    add_results = await asyncio.wait_for(
                        self.graphiti.add_episode(
                            name=f"Chat_Interaction_{chat_id}_{message_id}",
                            episode_body=episode_body,
                            source=EpisodeType.message,
                            source_description="Chat conversation",
                            reference_time=datetime.now(),
                            update_communities=self.valves.update_communities,
                        ),
                        timeout=self.valves.add_episode_timeout
                    )
            else:
                if group_id is not None:
                    add_results = await self.graphiti.add_episode(
                        name=f"Chat_Interaction_{chat_id}_{message_id}",
                        episode_body=episode_body,
                        source=EpisodeType.message,
                        source_description="Chat conversation",
                        reference_time=datetime.now(),
                        group_id=group_id,
                        update_communities=self.valves.update_communities,
                    )
                else:
                    add_results = await self.graphiti.add_episode(
                        name=f"Chat_Interaction_{chat_id}_{message_id}",
                        episode_body=episode_body,
                        source=EpisodeType.message,
                        source_description="Chat conversation",
                        reference_time=datetime.now(),
                        update_communities=self.valves.update_communities,
                    )
            
            if self.valves.debug_print:
                print(f"Added conversation to Graphiti memory: {episode_body[:100]}...")
                if add_results:
                    print(f"Extracted {len(add_results.nodes)} entities and {len(add_results.edges)} relationships")
            
            return {
                "success": True,
                "nodes_count": len(add_results.nodes) if add_results else 0,
                "edges_count": len(add_results.edges) if add_results else 0,
                "results": add_results,
            }
        
        except asyncio.TimeoutError:
            print(f"Timeout adding conversation to Graphiti memory after {self.valves.add_episode_timeout}s")
            return {"success": False, "error": "timeout"}
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            print(f"Graphiti error adding conversation: {e}")
            if "ValidationError" not in error_type:
                traceback.print_exc()
            return {"success": False, "error": error_type, "message": error_msg}

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Any] = None,
        __metadata__: Optional[dict] = None,
    ) -> AsyncGenerator[str, None] | dict:
        """
        Main pipeline method that:
        1. Searches for relevant memories and injects them
        2. Forwards the request to the target LLM
        3. Streams the response back to OpenWebUI
        4. Asynchronously stores new memories (non-blocking)
        
        Args:
            body: Request body from OpenWebUI
            __user__: User information
            __event_emitter__: Event emitter for status updates
            __metadata__: Metadata (chat_id, message_id, etc.)
            
        Returns:
            AsyncGenerator yielding response chunks, or dict for non-streaming responses
        """
        if self.valves.debug_print:
            print(f"Pipeline pipe() called for user: {__user__}")
        
        # Get user valves
        user_valves = self.UserValves()
        if __user__:
            user_valves = __user__.get("valves", self.UserValves())
            if not user_valves.enabled:
                if self.valves.debug_print:
                    print("Graphiti Memory feature is disabled for this user.")
                # Just forward to LLM without memory processing
                return await self._forward_to_llm(body)
        
        # Ensure Graphiti is initialized
        if not await self._ensure_graphiti_initialized() or self.graphiti is None:
            if self.valves.debug_print:
                print("Graphiti initialization failed. Forwarding without memory processing.")
            if __event_emitter__ and user_valves.show_status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Memory service unavailable", "done": True},
                    }
                )
            return await self._forward_to_llm(body)
        
        # Set user headers in context
        chat_id = __metadata__.get('chat_id') if __metadata__ else None
        headers = self._get_user_info_headers(__user__, chat_id)
        if headers:
            user_headers_context.set(headers)
            if self.valves.debug_print:
                print(f"Set user headers in context: {list(headers.keys())}")
        
        # Search for memories and inject them
        if __event_emitter__ and user_valves.show_status:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "🔍 Searching Graphiti memory...", "done": False},
                }
            )
        
        modified_body, search_info = await self._search_and_inject_memories(
            body, __user__, user_valves
        )
        
        # Report search results
        if __event_emitter__ and user_valves.show_status and search_info:
            if search_info.get("error"):
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"Memory search failed", "done": True},
                    }
                )
            elif search_info.get("found"):
                status_parts = []
                if search_info.get("facts_count", 0) > 0:
                    status_parts.append(f"{search_info['facts_count']} fact{'s' if search_info['facts_count'] != 1 else ''}")
                if search_info.get("entities_count", 0) > 0:
                    status_parts.append(f"{search_info['entities_count']} entit{'ies' if search_info['entities_count'] != 1 else 'y'}")
                
                if status_parts:
                    status_msg = "🧠 " + " and ".join(status_parts) + f" found ({search_info['duration']:.2f}s)"
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": status_msg, "done": True},
                        }
                    )
            else:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"No relevant memories found ({search_info.get('duration', 0):.2f}s)", "done": True},
                    }
                )
        
        # Forward to LLM and get response
        # We need to handle both streaming and non-streaming responses
        is_streaming = body.get("stream", False)
        
        if is_streaming:
            # For streaming: yield chunks as they come, then store memories after
            async for chunk in self._forward_to_llm_streaming(modified_body):
                yield chunk
            
            # After streaming is complete, store memories asynchronously
            if __event_emitter__ and user_valves.show_status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "✍️ Storing memories...", "done": False},
                    }
                )
            
            # Store memories (this will not block the response)
            storage_result = await self._store_memories(
                modified_body, __user__, user_valves, __metadata__
            )
            
            if __event_emitter__ and user_valves.show_status and storage_result:
                if storage_result.get("success"):
                    detail_parts = []
                    if storage_result.get("nodes_count", 0) > 0:
                        detail_parts.append(f"{storage_result['nodes_count']} entit{'ies' if storage_result['nodes_count'] != 1 else 'y'}")
                    if storage_result.get("edges_count", 0) > 0:
                        detail_parts.append(f"{storage_result['edges_count']} relation{'s' if storage_result['edges_count'] != 1 else ''}")
                    
                    if detail_parts:
                        status_msg = "✅ Memories stored - " + " and ".join(detail_parts)
                    else:
                        status_msg = "✅ Memories stored"
                    
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": status_msg, "done": True},
                        }
                    )
                else:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": "⚠️ Failed to store memories", "done": True},
                        }
                    )
        else:
            # For non-streaming: get full response, store memories, then return
            response = await self._forward_to_llm(modified_body)
            
            # Store memories
            if __event_emitter__ and user_valves.show_status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "✍️ Storing memories...", "done": False},
                    }
                )
            
            storage_result = await self._store_memories(
                modified_body, __user__, user_valves, __metadata__
            )
            
            if __event_emitter__ and user_valves.show_status and storage_result:
                if storage_result.get("success"):
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": "✅ Memories stored", "done": True},
                        }
                    )
                else:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": "⚠️ Failed to store memories", "done": True},
                        }
                    )
            
            return response

    async def _forward_to_llm(self, body: dict) -> dict:
        """
        Forward request to target LLM and return the full response.
        
        Args:
            body: Request body to forward
            
        Returns:
            Response from LLM
        """
        url = f"{self.valves.target_llm_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.valves.target_llm_api_key}",
            "Content-Type": "application/json",
        }
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(url, json=body, headers=headers)
            response.raise_for_status()
            return response.json()

    async def _forward_to_llm_streaming(self, body: dict) -> AsyncGenerator[str, None]:
        """
        Forward request to target LLM and stream the response.
        
        Args:
            body: Request body to forward
            
        Yields:
            Response chunks from LLM in SSE format
        """
        # Ensure streaming is enabled
        body["stream"] = True
        
        url = f"{self.valves.target_llm_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.valves.target_llm_api_key}",
            "Content-Type": "application/json",
        }
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("POST", url, json=body, headers=headers) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.strip():
                        # Forward SSE lines as-is
                        yield f"{line}\n"
