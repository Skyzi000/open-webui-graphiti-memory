"""
title: Graphiti Memory
author: Skyzi000
description: Temporal knowledge graph-based memory system using Graphiti. Automatically extracts entities, facts, and their relationships from conversations, stores them with timestamps in a graph database, and retrieves relevant context for future conversations.
author_url: https://github.com/Skyzi000
repository_url: https://github.com/Skyzi000/open-webui-graphiti-memory
version: 0.10.4
requirements: graphiti-core[falkordb]

Design:
- Main class: Filter
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
- Memory search: inlet() retrieves relevant memories before chat processing
- Memory storage: outlet() stores new information after chat completion
- RAG context ingestion: outlet() captures retrieval-augmented context returned by Open WebUI so referenced material is persisted in memory alongside the conversation.
"""

import asyncio
import contextvars
import hashlib
import html
import json
import os
import re
import time
import traceback
from collections import defaultdict
from datetime import datetime
from typing import Optional, Callable, Awaitable, Any
from urllib.parse import quote

from pydantic import BaseModel, Field

from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.nodes import EpisodeType, EntityNode
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

class Filter:
    """
    Open WebUI Filter for Graphiti-based memory management.
    
    Design References:
    - See module docstring for overall architecture
    - Graphiti documentation: https://github.com/getzep/graphiti-core
    
    Related Classes:
    - Valves: Configuration settings for the filter
    - UserValves: Per-user configuration settings
    
    Key Methods:
    - _initialize_graphiti(): Initialize the graph database connection
    - _ensure_graphiti_initialized(): Lazy initialization with retry logic
    - inlet(): Pre-process messages, inject relevant memories
    - outlet(): Post-process messages, store new memories
    
    Flow:
    1. User sends message → inlet() is called
    2. Search for relevant memories in graph database
    3. Inject found memories into conversation context
    4. LLM processes message with memory context
    5. outlet() is called with LLM response
    6. Extract and store new memories in graph database
    """
    class Valves(BaseModel):
        llm_client_type: str = Field(
            default="openai",
            description="Type of LLM client to use: 'openai' for OpenAI client, 'generic' for OpenAI-compatible generic client. Try both to see which works better with your LLM provider.",
        )
        openai_api_url: str = Field(
            default="https://api.openai.com/v1",
            description="openai compatible endpoint",
        )
        model: str = Field(
            default="gpt-5-mini",
            description="Model to use for memory processing.",
        )
        small_model: str = Field(
            default="gpt-5-nano",
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
            default="", description="API key for OpenAI compatible endpoint"
        )

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
            description="Forward user information headers (User-Name, User-Id, User-Email, User-Role, Chat-Id) to OpenAI API. Options: 'default' (follow environment variable ENABLE_FORWARD_USER_INFO_HEADERS, defaults to false if not set), 'true' (always forward), 'false' (never forward).",
        )
        
        use_user_name_in_episode: bool = Field(
            default=True,
            description="Use actual user name instead of 'User' label when saving conversations to memory. When enabled, episodes will be saved as '{user_name}: {message}' instead of 'User: {message}'.",
        )
        
        debug_print: bool = Field(
            default=False,
            description="Enable debug printing to console. When enabled, prints detailed information about search results, memory injection, and processing steps. Useful for troubleshooting.",
        )
        citation_source_id: str = Field(
            default="graphiti-memory",
            description="Source ID used for emitted citation events (controls grouping in the UI).",
        )
        citation_source_name: str = Field(
            default="Graphiti Memory",
            description="Source display name used for emitted citation events.",
        )

    class UserValves(BaseModel):
        enabled: bool = Field(
            default=True,
            description="Enable or disable Graphiti Memory feature for this user. When disabled, no memory search or storage will be performed.",
        )
        show_status: bool = Field(
            default=True, description="Show status of the action."
        )
        show_citation: bool = Field(
            default=True,
            description="Emit retrieval results as citation events (affects fact/entity previews).",
        )
        rich_html_citations: bool = Field(
            default=True,
            description="Render citation results as rich HTML with interactive graph visualizations. When disabled, uses plain text format.",
        )
        show_citation_parameters: bool = Field(
            default=False,
            description="Include detailed metadata parameters (type, index, valid_from, etc.) in citation events.",
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
        search_history_turns: int = Field(
            default=1,
            description="How many of the most recent user/assistant messages to include in the Graphiti search query. 1 uses only the latest user message, 2 also appends the assistant reply before it, 3 adds the previous user turn, and so on. Values <= 0 include as many alternating user/assistant messages as possible, still respecting the max_search_message_length limit. Messages from other roles are ignored.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.graphiti = None
        self._indices_built = False  # Track if indices have been built
        self._last_config = None  # Track configuration for change detection
        # Defer Graphiti initialization until inlet()/outlet() needs it
        if self.valves.debug_print:
            print("Graphiti initialization deferred until first use")
    
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
    
    def _build_search_history_query(
        self,
        messages: list[dict],
        history_turns: int,
        max_chars: int,
    ) -> Optional[str]:
        """
        Build a chronological snippet of the most recent conversation turns for Graphiti search.
        
        Args:
            messages: Full conversation message list.
            history_turns: Number of user/assistant messages to include. Values <= 0 include all.
            max_chars: Character budget used to stop collecting additional turns (<=0 disables).
        
        Returns:
            String formatted as alternating "User：..." / "Assistant：..." blocks, or None if no user turn found.
        """
        if not messages:
            return None
        
        turn_limit = history_turns if history_turns and history_turns > 0 else None
        char_limit = max_chars if isinstance(max_chars, int) and max_chars > 0 else None
        separator = "\n\n"
        collected: list[str] = []
        total_chars = 0
        turns_collected = 0
        found_latest_user = False
        
        for idx in range(len(messages) - 1, -1, -1):
            message = messages[idx]
            role = message.get("role")
            
            if not found_latest_user:
                if role != "user":
                    continue
                found_latest_user = True
                is_latest_user = True
            else:
                if role not in {"user", "assistant"}:
                    continue
                is_latest_user = False
            
            text = self._get_content_from_message(message)
            if text is None or len(text) == 0:
                if is_latest_user:
                    # Latest user message has no text, treat as missing to match previous behavior
                    return None
                continue
            
            role_label = "User" if role == "user" else "Assistant"
            segment = f"{role_label}：{text}"
            
            addition = len(segment) if not collected else len(separator) + len(segment)
            collected.append(segment)
            turns_collected += 1
            total_chars += addition
            
            if turn_limit is not None and turns_collected >= turn_limit:
                break
            
            if char_limit is not None and total_chars >= char_limit:
                break
        
        if not collected:
            return None
        
        return separator.join(reversed(collected))
    
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

    def _build_fact_citation_event(
        self,
        result: Any,
        idx: int,
        total: int,
        entity_lookup: dict[str, dict[str, str]],
        use_rich_html: bool = True,
        include_parameters: bool = False,
    ) -> Optional[dict]:
        """Convert a Graphiti fact result into a citation payload for Open WebUI."""
        fact_text = getattr(result, "fact", None)
        if not fact_text:
            return None

        source_id = self.valves.citation_source_id or "graphiti-memory"
        source_name = self.valves.citation_source_name or "Graphiti Memory"

        metadata: dict[str, Any] = {"source": source_id}
        valid_from = getattr(result, "valid_at", None)
        if valid_from:
            metadata["valid_from"] = str(valid_from)
        valid_until = getattr(result, "invalid_at", None)
        if valid_until:
            metadata["valid_until"] = str(valid_until)

        if include_parameters:
            parameters = self._build_fact_parameters(
                index=idx,
                total=total,
                uuid=getattr(result, "uuid", None),
                group_id=getattr(result, "group_id", None),
                name=getattr(result, "name", None),
                source_node_uuid=getattr(result, "source_node_uuid", None),
                target_node_uuid=getattr(result, "target_node_uuid", None),
                created_at=getattr(result, "created_at", None),
                valid_at=valid_from,
                invalid_at=valid_until,
                expired_at=getattr(result, "expired_at", None),
                episodes=getattr(result, "episodes", None),
                attributes=getattr(result, "attributes", None),
            )
            if parameters:
                metadata["parameters"] = parameters

        if use_rich_html:
            graph_html = self._render_fact_graph_html(
                fact_text=fact_text,
                relation_name=getattr(result, "name", None),
                source_entity=self._get_entity_display(entity_lookup, getattr(result, "source_node_uuid", None)),
                target_entity=self._get_entity_display(entity_lookup, getattr(result, "target_node_uuid", None)),
                valid_from=valid_from,
                valid_until=valid_until,
                idx=idx,
                total=total,
            )
            metadata["html"] = graph_html
            document_content = graph_html
        else:
            emoji = "🔚" if valid_until else "🔛"
            document_content = f"{emoji} Fact {idx}/{total}: {fact_text}"

        return {
            "source": {
                "id": source_id,
                "name": source_name,
                "type": "graphiti_memory",
            },
            "document": [document_content],
            "metadata": [metadata],
        }

    def _build_entity_citation_event(
        self,
        result: Any,
        idx: int,
        total: int,
        entity_lookup: dict[str, dict[str, str]],
        entity_connections: dict[str, list[dict[str, Any]]],
        use_rich_html: bool = True,
        include_parameters: bool = False,
    ) -> Optional[dict]:
        """Convert a Graphiti entity result into a citation payload for Open WebUI."""
        name = getattr(result, "name", None)
        summary = getattr(result, "summary", None)
        if not name or not summary:
            return None

        source_id = self.valves.citation_source_id or "graphiti-memory"
        metadata: dict[str, Any] = {"source": source_id}

        if include_parameters:
            parameters = self._build_entity_parameters(
                index=idx,
                total=total,
                uuid=getattr(result, "uuid", None),
                group_id=getattr(result, "group_id", None),
                name=name,
                labels=getattr(result, "labels", None),
                created_at=getattr(result, "created_at", None),
                attributes=getattr(result, "attributes", None),
            )
            if parameters:
                metadata["parameters"] = parameters

        if use_rich_html:
            entity_uuid = getattr(result, "uuid", None)
            entity_details = self._get_entity_display(entity_lookup, entity_uuid)
            if not entity_details.get("summary"):
                entity_details = {
                    **entity_details,
                    "summary": summary,
                }
            connections = entity_connections.get(str(entity_uuid) if entity_uuid else "", [])

            graph_html = self._render_entity_graph_html(
                entity=entity_details,
                connections=connections,
                idx=idx,
                total=total,
            )
            metadata["html"] = graph_html
            document_content = graph_html
        else:
            document_content = f"👤 Entity {idx}/{total}: {name} - {summary}"

        return {
            "source": {
                "id": source_id,
                "name": self.valves.citation_source_name or "Graphiti Memory",
                "type": "graphiti_memory",
            },
            "document": [document_content],
            "metadata": [metadata],
        }

    @staticmethod
    def _build_fact_parameters(
        *,
        index: Optional[int] = None,
        total: Optional[int] = None,
        uuid: Optional[str] = None,
        group_id: Optional[str] = None,
        name: Optional[str] = None,
        source_node_uuid: Optional[str] = None,
        target_node_uuid: Optional[str] = None,
        created_at: Optional[Any] = None,
        valid_at: Optional[Any] = None,
        invalid_at: Optional[Any] = None,
        expired_at: Optional[Any] = None,
        episodes: Optional[list[str]] = None,
        attributes: Optional[dict[str, Any]] = None,
    ) -> Optional[dict]:
        """Pack Graphiti EntityEdge fields into metadata.parameters."""
        payload: dict[str, Any] = {"type": "fact"}
        if isinstance(index, int):
            payload["index"] = index
        if isinstance(total, int):
            payload["total"] = total
        if uuid:
            payload["uuid"] = str(uuid)
        if group_id:
            payload["group_id"] = str(group_id)
        if name:
            payload["name"] = name
        if source_node_uuid:
            payload["source_node_uuid"] = str(source_node_uuid)
        if target_node_uuid:
            payload["target_node_uuid"] = str(target_node_uuid)
        if created_at:
            payload["created_at"] = str(created_at)
        if valid_at:
            payload["valid_at"] = str(valid_at)
        if invalid_at:
            payload["invalid_at"] = str(invalid_at)
        if expired_at:
            payload["expired_at"] = str(expired_at)
        if episodes:
            payload["episodes"] = episodes
        if attributes:
            payload["attributes"] = attributes
        return {"graphiti": payload} if len(payload) > 1 else None

    @staticmethod
    def _build_entity_parameters(
        *,
        index: Optional[int] = None,
        total: Optional[int] = None,
        uuid: Optional[str] = None,
        group_id: Optional[str] = None,
        name: Optional[str] = None,
        labels: Optional[list[str]] = None,
        created_at: Optional[Any] = None,
        attributes: Optional[dict[str, Any]] = None,
    ) -> Optional[dict]:
        """Pack Graphiti EntityNode fields into metadata.parameters."""
        payload: dict[str, Any] = {"type": "entity"}
        if isinstance(index, int):
            payload["index"] = index
        if isinstance(total, int):
            payload["total"] = total
        if uuid:
            payload["uuid"] = str(uuid)
        if group_id:
            payload["group_id"] = str(group_id)
        if name:
            payload["name"] = name
        if labels:
            payload["labels"] = labels
        if created_at:
            payload["created_at"] = str(created_at)
        if attributes:
            payload["attributes"] = attributes
        return {"graphiti": payload} if len(payload) > 1 else None

    async def _build_entity_lookup(
        self,
        node_results: list[Any] | None,
        edge_results: list[Any] | None,
    ) -> dict[str, dict[str, str]]:
        """Resolve entity UUIDs to display-friendly name/summary pairs for visualization."""
        lookup: dict[str, dict[str, str]] = {}

        if node_results:
            for node in node_results:
                node_uuid = getattr(node, "uuid", None)
                if not node_uuid:
                    continue
                uuid_str = str(node_uuid)
                lookup[uuid_str] = {
                    "uuid": uuid_str,
                    "name": getattr(node, "name", "") or f"Entity {uuid_str[:8]}",
                    "summary": getattr(node, "summary", "") or "",
                }

        required: set[str] = set()
        if edge_results:
            for edge in edge_results:
                for attr in ("source_node_uuid", "target_node_uuid"):
                    value = getattr(edge, attr, None)
                    if value is None:
                        continue
                    value_str = str(value)
                    if value_str not in lookup:
                        required.add(value_str)

        if required and self.graphiti is not None:
            try:
                fetched_nodes = await EntityNode.get_by_uuids(self.graphiti.driver, list(required))
                for node in fetched_nodes:
                    node_uuid = getattr(node, "uuid", None)
                    if not node_uuid:
                        continue
                    uuid_str = str(node_uuid)
                    lookup.setdefault(
                        uuid_str,
                        {
                            "uuid": uuid_str,
                            "name": getattr(node, "name", "") or f"Entity {uuid_str[:8]}",
                            "summary": getattr(node, "summary", "") or "",
                        },
                    )
            except Exception as exc:  # pragma: no cover - debug aid
                if self.valves.debug_print:
                    print(f"Failed to fetch entity summaries for visualization: {exc}")

        return lookup

    def _build_entity_connections(
        self,
        edges: list[Any] | None,
        entity_lookup: dict[str, dict[str, str]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Build adjacency lists so entity previews can highlight nearby facts."""
        if not edges:
            return {}

        adjacency: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for edge in edges:
            src_uuid = getattr(edge, "source_node_uuid", None)
            tgt_uuid = getattr(edge, "target_node_uuid", None)
            if not src_uuid or not tgt_uuid:
                continue

            base_payload = {
                "relation": getattr(edge, "name", "") or "",
                "fact": getattr(edge, "fact", "") or "",
                "valid_at": getattr(edge, "valid_at", None),
                "invalid_at": getattr(edge, "invalid_at", None),
            }

            src_str = str(src_uuid)
            tgt_str = str(tgt_uuid)
            adjacency[src_str].append(
                {
                    **base_payload,
                    "direction": "out",
                    "other_uuid": tgt_str,
                    "other": self._get_entity_display(entity_lookup, tgt_str),
                }
            )
            adjacency[tgt_str].append(
                {
                    **base_payload,
                    "direction": "in",
                    "other_uuid": src_str,
                    "other": self._get_entity_display(entity_lookup, src_str),
                }
            )

        return {uuid: links for uuid, links in adjacency.items() if links}

    @staticmethod
    def _get_entity_display(
        entity_lookup: dict[str, dict[str, str]],
        uuid_value: Optional[str],
    ) -> dict[str, str]:
        if not uuid_value:
            return {"uuid": "unknown", "name": "Unknown Entity", "summary": ""}
        uuid_str = str(uuid_value)
        details = entity_lookup.get(uuid_str)
        if details:
            return details
        return {"uuid": uuid_str, "name": f"Entity {uuid_str[:8]}", "summary": ""}

    @staticmethod
    def _escape_html_text(value: Any) -> str:
        if value is None:
            return ""
        return html.escape(str(value), quote=False)

    @staticmethod
    def _format_text_block(value: Any, limit: int | None = None) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        if not text:
            return ""
        if limit and limit > 3 and len(text) > limit:
            text = text[: limit - 3].rstrip() + "..."
        escaped = html.escape(text, quote=False)
        return escaped.replace("\n", "<br />")

    @staticmethod
    def _format_timestamp(value: Any) -> str:
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%d %H:%M")
        if value is None:
            return ""
        return str(value).strip()

    @classmethod
    def _format_valid_range(cls, valid_from: Any, valid_until: Any) -> str:
        start = cls._format_timestamp(valid_from)
        end = cls._format_timestamp(valid_until)
        if not start and not end:
            return ""
        if not start:
            start = "Unknown"
        if not end:
            end = "Present"
        return f"{start} -> {end}"

    @staticmethod
    def _wrap_rich_html(
        inner: str,
        min_height: int = 320,
        extra_head: Optional[str] = None,
        extra_body_script: Optional[str] = None,
    ) -> str:
        base_css = """
:root {
  color-scheme: light dark;
  --card-bg-light: #ffffff;
  --card-bg-dark: #111827;
  --surface-light: #f3f4f6;
  --surface-dark: #05070f;
  --border-light: #e5e7eb;
  --border-dark: #374151;
  --text-light: #0f172a;
  --text-dark: #f3f4f6;
  --accent-light: #2563eb;
  --accent-dark: #60a5fa;
  --muted-light: #6b7280;
  --muted-dark: #9ca3af;
  --graph-bg-light: rgba(37, 99, 235, 0.08);
  --graph-bg-dark: rgba(96, 165, 250, 0.12);
  --card-bg: var(--card-bg-light);
  --surface: var(--surface-light);
  --border: var(--border-light);
  --text: var(--text-light);
  --accent: var(--accent-light);
  --muted: var(--muted-light);
  --graph-bg: var(--graph-bg-light);
}
[data-theme="dark"] {
  --card-bg: var(--card-bg-dark);
  --surface: var(--surface-dark);
  --border: var(--border-dark);
  --text: var(--text-dark);
  --accent: var(--accent-dark);
  --muted: var(--muted-dark);
  --graph-bg: var(--graph-bg-dark);
}
body {
  margin: 0;
  padding: 12px;
  font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: var(--surface);
  color: var(--text);
  min-height: __MIN_HEIGHT__px;
}
.graphiti-card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 16px;
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
}
.card-heading {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
  margin-bottom: 12px;
}
.card-heading .title {
  font-size: 1rem;
  font-weight: 600;
}
.chip {
  font-size: 0.75rem;
  padding: 2px 8px;
  border-radius: 999px;
  background: rgba(37, 99, 235, 0.15);
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.chip-muted {
  background: transparent;
  border: 1px solid var(--border);
  color: var(--muted);
}
.graph-band {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
  align-items: stretch;
}
.node {
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 12px;
  background: var(--graph-bg);
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-height: 120px;
}
.node-label {
  font-size: 0.75rem;
  color: var(--muted);
  letter-spacing: 0.05em;
  text-transform: uppercase;
}
.node-title {
  font-size: 1rem;
  font-weight: 600;
}
.node-summary {
  font-size: 0.85rem;
  color: var(--text);
  opacity: 0.9;
  margin: 0;
}
.edge-visual {
  border: 1px dashed var(--border);
  border-radius: 12px;
  padding: 12px;
  background: rgba(149, 156, 177, 0.08);
  display: flex;
  flex-direction: column;
  gap: 8px;
  justify-content: center;
  min-height: 120px;
}
.edge-name {
  font-weight: 600;
}
.edge-fact {
  font-size: 0.85rem;
  color: var(--text);
}
.edge-range {
  font-size: 0.75rem;
  color: var(--muted);
}
.connection-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px;
  margin-top: 12px;
}
.connection-card {
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 12px;
  background: rgba(37, 99, 235, 0.05);
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.connection-card[data-direction="in"] {
  background: rgba(249, 115, 22, 0.1);
}
.connection-relation {
  font-weight: 600;
}
.connection-fact {
  font-size: 0.85rem;
  color: var(--text);
}
.connection-entity {
  font-size: 0.9rem;
  font-weight: 600;
}
.empty-state,
.more-note,
.legend,
.legend-item {
  font-size: 0.8rem;
  color: var(--muted);
  margin-top: 12px;
}
.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 0;
  margin-bottom: 8px;
}
.legend-item {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  margin-top: 0;
}
.legend-dot {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  display: inline-block;
  background: var(--muted);
}
.legend-spotlight {
  background: var(--accent);
}
.legend-context {
  background: #f97316;
}
.graph-canvas {
  width: 100%;
  min-height: 360px;
  border: 1px dashed var(--border);
  border-radius: 12px;
  background: rgba(37, 99, 235, 0.05);
  position: relative;
}
.graph-tooltip {
  position: absolute;
  pointer-events: none;
  background: rgba(15, 23, 42, 0.9);
  color: #f8fafc;
  padding: 8px 10px;
  border-radius: 6px;
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
  font-size: 0.75rem;
  max-width: 260px;
  z-index: 5;
}
.graph-tooltip.hidden {
  display: none;
}
.muted {
  color: var(--muted);
}
""".replace("__MIN_HEIGHT__", str(min_height))

        script = """
(function () {
  const docEl = document.documentElement;
  const prefersDark = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;

  function detectTheme() {
    try {
      const parentDoc = window.parent && window.parent.document;
      if (parentDoc && parentDoc.documentElement && parentDoc.documentElement.classList.contains('dark')) {
        return 'dark';
      }
    } catch (err) {
      // ignore cross-origin issues
    }
    if (prefersDark && prefersDark.matches) {
      return 'dark';
    }
    return 'light';
  }

  function applyTheme() {
    docEl.setAttribute('data-theme', detectTheme());
  }

  applyTheme();
  if (prefersDark && prefersDark.addEventListener) {
    prefersDark.addEventListener('change', applyTheme);
  }

  function resizeFrame() {
    const frame = window.frameElement;
    if (!frame) {
      return;
    }
    frame.style.height = Math.ceil(document.body.scrollHeight + 16) + 'px';
    frame.style.width = '100%';
  }

  if (window.ResizeObserver) {
    const ro = new ResizeObserver(resizeFrame);
    ro.observe(document.body);
  } else {
    window.setInterval(resizeFrame, 750);
  }

  window.addEventListener('load', resizeFrame);
  resizeFrame();
})();
"""

        head_parts = [f"<style>{base_css}</style>"]
        if extra_head:
            head_parts.append(extra_head)

        body_scripts = [f"<script>{script}</script>"]
        if extra_body_script:
            body_scripts.append(f"<script>{extra_body_script}</script>")

        return (
            "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\"/>"
            + "".join(head_parts)
            + f"</head><body>{inner}{''.join(body_scripts)}</body></html>"
        )

    def _render_fact_graph_html(
        self,
        *,
        fact_text: str,
        relation_name: Optional[str],
        source_entity: dict[str, str],
        target_entity: dict[str, str],
        valid_from: Any,
        valid_until: Any,
        idx: int,
        total: int,
    ) -> str:
        relation_text = relation_name or "Fact"
        relation_html = self._escape_html_text(relation_text)
        position_html = self._escape_html_text(f"Fact {idx}/{total}")

        # Build title: SOURCE → RELATION → TARGET
        source_name = source_entity.get("name", "Unknown")
        target_name = target_entity.get("name", "Unknown")
        title_text = f"{source_name} → {relation_text} → {target_name}"
        title_html = self._escape_html_text(title_text)

        source_summary = self._format_text_block(source_entity.get("summary", ""), 200)
        target_summary = self._format_text_block(target_entity.get("summary", ""), 200)
        fact_html = self._format_text_block(fact_text, 420) or ""

        source_block = (
            f"<p class=\"node-summary\">{source_summary}</p>"
            if source_summary
            else "<p class=\"node-summary muted\">No summary stored.</p>"
        )
        target_block = (
            f"<p class=\"node-summary\">{target_summary}</p>"
            if target_summary
            else "<p class=\"node-summary muted\">No summary stored.</p>"
        )
        range_text = self._format_valid_range(valid_from, valid_until)
        range_html = f"<div class=\"edge-range\">{self._escape_html_text(range_text)}</div>" if range_text else ""

        inner = f"""
<div class="graphiti-card fact-card">
  <div class="card-heading">
    <span class="chip">{position_html}</span>
    <span class="title">{title_html}</span>
  </div>
  <div class="graph-band">
    <div class="node">
      <div class="node-label">Source</div>
      <div class="node-title">{self._escape_html_text(source_entity.get('name'))}</div>
      {source_block}
    </div>
    <div class="edge-visual">
      <div class="edge-name">{relation_html}</div>
      <div class="edge-fact">{fact_html}</div>
      {range_html}
    </div>
    <div class="node">
      <div class="node-label">Target</div>
      <div class="node-title">{self._escape_html_text(target_entity.get('name'))}</div>
      {target_block}
    </div>
  </div>
</div>
"""

        return self._wrap_rich_html(inner.strip(), min_height=200)

    def _render_entity_graph_html(
        self,
        *,
        entity: dict[str, str],
        connections: list[dict[str, Any]],
        idx: int,
        total: int,
    ) -> str:
        title_html = self._escape_html_text(entity.get("name"))
        summary_html = self._format_text_block(entity.get("summary", ""), 360)
        summary_block = (
            f"<p class=\"node-summary\">{summary_html}</p>"
            if summary_html
            else "<p class=\"node-summary muted\">No summary stored.</p>"
        )
        connection_total = len(connections)
        connection_label = "0 related facts" if connection_total == 0 else (
            f"{connection_total} related fact{'s' if connection_total != 1 else ''}"
        )

        max_cards = 4
        cards: list[str] = []
        for conn in connections[:max_cards]:
            direction = conn.get("direction", "out")
            arrow = "->" if direction == "out" else "<-"
            relation_html = self._escape_html_text(conn.get("relation") or "Fact")
            fact_block = self._format_text_block(conn.get("fact", ""), 260)
            other = conn.get("other") or {}
            other_name = self._escape_html_text(other.get("name", "Unknown Entity"))
            other_summary = self._format_text_block(other.get("summary", ""), 180)
            other_summary_block = (
                f"<p class=\"node-summary\">{other_summary}</p>"
                if other_summary
                else ""
            )
            range_text = self._format_valid_range(conn.get("valid_at"), conn.get("invalid_at"))
            range_html = f"<div class=\"edge-range\">{self._escape_html_text(range_text)}</div>" if range_text else ""
            fact_section = f"<div class=\"connection-fact\">{fact_block}</div>" if fact_block else ""

            cards.append(
                f"""
<div class="connection-card" data-direction="{direction}">
  <div class="connection-relation">{arrow} {relation_html}</div>
  <div class="connection-entity">{other_name}</div>
  {fact_section}
  {range_html}
  {other_summary_block}
</div>
"""
            )

        if cards:
            connections_html = "<div class=\"connection-grid\">" + "".join(cards) + "</div>"
            if connection_total > max_cards:
                remaining = connection_total - max_cards
                more_label = f"+{remaining} more connection{'s' if remaining != 1 else ''} in Graphiti"
                connections_html += f"<div class=\"more-note\">{self._escape_html_text(more_label)}</div>"
        else:
            connections_html = "<div class=\"empty-state\">No related facts were returned for this entity in the current search window.</div>"

        inner = f"""
<div class="graphiti-card entity-card">
  <div class="card-heading">
    <span class="chip">Entity {idx}/{total}</span>
    <span class="title">{title_html}</span>
    <span class="chip chip-muted">{self._escape_html_text(connection_label)}</span>
  </div>
  <div class="node">
    <div class="node-label">Summary</div>
    <div class="node-title">{title_html}</div>
    {summary_block}
  </div>
  {connections_html}
</div>
"""

        height = 200 if connection_total == 0 else 420
        return self._wrap_rich_html(inner.strip(), min_height=height)

    def _build_overview_graph_data(
        self,
        entity_lookup: dict[str, dict[str, str]],
        nodes: list[Any] | None,
        edges: list[Any] | None,
        max_nodes: int = 14,
        max_edges: int = 24,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not entity_lookup:
            return [], []

        edge_count = len(edges or [])
        dynamic_node_cap = 30 if edge_count <= 12 else max_nodes
        dynamic_edge_cap = 60 if edge_count <= 12 else max_edges

        selected: dict[str, dict[str, Any]] = {}

        def add_node(node_uuid: Any, kind: str) -> None:
            if not node_uuid or len(selected) >= dynamic_node_cap:
                return
            uuid_str = str(node_uuid)
            if uuid_str in selected:
                return
            details = self._get_entity_display(entity_lookup, uuid_str)
            selected[uuid_str] = {
                "id": uuid_str,
                "label": details.get("name", f"Entity {uuid_str[:8]}") or f"Entity {uuid_str[:8]}",
                "summary": details.get("summary", "") or "",
                "kind": kind,
            }

        if nodes:
            for node in nodes:
                add_node(getattr(node, "uuid", None), "spotlight")
                if len(selected) >= dynamic_node_cap:
                    break

        if edges:
            for edge in edges:
                add_node(getattr(edge, "source_node_uuid", None), "context")
                add_node(getattr(edge, "target_node_uuid", None), "context")

        if not selected:
            return [], []

        node_ids = set(selected.keys())
        links: list[dict[str, Any]] = []
        if edges:
            for edge in edges:
                if len(links) >= dynamic_edge_cap:
                    break
                src = getattr(edge, "source_node_uuid", None)
                tgt = getattr(edge, "target_node_uuid", None)
                if not src or not tgt:
                    continue
                src_id = str(src)
                tgt_id = str(tgt)
                if src_id not in node_ids or tgt_id not in node_ids:
                    continue
                links.append(
                    {
                        "source": src_id,
                        "target": tgt_id,
                        "relation": getattr(edge, "name", "") or "",
                        "fact": getattr(edge, "fact", "") or "",
                    }
                )

        connected_ids: set[str] = set()
        for link in links:
            connected_ids.add(str(link["source"]))
            connected_ids.add(str(link["target"]))

        nodes_payload = [
            node
            for node in selected.values()
            if node["id"] in connected_ids or node["kind"] == "spotlight"
        ]

        if not links and nodes_payload:
            # Keep at most 8 spotlight-only nodes to prevent excessive spread when no edges
            nodes_payload = nodes_payload[:8]

        return nodes_payload, links

    def _render_overview_graph_html(
        self,
        nodes_payload: list[dict[str, Any]],
        links_payload: list[dict[str, Any]],
    ) -> str:
        if not nodes_payload:
            return self._wrap_rich_html(
                "<div class=\"graphiti-card\"><div class=\"empty-state\">No graph overview available.</div></div>",
                min_height=200,
            )

        node_count = len(nodes_payload)
        edge_count = len(links_payload)
        chip_text = f"{node_count} entit{'ies' if node_count != 1 else 'y'} · {edge_count} fact{'s' if edge_count != 1 else ''}"
        legend_html = """
<div class="legend">
  <span class="legend-item"><span class="legend-dot legend-spotlight"></span> Spotlight Entities</span>
  <span class="legend-item"><span class="legend-dot legend-context"></span> Connected Entities</span>
</div>
"""

        inner = f"""
<div class="graphiti-card overview-card">
  <div class="card-heading">
    <span class="chip">Overview</span>
    <span class="title">Retrieved Memory Graph</span>
    <span class="chip chip-muted">{self._escape_html_text(chip_text)}</span>
  </div>
  {legend_html}
  <div id="graph-container" class="graph-canvas" style="height: 350px;">
    <svg id="graph-svg" role="img" aria-label="Graph overview"></svg>
    <div id="graph-tooltip" class="graph-tooltip hidden"></div>
  </div>
  <div class="more-note">Interactive snapshot of the entities and facts returned in this search.</div>
</div>
"""

        extra_head = "<script src=\"https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js\"></script>"
        data_json = json.dumps({"nodes": nodes_payload, "links": links_payload})
        body_script = (
            "const overviewData = "
            + data_json
            + ";\n"
            + """
const container = document.getElementById('graph-container');
const svg = d3.select('#graph-svg');
const tooltip = document.getElementById('graph-tooltip');
let rootGroup;
let zoomBehaviorGlobal;
let hasInitialFit = false;

const nodesData = overviewData.nodes.map((node) => ({ ...node }));
const linksData = overviewData.links.map((link) => ({ ...link }));

// Group links by source-target pair for curve calculation
const linkGroups = {};
linksData.forEach((link) => {
  const key = link.source < link.target
    ? `${link.source}-${link.target}`
    : `${link.target}-${link.source}`;
  if (!linkGroups[key]) {
    linkGroups[key] = [];
  }
  linkGroups[key].push(link);
});

// Assign curve strength to each link
Object.values(linkGroups).forEach((group) => {
  const count = group.length;
  if (count === 1) {
    group[0].curveStrength = 0;
  } else {
    const baseStrength = 0.3;
    group.forEach((link, i) => {
      link.curveStrength = -baseStrength + (i * 2 * baseStrength) / (count - 1);
    });
  }
});

// Theme colors
const theme = {
  node: {
    stroke: 'var(--border)',
    text: 'var(--text)',
    selected: '#3b82f6',
    hover: '#60a5fa',
  },
  link: {
    stroke: 'var(--border)',
    selected: '#3b82f6',
    label: {
      bg: 'var(--surface)',
      text: 'var(--text)',
    },
  },
};

function nodeRadius(d) {
  return d.kind === 'spotlight' ? 12 : 10;
}

function colorFor(d) {
  return d.kind === 'spotlight' ? 'var(--accent)' : '#f97316';
}

function moveTooltip(event) {
  const bounds = container.getBoundingClientRect();
  tooltip.style.left = `${event.clientX - bounds.left + 12}px`;
  tooltip.style.top = `${event.clientY - bounds.top + 12}px`;
}

function showTooltip(event, datum) {
  tooltip.innerHTML = '';
  const title = document.createElement('div');
  title.style.fontWeight = '600';
  title.textContent = datum.label || datum.relation || '';
  tooltip.appendChild(title);
  if (datum.summary) {
    const summary = document.createElement('div');
    summary.style.marginTop = '4px';
    summary.textContent = datum.summary;
    tooltip.appendChild(summary);
  }
  if (datum.fact) {
    const fact = document.createElement('div');
    fact.style.marginTop = '4px';
    fact.style.fontStyle = 'italic';
    fact.textContent = datum.fact;
    tooltip.appendChild(fact);
  }
  tooltip.classList.remove('hidden');
  moveTooltip(event);
}

function hideTooltip() {
  tooltip.classList.add('hidden');
}

function fitToView(zoomBehavior, width, height) {
  if (!nodesData.length) {
    return;
  }
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  nodesData.forEach((node) => {
    if (typeof node.x === 'number' && typeof node.y === 'number') {
      minX = Math.min(minX, node.x);
      minY = Math.min(minY, node.y);
      maxX = Math.max(maxX, node.x);
      maxY = Math.max(maxY, node.y);
    }
  });
  if (!isFinite(minX) || !isFinite(minY) || !isFinite(maxX) || !isFinite(maxY)) {
    return;
  }
  const padding = 100;
  const dx = Math.max(maxX - minX, 10) + padding;
  const dy = Math.max(maxY - minY, 10) + padding;
  const scale = Math.min(1.2, 0.8 * Math.min(width / dx, height / dy));
  const translateX = width / 2 - scale * ((minX + maxX) / 2);
  const translateY = height / 2 - scale * ((minY + maxY) / 2);
  svg.transition().duration(750).ease(d3.easeCubicInOut).call(
    zoomBehavior.transform,
    d3.zoomIdentity.translate(translateX, translateY).scale(scale),
  );
}

function resetAllStyles(linkSelection, nodeSelection) {
  linkSelection.selectAll('path')
    .attr('stroke', theme.link.stroke)
    .attr('stroke-opacity', 0.6)
    .attr('stroke-width', 1);
  linkSelection.selectAll('.link-label rect')
    .attr('fill', theme.link.label.bg);
  linkSelection.selectAll('.link-label text')
    .attr('fill', theme.link.label.text);
  nodeSelection.selectAll('circle')
    .attr('fill', (d) => colorFor(d))
    .attr('stroke', theme.node.stroke)
    .attr('stroke-width', 2);
}

function renderGraph() {
  const rect = container.getBoundingClientRect();
  const width = Math.max(rect.width, 320);
  const height = Math.max(rect.height, 360);
  svg.attr('viewBox', `0 0 ${width} ${height}`);
  svg.attr('width', width);
  svg.attr('height', height);
  svg.selectAll('*').remove();

  rootGroup = svg.append('g').attr('class', 'graph-root');
  const linkLayer = rootGroup.append('g').attr('class', 'graph-links');
  const nodeLayer = rootGroup.append('g').attr('class', 'graph-nodes');

  // Create link groups (one per link)
  const linkGroup = linkLayer.selectAll('g')
    .data(linksData)
    .enter()
    .append('g')
    .attr('class', 'link-group');

  // Add curved paths for links
  linkGroup.append('path')
    .attr('stroke', theme.link.stroke)
    .attr('stroke-opacity', 0.6)
    .attr('stroke-width', 1)
    .attr('fill', 'none')
    .attr('cursor', 'pointer')
    .attr('data-curve-strength', (d) => d.curveStrength || 0);

  // Add edge labels
  const labelGroup = linkGroup.append('g')
    .attr('class', 'link-label')
    .attr('cursor', 'pointer');

  labelGroup.append('rect')
    .attr('fill', theme.link.label.bg)
    .attr('rx', 4)
    .attr('ry', 4)
    .attr('opacity', 0.9);

  labelGroup.append('text')
    .attr('fill', theme.link.label.text)
    .attr('font-size', '9px')
    .attr('text-anchor', 'middle')
    .attr('dominant-baseline', 'middle')
    .attr('pointer-events', 'none')
    .text((d) => d.relation || '');

  // Edge click handler
  function handleEdgeClick(event, d) {
    event.stopPropagation();
    resetAllStyles(linkGroup, node);

    // Highlight clicked edge
    const clickedGroup = d3.select(event.currentTarget.closest('.link-group'));
    clickedGroup.select('path')
      .attr('stroke', theme.link.selected)
      .attr('stroke-opacity', 1)
      .attr('stroke-width', 2);
    clickedGroup.select('.link-label rect')
      .attr('fill', theme.link.selected);
    clickedGroup.select('.link-label text')
      .attr('fill', '#ffffff');

    // Highlight connected nodes
    const sourceId = typeof d.source === 'object' ? d.source.id : d.source;
    const targetId = typeof d.target === 'object' ? d.target.id : d.target;
    node.selectAll('circle')
      .filter((n) => n.id === sourceId || n.id === targetId)
      .attr('fill', theme.node.selected)
      .attr('stroke', theme.node.selected)
      .attr('stroke-width', 3);

    // Zoom to edge
    const sourceNode = nodesData.find((n) => n.id === sourceId);
    const targetNode = nodesData.find((n) => n.id === targetId);
    if (sourceNode && targetNode && zoomBehaviorGlobal) {
      const padding = 100;
      const minX = Math.min(sourceNode.x, targetNode.x) - padding;
      const minY = Math.min(sourceNode.y, targetNode.y) - padding;
      const maxX = Math.max(sourceNode.x, targetNode.x) + padding;
      const maxY = Math.max(sourceNode.y, targetNode.y) + padding;
      const boundWidth = maxX - minX;
      const boundHeight = maxY - minY;
      const scale = 0.9 * Math.min(width / boundWidth, height / boundHeight);
      const midX = (minX + maxX) / 2;
      const midY = (minY + maxY) / 2;
      if (isFinite(scale) && isFinite(midX) && isFinite(midY)) {
        const transform = d3.zoomIdentity
          .translate(width / 2 - midX * scale, height / 2 - midY * scale)
          .scale(scale);
        svg.transition().duration(750).ease(d3.easeCubicInOut)
          .call(zoomBehaviorGlobal.transform, transform);
      }
    }
  }

  linkGroup.selectAll('path').on('click', handleEdgeClick);
  linkGroup.selectAll('.link-label').on('click', handleEdgeClick);

  // Edge hover
  linkGroup.on('mouseenter', (event, d) => showTooltip(event, d))
    .on('mousemove', (event) => moveTooltip(event))
    .on('mouseleave', () => hideTooltip());

  const node = nodeLayer.selectAll('g')
    .data(nodesData)
    .enter()
    .append('g')
    .attr('cursor', 'pointer')
    .call(d3.drag()
      .on('start', dragStarted)
      .on('drag', dragged)
      .on('end', dragEnded)
    );

  node.append('circle')
    .attr('r', (d) => nodeRadius(d))
    .attr('fill', (d) => colorFor(d))
    .attr('fill-opacity', 0.9)
    .attr('stroke', theme.node.stroke)
    .attr('stroke-width', 2)
    .attr('filter', 'drop-shadow(0 2px 4px rgba(0,0,0,0.2))');

  node.append('text')
    .text((d) => d.label)
    .attr('x', 15)
    .attr('y', '0.3em')
    .attr('text-anchor', 'start')
    .attr('fill', theme.node.text)
    .attr('font-weight', '500')
    .style('font-size', '0.7rem');

  // Node click handler
  function handleNodeClick(event, d) {
    event.stopPropagation();
    resetAllStyles(linkGroup, node);

    // Highlight selected node
    d3.select(event.currentTarget).select('circle')
      .attr('fill', theme.node.selected)
      .attr('stroke', theme.node.selected)
      .attr('stroke-width', 3);

    // Highlight connected edges
    const connectedNodes = new Set([d.id]);
    linkGroup.each(function(linkData) {
      const sourceId = typeof linkData.source === 'object' ? linkData.source.id : linkData.source;
      const targetId = typeof linkData.target === 'object' ? linkData.target.id : linkData.target;
      if (sourceId === d.id || targetId === d.id) {
        connectedNodes.add(sourceId);
        connectedNodes.add(targetId);
        d3.select(this).select('path')
          .attr('stroke', '#ffffff')
          .attr('stroke-width', 2);
      }
    });

    // Zoom to connected nodes
    if (connectedNodes.size > 0 && zoomBehaviorGlobal) {
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      connectedNodes.forEach((nodeId) => {
        const n = nodesData.find((nd) => nd.id === nodeId);
        if (n && n.x !== undefined && n.y !== undefined) {
          minX = Math.min(minX, n.x);
          minY = Math.min(minY, n.y);
          maxX = Math.max(maxX, n.x);
          maxY = Math.max(maxY, n.y);
        }
      });
      const padding = 50;
      minX -= padding;
      minY -= padding;
      maxX += padding;
      maxY += padding;
      const boundWidth = maxX - minX;
      const boundHeight = maxY - minY;
      const scale = 0.9 * Math.min(width / boundWidth, height / boundHeight);
      const midX = (minX + maxX) / 2;
      const midY = (minY + maxY) / 2;
      if (isFinite(scale) && isFinite(midX) && isFinite(midY)) {
        const transform = d3.zoomIdentity
          .translate(width / 2 - midX * scale, height / 2 - midY * scale)
          .scale(scale);
        svg.transition().duration(750).ease(d3.easeCubicInOut)
          .call(zoomBehaviorGlobal.transform, transform);
      }
    }
  }

  node.on('click', handleNodeClick);

  node.on('mouseenter', (event, d) => showTooltip(event, d))
    .on('mousemove', (event) => moveTooltip(event))
    .on('mouseleave', () => hideTooltip());

  // Background click to reset
  svg.on('click', function(event) {
    if (event.target === this) {
      resetAllStyles(linkGroup, node);
    }
  });

  const zoomBehavior = d3.zoom()
    .scaleExtent([0.1, 4])
    .on('zoom', (event) => {
      rootGroup.attr('transform', event.transform);
    });

  zoomBehaviorGlobal = zoomBehavior;
  svg.call(zoomBehavior);
  svg.call(zoomBehavior.transform, d3.zoomIdentity.scale(0.8));

  // Identify isolated nodes (not connected by any edge)
  const linkedNodeIds = new Set();
  linksData.forEach((link) => {
    linkedNodeIds.add(link.source);
    linkedNodeIds.add(link.target);
  });
  const isolatedNodeIds = new Set();
  nodesData.forEach((node) => {
    if (!linkedNodeIds.has(node.id)) {
      isolatedNodeIds.add(node.id);
    }
  });

  const simulation = d3.forceSimulation(nodesData)
    .force('link', d3.forceLink(linksData).id((d) => d.id).distance(200).strength(0.2))
    .force('charge', d3.forceManyBody()
      .strength((d) => isolatedNodeIds.has(d.id) ? -500 : -3000)
      .distanceMin(20)
      .distanceMax(500))
    .force('center', d3.forceCenter(width / 2, height / 2).strength(0.05))
    .force('collision', d3.forceCollide().radius((d) => nodeRadius(d) + 50).strength(0.3))
    .force('isolatedGravity', d3.forceRadial(100, width / 2, height / 2)
      .strength((d) => isolatedNodeIds.has(d.id) ? 0.15 : 0.01))
    .velocityDecay(0.4)
    .alphaDecay(0.05);

  simulation.on('tick', () => {
    // Update curved paths
    linkGroup.each(function(d) {
      const group = d3.select(this);
      const path = group.select('path');
      const curveStrength = d.curveStrength || 0;

      const sourceX = typeof d.source === 'object' ? d.source.x : nodesData.find((n) => n.id === d.source).x;
      const sourceY = typeof d.source === 'object' ? d.source.y : nodesData.find((n) => n.id === d.source).y;
      const targetX = typeof d.target === 'object' ? d.target.x : nodesData.find((n) => n.id === d.target).x;
      const targetY = typeof d.target === 'object' ? d.target.y : nodesData.find((n) => n.id === d.target).y;

      // Self-referencing edge
      if (d.source === d.target || (d.source.id && d.source.id === d.target.id)) {
        const radiusX = 40;
        const radiusY = 90;
        const cx = sourceX;
        const cy = sourceY - radiusY - 20;
        const pathD = `M${sourceX},${sourceY} C${cx - radiusX},${cy} ${cx + radiusX},${cy} ${sourceX},${sourceY}`;
        path.attr('d', pathD);

        // Position label for self-loop
        const labelG = group.select('.link-label');
        const text = labelG.select('text');
        const rect = labelG.select('rect');
        labelG.attr('transform', `translate(${cx}, ${cy - 10})`);
        const textNode = text.node();
        if (textNode) {
          const textBBox = textNode.getBBox();
          rect.attr('x', -textBBox.width / 2 - 6)
            .attr('y', -textBBox.height / 2 - 4)
            .attr('width', textBBox.width + 12)
            .attr('height', textBBox.height + 8);
          text.attr('x', 0).attr('y', 0);
        }
      } else {
        const dx = targetX - sourceX;
        const dy = targetY - sourceY;
        const dr = Math.sqrt(dx * dx + dy * dy);
        const midX = (sourceX + targetX) / 2;
        const midY = (sourceY + targetY) / 2;
        const normalX = -dy / dr;
        const normalY = dx / dr;
        const curveMagnitude = dr * curveStrength;
        const controlX = midX + normalX * curveMagnitude;
        const controlY = midY + normalY * curveMagnitude;
        const pathD = `M${sourceX},${sourceY} Q${controlX},${controlY} ${targetX},${targetY}`;
        path.attr('d', pathD);

        // Position label at midpoint of curve
        const pathNode = path.node();
        if (pathNode) {
          const pathLength = pathNode.getTotalLength();
          const midPoint = pathNode.getPointAtLength(pathLength / 2);
          const labelG = group.select('.link-label');
          const text = labelG.select('text');
          const rect = labelG.select('rect');
          const textNode = text.node();
          if (textNode) {
            const textBBox = textNode.getBBox();

            const angle = (Math.atan2(targetY - sourceY, targetX - sourceX) * 180) / Math.PI;
            const rotationAngle = angle > 90 || angle < -90 ? angle - 180 : angle;

            labelG.attr('transform', `translate(${midPoint.x}, ${midPoint.y}) rotate(${rotationAngle})`);
            rect.attr('x', -textBBox.width / 2 - 6)
              .attr('y', -textBBox.height / 2 - 4)
              .attr('width', textBBox.width + 12)
              .attr('height', textBBox.height + 8);
            text.attr('x', 0).attr('y', 0);
          }
        }
      }
    });

    node.attr('transform', (d) => `translate(${d.x}, ${d.y})`);
  });

  simulation.on('end', () => {
    if (!hasInitialFit) {
      hasInitialFit = true;
      fitToView(zoomBehavior, width, height);
    }
  });
  setTimeout(() => {
    if (!hasInitialFit) {
      hasInitialFit = true;
      fitToView(zoomBehavior, width, height);
    }
  }, 900);

  function dragStarted(event) {
    if (!event.active) {
      simulation.velocityDecay(0.7).alphaDecay(0.1).alphaTarget(0.1).restart();
    }
    d3.select(event.sourceEvent.target.parentNode).select('circle')
      .attr('stroke', theme.node.hover)
      .attr('stroke-width', 3);
    event.subject.fx = event.subject.x;
    event.subject.fy = event.subject.y;
  }

  function dragged(event) {
    event.subject.x = event.x;
    event.subject.y = event.y;
    event.subject.fx = event.x;
    event.subject.fy = event.y;
  }

  function dragEnded(event) {
    if (!event.active) {
      simulation.velocityDecay(0.4).alphaDecay(0.05).alphaTarget(0);
    }
    d3.select(event.sourceEvent.target.parentNode).select('circle')
      .attr('stroke', theme.node.stroke)
      .attr('stroke-width', 2);
    // Keep node fixed at final position
    event.subject.fx = event.x;
    event.subject.fy = event.y;
  }
}

renderGraph();
window.addEventListener('resize', renderGraph, { passive: true });
"""
        )

        return self._wrap_rich_html(inner.strip(), min_height=350, extra_head=extra_head, extra_body_script=body_script)
    
    def _build_overview_citation_event(
        self,
        entity_lookup: dict[str, dict[str, str]],
        nodes: list[Any] | None,
        edges: list[Any] | None,
    ) -> Optional[dict]:
        nodes_payload, links_payload = self._build_overview_graph_data(
            entity_lookup,
            nodes,
            edges,
        )
        if not nodes_payload:
            return None

        overview_html = self._render_overview_graph_html(nodes_payload, links_payload)
        source_id = self.valves.citation_source_id or "graphiti-memory"
        source_name = self.valves.citation_source_name or "Graphiti Memory"
        metadata: dict[str, Any] = {
            "source": source_id,
            "html": overview_html,
        }

        return {
            "source": {
                "id": source_id,
                "name": source_name,
                "type": "graphiti_memory",
            },
            "document": [overview_html],
            "metadata": [metadata],
        }
    
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
        sanitized = re.sub(r'[@:"()|]', ' ', query)
        
        # Replace multiple spaces with single space
        sanitized = re.sub(r'\s+', ' ', sanitized)
        
        # Trim whitespace
        sanitized = sanitized.strip()
        
        return sanitized

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,
        __metadata__: Optional[dict] = None,
    ) -> dict:
        if self.valves.debug_print:
            print(f"inlet:{__name__}")
            print(f"inlet:user:{__user__}")
        
        user_valves: Filter.UserValves = self.UserValves()
        # Check if user has disabled the feature
        if __user__:
            user_valves = __user__.get("valves", self.UserValves())
            if not user_valves.enabled:
                if self.valves.debug_print:
                    print("Graphiti Memory feature is disabled for this user.")
                return body
        
        # Check if graphiti is initialized, retry if not
        if not await self._ensure_graphiti_initialized() or self.graphiti is None:
            if self.valves.debug_print:
                print("Graphiti initialization failed. Skipping memory search.")
            if __user__ and user_valves.show_status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Memory service unavailable", "done": True},
                    }
                )
            return body
        
        # Set user headers in context variable (before any API calls)
        chat_id = __metadata__.get('chat_id') if __metadata__ else None
        headers = self._get_user_info_headers(__user__, chat_id)
        if headers:
            user_headers_context.set(headers)
            if self.valves.debug_print:
                print(f"Set user headers in context: {list(headers.keys())}")
        
        if __user__ is None:
            if self.valves.debug_print:
                print("User information is not available. Skipping memory search.")
            return body
        
        # Check if this is a "Continue Response" action
        # When user clicks "Continue Response" button, the last message is an assistant message
        # In this case, we should skip memory search to avoid injecting memories again
        messages = body.get("messages", [])
        if messages and messages[-1].get("role") == "assistant":
            if self.valves.debug_print:
                print("Detected 'Continue Response' action (last message is assistant). Skipping memory search.")
            return body
        
        max_length = self.valves.max_search_message_length
        history_turns = getattr(user_valves, "search_history_turns", 1)
        search_source_text = self._build_search_history_query(
            messages,
            history_turns,
            max_length,
        )
        
        if not search_source_text:
            if self.valves.debug_print:
                print("No user message found. Skipping memory search.")
            return body
        
        # Sanitize query for FalkorDB/RediSearch compatibility (before truncation)
        sanitized_query = search_source_text
        if self.valves.sanitize_search_query:
            sanitized_query = self._sanitize_search_query(search_source_text)
            if not sanitized_query:
                if self.valves.debug_print:
                    print("Search query is empty after sanitization. Skipping memory search.")
                return body
            
            if sanitized_query != search_source_text:
                if self.valves.debug_print:
                    print(f"Search query sanitized: removed problematic characters")
        
        # Truncate message if too long (keep first and last parts, drop middle)
        pre_trunc_length = len(sanitized_query)
        if max_length > 0 and len(sanitized_query) > max_length:
            keep_length = max_length // 2 - 25  # Leave room for separator
            sanitized_query = (
                sanitized_query[:keep_length] 
                + "\n\n[...]\n\n" 
                + sanitized_query[-keep_length:]
            )
            if self.valves.debug_print:
                print(f"Search query truncated from {pre_trunc_length} to {len(sanitized_query)} characters")
        
        # Get search configuration based on strategy
        search_config = self._get_search_config()
        
        if self.valves.debug_print:
            print(f"Using search strategy: {self.valves.search_strategy}")
        
        if user_valves.show_status:
            preview = sanitized_query[:100] + "..." if len(sanitized_query) > 100 else sanitized_query
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": f"🔍 Searching Graphiti: {preview}", "done": False},
                }
            )
        
        # Generate group_id using configured method (email or user ID)
        group_id = self._get_group_id(__user__)
        
        # Perform search with error handling for FalkorDB/RediSearch syntax issues
        # Measure search time
        search_start_time = time.time()
        
        try:
            # Use search_() for advanced search that returns SearchResults with nodes, edges, episodes, communities
            # Search configuration is determined by search_strategy setting
            # Only pass group_ids if group_id is not None
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
            
            # Calculate search duration
            search_duration = time.time() - search_start_time
            
            if self.valves.debug_print:
                print(f"Search completed in {search_duration:.2f}s")
        except Exception as e:
            search_duration = time.time() - search_start_time
            error_msg = str(e)
            if "Syntax error" in error_msg or "RediSearch" in error_msg:
                print(f"FalkorDB/RediSearch syntax error during search (after {search_duration:.2f}s): {error_msg}")
                if user_valves.show_status:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": f"Memory search unavailable (syntax error, {search_duration:.2f}s)", "done": True},
                        }
                    )
            else:
                print(f"Unexpected error during Graphiti search (after {search_duration:.2f}s): {e}")
                if user_valves.show_status:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": f"Memory search failed ({search_duration:.2f}s)", "done": True},
                        }
                    )
            return body
        
        # Check if any results were found
        if len(results.edges) == 0 and len(results.nodes) == 0:
            if user_valves.show_status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"No relevant memories found ({search_duration:.2f}s)", "done": True},
                    }
                )
            return body

        # Print search results (if debug mode enabled)
        if self.valves.debug_print:
            print('\nSearch Results:')

        facts = []
        entities = {}  # Dictionary to store unique entities: {name: summary}

        entity_lookup = await self._build_entity_lookup(results.nodes, results.edges)
        entity_connections = self._build_entity_connections(results.edges, entity_lookup)

        if user_valves.show_citation:
            overview_citation = self._build_overview_citation_event(
                entity_lookup=entity_lookup,
                nodes=results.nodes,
                edges=results.edges,
            )
            if overview_citation:
                await __event_emitter__(
                    {
                        "type": "citation",
                        "data": overview_citation,
                    }
                )
        
        # Process EntityEdge results (relations/facts) only if enabled
        if user_valves.inject_facts:
            for idx, result in enumerate(results.edges, 1):
                if self.valves.debug_print:
                    print(f'Edge UUID: {result.uuid}')
                    print(f'Fact({result.name}): {result.fact}')
                    if hasattr(result, 'valid_at') and result.valid_at:
                        print(f'Valid from: {result.valid_at}')
                    if hasattr(result, 'invalid_at') and result.invalid_at:
                        print(f'Valid until: {result.invalid_at}')

                facts.append((result.fact, result.valid_at, result.invalid_at, result.name))
                
                # Emit citation for each fact found when citations are enabled
                if user_valves.show_citation:
                    fact_citation = self._build_fact_citation_event(
                        result,
                        idx,
                        len(results.edges),
                        entity_lookup,
                        use_rich_html=user_valves.rich_html_citations,
                        include_parameters=user_valves.show_citation_parameters,
                    )
                    if fact_citation:
                        await __event_emitter__(
                            {
                                "type": "citation",
                                "data": fact_citation,
                            }
                        )
                
                if self.valves.debug_print:
                    print('---')
        else:
            if self.valves.debug_print:
                print(f'Skipping {len(results.edges)} facts (inject_facts is disabled)')
        
        # Process EntityNode results (entities with summaries) only if enabled
        if user_valves.inject_entities:
            for idx, result in enumerate(results.nodes, 1):
                if self.valves.debug_print:
                    print(f'Node UUID: {result.uuid}')
                    print(f'Entity({result.name}): {result.summary}')
                
                # Store entity information
                if result.name and result.summary:
                    entities[result.name] = result.summary
                    
                    # Emit citation for each entity found when citations are enabled
                    if user_valves.show_citation:
                        entity_citation = self._build_entity_citation_event(
                            result,
                            idx,
                            len(results.nodes),
                            entity_lookup,
                            entity_connections,
                            use_rich_html=user_valves.rich_html_citations,
                            include_parameters=user_valves.show_citation_parameters,
                        )
                        if entity_citation:
                            await __event_emitter__(
                                {
                                    "type": "citation",
                                    "data": entity_citation,
                                }
                            )
                
                if self.valves.debug_print:
                    print('---')
        else:
            if self.valves.debug_print:
                print(f'Skipping {len(results.nodes)} entities (inject_entities is disabled)')

        # Insert memory message if we have facts OR entities
        if len(facts) > 0 or len(entities) > 0:
            # Find the index of the last user message
            last_user_msg_index = None
            for i in range(len(body['messages']) - 1, -1, -1):
                if body['messages'][i].get("role") == "user":
                    last_user_msg_index = i
                    break
            
            # Determine the role to use for memory message (default to system if invalid value)
            memory_role = self.valves.memory_message_role.lower()
            if memory_role not in ["system", "user"]:
                if self.valves.debug_print:
                    print(f"Invalid memory_message_role '{memory_role}', using 'system'")
                memory_role = "system"
            
            # Format memory content with improved structure
            memory_content = "FACTS and ENTITIES represent relevant context to the current conversation.  \n"
            
            # Add facts section if any facts were found
            if len(facts) > 0:
                memory_content += "# These are the most relevant facts and their valid date ranges  \n"
                memory_content += "# format: FACT (Date range: from - to)  \n"
                memory_content += "<FACTS>  \n"
                
                for fact, valid_at, invalid_at, name in facts:
                    # Format date range
                    valid_str = str(valid_at) if valid_at else "unknown"
                    invalid_str = str(invalid_at) if invalid_at else "present"
                    
                    memory_content += f"  - {fact} ({valid_str} - {invalid_str})  \n"
                
                memory_content += "</FACTS>"
            
            # Add entities section if any entities were found
            if len(entities) > 0:
                if len(facts) > 0:
                    memory_content += "  \n\n"  # Add spacing between sections
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
                # Fallback: if no user message found, append to end
                body['messages'].append(memory_message)
            
            if user_valves.show_status:
                # Build status message showing what was found
                status_parts = []
                if len(facts) > 0:
                    status_parts.append(f"{len(facts)} fact{'s' if len(facts) != 1 else ''}")
                if len(entities) > 0:
                    status_parts.append(f"{len(entities)} entit{'ies' if len(entities) != 1 else 'y'}")
                
                status_msg = "🧠 " + " and ".join(status_parts) + f" found ({search_duration:.2f}s)"
                
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": status_msg, "done": True},
                    }
                )
        return body

    async def outlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,
        __metadata__: Optional[dict] = None,
    ) -> dict:
        # Check if user has disabled the feature
        if __user__:
            user_valves: Filter.UserValves = __user__.get("valves", self.UserValves())
            if not user_valves.enabled:
                if self.valves.debug_print:
                    print("Graphiti Memory feature is disabled for this user.")
                return body
        
        # Check if graphiti is initialized, retry if not
        if not await self._ensure_graphiti_initialized() or self.graphiti is None:
            if self.valves.debug_print:
                print("Graphiti initialization failed. Skipping memory addition.")
            return body
            
        if __user__ is None:
            if self.valves.debug_print:
                print("User information is not available. Skipping memory addition.")
            return body
        chat_id = __metadata__.get('chat_id', 'unknown') if __metadata__ else 'unknown'
        message_id = __metadata__.get('message_id', 'unknown') if __metadata__ else 'unknown'
        if self.valves.debug_print:
            print(f"outlet:{__name__}, chat_id:{chat_id}, message_id:{message_id}")
        
        # Set user headers in context variable (before any API calls)
        headers = self._get_user_info_headers(__user__, chat_id)
        if headers:
            user_headers_context.set(headers)
            if self.valves.debug_print:
                print(f"Set user headers in context: {list(headers.keys())}")


        user_valves: Filter.UserValves = __user__.get("valves", self.UserValves())
        messages = body.get("messages", [])
        if len(messages) == 0:
            return body
        
        # Determine which messages to save based on settings
        messages_to_save = []
        
        # Find the last user message, last assistant message, and previous assistant message
        last_user_message = None
        last_assistant_message = None
        previous_assistant_message = None
        
        for msg in reversed(messages):
            if msg.get("role") == "user" and last_user_message is None:
                last_user_message = msg
            elif msg.get("role") == "assistant":
                if last_user_message is None:
                    # This is after the last user message (the latest assistant response)
                    if last_assistant_message is None:
                        last_assistant_message = msg
                else:
                    # This is before the last user message (the assistant message user is responding to)
                    if previous_assistant_message is None:
                        previous_assistant_message = msg
                        break  # We found everything we need
        
        # Build messages_to_save list based on UserValves settings
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
            return body
        
        # Construct episode body in "Assistant: {message}\nUser: {message}\nAssistant: {message}" format for EpisodeType.message
        # Sort messages to maintain chronological order: previous_assistant -> user -> assistant
        role_order = {"previous_assistant": 0, "user": 1, "assistant": 2}
        messages_to_save.sort(key=lambda x: role_order.get(x[0], 99))
        
        episode_parts = []
        for role, content in messages_to_save:
            if role == "user":
                # Use actual user name if enabled, otherwise use "User"
                if self.valves.use_user_name_in_episode and __user__.get('name'):
                    role_label = __user__['name']
                else:
                    role_label = "User"
            elif role in ("assistant", "previous_assistant"):
                role_label = "Assistant"
            else:
                role_label = role.capitalize()
            episode_parts.append(f"{role_label}: {content}")
        
        episode_body = "\n".join(episode_parts)
        
        if user_valves.show_status:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": f"✍️ Adding conversation to Graphiti memory...", "done": False},
                }
            )
        
        # Generate group_id using configured method (email or user ID)
        group_id = self._get_group_id(__user__)
        saved_count = 0
        
        # Store add_episode results for detailed status display
        add_results = None
        
        try:
            # Apply timeout if configured
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
            saved_count = 1

            # Display extracted entities and facts in status
            if user_valves.show_status and add_results:
                # Show all Facts
                if add_results.edges:
                    for idx, edge in enumerate(add_results.edges, 1):
                        emoji = "🔚" if edge.invalid_at else "🔛"
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {"description": f"{emoji} Fact {idx}/{len(add_results.edges)}: {edge.fact}", "done": False},
                            }
                        )
                
                # Show all entities
                if add_results.nodes:
                    for idx, node in enumerate(add_results.nodes, 1):
                        # Display entity name and summary (if available)
                        entity_display = f"{node.name}"
                        if hasattr(node, 'summary') and node.summary:
                            entity_display += f" - {node.summary}"
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {"description": f"👤 Entity {idx}/{len(add_results.nodes)}: {entity_display}", "done": False},
                            }
                        )
        except asyncio.TimeoutError:
            print(f"Timeout adding conversation to Graphiti memory after {self.valves.add_episode_timeout}s")
            if user_valves.show_status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"Warning: Memory save timed out", "done": False},
                    }
                )
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            
            # Provide more specific error messages for common issues
            if "ValidationError" in error_type:
                print(f"Graphiti LLM response validation error for conversation: {error_msg}")
                user_msg = "Graphiti: LLM response format error (will retry on next message)"
            elif "ConnectionError" in error_type or "timeout" in error_msg.lower():
                print(f"Graphiti connection error adding conversation: {error_msg}")
                user_msg = "Graphiti: Connection error (temporary)"
            else:
                print(f"Graphiti error adding conversation: {e}")
                user_msg = f"Graphiti: Memory save failed ({error_type})"
            
            # Only print full traceback for unexpected errors
            if "ValidationError" not in error_type:
                traceback.print_exc()
            
            if user_valves.show_status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"Warning: {user_msg}", "done": False},
                    }
                )
        
        # Only increment count for successfully saved messages
        if saved_count > 0:
            pass  # Successfully saved messages

        if user_valves.show_status:
            if saved_count == 0:
                status_msg = "❌ Failed to save conversation to Graphiti memory"
            else:
                # Build detailed status message with entity and relationship counts
                status_parts = ["✅ Added conversation to Graphiti memory"]
                if add_results:
                    detail_parts = []
                    if add_results.nodes:
                        detail_parts.append(f"{len(add_results.nodes)} entit{'ies' if len(add_results.nodes) != 1 else 'y'}")
                    if add_results.edges:
                        detail_parts.append(f"{len(add_results.edges)} relation{'s' if len(add_results.edges) != 1 else ''}")
                    if detail_parts:
                        status_msg = " - ".join(status_parts + [" and ".join(detail_parts) + " extracted"])
                    else:
                        status_msg = status_parts[0]
                else:
                    status_msg = status_parts[0]
            
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": status_msg, "done": True},
                }
            )
        
        return body
