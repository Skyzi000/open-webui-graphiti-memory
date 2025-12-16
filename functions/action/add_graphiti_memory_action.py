"""
title: Add Graphiti Memory Action Button
description: Action button to save clicked messages to Graphiti knowledge graph memory.
author: Skyzi000
author_url: https://github.com/Skyzi000
repository_url: https://github.com/Skyzi000/open-webui-graphiti-memory
version: 0.3.1
requirements: graphiti-core
icon_url: data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzMiIgaGVpZ2h0PSIzMiIgdmlld0JveD0iMCAwIDMyIDMyIj4KICA8cmVjdCB4PSI2IiB5PSI0IiB3aWR0aD0iMjAiIGhlaWdodD0iMjQiIHJ4PSIyLjUiIGZpbGw9IiNmNmY2ZjAiIHN0cm9rZT0iIzRjNGM0YyIgc3Ryb2tlLXdpZHRoPSIxLjUiLz4KICA8cmVjdCB4PSIxMCIgeT0iOCIgd2lkdGg9IjEyIiBoZWlnaHQ9IjYiIHJ4PSIxIiBmaWxsPSIjZDBlNmZmIiBzdHJva2U9IiM0YzRjNGMiIHN0cm9rZS13aWR0aD0iMSIvPgogIDxyZWN0IHg9IjEyIiB5PSIyMCIgd2lkdGg9IjgiIGhlaWdodD0iNiIgcng9IjAuNyIgZmlsbD0iI2ZmZmJlNiIgc3Ryb2tlPSIjNGM0YzRjIiBzdHJva2Utd2lkdGg9IjEiLz4KICA8cmVjdCB4PSIxMyIgeT0iMjEiIHdpZHRoPSI2IiBoZWlnaHQ9IjIuNSIgcng9IjAuNyIgZmlsbD0iIzRjNGM0YyIvPgo8L3N2Zz4=

Note on FalkorDB backend:
  FalkorDB requires additional setup due to Redis version conflicts.
  See the README for details: https://github.com/Skyzi000/open-webui-graphiti-memory#falkordb-alternative-not-recently-tested

Design:
- Main class: Action
- Related components:
  - Graphiti: Knowledge graph memory system
  - FalkorDriver: FalkorDB backend driver for graph storage
  - MultiUserOpenAIClient: OpenAI client with per-user header injection
  - MultiUserOpenAIGenericClient: Generic OpenAI-compatible client with per-user header injection
  - MultiUserOpenAIEmbedder: Embedding model with per-user header injection

Architecture:
- Initialization: _initialize_graphiti() sets up graph database connection
- Multi-user support: contextvars-based header injection for concurrent requests
- Action handler: Saves clicked message to Graphiti memory
- Manual save: User can explicitly save important messages via button
"""

import asyncio
import contextvars
import hashlib
import os
import re
import time
import traceback
from datetime import datetime, timezone
from typing import Optional, Callable, Awaitable, Any
from urllib.parse import quote

from pydantic import BaseModel, Field

from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.nodes import EpisodeType
from openai import AsyncOpenAI
# Note: FalkorDriver is imported lazily only when FalkorDB backend is selected
# to avoid requiring the falkordb package when using Neo4j backend

# Chats import may fail if running outside Open WebUI core
try:
    from open_webui.models.chats import Chats  # type: ignore
except Exception:
    Chats = None

# Context variable to store user-specific headers for each async request
user_headers_context = contextvars.ContextVar('user_headers', default={})


class MultiUserOpenAIClient(OpenAIClient):
    """
    Custom OpenAI LLM client that retrieves user-specific headers from context variables.
    """
    
    def __init__(self, config: LLMConfig | None = None, cache: bool = False, **kwargs):
        if config is None:
            config = LLMConfig()
        
        # Store base client for dynamic header injection
        self._base_client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        
        # Initialize parent with our base client
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


class Action:
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
        
        group_id_format: str = Field(
            default="{user_id}",
            description="Format string for group_id. Available placeholders: {user_id}, {user_email}, {user_name}. Email addresses are automatically sanitized (@ becomes _at_, . becomes _). Examples: '{user_id}', '{user_id}_chat', 'user_{user_id}'. Set to 'none' to disable group filtering (all users share the same memory space). Recommended: Use {user_id} (default) as it's stable; email/name changes could cause memory access issues.",
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

    class UserValves(BaseModel):
        show_status: bool = Field(
            default=True, description="Show status of the action."
        )
        save_user_message: bool = Field(
            default=True,
            description="Save user messages as memories.",
        )
        save_assistant_response: bool = Field(
            default=True,
            description="Save assistant responses as memories.",
        )
        ui_language: str = Field(
            default="en",
            description="Language for UI labels and status messages: 'en' (English) or 'ja' (Japanese)",
        )
        

    def __init__(self):
        self.valves = self.Valves()
        self.graphiti = None
        self._indices_built = False
        self._last_config = None
        # Defer initialization until ensure_graphiti_initialized is called
        if self.valves.debug_print:
            print("Graphiti initialization deferred until first use")
    
    def _get_config_hash(self) -> str:
        """Generate hash of current configuration for change detection"""
        config_str = f"{self.valves.llm_client_type}_{self.valves.openai_api_url}_{self.valves.model}_" \
                    f"{self.valves.embedding_model}_{self.valves.api_key}_{self.valves.graph_db_backend}_" \
                    f"{self.valves.neo4j_uri}_{self.valves.neo4j_user}_{self.valves.neo4j_password}_" \
                    f"{self.valves.falkordb_host}_{self.valves.falkordb_port}_{self.valves.falkordb_username}_" \
                    f"{self.valves.falkordb_password}"
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _config_changed(self) -> bool:
        """Check if configuration has changed since last initialization"""
        current_hash = self._get_config_hash()
        return self._last_config != current_hash

    def _config_ready(self) -> tuple[bool, str]:
        """Prevent init when valves are still placeholders (e.g., right after restart)."""
        if not (self.valves.api_key or "").strip():
            return False, "api_key is empty"

        backend = (self.valves.graph_db_backend or "").lower().strip()

        if backend == "neo4j":
            # Allow default credentials; assume explicit API key means the admin intends this config
            pass
        elif backend == "falkordb":
            # Allow default host/port; API key guard already applied
            pass
        else:
            return False, f"Unsupported backend '{self.valves.graph_db_backend}'"

        return True, ""
    
    def _initialize_graphiti(self) -> bool:
        """Initialize Graphiti instance with configured backend"""
        try:
            os.environ['GRAPHITI_TELEMETRY_ENABLED'] = 'true' if self.valves.graphiti_telemetry_enabled else 'false'
            os.environ['SEMAPHORE_LIMIT'] = str(self.valves.semaphore_limit)
            
            llm_config = LLMConfig(
                api_key=self.valves.api_key,
                model=self.valves.model,
                small_model=self.valves.small_model,
                base_url=self.valves.openai_api_url,
            )

            # Select LLM client - use multi-user versions
            if self.valves.llm_client_type.lower() == "openai":
                llm_client = MultiUserOpenAIClient(config=llm_config)
                if self.valves.debug_print:
                    print("Using Multi-User OpenAI client")
            elif self.valves.llm_client_type.lower() == "generic":
                llm_client = MultiUserOpenAIGenericClient(config=llm_config)
                if self.valves.debug_print:
                    print("Using Multi-User OpenAI-compatible generic client")
            else:
                llm_client = MultiUserOpenAIClient(config=llm_config)
                if self.valves.debug_print:
                    print(f"Unknown client type '{self.valves.llm_client_type}', defaulting to Multi-User OpenAI client")

            falkor_driver = None
            if self.valves.graph_db_backend.lower() == "falkordb":
                from graphiti_core.driver.falkordb_driver import FalkorDriver
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
                    cross_encoder=OpenAIRerankerClient(client=llm_client._base_client, config=llm_config),
                )
            else:
                print(f"Unsupported graph database backend: {self.valves.graph_db_backend}")
                return False
            
            self._last_config = self._get_config_hash()
            if self.valves.debug_print:
                print("Graphiti initialized successfully.")
            return True
            
        except Exception as e:
            print(f"Graphiti initialization failed: {e}")
            return False
    
    async def _build_indices(self) -> bool:
        """Build database indices and constraints"""
        if self.graphiti is None:
            return False
        if self._indices_built:
            return True
        try:
            if self.valves.debug_print:
                print("Building Graphiti database indices...")
            await self.graphiti.build_indices_and_constraints()
            self._indices_built = True
            if self.valves.debug_print:
                print("Graphiti indices built successfully.")
            return True
        except Exception as e:
            print(f"Failed to build Graphiti indices: {e}")
            return False
    
    async def _ensure_graphiti_initialized(self) -> bool:
        """Ensure Graphiti is initialized and ready"""
        ready, reason = self._config_ready()
        if not ready:
            if self.valves.debug_print:
                print(f"Graphiti init skipped: {reason}")
            return False
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
        
        if not self._indices_built:
            if not await self._build_indices():
                return False
        
        return True
    
    def _get_group_id(self, user: dict) -> Optional[str]:
        """Generate group_id for the user based on format string configuration"""
        if self.valves.group_id_format.lower().strip() == "none":
            return None
        
        user_id = user.get('id', 'unknown')
        user_email = user.get('email', user_id)
        user_name = user.get('name', user_id)
        
        sanitized_email = user_email.replace('@', '_at_').replace('.', '_')
        sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '_', user_name)
        
        group_id = self.valves.group_id_format.format(
            user_id=user_id,
            user_email=sanitized_email,
            user_name=sanitized_name,
        )
        
        group_id = re.sub(r'[^a-zA-Z0-9_-]', '_', group_id)
        return group_id
    
    def _get_user_info_headers(self, user: Optional[dict] = None, chat_id: Optional[str] = None) -> dict:
        """Build user information headers dictionary"""
        valves_setting = self.valves.forward_user_info_headers.lower()
        
        if valves_setting == 'true':
            enable_forward = True
        elif valves_setting == 'false':
            enable_forward = False
        elif valves_setting == 'default':
            env_setting = os.environ.get('ENABLE_FORWARD_USER_INFO_HEADERS', 'false').lower()
            enable_forward = env_setting == 'true'
        else:
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
    
    def _get_content_from_message(self, message: dict) -> Optional[str]:
        """Extract text content from a message"""
        content = message.get("content")

        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    return item.get("text", "")
            return ""

        return content if isinstance(content, str) else ""

    def _get_chat_title(self, chat_id: Optional[str]) -> Optional[str]:
        """Resolve chat title from Open WebUI Chats model."""
        if Chats is None:
            return None
        if not chat_id or chat_id == "unknown":
            return None
        try:
            return Chats.get_chat_title_by_id(str(chat_id))
        except Exception:
            return None

    def _is_japanese_preferred(self, user_valves: "Action.UserValves") -> bool:
        """
        Check if user prefers Japanese language based on UserValves settings.

        Args:
            user_valves: UserValves object with ui_language setting

        Returns:
            True if user prefers Japanese (ja), False otherwise (default: English)
        """
        if hasattr(user_valves, 'ui_language'):
            return user_valves.ui_language.lower() == 'ja'
        return False

    async def action(
        self,
        body: dict,
        __user__=None,
        __event_emitter__=None,
        __event_call__=None,
    ) -> Optional[dict]:
        print(f"action:{__name__}")

        # Get user valves
        user_valves = self.UserValves.model_validate((__user__ or {}).get("valves", {}))

        is_ja = self._is_japanese_preferred(user_valves)

        # Check if Graphiti is initialized
        if not await self._ensure_graphiti_initialized() or self.graphiti is None:
            if __event_emitter__ and user_valves.show_status:
                msg = "❌ Graphitiが初期化されていません" if is_ja else "❌ Graphiti not initialized"
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": msg, "done": True},
                    }
                )
            return None

        if __user__ is None:
            if __event_emitter__ and user_valves.show_status:
                msg = "❌ ユーザー情報が利用できません" if is_ja else "❌ User information not available"
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": msg, "done": True},
                    }
                )
            return None

        messages = body.get("messages", [])

        # The clicked message is always the last in the list
        # (createMessagesList builds root -> clicked message)
        if not messages:
            if __event_emitter__ and user_valves.show_status:
                msg = "❌ メッセージがありません" if is_ja else "❌ No messages"
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": msg, "done": True},
                    }
                )
            return None

        clicked_message = messages[-1]

        # Find the user/assistant pair based on clicked message
        last_user_message = None
        last_assistant_message = None

        if clicked_message.get("role") == "assistant":
            last_assistant_message = clicked_message
            # Find the previous user message
            for msg in reversed(messages[:-1]):
                if msg.get("role") == "user":
                    last_user_message = msg
                    break
        elif clicked_message.get("role") == "user":
            last_user_message = clicked_message
            # No assistant response for this user message yet

        if __event_emitter__:
            # Get chat_id from metadata if available
            chat_id = body.get("chat_id", "unknown")
            
            # Set user headers in context variable
            headers = self._get_user_info_headers(__user__, chat_id)
            if headers:
                user_headers_context.set(headers)
                if self.valves.debug_print:
                    print(f"Set user headers in context: {list(headers.keys())}")

            if user_valves.show_status:
                msg = "✍️ Graphitiメモリに手動保存中..." if is_ja else "✍️ Manually saving to Graphiti memory..."
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": msg, "done": False},
                    }
                )

            # Add the conversation to memories
            try:
                messages_to_save = []
                
                if user_valves.save_user_message and last_user_message:
                    user_content = self._get_content_from_message(last_user_message)
                    if user_content:
                        messages_to_save.append(("user", user_content))
                
                if user_valves.save_assistant_response and last_assistant_message:
                    assistant_content = self._get_content_from_message(last_assistant_message)
                    if assistant_content:
                        messages_to_save.append(("assistant", assistant_content))
                
                if len(messages_to_save) == 0:
                    if user_valves.show_status:
                        msg = "ℹ️ 保存するメッセージがありません" if is_ja else "ℹ️ No messages to save"
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {"description": msg, "done": True},
                            }
                        )
                    return None
                
                # Construct episode body
                episode_parts = []
                for role, content in messages_to_save:
                    if role == "user":
                        if self.valves.use_user_name_in_episode and __user__.get('name'):
                            role_label = __user__['name']
                        else:
                            role_label = "User"
                    else:
                        role_label = "Assistant"
                    episode_parts.append(f"{role_label}: {content}")
                
                separator = "\n\n---\n\n"
                episode_body = separator.join(episode_parts)

                # Get group_id
                group_id = self._get_group_id(__user__)

                # Build episode name and source_description (matching Filter format)
                user_turn_index = sum(1 for m in messages if m.get("role") == "user")
                chat_title = self._get_chat_title(chat_id) or "Manual_Save"
                episode_name = (
                    f"{chat_title}_turn{user_turn_index}"
                    if user_turn_index > 0
                    else chat_title
                )
                timestamp_prefix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                source_description = f"{timestamp_prefix}_Chat_{chat_id}_turn{user_turn_index}_Manual"

                # Add episode
                if self.valves.add_episode_timeout > 0:
                    if group_id is not None:
                        add_results = await asyncio.wait_for(
                            self.graphiti.add_episode(
                                name=episode_name,
                                episode_body=episode_body,
                                source=EpisodeType.message,
                                source_description=source_description,
                                reference_time=datetime.now(timezone.utc),
                                group_id=group_id,
                                update_communities=self.valves.update_communities,
                            ),
                            timeout=self.valves.add_episode_timeout
                        )
                    else:
                        add_results = await asyncio.wait_for(
                            self.graphiti.add_episode(
                                name=episode_name,
                                episode_body=episode_body,
                                source=EpisodeType.message,
                                source_description=source_description,
                                reference_time=datetime.now(timezone.utc),
                                update_communities=self.valves.update_communities,
                            ),
                            timeout=self.valves.add_episode_timeout
                        )
                else:
                    if group_id is not None:
                        add_results = await self.graphiti.add_episode(
                            name=episode_name,
                            episode_body=episode_body,
                            source=EpisodeType.message,
                            source_description=source_description,
                            reference_time=datetime.now(timezone.utc),
                            group_id=group_id,
                            update_communities=self.valves.update_communities,
                        )
                    else:
                        add_results = await self.graphiti.add_episode(
                            name=episode_name,
                            episode_body=episode_body,
                            source=EpisodeType.message,
                            source_description=source_description,
                            reference_time=datetime.now(timezone.utc),
                            update_communities=self.valves.update_communities,
                        )
                
                if self.valves.debug_print:
                    print(f"Added to memory: {episode_body[:100]}...")
                    if add_results:
                        print(f"Extracted {len(add_results.nodes)} entities and {len(add_results.edges)} relationships")
                
                # Display extracted entities and facts in status
                if user_valves.show_status and add_results:
                    # Show all Facts
                    if add_results.edges:
                        fact_label = "ファクト" if is_ja else "Fact"
                        for idx, edge in enumerate(add_results.edges, 1):
                            emoji = "🔚" if edge.invalid_at else "🔛"
                            await __event_emitter__(
                                {
                                    "type": "status",
                                    "data": {"description": f"{emoji} {fact_label} {idx}/{len(add_results.edges)}: {edge.fact}", "done": False},
                                }
                            )

                    # Show all entities
                    if add_results.nodes:
                        entity_label = "エンティティ" if is_ja else "Entity"
                        for idx, node in enumerate(add_results.nodes, 1):
                            entity_display = f"{node.name}"
                            if hasattr(node, 'summary') and node.summary:
                                entity_display += f" - {node.summary}"
                            await __event_emitter__(
                                {
                                    "type": "status",
                                    "data": {"description": f"👤 {entity_label} {idx}/{len(add_results.nodes)}: {entity_display}", "done": False},
                                }
                            )
                
                # Final success status
                if user_valves.show_status:
                    if is_ja:
                        status_base = "✅ Graphitiメモリに追加しました"
                        if add_results:
                            detail_parts = []
                            if add_results.nodes:
                                detail_parts.append(f"{len(add_results.nodes)}件のエンティティ")
                            if add_results.edges:
                                detail_parts.append(f"{len(add_results.edges)}件のファクト")
                            if detail_parts:
                                status_msg = status_base + " - " + "と".join(detail_parts) + "を抽出"
                            else:
                                status_msg = status_base
                        else:
                            status_msg = status_base
                    else:
                        status_base = "✅ Added to Graphiti memory"
                        if add_results:
                            detail_parts = []
                            if add_results.nodes:
                                detail_parts.append(f"{len(add_results.nodes)} entit{'ies' if len(add_results.nodes) != 1 else 'y'}")
                            if add_results.edges:
                                detail_parts.append(f"{len(add_results.edges)} fact{'s' if len(add_results.edges) != 1 else ''}")
                            if detail_parts:
                                status_msg = status_base + " - " + " and ".join(detail_parts) + " extracted"
                            else:
                                status_msg = status_base
                        else:
                            status_msg = status_base

                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": status_msg, "done": True},
                        }
                    )
                    
            except asyncio.TimeoutError:
                print(f"Timeout adding to memory after {self.valves.add_episode_timeout}s")
                if user_valves.show_status:
                    msg = "⚠️ メモリ保存がタイムアウトしました" if is_ja else "⚠️ Memory save timed out"
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": msg, "done": True},
                        }
                    )
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)

                if "ValidationError" in error_type:
                    print(f"Graphiti LLM response validation error: {error_msg}")
                    user_msg = "Graphiti: LLMレスポンス形式エラー" if is_ja else "Graphiti: LLM response format error"
                elif "ConnectionError" in error_type or "timeout" in error_msg.lower():
                    print(f"Graphiti connection error: {error_msg}")
                    user_msg = "Graphiti: 接続エラー" if is_ja else "Graphiti: Connection error"
                else:
                    print(f"Graphiti error: {e}")
                    user_msg = f"Graphiti: エラー ({error_type})" if is_ja else f"Graphiti: Error ({error_type})"
                    traceback.print_exc()

                if user_valves.show_status:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": f"❌ {user_msg}", "done": True},
                        }
                    )
        
        return None
