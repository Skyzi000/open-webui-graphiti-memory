"""
title: Graphiti Memory Manage Tool
author: Skyzi000
description: Manage specific entities, relationships, or episodes in Graphiti knowledge graph memory.
author_url: https://github.com/Skyzi000
repository_url: https://github.com/Skyzi000/open-webui-graphiti-memory
version: 0.5.5
requirements: graphiti-core

Note on FalkorDB backend:
  FalkorDB requires additional setup due to Redis version conflicts.
  See the README for details: https://github.com/Skyzi000/open-webui-graphiti-memory#falkordb-alternative-not-recently-tested

Design:
- Main class: Tools
- Helper class: GraphitiHelper (handles initialization, not exposed to AI)
- Related components:
  - Graphiti: Knowledge graph memory system
  - FalkorDriver: FalkorDB backend driver for graph storage
  - OpenAIClient: OpenAI client with JSON structured output support
  - OpenAIGenericClient: Generic OpenAI-compatible client
  - OpenAIEmbedder: Embedding model for semantic search

Architecture:
- Search and Delete: Search for specific entities, edges, or episodes, then delete them via Cypher queries
- Episode Deletion: Uses Graphiti's remove_episode() method
- Node/Edge Deletion: Uses driver's execute_query() with Cypher DELETE statements
- UUID-based Deletion: Delete by UUID for precise control
- Batch Operations: Delete multiple items at once
- Group Isolation: Only delete from user's own memory space (respects group_id)

Related Filter:
- functions/filter/graphiti_memory.py: Main memory management filter
"""

import os
import re
import json
import copy
import asyncio
import contextvars
import hashlib
import traceback
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Callable
from urllib.parse import quote

from pydantic import BaseModel, Field

from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF
# Note: FalkorDriver is imported lazily only when FalkorDB backend is selected
# to avoid requiring the falkordb package when using Neo4j backend
from graphiti_core.nodes import EntityNode, EpisodicNode, EpisodeType, get_episodic_node_from_record
from graphiti_core.edges import EntityEdge
from openai import AsyncOpenAI

# Context variable to store user-specific headers for each async request
# This ensures complete isolation between concurrent requests without locks
user_headers_context = contextvars.ContextVar('user_headers', default={})


class MultiUserOpenAIClient(OpenAIClient):
    """
    Custom OpenAI LLM client that retrieves user-specific headers from context variables.
    This allows a single Graphiti instance to safely handle concurrent requests from multiple users.
    
    Overrides self.client property to inject user headers dynamically without copying parent logic.
    This ensures automatic compatibility with future Graphiti updates.
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
    
    Overrides self.client property to inject user headers dynamically without copying parent logic.
    This ensures automatic compatibility with future Graphiti updates.
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
    
    Overrides self.client property to inject user headers dynamically without copying parent logic.
    This ensures automatic compatibility with future Graphiti updates.
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


class GraphitiHelper:
    def __init__(self, tools_instance):
        self.tools = tools_instance
        self.graphiti = None
        self._last_config = None
    
    @property
    def valves(self):
        """Always get fresh valves from Tools instance."""
        return self.tools.valves
    
    @staticmethod
    def validate_message_format(content: str) -> tuple[bool, Optional[str]]:
        """
        Validate message format content.
        
        :param content: Content to validate
        :return: Tuple of (is_valid, error_message). error_message is None if valid.
        """
        if ':' not in content:
            error_msg = (
                "❌ Error: For source='message', content must be in 'speaker: content' format.\n"
                "Example: 'user: Hello' or 'assistant: How can I help?'\n"
                "Each line should start with a speaker name followed by a colon."
            )
            return False, error_msg
        
        # Check if at least one line follows the format
        lines = content.strip().split('\n')
        valid_format = False
        for line in lines:
            if ':' in line and line.split(':', 1)[0].strip():
                valid_format = True
                break
        
        if not valid_format:
            error_msg = (
                "❌ Error: For source='message', at least one line must follow 'speaker: content' format.\n"
                "Provided content does not have any valid message lines.\n"
                "Example format:\n"
                "user: What's the weather?\n"
                "assistant: It's sunny today."
            )
            return False, error_msg
        
        return True, None
    
    def get_config_hash(self) -> str:
        """Generate configuration hash for change detection."""
        # Get all valve values as dict, excluding non-config fields
        valve_dict = self.valves.model_dump(
            exclude={
                'debug_print',  # Debugging settings don't affect initialization
                'group_id_format',  # Group ID format doesn't affect Graphiti init
                'confirmation_timeout',  # UI timeout doesn't affect Graphiti init
            }
        )
        # Sort keys for consistent hashing
        config_str = '|'.join(f"{k}={v}" for k, v in sorted(valve_dict.items()))
        return hashlib.md5(config_str.encode()).hexdigest()

    def config_ready(self) -> tuple[bool, str]:
        """Return (ready, reason). Prevent init with placeholder defaults after restart."""
        if not (self.valves.api_key or "").strip():
            return False, "api_key is empty"

        backend = (self.valves.graph_db_backend or "").lower().strip()
        if backend == "neo4j":
            # Allow default credentials; assume explicit API key means the admin intends this config
            pass
        elif backend == "falkordb":
            # Accept default host/port; API key guard already applied
            pass
        else:
            return False, f"Unsupported backend '{self.valves.graph_db_backend}'"

        return True, ""
    
    def config_changed(self) -> bool:
        """Check if configuration has changed."""
        current_hash = self.get_config_hash()
        if self._last_config != current_hash:
            if self._last_config is not None and self.valves.debug_print:
                print("Configuration changed, will reinitialize Graphiti")
            return True
        return False
    
    def initialize_graphiti(self):
        """Initialize Graphiti with configured settings."""
        if self.graphiti is not None and not self.config_changed():
            return

        ready, reason = self.config_ready()
        if not ready:
            if self.valves.debug_print:
                print(f"Graphiti init skipped: {reason}")
            return
        
        if self.valves.debug_print:
            print("Initializing Graphiti for memory deletion...")
        
        # Disable telemetry if configured
        if not self.valves.graphiti_telemetry_enabled:
            os.environ['GRAPHITI_TELEMETRY_ENABLED'] = 'false'
        
        # Set semaphore limit via environment variable
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
        # Initialize embedder
        embedder = MultiUserOpenAIEmbedder(
            config=OpenAIEmbedderConfig(
                api_key=self.valves.api_key,
                base_url=self.valves.openai_api_url,
                embedding_model=self.valves.embedding_model,
                embedding_dim=self.valves.embedding_dim,
            )
        )
        
        # Initialize based on backend
        if self.valves.debug_print:
            print(f"Graph DB Backend: {self.valves.graph_db_backend}")
            print(f"Neo4j URI: {self.valves.neo4j_uri}")
            print(f"FalkorDB Host: {self.valves.falkordb_host}:{self.valves.falkordb_port}")
        
        falkor_driver = None
        if self.valves.graph_db_backend.lower() == "falkordb":
            if self.valves.debug_print:
                print("Initializing FalkorDB driver...")
            from graphiti_core.driver.falkordb_driver import FalkorDriver
            falkor_driver = FalkorDriver(
                host=self.valves.falkordb_host,
                port=self.valves.falkordb_port,
                username=self.valves.falkordb_username,
                password=self.valves.falkordb_password,
            )
        # Initialize Graphiti
        if falkor_driver:
            if self.valves.debug_print:
                print("Creating Graphiti instance with FalkorDB...")
            self.graphiti = Graphiti(
                graph_driver=falkor_driver,
                llm_client=llm_client,
                embedder=embedder,
                # OpenAIRerankerClient requires AsyncOpenAI client
                # Use _base_client from our custom multi-user client
                cross_encoder=OpenAIRerankerClient(client=llm_client._base_client, config=llm_config),
            )
        elif self.valves.graph_db_backend.lower() == "neo4j":
            if self.valves.debug_print:
                print("Creating Graphiti instance with Neo4j...")
            self.graphiti = Graphiti(
                self.valves.neo4j_uri,
                self.valves.neo4j_user,
                self.valves.neo4j_password,
                llm_client=llm_client,
                embedder=embedder,
                # OpenAIRerankerClient requires AsyncOpenAI client
                # Use _base_client from our custom multi-user client
                cross_encoder=OpenAIRerankerClient(client=llm_client._base_client, config=llm_config),
            )
        else:
            raise ValueError(f"Unsupported graph database backend: {self.valves.graph_db_backend}. Supported backends are 'neo4j' and 'falkordb'.")
        
        self._last_config = self.get_config_hash()
        
        if self.valves.debug_print:
            print("Graphiti initialized successfully")
    
    async def ensure_graphiti_initialized(self) -> bool:
        """Ensure Graphiti is initialized, retry if needed."""
        ready, reason = self.config_ready()
        if not ready:
            if self.valves.debug_print:
                print(f"Graphiti init skipped: {reason}")
            return False
        if self.graphiti is None or self.config_changed():
            try:
                if self.valves.debug_print:
                    print("=== ensure_graphiti_initialized: Attempting initialization ===")
                self.initialize_graphiti()
                return True
            except Exception as e:
                print(f"Failed to initialize Graphiti: {e}")
                if self.valves.debug_print:
                    traceback.print_exc()
                return False
        return True
    
    def get_group_id(self, user: dict) -> Optional[str]:
        """
        Generate group_id from user information based on configured format.
        
        Args:
            user: User dictionary containing 'id', 'email', 'name'
            
        Returns:
            Generated group_id or None if group filtering is disabled
        """
        if self.valves.group_id_format.lower() == 'none':
            return None
        
        user_id = user.get('id', 'unknown')
        user_email = user.get('email', '')
        user_name = user.get('name', '')
        
        # Sanitize email and name
        sanitized_email = re.sub(r'[@.]', lambda m: '_at_' if m.group() == '@' else '_', user_email)
        sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '_', user_name)
        
        group_id = self.valves.group_id_format.format(
            user_id=user_id,
            user_email=sanitized_email,
            user_name=sanitized_name,
        )
        
        # Final sanitization
        group_id = re.sub(r'[^a-zA-Z0-9_-]', '_', group_id)
        
        return group_id
    
    def is_japanese_preferred(self, user: dict) -> bool:
        """
        Check if user prefers Japanese language based on UserValves settings.
        
        Args:
            user: User dictionary containing 'valves' with message_language setting
            
        Returns:
            True if user prefers Japanese (ja), False otherwise (default: English)
        """
        user_valves = user.get("valves")
        if user_valves and hasattr(user_valves, 'message_language'):
            return user_valves.message_language.lower() == 'ja'
        return False
    
    async def delete_nodes_by_uuids(self, uuids: List[str], group_id: Optional[str] = None) -> int:
        """Delete nodes by UUIDs using EntityNode.delete_by_uuids()."""
        if not uuids or not self.graphiti:
            return 0
        
        if self.valves.debug_print:
            print(f"=== delete_nodes_by_uuids: Attempting to delete {len(uuids)} nodes ===")
            print(f"Group ID filter: {group_id}")
            print(f"UUIDs: {uuids}")
        
        try:
            # Use EntityNode.delete_by_uuids() static method
            await EntityNode.delete_by_uuids(self.graphiti.driver, uuids)
            
            if self.valves.debug_print:
                print(f"=== Successfully deleted {len(uuids)} nodes ===")
            
            return len(uuids)
        except Exception as e:
            print(f"Failed to delete nodes: {e}")
            if self.valves.debug_print:
                traceback.print_exc()
            return 0
    
    async def delete_edges_by_uuids(self, uuids: List[str], group_id: Optional[str] = None) -> int:
        """Delete edges by UUIDs using EntityEdge.delete_by_uuids()."""
        if not uuids or not self.graphiti:
            return 0
        
        if self.valves.debug_print:
            print(f"=== delete_edges_by_uuids: Attempting to delete {len(uuids)} edges ===")
            print(f"Group ID filter: {group_id}")
            print(f"UUIDs: {uuids}")
        
        try:
            # Use EntityEdge.delete_by_uuids() static method
            await EntityEdge.delete_by_uuids(self.graphiti.driver, uuids)
            
            if self.valves.debug_print:
                print(f"=== Successfully deleted {len(uuids)} edges ===")
            
            return len(uuids)
        except Exception as e:
            print(f"Failed to delete edges: {e}")
            if self.valves.debug_print:
                traceback.print_exc()
            return 0
    
    async def delete_episodes_by_uuids(self, uuids: List[str]) -> int:
        """Delete episodes by UUIDs using Graphiti.remove_episode()."""
        if not uuids or not self.graphiti:
            return 0
        
        if self.valves.debug_print:
            print(f"=== delete_episodes_by_uuids: Attempting to delete {len(uuids)} episodes ===")
            print(f"UUIDs: {uuids}")
        
        deleted_count = 0
        for uuid in uuids:
            try:
                if self.valves.debug_print:
                    print(f"Deleting episode with UUID: {uuid}")
                
                await self.graphiti.remove_episode(uuid)
                deleted_count += 1
                
                if self.valves.debug_print:
                    print(f"Successfully deleted episode {uuid}")
            except Exception as e:
                print(f"Failed to delete episode {uuid}: {e}")
                if self.valves.debug_print:
                    traceback.print_exc()
        
        if self.valves.debug_print:
            print(f"=== Total deleted: {deleted_count} episodes ===")
        
        return deleted_count
    
    async def show_confirmation_dialog(
        self,
        title: str,
        items: List[str],
        warning_message: str,
        timeout: int,
        __user__: dict = {},
        __event_call__: Optional[Callable[[dict], Any]] = None,
    ) -> tuple[bool, str]:
        """
        Show confirmation dialog and wait for user response.
        Helper method for Tools class - not exposed to AI.

        This method can be used for any operation requiring user confirmation,
        not limited to deletion operations.

        :param title: Dialog title
        :param items: List of items to display for confirmation
        :param warning_message: Warning or informational message to show
        :param timeout: Timeout in seconds
        :param __user__: User information dictionary
        :param __event_call__: Event caller for confirmation dialog
        :return: Tuple of (confirmed, error_message) where:
                 - confirmed: True if user confirmed, False otherwise
                 - error_message: Empty string if confirmed, error message otherwise
        """
        if not __event_call__:
            return False, "❌ Error: Confirmation dialog is not available. Cannot perform operation without user confirmation."

        preview_text = "  \n".join(items)

        # Get user's language preference from UserValves
        is_japanese = self.is_japanese_preferred(__user__)

        if is_japanese:
            confirmation_message = f"""以下の操作を実行しますか？

{preview_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{warning_message}
⏰ {timeout}秒以内に選択しないと自動的にキャンセルされます。"""
        else:
            confirmation_message = f"""Do you want to proceed with the following operation?

{preview_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{warning_message}
⏰ Auto-cancel in {timeout} seconds if no selection is made."""
        
        try:
            confirmation_task = __event_call__(
                {
                    "type": "confirmation",
                    "data": {
                        "title": title,
                        "message": confirmation_message,
                    },
                }
            )
            
            try:
                result = await asyncio.wait_for(confirmation_task, timeout=timeout)
                if result:
                    return True, ""
                else:
                    return False, "🚫 User cancelled the operation"
            except asyncio.TimeoutError:
                return False, "⏰ Operation timed out - user did not respond within the time limit"
        except Exception:
            return False, "❌ Error: Failed to show confirmation dialog"


def get_user_info_headers(valves, user: Optional[dict] = None, chat_id: Optional[str] = None) -> dict:
    """
    Build user information headers dictionary.

    Args:
        valves: Valves object containing forward_user_info_headers setting
        user: User dictionary containing 'id', 'email', 'name', 'role'
        chat_id: Current chat ID

    Returns:
        Dictionary of headers to send to OpenAI API
    """
    # Check Valves setting first
    valves_setting = valves.forward_user_info_headers.lower()

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


class Tools:
    class Valves(BaseModel):
        llm_client_type: str = Field(
            default="openai",
            description="Type of LLM client to use: 'openai' or 'generic'",
        )
        openai_api_url: str = Field(
            default="https://api.openai.com/v1",
            description="OpenAI compatible endpoint",
        )
        model: str = Field(
            default="gpt-5-mini",
            description="Model to use for memory processing",
        )
        small_model: str = Field(
            default="gpt-5-nano",
            description="Smaller model for memory processing in legacy mode",
        )
        embedding_model: str = Field(
            default="text-embedding-3-small",
            description="Model to use for embedding memories",
        )
        embedding_dim: int = Field(
            default=1536,
            description="Dimension of the embedding model",
        )
        api_key: str = Field(
            default="",
            description="API key for OpenAI compatible endpoint",
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
        
        semaphore_limit: int = Field(
            default=10,
            description="Maximum number of concurrent LLM operations",
        )
        
        group_id_format: str = Field(
            default="{user_id}",
            description="Format string for group_id. Available placeholders: {user_id}, {user_email}, {user_name}. Set to 'none' to disable group filtering.",
        )
        
        forward_user_info_headers: str = Field(
            default="default",
            description="Forward user information headers (User-Name, User-Id, User-Email, User-Role, Chat-Id) to OpenAI API. Options: 'default' (follow environment variable ENABLE_FORWARD_USER_INFO_HEADERS, defaults to false if not set), 'true' (always forward), 'false' (never forward).",
        )
        
        debug_print: bool = Field(
            default=False,
            description="Enable debug printing to console",
        )
        
        confirmation_timeout: int = Field(
            default=60,
            description="Timeout in seconds for confirmation dialogs",
        )
    
    class UserValves(BaseModel):
        message_language: str = Field(
            default="en",
            description="Language for confirmation dialog messages: 'en' (English) or 'ja' (Japanese)",
        )
        show_extracted_details: bool = Field(
            default=True,
            description="Show extracted entities and facts in status after adding memory or during migration. Set to False to reduce status message noise.",
        )
        episode_preview_length: int = Field(
            default=500,
            description="Maximum characters to show in episode content previews. Lower values reduce context usage.",
        )
        pass

    def __init__(self):
        self.valves = self.Valves()
        self.helper = GraphitiHelper(self)
        
        # Don't initialize here - Valves may not be loaded yet
        # Initialization happens lazily on first use via ensure_graphiti_initialized()

# NOTE: add_memory is commented out by default because Filter handles automatic memory saving.
# Uncomment this method if you need AI to explicitly add memories on demand (e.g., when Filter is disabled).
#
#    async def add_memory(
#        self,
#        name: str,
#        content: str,
#        source: str = "text",
#        source_description: str = "",
#        __user__: dict = {},
#        __event_emitter__: Optional[Callable[[dict], Any]] = None,
#    ) -> str:
#        """
#        Add a new memory episode to the knowledge graph.
#        
#        This tool creates a new episode and automatically extracts entities and relationships.
#        The episode is processed to identify key information and integrate it into the knowledge graph.
#        
#        :param name: Name/title of the episode (e.g., "Meeting with client", "Product launch announcement")
#        :param content: The content to store as memory. Can be text, conversation, or JSON data.
#        :param source: Type of source content. Options are:
#                      - "text": Plain text content (default) - Most reliable option
#                      - "message": Conversation-style content in "speaker: content" format.
#                                   REQUIRED FORMAT: Each line must be "speaker_name: message_content"
#                                   The speaker name will be automatically extracted as an entity.
#                                   Example: "user: Hello" or "assistant: How can I help?"
#                      - "json": Structured JSON data (must be valid JSON string)
#                                ⚠️ WARNING: Experimental feature with known compatibility issues.
#                                May fail with ValidationError depending on the LLM endpoint configuration.
#                                If you encounter errors, use "text" source type instead (works with JSON content too).
#        :param source_description: Description of where this memory came from (e.g., "team meeting notes", "customer email")
#        :return: Result message with episode details
#        
#        Examples:
#        - add_memory(name="Client Meeting", content="Discussed Q1 targets with John", source="text", source_description="meeting notes")
#        - add_memory(name="Customer Chat", content="user: What's the return policy?\nassistant: 30-day returns", source="message", source_description="support chat")
#        - add_memory(name="Product Data", content='{"product": "Widget X", "price": 99.99}', source="json", source_description="inventory system")
#        
#        Important Notes:
#        - For source="message": Content MUST follow "speaker: content" format for each line
#        - For source="json": Content must be a valid JSON string (will be validated)
#        """
#        if not await self.helper.ensure_graphiti_initialized() or self.helper.graphiti is None:
#            return "❌ Error: Memory service is not available"
#        
#        
#        # Set user headers in context variable (before any API calls)
#        headers = get_user_info_headers(self.valves, __user__, None)
#        if headers:
#            user_headers_context.set(headers)
#            if self.valves.debug_print:
#                print(f"Set user headers in context: {list(headers.keys())}")
#        
#        source_type = None  # Initialize for error handling scope
#        
#        try:
#            # Validate source type
#            source_lower = source.lower()
#            if source_lower == "text":
#                source_type = EpisodeType.text
#            elif source_lower == "message":
#                source_type = EpisodeType.message
#                # Validate message format using helper method
#                is_valid, error_msg = GraphitiHelper.validate_message_format(content)
#                if not is_valid:
#                    return error_msg if error_msg else "❌ Error: Invalid message format"
#            elif source_lower == "json":
#                source_type = EpisodeType.json
#                # Validate JSON format
#                try:
#                    json.loads(content)
#                except json.JSONDecodeError as e:
#                    return f"❌ Error: Invalid JSON content: {str(e)}\nPlease ensure the content is a valid JSON string."
#            else:
#                return f"❌ Error: Invalid source type '{source}'. Must be 'text', 'message', or 'json'"
#            
#            # Get user's group_id
#            group_id = self.helper.get_group_id(__user__)
#            if not group_id:
#                return "❌ Error: Group ID is required. Please check your group_id_format configuration."
#            
#            if self.valves.debug_print:
#                print(f"=== add_memory: Adding episode ===")
#                print(f"Name: {name}")
#                print(f"Source: {source_type}")
#                print(f"Group ID: {group_id}")
#                print(f"Content length: {len(content)} chars")
#            
#            # Add episode using Graphiti's add_episode method
#            result = await self.helper.graphiti.add_episode(
#                name=name,
#                episode_body=content,
#                source=source_type,
#                source_description=source_description,
#                reference_time=datetime.now(timezone.utc),
#                group_id=group_id,
#            )
#            
#            if self.valves.debug_print:
#                print(f"=== Episode added successfully ===")
#                print(f"Episode UUID: {result.episode.uuid}")
#                print(f"Extracted {len(result.nodes)} entities")
#                print(f"Extracted {len(result.edges)} relationships")
#            
#            # Get user's language and display preferences
#            user_valves = self.UserValves.model_validate(
#                (__user__ or {}).get("valves", {})
#            )
#            is_ja = user_valves.message_language == "ja"
#
#            # Display extracted entities and facts in status (if enabled)
#            if __event_emitter__ and result and user_valves.show_extracted_details:
#                # Show extracted facts
#                if result.edges:
#                    for edge_idx, edge in enumerate(result.edges, 1):
#                        emoji = "🔚" if edge.invalid_at else "🔛"
#                        label = "ファクト" if is_ja else "Fact"
#                        await __event_emitter__({
#                            "type": "status",
#                            "data": {
#                                "description": f"{emoji} {label} {edge_idx}/{len(result.edges)}: {edge.fact}",
#                                "done": False
#                            }
#                        })
#
#                # Show extracted entities
#                if result.nodes:
#                    for node_idx, node in enumerate(result.nodes, 1):
#                        entity_display = f"{node.name}"
#                        if hasattr(node, 'summary') and node.summary:
#                            entity_display += f" - {node.summary}"
#                        label = "エンティティ" if is_ja else "Entity"
#                        await __event_emitter__({
#                            "type": "status",
#                            "data": {
#                                "description": f"👤 {label} {node_idx}/{len(result.nodes)}: {entity_display}",
#                                "done": False
#                            }
#                        })
#
#            # Final status
#            if __event_emitter__:
#                complete_msg = "✅ メモリ追加完了" if is_ja else "✅ Memory added"
#                await __event_emitter__({
#                    "type": "status",
#                    "data": {
#                        "description": complete_msg,
#                        "done": True
#                    }
#                })
#
#            # Build response message
#            response = f"✅ Memory added successfully!\n\n"
#            response += f"**Episode:** {name}\n"
#            response += f"**UUID:** `{result.episode.uuid}`\n"
#            response += f"**Source Type:** {source}\n"
#            
#            if result.nodes:
#                response += f"\n**Extracted Entities ({len(result.nodes)}):**\n"
#                for i, node in enumerate(result.nodes[:5], 1):  # Show first 5
#                    response += f"{i}. {node.name}\n"
#                if len(result.nodes) > 5:
#                    response += f"   ... and {len(result.nodes) - 5} more\n"
#            
#            if result.edges:
#                response += f"\n**Extracted Relationships ({len(result.edges)}):**\n"
#                for i, edge in enumerate(result.edges[:5], 1):  # Show first 5
#                    response += f"{i}. {edge.name}: {edge.fact}\n"
#                if len(result.edges) > 5:
#                    response += f"   ... and {len(result.edges) - 5} more\n"
#            
#            return response
#            
#        except Exception as e:
#            error_type = type(e).__name__
#            error_str = str(e)
#            
#            # Provide more specific error messages for common issues
#            if "ValidationError" in error_type or "validation error" in error_str.lower():
#                if source_type == EpisodeType.message:
#                    error_msg = f"❌ Error: LLM validation failed for message format.\n\n"
#                    error_msg += f"The content does not follow the required 'speaker: content' format.\n"
#                    error_msg += f"Each line must be formatted as: 'speaker_name: message_content'\n\n"
#                    error_msg += f"Valid format examples:\n"
#                    error_msg += f"  user: Hello, how are you?\n"
#                    error_msg += f"  assistant: I'm doing well, thank you!\n\n"
#                    error_msg += f"Technical details: {error_str}"
#                elif source_type == EpisodeType.json:
#                    error_msg = f"❌ Error: JSON type processing failed.\n\n"
#                    error_msg += f"Graphiti's JSON extraction encountered a validation error.\n"
#                    error_msg += f"This may be due to endpoint compatibility or JSON structure complexity.\n\n"
#                    error_msg += f"✅ Possible solutions:\n"
#                    error_msg += f"  1. USE TEXT TYPE: source='text' is more reliable across endpoints\n"
#                    error_msg += f"     (Still extracts entities and relationships from JSON content)\n\n"
#                    error_msg += f"  2. SIMPLIFY JSON: Flatten nested structures or reduce field count\n\n"
#                    error_msg += f"  3. CONFIGURATION: The administrator may need to adjust llm_client_type in Valves\n\n"
#                    if self.valves.debug_print:
#                        error_msg += f"\nTechnical details: {error_str}"
#                else:
#                    error_msg = f"❌ Error: LLM validation failed.\n\n"
#                    error_msg += f"The LLM response format was unexpected.\n"
#                    error_msg += f"This may indicate endpoint compatibility issues.\n\n"
#                    error_msg += f"The administrator may need to adjust llm_client_type or endpoint configuration in Valves.\n"
#                    if self.valves.debug_print:
#                        error_msg += f"\n\nTechnical details: {error_str}"
#            elif "ConnectionError" in error_type or "timeout" in error_str.lower():
#                error_msg = f"❌ Error: Connection error to LLM service.\n"
#                error_msg += f"Network connection or endpoint configuration issue.\n"
#                if self.valves.debug_print:
#                    error_msg += f"\nDetails: {error_str}"
#            elif "api_key" in error_str.lower() or "authentication" in error_str.lower() or "unauthorized" in error_str.lower():
#                error_msg = f"❌ Error: Authentication failed.\n"
#                error_msg += f"API key configuration issue in Valves.\n"
#                if self.valves.debug_print:
#                    error_msg += f"\nDetails: {error_str}"
#            else:
#                error_msg = f"❌ Error adding memory: {error_str}\n\n"
#                error_msg += f"Possible solutions:\n"
#                error_msg += f"  - Using 'text' source type may work better\n"
#                error_msg += f"  - The administrator may need to verify endpoint configuration in Valves\n"
#                if self.valves.debug_print:
#                    error_msg += f"\n\nNote: Enable debug_print in Valves for detailed error information."
#            
#            if self.valves.debug_print:
#                print(f"Exception type: {error_type}")
#                traceback.print_exc()
#            
#            return error_msg

    async def migrate_builtin_memories(
        self,
        dry_run: bool = False,
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
        __event_call__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Migrate Open WebUI built-in memories to Graphiti knowledge graph.

        This method safely migrates all built-in memories for the current user to Graphiti episodes.
        The migration is idempotent - running it multiple times will not create duplicates.

        Migration Strategy:
        - Each built-in memory is converted to a Graphiti episode
        - Source type: EpisodeType.text (built-in memories are text-based)
        - Episode name: {timestamp}_Migrated_{memory_id} format (e.g., "20241225T093000Z_Migrated_uuid")
        - Timestamp: Preserved from built-in memory's created_at in UTC
        - Idempotency marker: source_description = "migrated_builtin_memory:{memory.id}"

        Parameters:
        - dry_run: If True, preview what would be migrated without making changes
                   Confirmation dialog is skipped in dry-run mode
                   Default: False

        Returns:
        - str: Migration summary report with statistics

        Examples:
        - migrate_builtin_memories() → Migrate all memories (with confirmation)
        - migrate_builtin_memories(dry_run=True) → Preview migration without changes

        Safety Features:
        - Idempotent: Safe to run multiple times (checks source_description)
        - Graceful failure: Individual failures don't stop the migration
        - Confirmation dialog: Requires user approval when dry_run=False
        - Detailed reporting: Shows success, failures, and already migrated counts
        - Preserves timestamps: Maintains original creation time in UTC

        Note: Built-in memories are NOT deleted after migration. Users can manually
              delete them from Open WebUI UI (Settings → Personalization → Memory).
        """

        # === Phase 1: Initialization & Validation ===

        if self.valves.debug_print:
            print("=== migrate_builtin_memories: Starting ===")

        # 1. Ensure Graphiti is initialized
        if not await self.helper.ensure_graphiti_initialized() or self.helper.graphiti is None:
            return "❌ Error: Memory service is not available. Please check Graphiti configuration."

        # 2. Set user headers for API calls
        headers = get_user_info_headers(self.valves, __user__, None)
        if headers:
            user_headers_context.set(headers)
            if self.valves.debug_print:
                print(f"Set user headers: {list(headers.keys())}")

        # 3. Get group_id for episode filtering
        group_id = self.helper.get_group_id(__user__)
        if not group_id:
            return "❌ Error: Group ID required. Check group_id_format configuration."

        # 4. Get user_id for Memories query
        user_id = __user__.get("id")
        if not user_id:
            return "❌ Error: User ID required for migration."

        if self.valves.debug_print:
            print(f"User ID: {user_id}")
            print(f"Group ID: {group_id}")

        # === Phase 2: Fetch Built-in Memories ===

        # Import Memories module
        try:
            from open_webui.models.memories import Memories
        except Exception as e:
            return f"❌ Error: Cannot import Memories module: {e}\nPlease ensure you are running in Open WebUI environment."

        # Fetch all built-in memories for user
        try:
            builtin_memories = Memories.get_memories_by_user_id(user_id)
            if not builtin_memories:
                return "ℹ️ No built-in memories found for this user."

            if self.valves.debug_print:
                print(f"Found {len(builtin_memories)} built-in memories")
        except Exception as e:
            error_msg = f"❌ Error: Failed to fetch built-in memories: {e}"
            if self.valves.debug_print:
                traceback.print_exc()
            return error_msg

        # === Phase 3: Check Existing Migrations (Idempotency) ===

        # Fetch all existing Graphiti episodes for this group
        try:
            existing_episodes = await EpisodicNode.get_by_group_ids(
                self.helper.graphiti.driver,
                [group_id]
            )

            if self.valves.debug_print:
                print(f"Found {len(existing_episodes)} existing Graphiti episodes")
        except Exception as e:
            error_msg = f"❌ Error: Failed to fetch existing episodes: {e}"
            if self.valves.debug_print:
                traceback.print_exc()
            return error_msg

        # Extract already-migrated memory IDs
        migrated_memory_ids = set()
        for episode in existing_episodes:
            source_desc = getattr(episode, 'source_description', '')
            if source_desc.startswith('migrated_builtin_memory:'):
                memory_id = source_desc.split(':', 1)[1]
                migrated_memory_ids.add(memory_id)

        if self.valves.debug_print:
            print(f"Already migrated: {len(migrated_memory_ids)} memories")

        # Filter to unmigrated memories only and sort by created_at (oldest first)
        memories_to_migrate = sorted(
            [mem for mem in builtin_memories if mem.id not in migrated_memory_ids],
            key=lambda m: m.created_at
        )

        if not memories_to_migrate:
            total = len(builtin_memories)
            return f"✅ All {total} built-in memories have already been migrated."

        if self.valves.debug_print:
            print(f"To migrate: {len(memories_to_migrate)} memories")

        # === Phase 4: Confirmation Dialog (if dry_run=False) ===

        if not dry_run:
            # Get user's language preference
            user_valves = self.UserValves.model_validate(
                (__user__ or {}).get("valves", {})
            )
            is_ja = user_valves.message_language == "ja"

            # Create confirmation dialog content
            if is_ja:
                title = "メモリ移行の確認"
                items = [
                    f"移行対象: {len(memories_to_migrate)} 件のビルトインメモリ",
                    f"移行先グループ: {group_id}",
                    f"既に移行済み: {len(migrated_memory_ids)} 件",
                ]
                warning = "ℹ️ ビルトインメモリは削除されません。移行後、必要に応じて手動で削除してください。"
            else:
                title = "Confirm Memory Migration"
                items = [
                    f"Memories to migrate: {len(memories_to_migrate)} built-in memories",
                    f"Target group: {group_id}",
                    f"Already migrated: {len(migrated_memory_ids)} memories",
                ]
                warning = "ℹ️ Built-in memories will NOT be deleted. You can manually delete them after migration if needed."

            # Show confirmation dialog
            try:
                confirmed, error_msg = await self.helper.show_confirmation_dialog(
                    title=title,
                    items=items,
                    warning_message=warning,
                    timeout=self.valves.confirmation_timeout,
                    __user__=__user__,
                    __event_call__=__event_call__,
                )

                if not confirmed:
                    return error_msg if error_msg else "❌ Migration cancelled by user."
            except Exception as e:
                error_msg = f"❌ Confirmation dialog error: {e}"
                if self.valves.debug_print:
                    traceback.print_exc()
                return error_msg

        # === Phase 5: Migration Execution ===

        # Get user's language preference for progress messages
        user_valves = self.UserValves.model_validate(
            (__user__ or {}).get("valves", {})
        )
        is_ja = user_valves.message_language == "ja"

        migrated_count = 0
        failed_count = 0
        failed_memories = []

        for idx, memory in enumerate(memories_to_migrate, 1):
            try:
                # Format memory info for status display
                memory_date = datetime.fromtimestamp(memory.created_at, tz=timezone.utc).strftime("%Y-%m-%d")
                content_preview = memory.content or "(empty)"

                # Progress notification
                if __event_emitter__:
                    if dry_run:
                        progress_msg = f"🔍 [ドライラン] メモリを確認中 {idx}/{len(memories_to_migrate)}: [{memory_date}] {content_preview}" if is_ja else f"🔍 [Dry Run] Checking memory {idx}/{len(memories_to_migrate)}: [{memory_date}] {content_preview}"
                    else:
                        progress_msg = f"🔄 メモリを移行中 {idx}/{len(memories_to_migrate)}: [{memory_date}] {content_preview}" if is_ja else f"🔄 Migrating memory {idx}/{len(memories_to_migrate)}: [{memory_date}] {content_preview}"
                    await __event_emitter__({
                        "type": "status",
                        "data": {
                            "description": progress_msg,
                            "done": False
                        }
                    })

                if dry_run:
                    # Dry-run: skip actual migration
                    continue

                # Generate episode name with timestamp prefix (consistent with Filter format)
                timestamp_prefix = datetime.fromtimestamp(memory.created_at, tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                episode_name = f"{timestamp_prefix}_Migrated_{memory.id}"

                # Create episode with idempotency marker
                result = await self.helper.graphiti.add_episode(
                    name=episode_name,
                    episode_body=memory.content,
                    source=EpisodeType.text,
                    source_description=f"migrated_builtin_memory:{memory.id}",
                    reference_time=datetime.fromtimestamp(
                        memory.created_at,
                        tz=timezone.utc
                    ),
                    group_id=group_id,
                )

                migrated_count += 1

                if self.valves.debug_print:
                    print(f"✓ Migrated memory {memory.id} -> episode {result.episode.uuid}")

                # Display extracted entities and facts in status (if enabled)
                if __event_emitter__ and result and user_valves.show_extracted_details:
                    # Show extracted facts
                    if result.edges:
                        for edge_idx, edge in enumerate(result.edges, 1):
                            emoji = "🔚" if edge.invalid_at else "🔛"
                            label = "ファクト" if is_ja else "Fact"
                            await __event_emitter__({
                                "type": "status",
                                "data": {
                                    "description": f"{emoji} {label} {edge_idx}/{len(result.edges)}: {edge.fact}",
                                    "done": False
                                }
                            })

                    # Show extracted entities
                    if result.nodes:
                        for node_idx, node in enumerate(result.nodes, 1):
                            entity_display = f"{node.name}"
                            if hasattr(node, 'summary') and node.summary:
                                entity_display += f" - {node.summary}"
                            label = "エンティティ" if is_ja else "Entity"
                            await __event_emitter__({
                                "type": "status",
                                "data": {
                                    "description": f"👤 {label} {node_idx}/{len(result.nodes)}: {entity_display}",
                                    "done": False
                                }
                            })

            except asyncio.CancelledError:
                # User pressed stop button - cancel migration gracefully
                # Already migrated memories are saved in Graphiti and will be skipped on retry (idempotent)
                if __event_emitter__:
                    cancel_msg = f"⏹️ 移行が中断されました ({migrated_count}/{len(memories_to_migrate)} 完了)" if is_ja else f"⏹️ Migration interrupted ({migrated_count}/{len(memories_to_migrate)} completed)"
                    await __event_emitter__({
                        "type": "status",
                        "data": {
                            "description": cancel_msg,
                            "done": True
                        }
                    })
                if self.valves.debug_print:
                    print(f"⚠ Migration cancelled by user. Progress: {migrated_count}/{len(memories_to_migrate)} migrated")
                # Re-raise to let Open WebUI middleware handle the cancellation properly
                raise

            except Exception as e:
                failed_count += 1
                error_str = str(e)
                content_preview = memory.content[:50] if memory.content else "(empty)"
                failed_memories.append({
                    "id": memory.id,
                    "error": error_str,
                    "content_preview": content_preview
                })

                # Show error status notification
                if __event_emitter__:
                    error_msg = f"⚠️ 移行失敗 ({idx}/{len(memories_to_migrate)}): {content_preview}..." if is_ja else f"⚠️ Migration failed ({idx}/{len(memories_to_migrate)}): {content_preview}..."
                    await __event_emitter__({
                        "type": "status",
                        "data": {
                            "description": error_msg,
                            "done": False
                        }
                    })

                if self.valves.debug_print:
                    print(f"✗ Failed to migrate memory {memory.id}: {e}")
                    traceback.print_exc()

                # Continue with remaining memories
                continue

        # Final progress notification
        if __event_emitter__:
            complete_msg = "✅ 移行完了" if is_ja else "✅ Migration complete"
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": complete_msg,
                    "done": True
                }
            })

        # === Phase 6: Result Report ===

        # Build result summary
        result_lines = []

        if dry_run:
            result_lines.append("🔍 **Dry Run Mode** (No actual changes made)")
            result_lines.append("")

        result_lines.append("# Migration Summary")
        result_lines.append("")
        result_lines.append(f"**Total built-in memories:** {len(builtin_memories)}")
        result_lines.append(f"**Already migrated:** {len(migrated_memory_ids)}")
        result_lines.append(f"**To migrate:** {len(memories_to_migrate)}")
        result_lines.append("")

        if not dry_run:
            result_lines.append(f"**Successfully migrated:** {migrated_count}")
            if failed_count > 0:
                result_lines.append(f"**Failed:** {failed_count}")
        else:
            result_lines.append(f"**Would migrate:** {len(memories_to_migrate)} memories")

        if failed_memories and not dry_run:
            result_lines.append("")
            result_lines.append("## Failed Migrations")
            for failure in failed_memories[:5]:  # Show first 5 failures
                result_lines.append(f"- **ID:** {failure['id']}")
                result_lines.append(f"  - Content: {failure['content_preview']}...")
                result_lines.append(f"  - Error: {failure['error']}")

            if len(failed_memories) > 5:
                result_lines.append(f"- ... and {len(failed_memories) - 5} more failures")

        if self.valves.debug_print:
            print("=== migrate_builtin_memories: Complete ===")

        return "\n".join(result_lines)

    async def search_entities(
        self,
        query: str,
        limit: int = 10,
        show_uuid: bool = False,
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Search for entities by name or description without deleting them.
        
        This tool allows you to preview entities before deciding to delete them.
        Use this to verify what will be deleted before calling search_and_delete_entities.
        
        :param query: Search query to find entities (e.g., "John Smith", "Python programming")
        :param limit: Maximum number of entities to return (default: 10, max: 100)
        :param show_uuid: Whether to display UUID in search results (default: False). Set to True if you need to see UUIDs for debugging or manual deletion.
        :return: List of found entities with their details
        """
        
        if not await self.helper.ensure_graphiti_initialized() or self.helper.graphiti is None:
            return "❌ Error: Memory service is not available"
        
        
        # Set user headers in context variable (before any API calls)
        headers = get_user_info_headers(self.valves, __user__, None)
        if headers:
            user_headers_context.set(headers)
            if self.valves.debug_print:
                print(f"Set user headers in context: {list(headers.keys())}")
        # Validate and clamp limit
        limit = max(1, min(100, limit))
        
        try:
            group_id = self.helper.get_group_id(__user__)
            
            # Create a copy of config with custom limit
            search_config = copy.copy(COMBINED_HYBRID_SEARCH_RRF)
            search_config.limit = limit
            
            # Search for entities
            search_results = await self.helper.graphiti.search_(
                query=query,
                group_ids=[group_id] if group_id else None,
                config=search_config,
            )
            
            # Extract entity nodes
            entity_nodes = [node for node in search_results.nodes if hasattr(node, 'name')]
            
            if not entity_nodes:
                return f"ℹ️ No entities found matching '{query}'"
            
            total_count = len(entity_nodes)
            
            # Build result message
            result = f"🔍 Found {total_count} entities matching '{query}':\n\n"
            
            for i, node in enumerate(entity_nodes, 1):
                name = getattr(node, 'name', 'Unknown')
                summary = getattr(node, 'summary', 'No description')
                uuid = getattr(node, 'uuid', 'N/A')
                
                result += f"**{i}. {name}**\n"
                result += f"   Summary: {summary}\n"
                if show_uuid:
                    result += f"   UUID: `{uuid}`\n"
                result += "\n"
            
            result += f"💡 To delete these entities, use `search_and_delete_entities` with the same query and limit."
            
            return result
            
        except Exception as e:
            error_msg = f"❌ Error searching entities: {str(e)}"
            if self.valves.debug_print:
                traceback.print_exc()
            return error_msg
    
    async def search_facts(
        self,
        query: str,
        limit: int = 10,
        show_uuid: bool = False,
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Search for facts (relationships) without deleting them.
        
        This tool allows you to preview relationships before deciding to delete them.
        Use this to verify what will be deleted before calling search_and_delete_facts.
        
        :param query: Search query to find relationships (e.g., "works at", "friends with")
        :param limit: Maximum number of facts to return (default: 10, max: 100)
        :param show_uuid: Whether to display UUID in search results (default: False). Set to True if you need to see UUIDs for debugging or manual deletion.
        :return: List of found facts with their details
        """
        
        if not await self.helper.ensure_graphiti_initialized() or self.helper.graphiti is None:
            return "❌ Error: Memory service is not available"
        
        
        # Set user headers in context variable (before any API calls)
        headers = get_user_info_headers(self.valves, __user__, None)
        if headers:
            user_headers_context.set(headers)
            if self.valves.debug_print:
                print(f"Set user headers in context: {list(headers.keys())}")
        # Validate and clamp limit
        limit = max(1, min(100, limit))
        
        try:
            group_id = self.helper.get_group_id(__user__)
            
            # Create a copy of config with custom limit
            search_config = copy.copy(COMBINED_HYBRID_SEARCH_RRF)
            search_config.limit = limit
            
            # Search for facts
            search_results = await self.helper.graphiti.search_(
                query=query,
                group_ids=[group_id] if group_id else None,
                config=search_config,
            )
            
            # Extract edges
            edges = search_results.edges
            
            if not edges:
                return f"ℹ️ No facts found matching '{query}'"
            
            total_count = len(edges)
            
            # Build result message
            result = f"🔍 Found {total_count} facts matching '{query}':\n\n"
            
            for i, edge in enumerate(edges, 1):
                fact_text = getattr(edge, 'fact', 'Unknown relationship')
                valid_at = getattr(edge, 'valid_at', 'unknown')
                invalid_at = getattr(edge, 'invalid_at', 'present')
                uuid = getattr(edge, 'uuid', 'N/A')
                
                result += f"**{i}. {fact_text}**\n"
                result += f"   Period: {valid_at} → {invalid_at}\n"
                if show_uuid:
                    result += f"   UUID: `{uuid}`\n"
                result += "\n"
            
            result += f"💡 To delete these facts, use `search_and_delete_facts` with the same query and limit."
            
            return result
            
        except Exception as e:
            error_msg = f"❌ Error searching facts: {str(e)}"
            if self.valves.debug_print:
                traceback.print_exc()
            return error_msg
    
    async def search_episodes(
        self,
        query: str,
        limit: int = 10,
        show_uuid: bool = True,
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Search for episodes (conversation history) without deleting them.

        This tool allows you to preview episodes before deciding to delete them.
        Use this to verify what will be deleted before calling search_and_delete_episodes.

        :param query: Search query to find episodes (e.g., "conversation about Python")
        :param limit: Maximum number of episodes to return (default: 10, max: 100)
        :param show_uuid: Whether to display UUID in search results (default: True). UUIDs can be used with get_episode_content to retrieve full content.
        :return: List of found episodes with their details
        """

        if not await self.helper.ensure_graphiti_initialized() or self.helper.graphiti is None:
            return "❌ Error: Memory service is not available"


        # Set user headers in context variable (before any API calls)
        headers = get_user_info_headers(self.valves, __user__, None)
        if headers:
            user_headers_context.set(headers)
            if self.valves.debug_print:
                print(f"Set user headers in context: {list(headers.keys())}")
        # Validate and clamp limit
        limit = max(1, min(100, limit))

        try:
            group_id = self.helper.get_group_id(__user__)

            # Create a copy of config with custom limit
            search_config = copy.copy(COMBINED_HYBRID_SEARCH_RRF)
            search_config.limit = limit

            # Search for episodes
            search_results = await self.helper.graphiti.search_(
                query=query,
                group_ids=[group_id] if group_id else None,
                config=search_config,
            )

            # Extract episodes
            episodes = search_results.episodes

            if not episodes:
                return f"ℹ️ No episodes found matching '{query}'"

            total_count = len(episodes)

            # Get user's preview length preference
            user_valves = self.UserValves.model_validate(
                (__user__ or {}).get("valves", {})
            )
            preview_length = user_valves.episode_preview_length

            # Build result message
            result = f"🔍 Found {total_count} episodes matching '{query}':\n\n"

            for i, episode in enumerate(episodes, 1):
                name = getattr(episode, 'name', 'Unknown episode')
                content = getattr(episode, 'content', '')
                created_at = getattr(episode, 'created_at', 'unknown')
                uuid = getattr(episode, 'uuid', 'N/A')

                # Truncate content for preview
                total_chars = len(content)
                if total_chars > preview_length:
                    content_preview = content[:preview_length] + "..."
                    truncation_info = f"(showing {preview_length}/{total_chars} chars)"
                else:
                    content_preview = content
                    truncation_info = None

                result += f"**{i}. {name}**\n"
                result += f"   Content: {content_preview}\n"
                if truncation_info:
                    result += f"   {truncation_info}\n"
                result += f"   Created: {created_at}\n"
                if show_uuid:
                    result += f"   UUID: `{uuid}`\n"
                result += "\n"

            result += "💡 Previews may be truncated. Use `get_episode_content(uuid=\"...\")` to retrieve full content before answering if details are needed.\n\n"
            result += f"To delete these episodes, use `search_and_delete_episodes` with the same query and limit."

            return result

        except Exception as e:
            error_msg = f"❌ Error searching episodes: {str(e)}"
            if self.valves.debug_print:
                traceback.print_exc()
            return error_msg

    async def get_recent_episodes(
        self,
        limit: int = 10,
        offset: int = 0,
        show_uuid: bool = True,
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Get recent episodes in chronological order (oldest first).

        This tool retrieves episodes sorted by time without requiring a search query.
        Useful for viewing your conversation history or recent memories in the order they occurred.

        :param limit: Maximum number of episodes to return (default: 10, max: 100)
        :param offset: Number of episodes to skip for pagination (default: 0)
        :param show_uuid: Whether to display UUID in results (default: True). UUIDs can be used with get_episode_content to retrieve full content.
        :return: List of recent episodes with their details

        Examples:
        - get_recent_episodes(limit=20) - Get the 20 most recent episodes
        - get_recent_episodes(limit=10, offset=10) - Get episodes 11-20 (pagination)
        - get_recent_episodes(limit=5, show_uuid=True) - Get 5 recent episodes with UUIDs
        """

        if not await self.helper.ensure_graphiti_initialized() or self.helper.graphiti is None:
            return "❌ Error: Memory service is not available"

        # Set user headers in context variable (before any API calls)
        headers = get_user_info_headers(self.valves, __user__, None)
        if headers:
            user_headers_context.set(headers)
            if self.valves.debug_print:
                print(f"Set user headers in context: {list(headers.keys())}")

        # Validate and clamp parameters
        limit = max(1, min(100, limit))
        offset = max(0, offset)

        try:
            group_id = self.helper.get_group_id(__user__)

            if self.valves.debug_print:
                print(f"=== get_recent_episodes: Fetching episodes ===")
                print(f"Group ID: {group_id}")
                print(f"Limit: {limit}, Offset: {offset}")

            # Retrieve episodes via Graphiti helper (chronological order, oldest→newest)
            # +1 to detect if there are more
            fetch_count = offset + limit + 1
            episodes_list = await self.helper.graphiti.retrieve_episodes(
                reference_time=datetime.now(timezone.utc),
                last_n=fetch_count,
                group_ids=[group_id] if group_id else None,
            )

            if not episodes_list:
                return "ℹ️ No episodes found"

            total_fetched = len(episodes_list)
            has_more_in_db = total_fetched > (offset + limit)

            # episodes_list is in chronological order.
            # The end side is "newer", so extract offset/limit based on the end.
            start = max(0, total_fetched - (offset + limit))
            end = total_fetched - offset
            episodes = episodes_list[start:end]

            if not episodes:
                return f"ℹ️ No episodes found at offset {offset}"

            # Get user's preview length preference
            user_valves = self.UserValves.model_validate(
                (__user__ or {}).get("valves", {})
            )
            preview_length = user_valves.episode_preview_length

            # Build result message
            result = f"📅 Recent episodes ({len(episodes)}"
            if has_more_in_db:
                result += ", more may be available"
            if offset > 0:
                result += f", starting from #{offset + 1}"
            result += "):\n\n"

            for i, episode in enumerate(episodes, 1):
                name = getattr(episode, 'name', 'Unknown episode')
                content = getattr(episode, 'content', '')
                valid_at = getattr(episode, 'valid_at', 'unknown')
                source = getattr(episode, 'source', 'unknown')
                uuid = getattr(episode, 'uuid', 'N/A')

                # Truncate content for preview
                total_chars = len(content)
                if total_chars > preview_length:
                    content_preview = content[:preview_length] + "..."
                    truncation_info = f"(showing {preview_length}/{total_chars} chars)"
                else:
                    content_preview = content
                    truncation_info = None

                # Calculate actual position in full list
                position = offset + i

                result += f"**{position}. {name}**\n"
                result += f"   Content: {content_preview}\n"
                if truncation_info:
                    result += f"   {truncation_info}\n"
                result += f"   Time: {valid_at}\n"
                result += f"   Source: {source}\n"
                if show_uuid:
                    result += f"   UUID: `{uuid}`\n"
                result += "\n"

            result += "💡 Previews may be truncated. Use `get_episode_content(uuid=\"...\")` to retrieve full content before answering if details are needed.\n"

            # Add pagination hints
            if has_more_in_db:
                result += f"\n📄 More episodes may be available. Use `offset={offset + len(episodes)}` to see the next page."

            return result

        except Exception as e:
            error_msg = f"❌ Error retrieving recent episodes: {str(e)}"
            if self.valves.debug_print:
                traceback.print_exc()
            return error_msg

    async def get_episode_content(
        self,
        uuid: str,
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Get the full content of a specific episode by UUID.

        Unlike search_episodes or get_recent_episodes which show truncated previews,
        this method returns the complete, untruncated content of an episode.

        :param uuid: The UUID of the episode to retrieve
        :return: Full episode content with metadata
        """

        if not await self.helper.ensure_graphiti_initialized() or self.helper.graphiti is None:
            return "❌ Error: Memory service is not available"

        # Set user headers in context variable (before any API calls)
        headers = get_user_info_headers(self.valves, __user__, None)
        if headers:
            user_headers_context.set(headers)
            if self.valves.debug_print:
                print(f"Set user headers in context: {list(headers.keys())}")

        if not uuid or not uuid.strip():
            return "❌ Error: UUID is required"

        uuid = uuid.strip()

        try:
            if self.valves.debug_print:
                print(f"=== get_episode_content: Fetching episode ===")
                print(f"UUID: {uuid}")

            # Fetch episode by UUID
            episodes = await EpisodicNode.get_by_uuids(self.helper.graphiti.driver, [uuid])

            if not episodes:
                return f"❌ Episode not found with UUID: {uuid}"

            episode = episodes[0]

            # Get episode attributes
            name = getattr(episode, 'name', 'Unknown episode')
            content = getattr(episode, 'content', '')
            source = getattr(episode, 'source', 'unknown')
            source_description = getattr(episode, 'source_description', '')
            valid_at = getattr(episode, 'valid_at', 'unknown')
            created_at = getattr(episode, 'created_at', 'unknown')

            # Build result message with full content
            result = f"📄 **Episode: {name}**\n\n"
            result += f"**UUID:** `{uuid}`\n"
            result += f"**Source:** {source}\n"
            if source_description:
                result += f"**Source Description:** {source_description}\n"
            result += f"**Valid At:** {valid_at}\n"
            result += f"**Created At:** {created_at}\n\n"
            result += f"**Full Content:**\n```\n{content}\n```"

            return result

        except Exception as e:
            error_msg = f"❌ Error retrieving episode content: {str(e)}"
            if self.valves.debug_print:
                traceback.print_exc()
            return error_msg

    async def delete_episodes_by_time_range(
        self,
        start_time: str,
        end_time: str,
        time_field: str = "created_at",
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
        __event_call__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Delete episodes within a specified time range after user confirmation.

        This tool deletes episodes whose timestamp falls within the specified time range.

        IMPORTANT NOTES FOR AI:
        - You MUST ask the user for specific date/time values, unless you have access to
          accurate current time information (e.g., from system prompts or time tools).
          Do NOT attempt to guess or calculate relative times without a reliable time source.
        - Example user request: "Delete episodes created between 2024-01-01 and 2024-01-31"

        - Time format: ISO 8601 strings with timezone offset
        - ⚠️ CRITICAL: Understand timezone offsets correctly!

          ❌ WRONG: "2024-01-01T00:00:00+09:00" is NOT midnight UTC!
                    It means midnight in JST (= 2023-12-31 15:00 UTC)

          ✅ Examples:
          - "2024-01-01T00:00:00Z" = Jan 1 midnight UTC
          - "2024-01-01T00:00:00+09:00" = Jan 1 midnight JST (= Dec 31 15:00 UTC)
          - "2024-01-01T09:00:00+09:00" = Jan 1 9AM JST (= Jan 1 00:00 UTC)

        - Prefer using the user's local timezone if known (e.g., +09:00 for Japan)
          This makes times more intuitive for users
        - Range is start-inclusive, end-exclusive: [start_time, end_time)

        :param start_time: Start of time range (ISO 8601 format with timezone). Episodes with timestamp >= this will be included.
                           Examples: "2024-01-01T00:00:00Z", "2024-01-01T09:00:00+09:00"
        :param end_time: End of time range (ISO 8601 format with timezone). Episodes with timestamp < this will be included.
                         Examples: "2024-02-01T00:00:00Z", "2024-02-01T09:00:00+09:00"
        :param time_field: (Optional) Timestamp field to filter by: "created_at" (default) or "valid_at".
                           Usually not needed - only specify if explicitly required for special cases.
                           Do NOT ask users about this parameter unless they specifically mention it.
        :return: Result message with deletion status

        Example:
        - delete_episodes_by_time_range(start_time="2024-01-01T00:00:00Z", end_time="2024-02-01T00:00:00Z")
          → Deletes all episodes created in January 2024
        """

        if not await self.helper.ensure_graphiti_initialized() or self.helper.graphiti is None:
            return "❌ Error: Memory service is not available"

        # Set user headers in context variable (before any API calls)
        headers = get_user_info_headers(self.valves, __user__, None)
        if headers:
            user_headers_context.set(headers)
            if self.valves.debug_print:
                print(f"Set user headers in context: {list(headers.keys())}")

        try:
            # 1. Parameter Validation
            # Validate time_field
            if time_field not in ["created_at", "valid_at"]:
                return f"❌ Error: time_field must be 'created_at' or 'valid_at'. Got: '{time_field}'"

            # Parse ISO 8601 strings to datetime
            try:
                # Replace 'Z' with '+00:00' for ISO format compatibility
                start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            except ValueError as e:
                return (
                    f"❌ Error: Invalid time format. Please use ISO 8601 format.\n"
                    f"Examples: '2024-01-01T00:00:00Z', '2024-01-01T00:00:00+00:00'\n"
                    f"Error details: {str(e)}"
                )

            # Ensure timezone-aware (assume UTC if missing)
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=timezone.utc)
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=timezone.utc)

            # Validate range
            if start_dt >= end_dt:
                return (
                    f"❌ Error: start_time must be before end_time.\n"
                    f"start_time: {start_time}\n"
                    f"end_time: {end_time}"
                )

            # 2. Fetch and Filter Episodes
            # Get group_id
            group_id = self.helper.get_group_id(__user__)
            if not group_id:
                return "❌ Error: Group ID is required. Please check your group_id_format configuration."

            if self.valves.debug_print:
                print(f"=== delete_episodes_by_time_range: Fetching episodes ===")
                print(f"Group ID: {group_id}")
                print(f"Time range: {start_time} to {end_time}")
                print(f"Time field: {time_field}")

            # Fetch all episodes for user's group
            episodes = await EpisodicNode.get_by_group_ids(
                self.helper.graphiti.driver,
                [group_id]
            )

            if self.valves.debug_print:
                print(f"Total episodes for group: {len(episodes)}")

            # Filter by time range [start_time, end_time)
            matched_episodes = []
            for ep in episodes:
                ep_time = getattr(ep, time_field, None)
                if ep_time is None:
                    continue

                # Ensure episode time is timezone-aware
                if ep_time.tzinfo is None:
                    ep_time = ep_time.replace(tzinfo=timezone.utc)

                if start_dt <= ep_time < end_dt:
                    matched_episodes.append(ep)

            if self.valves.debug_print:
                print(f"Matched episodes in time range: {len(matched_episodes)}")

            # Handle no matches
            if not matched_episodes:
                return f"ℹ️ No episodes found in time range {start_time} to {end_time} (filtered by {time_field})"

            # 3. Confirmation Dialog
            # Get user language preference
            is_japanese = self.helper.is_japanese_preferred(__user__)

            # Calculate statistics for summary
            # Normalize timezones to avoid comparison errors with legacy data
            ep_times = []
            for ep in matched_episodes:
                ep_time = getattr(ep, time_field, None)
                if ep_time is not None:
                    # Ensure timezone-aware (normalize to UTC if naive)
                    if ep_time.tzinfo is None:
                        ep_time = ep_time.replace(tzinfo=timezone.utc)
                    ep_times.append(ep_time)
            oldest_time = min(ep_times) if ep_times else None
            newest_time = max(ep_times) if ep_times else None

            # Get display timezone from user's input (start_time timezone)
            # This makes the dialog display times in the same timezone the user specified
            display_tz = start_dt.tzinfo

            # Convert times to display timezone for consistency
            if oldest_time and display_tz:
                oldest_time_display = oldest_time.astimezone(display_tz)
            else:
                oldest_time_display = oldest_time

            if newest_time and display_tz:
                newest_time_display = newest_time.astimezone(display_tz)
            else:
                newest_time_display = newest_time

            # Build preview items with summary header (limit to first 5 for display)
            preview_items = []

            # Add summary header
            if is_japanese:
                summary = f"📊 削除対象: {len(matched_episodes)}件"
                if oldest_time_display and newest_time_display:
                    summary += f"  \n最古: {oldest_time_display}  \n最新: {newest_time_display}"
                summary += "  \n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                preview_items.append(summary)
            else:
                summary = f"📊 Target: {len(matched_episodes)} episodes"
                if oldest_time_display and newest_time_display:
                    summary += f"  \nOldest: {oldest_time_display}  \nNewest: {newest_time_display}"
                summary += "  \n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                preview_items.append(summary)

            # Add episode previews (first 3 only, compact format)
            for i, ep in enumerate(matched_episodes[:3], 1):
                source_description = getattr(ep, 'source_description', '')
                name = getattr(ep, 'name', 'Unknown')
                ep_time = getattr(ep, time_field, None)

                # Convert episode time to display timezone
                if ep_time is not None and display_tz:
                    # Ensure timezone-aware
                    if ep_time.tzinfo is None:
                        ep_time = ep_time.replace(tzinfo=timezone.utc)
                    ep_time_display = ep_time.astimezone(display_tz)
                else:
                    ep_time_display = ep_time if ep_time is not None else 'unknown'

                # Use source_description if available, otherwise fall back to name
                display_label = source_description if source_description else name

                # Compact one-line format: [i] label (timestamp)
                preview_text = f"[{i}] {display_label} ({ep_time_display})"
                preview_items.append(preview_text)

            if len(matched_episodes) > 3:
                remaining = len(matched_episodes) - 3
                if is_japanese:
                    preview_items.append(f"... 他 {remaining}件")
                else:
                    preview_items.append(f"... and {remaining} more")

            # Show confirmation dialog
            if is_japanese:
                title = f"時間範囲指定エピソード削除の確認 ({len(matched_episodes)}件)"
                warning = (
                    f"⚠️ {start_time} から {end_time} の間（{time_field}基準）のエピソードを削除します。\n"
                    f"この操作は取り消すことができません。関連するエンティティと関係性も削除されます。"
                )
            else:
                title = f"Confirm Time-Range Episode Deletion ({len(matched_episodes)} items)"
                warning = (
                    f"⚠️ Deleting episodes from {start_time} to {end_time} (based on {time_field}).\n"
                    f"This operation cannot be undone. Related entities and relationships will also be deleted."
                )

            confirmed, error_msg = await self.helper.show_confirmation_dialog(
                title=title,
                items=preview_items,
                warning_message=warning,
                timeout=self.valves.confirmation_timeout,
                __user__=__user__,
                __event_call__=__event_call__,
            )

            if not confirmed:
                return error_msg

            # 4. Execute Deletion
            # Extract UUIDs
            episode_uuids = [ep.uuid for ep in matched_episodes]

            if self.valves.debug_print:
                print(f"Deleting {len(episode_uuids)} episodes")

            # Delete using existing helper
            deleted_count = await self.helper.delete_episodes_by_uuids(episode_uuids)

            # Build result message
            result = f"✅ Deleted {deleted_count} episodes\n"
            result += f"Time range: {start_time} to {end_time}\n"
            result += f"Filter field: {time_field}"

            return result

        except Exception as e:
            error_msg = f"❌ Error deleting episodes by time range: {str(e)}"
            if self.valves.debug_print:
                traceback.print_exc()
            return error_msg

    async def search_and_delete_entities(
        self,
        query: str,
        limit: int = 1,
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
        __event_call__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Search for entities by name or description and delete them after user confirmation.
        
        This tool searches for entities (people, places, concepts) in your memory
        and allows you to delete them along with their relationships.
        
        IMPORTANT: This operation requires user confirmation and cannot be undone.
        
        :param query: Search query to find entities (e.g., "John Smith", "Python programming")
        :param limit: Maximum number of entities to return (default: 1, max: 100)
        :return: Result message with deleted entities count
        
        Note: __user__, __event_emitter__, and __event_call__ are automatically injected by the system.
        """
        
        if not await self.helper.ensure_graphiti_initialized() or self.helper.graphiti is None:
            return "❌ Error: Memory service is not available"
        
        
        # Set user headers in context variable (before any API calls)
        headers = get_user_info_headers(self.valves, __user__, None)
        if headers:
            user_headers_context.set(headers)
            if self.valves.debug_print:
                print(f"Set user headers in context: {list(headers.keys())}")
        # Validate and clamp limit
        limit = max(1, min(100, limit))
        
        try:
            group_id = self.helper.get_group_id(__user__)
            
            # Create a copy of config with custom limit
            search_config = copy.copy(COMBINED_HYBRID_SEARCH_RRF)
            search_config.limit = limit
            
            # Search for entities using search_() which returns SearchResults
            search_results = await self.helper.graphiti.search_(
                query=query,
                group_ids=[group_id] if group_id else None,
                config=search_config,
            )
            
            # Extract entity nodes
            entity_nodes = [node for node in search_results.nodes if hasattr(node, 'name')]
            
            if not entity_nodes:
                return f"ℹ️ No entities found matching '{query}'"
            
            # Show all entities for confirmation (no limit - user must see everything being deleted)
            total_count = len(entity_nodes)
            
            entity_list = []
            preview_items = []
            for i, node in enumerate(entity_nodes, 1):
                summary = getattr(node, 'summary', 'No description')
                if len(summary) > 80:
                    summary = summary[:80] + "..."
                entity_list.append(f"{i}. {node.name}: {summary}")
                preview_items.append(f"[{i}] {node.name}:  \n{summary}")
            
            # Get user's language preference
            is_japanese = self.helper.is_japanese_preferred(__user__)
            
            # Show confirmation dialog
            confirmed, error_msg = await self.helper.show_confirmation_dialog(
                title=f"エンティティの削除確認 ({total_count}件)" if is_japanese else f"Confirm Entity Deletion ({total_count} items)",
                items=preview_items,
                warning_message="⚠️ この操作は取り消すことができません。関連する関係性も削除されます。" if is_japanese else "⚠️ This operation cannot be undone. Related relationships will also be deleted.",
                timeout=self.valves.confirmation_timeout,
                __user__=__user__,
                __event_call__=__event_call__,
            )
            
            if not confirmed:
                return error_msg
            
            # Show compact result message (list only first 10)
            result_list = entity_list[:10]
            result = f"🔍 Found {total_count} entities:\n" + "\n".join(result_list)
            if total_count > 10:
                result += f"\n... and {total_count - 10} more"
            
            # Delete entities using Cypher
            entity_uuids = [node.uuid for node in entity_nodes]
            deleted_count = await self.helper.delete_nodes_by_uuids(entity_uuids)
            
            result += f"\n\n✅ Deleted {deleted_count} entities and their relationships"
            return result
            
        except Exception as e:
            error_msg = f"❌ Error searching/deleting entities: {str(e)}"
            if self.valves.debug_print:
                traceback.print_exc()
            return error_msg
    
    async def search_and_delete_facts(
        self,
        query: str,
        limit: int = 1,
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
        __event_call__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Search for facts (relationships/edges) and delete them after user confirmation.
        
        This tool searches for relationships between entities (e.g., "John works at Company X")
        and allows you to delete them.
        
        IMPORTANT: This operation requires user confirmation and cannot be undone.
        
        :param query: Search query to find relationships (e.g., "works at", "friends with")
        :param limit: Maximum number of facts to return (default: 1, max: 100)
        :return: Result message with deleted facts count
        
        Note: __user__, __event_emitter__, and __event_call__ are automatically injected by the system.
        """
        
        if not await self.helper.ensure_graphiti_initialized() or self.helper.graphiti is None:
            return "❌ Error: Memory service is not available"
        
        
        # Set user headers in context variable (before any API calls)
        headers = get_user_info_headers(self.valves, __user__, None)
        if headers:
            user_headers_context.set(headers)
            if self.valves.debug_print:
                print(f"Set user headers in context: {list(headers.keys())}")
        # Validate and clamp limit
        limit = max(1, min(100, limit))
        
        try:
            group_id = self.helper.get_group_id(__user__)
            
            # Create a copy of config with custom limit
            search_config = copy.copy(COMBINED_HYBRID_SEARCH_RRF)
            search_config.limit = limit
            
            # Search for facts using search_() which returns SearchResults
            search_results = await self.helper.graphiti.search_(
                query=query,
                group_ids=[group_id] if group_id else None,
                config=search_config,
            )
            
            # Extract edges (facts/relationships)
            edges = search_results.edges
            
            if not edges:
                return f"ℹ️ No facts found matching '{query}'"
            
            # Show all facts for confirmation (no limit - user must see everything being deleted)
            total_count = len(edges)
            
            # Get user's language preference
            is_japanese = self.helper.is_japanese_preferred(__user__)
            
            fact_list = []
            preview_items = []
            period_label = "期間" if is_japanese else "Period"
            for i, edge in enumerate(edges, 1):
                fact_text = getattr(edge, 'fact', 'Unknown relationship')
                if len(fact_text) > 80:
                    fact_text = fact_text[:80] + "..."
                valid_at = getattr(edge, 'valid_at', 'unknown')
                invalid_at = getattr(edge, 'invalid_at', 'present')
                fact_list.append(f"{i}. {fact_text} ({valid_at} - {invalid_at})")
                preview_items.append(f"[{i}] {fact_text}  \n{period_label}: {valid_at} - {invalid_at}")
            
            # Show confirmation dialog
            confirmed, error_msg = await self.helper.show_confirmation_dialog(
                title=f"関係性の削除確認 ({total_count}件)" if is_japanese else f"Confirm Fact Deletion ({total_count} items)",
                items=preview_items,
                warning_message="⚠️ この操作は取り消すことができません。" if is_japanese else "⚠️ This operation cannot be undone.",
                timeout=self.valves.confirmation_timeout,
                __user__=__user__,
                __event_call__=__event_call__,
            )
            
            if not confirmed:
                return error_msg
            
            # Show compact result message (list only first 10)
            result_list = fact_list[:10]
            result = f"🔍 Found {total_count} facts:\n" + "\n".join(result_list)
            if total_count > 10:
                result += f"\n... and {total_count - 10} more"
            
            # Delete edges using Cypher
            edge_uuids = [edge.uuid for edge in edges]
            deleted_count = await self.helper.delete_edges_by_uuids(edge_uuids)
            
            result += f"\n\n✅ Deleted {deleted_count} facts"
            return result
            
        except Exception as e:
            error_msg = f"❌ Error searching/deleting facts: {str(e)}"
            if self.valves.debug_print:
                traceback.print_exc()
            return error_msg
    
    async def search_and_delete_episodes(
        self,
        query: str,
        limit: int = 1,
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
        __event_call__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Search for episodes (conversation history) and delete them after user confirmation.
        
        This tool searches for past conversations and allows you to delete them
        along with the entities and relationships extracted from them.
        
        IMPORTANT: This operation requires user confirmation and cannot be undone.
        
        :param query: Search query to find episodes (e.g., "conversation about Python")
        :param limit: Maximum number of episodes to return (default: 1, max: 100)
        :return: Result message with deleted episodes count
        
        Note: __user__, __event_emitter__, and __event_call__ are automatically injected by the system.
        """
        
        if not await self.helper.ensure_graphiti_initialized() or self.helper.graphiti is None:
            return "❌ Error: Memory service is not available"
        
        
        # Set user headers in context variable (before any API calls)
        headers = get_user_info_headers(self.valves, __user__, None)
        if headers:
            user_headers_context.set(headers)
            if self.valves.debug_print:
                print(f"Set user headers in context: {list(headers.keys())}")
        # Validate and clamp limit
        limit = max(1, min(100, limit))
        
        try:
            group_id = self.helper.get_group_id(__user__)
            
            # Create a copy of config with custom limit
            search_config = copy.copy(COMBINED_HYBRID_SEARCH_RRF)
            search_config.limit = limit
            
            # Search for episodes using search_() which returns SearchResults
            search_results = await self.helper.graphiti.search_(
                query=query,
                group_ids=[group_id] if group_id else None,
                config=search_config,
            )
            
            # Extract episodes
            episodes = search_results.episodes
            
            if not episodes:
                return f"ℹ️ No episodes found matching '{query}'"
            
            # Show all episodes for confirmation (no limit - user must see everything being deleted)
            total_count = len(episodes)
            
            # Get user's language preference
            is_japanese = self.helper.is_japanese_preferred(__user__)
            
            episode_list = []
            preview_items = []
            created_label = "作成日時" if is_japanese else "Created"
            for i, episode in enumerate(episodes, 1):
                name = getattr(episode, 'name', 'Unknown episode')
                content = getattr(episode, 'content', '')
                if len(content) > 80:
                    content_preview = content[:80] + "..."
                else:
                    content_preview = content
                created_at = getattr(episode, 'created_at', 'unknown')
                episode_list.append(f"{i}. {name}: {content_preview} (created: {created_at})")
                preview_items.append(f"[{i}] {name}  \n{content_preview}  \n{created_label}: {created_at}")
            
            # Show confirmation dialog
            confirmed, error_msg = await self.helper.show_confirmation_dialog(
                title=f"エピソードの削除確認 ({total_count}件)" if is_japanese else f"Confirm Episode Deletion ({total_count} items)",
                items=preview_items,
                warning_message="⚠️ この操作は取り消すことができません。関連するエンティティと関係性も削除されます。" if is_japanese else "⚠️ This operation cannot be undone. Related entities and relationships will also be deleted.",
                timeout=self.valves.confirmation_timeout,
                __user__=__user__,
                __event_call__=__event_call__,
            )
            
            if not confirmed:
                return error_msg
            
            # Show compact result message (list only first 10)
            result_list = episode_list[:10]
            result = f"🔍 Found {total_count} episodes:\n" + "\n".join(result_list)
            if total_count > 10:
                result += f"\n... and {total_count - 10} more"
            
            # Delete episodes using remove_episode
            episode_uuids = [episode.uuid for episode in episodes]
            deleted_count = await self.helper.delete_episodes_by_uuids(episode_uuids)
            
            result += f"\n\n✅ Deleted {deleted_count} episodes"
            return result
            
        except Exception as e:
            error_msg = f"❌ Error searching/deleting episodes: {str(e)}"
            if self.valves.debug_print:
                traceback.print_exc()
            return error_msg
    
    async def delete_by_uuids(
        self,
        node_uuids: str = "",
        edge_uuids: str = "",
        episode_uuids: str = "",
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
        __event_call__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Delete specific nodes, edges, or episodes by their UUIDs after user confirmation.
        
        This tool allows precise deletion when you know the exact UUID of items to delete.
        UUIDs can be found in debug output or by using search tools first.
        
        IMPORTANT: This operation requires user confirmation and cannot be undone.
        
        :param node_uuids: Comma-separated list of node UUIDs to delete
        :param edge_uuids: Comma-separated list of edge UUIDs to delete
        :param episode_uuids: Comma-separated list of episode UUIDs to delete
        :return: Result message with deletion status
        
        Note: __user__, __event_emitter__, and __event_call__ are automatically injected by the system.
        """
        if not await self.helper.ensure_graphiti_initialized() or self.helper.graphiti is None:
            return "❌ Error: Memory service is not available"
        
        
        # Set user headers in context variable (before any API calls)
        headers = get_user_info_headers(self.valves, __user__, None)
        if headers:
            user_headers_context.set(headers)
            if self.valves.debug_print:
                print(f"Set user headers in context: {list(headers.keys())}")
        
        try:
            # Get user's language preference first
            is_japanese = self.helper.is_japanese_preferred(__user__)
            
            # Prepare preview items with actual content from database
            preview_items = []
            
            # Fetch and display node information
            if node_uuids.strip():
                uuids = [uuid.strip() for uuid in node_uuids.split(',') if uuid.strip()]
                for i, uuid in enumerate(uuids, 1):
                    try:
                        # Fetch node details from database
                        nodes = await EntityNode.get_by_uuids(self.helper.graphiti.driver, [uuid])
                        if nodes:
                            node = nodes[0]
                            name = getattr(node, 'name', 'Unknown')
                            summary = getattr(node, 'summary', 'No description')
                            if len(summary) > 80:
                                summary = summary[:80] + "..."
                            summary_label = "概要" if is_japanese else "Summary"
                            preview_items.append(f"[Node {i}] {name}  \nUUID: {uuid}  \n{summary_label}: {summary}")
                        else:
                            not_found_msg = "見つかりません" if is_japanese else "Not found"
                            preview_items.append(f"[Node {i}] ⚠️ {not_found_msg}  \nUUID: {uuid}")
                    except Exception as e:
                        error_msg = "詳細取得エラー" if is_japanese else "Error fetching details"
                        preview_items.append(f"[Node {i}] ⚠️ {error_msg}  \nUUID: {uuid}")
                        if self.valves.debug_print:
                            print(f"Error fetching node {uuid}: {e}")
            
            # Fetch and display edge information
            if edge_uuids.strip():
                uuids = [uuid.strip() for uuid in edge_uuids.split(',') if uuid.strip()]
                for i, uuid in enumerate(uuids, 1):
                    try:
                        # Fetch edge details from database
                        edges = await EntityEdge.get_by_uuids(self.helper.graphiti.driver, [uuid])
                        if edges:
                            edge = edges[0]
                            fact = getattr(edge, 'fact', 'Unknown relationship')
                            if len(fact) > 80:
                                fact = fact[:80] + "..."
                            valid_at = getattr(edge, 'valid_at', 'unknown')
                            invalid_at = getattr(edge, 'invalid_at', 'present')
                            period_label = "期間" if is_japanese else "Period"
                            preview_items.append(f"[Edge {i}] {fact}  \nUUID: {uuid}  \n{period_label}: {valid_at} → {invalid_at}")
                        else:
                            not_found_msg = "見つかりません" if is_japanese else "Not found"
                            preview_items.append(f"[Edge {i}] ⚠️ {not_found_msg}  \nUUID: {uuid}")
                    except Exception as e:
                        error_msg = "詳細取得エラー" if is_japanese else "Error fetching details"
                        preview_items.append(f"[Edge {i}] ⚠️ {error_msg}  \nUUID: {uuid}")
                        if self.valves.debug_print:
                            print(f"Error fetching edge {uuid}: {e}")
            
            # Fetch and display episode information
            if episode_uuids.strip():
                uuids = [uuid.strip() for uuid in episode_uuids.split(',') if uuid.strip()]
                for i, uuid in enumerate(uuids, 1):
                    try:
                        # Fetch episode details from database
                        episodes = await EpisodicNode.get_by_uuids(self.helper.graphiti.driver, [uuid])
                        if episodes:
                            episode = episodes[0]
                            name = getattr(episode, 'name', 'Unknown episode')
                            content = getattr(episode, 'content', '')
                            if len(content) > 80:
                                content = content[:80] + "..."
                            created_at = getattr(episode, 'created_at', 'unknown')
                            content_label = "内容" if is_japanese else "Content"
                            created_label = "作成" if is_japanese else "Created"
                            preview_items.append(f"[Episode {i}] {name}  \nUUID: {uuid}  \n{content_label}: {content}  \n{created_label}: {created_at}")
                        else:
                            not_found_msg = "見つかりません" if is_japanese else "Not found"
                            preview_items.append(f"[Episode {i}] ⚠️ {not_found_msg}  \nUUID: {uuid}")
                    except Exception as e:
                        error_msg = "詳細取得エラー" if is_japanese else "Error fetching details"
                        preview_items.append(f"[Episode {i}] ⚠️ {error_msg}  \nUUID: {uuid}")
                        if self.valves.debug_print:
                            print(f"Error fetching episode {uuid}: {e}")
            
            if not preview_items:
                return "ℹ️ No UUIDs provided for deletion"
            
            # Show confirmation dialog
            confirmed, error_msg = await self.helper.show_confirmation_dialog(
                title="UUID指定削除の確認" if is_japanese else "Confirm UUID-based Deletion",
                items=preview_items,
                warning_message="⚠️ この操作は取り消すことができません。UUIDを直接指定しての削除は慎重に行ってください。" if is_japanese else "⚠️ This operation cannot be undone. Please be careful when deleting by UUID.",
                timeout=self.valves.confirmation_timeout,
                __user__=__user__,
                __event_call__=__event_call__,
            )
            
            if not confirmed:
                return error_msg
            
            results = []
            
            # Delete nodes
            if node_uuids.strip():
                uuids = [uuid.strip() for uuid in node_uuids.split(',') if uuid.strip()]
                try:
                    deleted_count = await self.helper.delete_nodes_by_uuids(uuids)
                    results.append(f"✅ Deleted {deleted_count} node(s)")
                except Exception as e:
                    results.append(f"❌ Failed to delete nodes: {str(e)}")
            
            # Delete edges
            if edge_uuids.strip():
                uuids = [uuid.strip() for uuid in edge_uuids.split(',') if uuid.strip()]
                try:
                    deleted_count = await self.helper.delete_edges_by_uuids(uuids)
                    results.append(f"✅ Deleted {deleted_count} edge(s)")
                except Exception as e:
                    results.append(f"❌ Failed to delete edges: {str(e)}")
            
            # Delete episodes
            if episode_uuids.strip():
                uuids = [uuid.strip() for uuid in episode_uuids.split(',') if uuid.strip()]
                try:
                    deleted_count = await self.helper.delete_episodes_by_uuids(uuids)
                    results.append(f"✅ Deleted {deleted_count} episode(s)")
                except Exception as e:
                    results.append(f"❌ Failed to delete episodes: {str(e)}")
            
            if not results:
                return "ℹ️ No UUIDs provided for deletion"
            
            return "\n".join(results)
            
        except Exception as e:
            error_msg = f"❌ Error deleting by UUIDs: {str(e)}"
            if self.valves.debug_print:
                traceback.print_exc()
            return error_msg
    
    async def clear_all_memory(
        self,
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
        __event_call__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Clear ALL memory for the current user after confirmation. THIS CANNOT BE UNDONE!
        
        This tool deletes all entities, relationships, and episodes from your memory space.
        Use with extreme caution as this operation is irreversible.
        
        IMPORTANT: Requires user confirmation dialog before execution.
        
        :return: Result message with deletion status
        
        Note: __user__, __event_emitter__, and __event_call__ are automatically injected by the system.
        """
        if not await self.helper.ensure_graphiti_initialized() or self.helper.graphiti is None:
            return "❌ Error: Memory service is not available"
        
        
        # Set user headers in context variable (before any API calls)
        headers = get_user_info_headers(self.valves, __user__, None)
        if headers:
            user_headers_context.set(headers)
            if self.valves.debug_print:
                print(f"Set user headers in context: {list(headers.keys())}")
        try:
            group_id = self.helper.get_group_id(__user__)
            
            if not group_id:
                return "❌ Error: Group ID is required for memory clearing. Please check your group_id_format configuration."
            
            # Count existing items using get_by_group_ids methods
            try:
                nodes = await EntityNode.get_by_group_ids(self.helper.graphiti.driver, [group_id])
                node_count = len(nodes)
            except Exception as e:
                if self.valves.debug_print:
                    print(f"Error counting nodes: {e}")
                node_count = 0
            
            try:
                edges = await EntityEdge.get_by_group_ids(self.helper.graphiti.driver, [group_id])
                edge_count = len(edges)
            except Exception as e:
                if self.valves.debug_print:
                    print(f"Error counting edges: {e}")
                edge_count = 0
            
            try:
                episodes = await EpisodicNode.get_by_group_ids(self.helper.graphiti.driver, [group_id])
                episode_count = len(episodes)
            except Exception as e:
                if self.valves.debug_print:
                    print(f"Error counting episodes: {e}")
                episode_count = 0
            
            if node_count == 0 and edge_count == 0 and episode_count == 0:
                return "ℹ️ Memory is already empty"
            
            # Get user's language preference
            is_japanese = self.helper.is_japanese_preferred(__user__)
            
            # Show confirmation dialog
            if is_japanese:
                preview_items = [
                    f"エンティティ(Entity): {node_count}個",
                    f"関係性(Fact): {edge_count}個",
                    f"エピソード(Episode): {episode_count}個",
                ]
            else:
                preview_items = [
                    f"Entities: {node_count} items",
                    f"Facts: {edge_count} items",
                    f"Episodes: {episode_count} items",
                ]
            
            confirmed, error_msg = await self.helper.show_confirmation_dialog(
                title="⚠️ 全メモリ削除の最終確認" if is_japanese else "⚠️ Final Confirmation: Clear All Memory",
                items=preview_items,
                warning_message="🔥 この操作は完全に元に戻せません！全てのメモリデータが永久に失われます。" if is_japanese else "🔥 This operation is completely irreversible! All memory data will be permanently lost.",
                timeout=self.valves.confirmation_timeout,
                __user__=__user__,
                __event_call__=__event_call__,
            )
            
            if not confirmed:
                return error_msg
            # Require text input confirmation
            if __event_call__:
                try:
                    if is_japanese:
                        input_task = __event_call__(
                            {
                                "type": "input",
                                "data": {
                                    "title": "最終確認",
                                    "message": f"本当に全メモリ({node_count + edge_count + episode_count}件)を削除しますか？\n確認のため 'CLEAR_ALL_MEMORY' と入力してください。\n⏰ {self.valves.confirmation_timeout}秒以内に入力しないと自動的にキャンセルされます。",
                                    "placeholder": "CLEAR_ALL_MEMORY"
                                }
                            }
                        )
                    else:
                        input_task = __event_call__(
                            {
                                "type": "input",
                                "data": {
                                    "title": "Final Confirmation",
                                    "message": f"Are you sure you want to delete all memory ({node_count + edge_count + episode_count} items)?\nPlease type 'CLEAR_ALL_MEMORY' to confirm.\n⏰ Auto-cancel in {self.valves.confirmation_timeout} seconds if no input is provided.",
                                    "placeholder": "CLEAR_ALL_MEMORY"
                                }
                            }
                        )
                    
                    input_result = await asyncio.wait_for(input_task, timeout=self.valves.confirmation_timeout)
                    
                    if input_result != "CLEAR_ALL_MEMORY":
                        return "🚫 Confirmation text does not match. Memory clearing cancelled."
                except asyncio.TimeoutError:
                    return "🚫 Input timeout. Memory clearing cancelled."
                except Exception as e:
                    if self.valves.debug_print:
                        print(f"Input confirmation error: {e}")
                    return "🚫 Input confirmation cancelled."
            
            # Use Node.delete_by_group_id() - the correct method for clearing all data
            try:
                if self.valves.debug_print:
                    print(f"Deleting all data for group_id: {group_id}")
                
                await EntityNode.delete_by_group_id(self.helper.graphiti.driver, group_id)
                
                result = f"🗑️ All memory cleared:\n"
                result += f"  - {node_count} entities deleted\n"
                result += f"  - {edge_count} facts deleted\n"
                result += f"  - {episode_count} episodes deleted"
                
                return result
            except Exception as e:
                error_msg = f"❌ Error deleting memory: {str(e)}"
                if self.valves.debug_print:
                    traceback.print_exc()
                return error_msg
            
        except Exception as e:
            error_msg = f"❌ Error clearing memory: {str(e)}"
            if self.valves.debug_print:
                traceback.print_exc()
            return error_msg

# NOTE: One-off normalization utility for legacy Episodes.
# Episodes saved before Graphiti Memory v0.12.1 may have non-normalized valid_at values.
# If you need to bulk-normalize existing data, uncomment this method and ask the AI to run it once (one-time only).
# After migration, re-comment or remove it.
# Compatibility: Neo4j only. FalkorDB is not supported; this method will no-op there.
# IMPORTANT: Take a database backup before running. This operation updates stored data.
#
#    async def normalize_episode_valid_at_to_utc(
#        self,
#        __user__: dict = {},
#        __event_emitter__: Optional[Callable[[dict], Any]] = None,
#        __event_call__: Optional[Callable[[dict], Any]] = None,
#    ) -> str:
#        """
#        Normalize Episodic.valid_at from LocalDateTime to UTC DateTime in Neo4j.
#
#        Executes a single Cypher (for reference):
#        MATCH (e:Episodic)
#        WITH e, toString(e.valid_at) AS s
#        WHERE e.valid_at IS NOT NULL AND NOT s ENDS WITH 'Z'  // localdatetime 判定
#        WITH e, datetime(e.valid_at) AS dt_utc
#        SET e.valid_at = datetime({epochMillis: dt_utc.epochMillis, timezone:'UTC'})
#        RETURN count(e) AS updated
#
#        ⚠️ Irreversible. Back up the database before running.
#        """
#        if not await self.helper.ensure_graphiti_initialized() or self.helper.graphiti is None:
#            return "❌ Error: Memory service is not available"
#
#        # Reject if backend is not Neo4j (valves setting)
#        if self.valves.graph_db_backend.lower() != "neo4j":
#            return "❌ This migration is only supported on Neo4j backend."
#
#        # Set user headers in context variable (before any API calls)
#        headers = get_user_info_headers(self.valves, __user__, None)
#        if headers:
#            user_headers_context.set(headers)
#
#        # Confirmation dialog is mandatory
#        if not __event_call__:
#            return "❌ Error: Confirmation dialog is not available. Cannot run migration without confirmation."
#
#        is_japanese = self.helper.is_japanese_preferred(__user__)
#        title = "valid_at を UTC に正規化しますか？" if is_japanese else "Normalize valid_at to UTC?"
#        warning = "⚠️ この操作は取り消せません。実行前にバックアップを取ってください。" if is_japanese else "⚠️ This operation cannot be undone. Please back up first."
#        items = [
#            "対象: valid_at が LocalDateTime の Episodic ノード（末尾にZが付かないもの）" if is_japanese else "Scope: Episodic nodes whose valid_at is LocalDateTime (strings without trailing 'Z')",
#            "操作: localdatetime を UTC datetime に変換" if is_japanese else "Action: convert localdatetime to UTC datetime",
#        ]
#
#        confirmed, error_msg = await self.helper.show_confirmation_dialog(
#            title=title,
#            items=items,
#            warning_message=warning,
#            timeout=self.valves.confirmation_timeout,
#            __user__=__user__,
#            __event_call__=__event_call__,
#        )
#        if not confirmed:
#            return error_msg
#
#        cypher = """
#        MATCH (e:Episodic)
#        WITH e, toString(e.valid_at) AS s
#        WHERE e.valid_at IS NOT NULL AND NOT s ENDS WITH 'Z'
#        WITH e, datetime(e.valid_at) AS dt_utc
#        SET e.valid_at = datetime({epochMillis: dt_utc.epochMillis, timezone:'UTC'})
#        RETURN count(e) AS updated
#        """
#        try:
#            records, _, _ = await self.helper.graphiti.driver.execute_query(cypher)
#            updated = records[0]["updated"] if records else 0
#            return f"✅ normalized valid_at to UTC datetime. Updated: {updated}"
#        except Exception as e:
#            if self.valves.debug_print:
#                traceback.print_exc()
#            return f"❌ Error during normalization: {e}"
