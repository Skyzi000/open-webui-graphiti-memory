"""
title: Graphiti Search Tool
author: Skyzi000
description: Read-only search access for entities, facts, and episodes in Graphiti knowledge graph memory.
author_url: https://github.com/Skyzi000
repository_url: https://github.com/Skyzi000/open-webui-graphiti-memory
version: 0.1.0
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
- Search Only: Search for entities, edges, or episodes with Graphiti search.
- Read-only: No delete or write operations are exposed in this tool.
- Group Isolation: Reads are scoped to configured group_id (supports group_id selection).
"""

import os
import re
import json
import copy
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
from graphiti_core.nodes import EpisodicNode
from openai import AsyncOpenAI

# Context variable to store user-specific headers for each async request
# This ensures complete isolation between concurrent requests without locks
user_headers_context = contextvars.ContextVar('user_headers', default={})


def _normalize_group_id(value: Optional[str]) -> str:
    if value is None:
        return ""
    return value.strip()


def _sanitize_group_id(value: str) -> str:
    if value == "":
        return ""
    return re.sub(r'[^a-zA-Z0-9_-]', '_', value)


def _parse_group_id_formats(raw: Optional[str]) -> List[tuple[str, str]]:
    normalized = _normalize_group_id(raw)
    if not normalized:
        return []

    formats: List[tuple[str, str]] = []
    seen = set()
    for part in (part.strip() for part in normalized.split(",") if part.strip()):
        template, _, description = part.partition(":")
        template = template.strip()
        description = description.strip()
        if not template or template in seen:
            continue
        seen.add(template)
        formats.append((template, description))
    return formats


def _format_group_id(template: str, user: dict) -> str:
    user_id = user.get('id', 'unknown')
    user_email = user.get('email', '')
    user_name = user.get('name', '')

    sanitized_email = re.sub(r'[@.]', lambda m: '_at_' if m.group() == '@' else '_', user_email)
    sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '_', user_name)

    group_id = template.format(
        user_id=user_id,
        user_email=sanitized_email,
        user_name=sanitized_name,
    )
    return re.sub(r'[^a-zA-Z0-9_-]', '_', group_id)


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

    def get_config_hash(self) -> str:
        """Generate configuration hash for change detection."""
        # Get all valve values as dict, excluding non-config fields
        valve_dict = self.valves.model_dump(
            exclude={
                'debug_print',  # Debugging settings don't affect initialization
                'group_id_formats',  # Group ID formats don't affect Graphiti init
                'default_group_id',  # Default group selection doesn't affect Graphiti init
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
            print("Initializing Graphiti for memory search...")

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
            raise ValueError(
                f"Unsupported graph database backend: {self.valves.graph_db_backend}. "
                "Supported backends are 'neo4j' and 'falkordb'."
            )

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

    def get_user_group_ids(self, user: dict) -> List[str]:
        """
        Generate user-based group_ids from configured formats.

        Args:
            user: User dictionary containing 'id', 'email', 'name'

        Returns:
            Generated group_ids (sanitized, de-duplicated).
        """
        formats = _parse_group_id_formats(self.valves.group_id_formats)
        if not formats or any(template.lower() == 'none' for template, _ in formats):
            return []

        group_ids: List[str] = []
        for group_id_format, _ in formats:
            group_id = _format_group_id(group_id_format, user)
            if group_id:
                group_ids.append(group_id)

        return list(dict.fromkeys(group_ids))

    def get_allowed_group_ids_with_desc(
        self,
        user: dict,
    ) -> tuple[List[str], Dict[str, str]]:
        """
        Build the list of allowed group_ids and optional descriptions for this user.

        Args:
            user: User dictionary containing 'id', 'email', 'name'

        Returns:
            Tuple of (allowed_group_ids, descriptions).
        """
        formats = _parse_group_id_formats(self.valves.group_id_formats)
        allowed_group_ids: List[str] = []
        descriptions: Dict[str, str] = {}

        for template, description in formats:
            if template.lower() == 'none':
                continue
            group_id = _format_group_id(template, user)
            if not group_id:
                continue
            allowed_group_ids.append(group_id)
            if description:
                descriptions[group_id] = description

        default_template = _normalize_group_id(self.valves.default_group_id)
        default_group_id = _format_group_id(default_template, user) if default_template else ""
        if default_group_id:
            allowed_group_ids.append(default_group_id)

        deduped_ids = list(dict.fromkeys(allowed_group_ids))
        return deduped_ids, descriptions

    def get_allowed_group_ids(
        self,
        user: dict,
    ) -> List[str]:
        """
        Build the list of allowed group_ids for this user.

        Args:
            user: User dictionary containing 'id', 'email', 'name'

        Returns:
            List of allowed group_ids (sanitized, de-duplicated).
        """
        allowed_group_ids, _ = self.get_allowed_group_ids_with_desc(user)
        return allowed_group_ids

    def get_group_id_selection(
        self,
        user: dict,
        group_id: Optional[str] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Resolve group_id with validation against allowed group_ids.

        Args:
            user: User dictionary containing 'id', 'email', 'name'
            group_id: Optional group_id override. None uses default user-based group_id.

        Returns:
            Tuple of (group_id, error_message). error_message is None if valid.
        """
        allowed_group_ids = self.get_allowed_group_ids(user)
        base_group_ids = self.get_user_group_ids(user)
        default_template = _normalize_group_id(self.valves.default_group_id)
        default_group_id = _format_group_id(default_template, user) if default_template else ""

        if group_id is None:
            if default_group_id:
                return default_group_id, None
            if base_group_ids:
                return base_group_ids[0], None
            return None, "❌ Error: Group ID is required. Please configure group_id_formats or default_group_id."

        if not isinstance(group_id, str):
            return None, "❌ Error: group_id must be a string or null."

        normalized_group_id = _normalize_group_id(group_id)
        sanitized_group_id = _sanitize_group_id(normalized_group_id)

        if sanitized_group_id not in allowed_group_ids:
            return None, f"❌ Error: group_id '{sanitized_group_id}' is not allowed."

        return sanitized_group_id, None


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

        group_id_formats: str = Field(
            default="{user_id}",
            description=(
                "Comma-separated group_id formats or fixed IDs. "
                "Each entry can include an optional description: \"group_id_or_format:description\". "
                "Available placeholders: {user_id}, {user_email}, {user_name}. "
                "Include 'none' to disable user-based group filtering."
            ),
        )
        default_group_id: str = Field(
            default="{user_id}",
            description=(
                "Default group_id format to use when group_id is not specified. "
                "Available placeholders: {user_id}, {user_email}, {user_name}."
            ),
        )

        forward_user_info_headers: str = Field(
            default="default",
            description=(
                "Forward user information headers (User-Name, User-Id, User-Email, User-Role, Chat-Id) to OpenAI API. "
                "Options: 'default' (follow environment variable ENABLE_FORWARD_USER_INFO_HEADERS, defaults to false if not set), "
                "'true' (always forward), 'false' (never forward)."
            ),
        )

        debug_print: bool = Field(
            default=False,
            description="Enable debug printing to console",
        )

    class UserValves(BaseModel):
        message_language: str = Field(
            default="en",
            description="Language for messages: 'en' (English) or 'ja' (Japanese)",
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

    async def get_available_group_ids(
        self,
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Get available group_id options for the current user.

        Call this BEFORE using group_id parameter in other tools to:
        - See which group_id values are allowed for search
        - Know the default group_id (used when group_id is not specified)

        :return: JSON with resolved group_ids the AI should use, with descriptions embedded
        """
        user_group_ids = self.helper.get_user_group_ids(__user__)
        default_template = _normalize_group_id(self.valves.default_group_id)
        default_group_id = _format_group_id(default_template, __user__) if default_template else (
            user_group_ids[0] if user_group_ids else None
        )

        allowed_group_ids, descriptions = self.helper.get_allowed_group_ids_with_desc(__user__)
        allowed_group_ids_with_desc = [
            f"{group_id}:{descriptions[group_id]}"
            if group_id in descriptions and descriptions[group_id]
            else group_id
            for group_id in allowed_group_ids
        ]

        payload = {
            "allowed_group_ids": allowed_group_ids_with_desc,
            "default_group_id": default_group_id,
        }

        return json.dumps(payload)

    async def search_entities(
        self,
        query: str,
        limit: int = 10,
        show_uuid: bool = False,
        group_id: Optional[str] = None,
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Search memory for people, places, concepts, projects, or other entities.

        Use this proactively when:
        - User mentions someone/something that might be in memory
        - User asks "Do you remember...?" or refers to past conversations
        - You need context about a previously discussed topic
        - You want to provide more personalized, informed responses

        Semantic search: queries like "development tools" find related tech entities.

        :param query: Search query in user's language (memories are stored in original language)
        :param limit: Maximum results (default: 10, max: 100)
        :param show_uuid: Show UUIDs for manual reference (default: False)
        :param group_id: Optional group_id override. None uses the default user-based group_id.
        :return: Found entities with details (name, summary)
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
            group_id, error_msg = self.helper.get_group_id_selection(__user__, group_id)
            if error_msg:
                return error_msg

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
        group_id: Optional[str] = None,
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Search memory for facts (extracted relationships between entities).

        Use this proactively when:
        - User asks about relationships (e.g., "Who does X work with?", "What's related to Y?")
        - You need to recall connections between people, places, or concepts
        - User references a relationship that might be stored

        Facts are extracted relationships, not raw conversation content.
        For original conversation text, use search_episodes instead.

        :param query: Search query in user's language (memories are stored in original language)
        :param limit: Maximum results (default: 10, max: 100)
        :param show_uuid: Show UUIDs for manual reference (default: False)
        :param group_id: Optional group_id override. None uses the default user-based group_id.
        :return: Found facts with validity period
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
            group_id, error_msg = self.helper.get_group_id_selection(__user__, group_id)
            if error_msg:
                return error_msg

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
        group_id: Optional[str] = None,
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Search past conversations and interaction history by content.

        Use this proactively when:
        - User asks "What did we talk about...?" or "Do you remember when...?"
        - You need to recall details from a specific past conversation
        - User references a topic discussed previously

        Episodes are raw conversation records. Use get_episode_content(uuid) to retrieve full content.

        :param query: Search query in user's language (memories are stored in original language)
        :param limit: Maximum results (default: 10, max: 100)
        :param show_uuid: Show UUIDs for get_episode_content (default: True)
        :param group_id: Optional group_id override. None uses the default user-based group_id.
        :return: Found episodes with previews
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
            group_id, error_msg = self.helper.get_group_id_selection(__user__, group_id)
            if error_msg:
                return error_msg

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

            result += (
                "💡 Previews may be truncated. Use `get_episode_content(uuid=\"...\")` "
                "to retrieve full content before answering if details are needed."
            )

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
        group_id: Optional[str] = None,
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Browse recent conversation history in chronological order.

        Use this when:
        - User asks "What have we discussed recently?"
        - You want to review recent interactions without a specific search query
        - User wants to see their conversation timeline

        Unlike search_episodes, this doesn't require a query - just retrieves recent history.

        :param limit: Maximum results (default: 10, max: 100)
        :param offset: Skip N episodes for pagination (default: 0)
        :param show_uuid: Show UUIDs for get_episode_content (default: True)
        :param group_id: Optional group_id override. None uses the default user-based group_id.
        :return: Recent episodes in chronological order
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
            group_id, error_msg = self.helper.get_group_id_selection(__user__, group_id)
            if error_msg:
                return error_msg

            if self.valves.debug_print:
                print("=== get_recent_episodes: Fetching episodes ===")
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

            result += (
                "💡 Previews may be truncated. Use `get_episode_content(uuid=\"...\")` "
                "to retrieve full content before answering if details are needed."
            )

            # Add pagination hints
            if has_more_in_db:
                result += f"\n\n📄 More episodes may be available. Use `offset={offset + len(episodes)}` to see the next page."

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
        Get full content of an episode by UUID (from search_episodes or get_recent_episodes).

        Use this when:
        - Search results show truncated previews and you need complete content
        - User wants to see full details of a specific past conversation

        :param uuid: Episode UUID from search_episodes or get_recent_episodes results
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
                print("=== get_episode_content: Fetching episode ===")
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
