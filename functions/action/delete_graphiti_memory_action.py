"""
title: Delete Graphiti Memory Action Button
description: Action button to search and delete an episode from Graphiti knowledge graph memory based on clicked message content.
author: Skyzi000
author_url: https://github.com/Skyzi000
repository_url: https://github.com/Skyzi000/open-webui-graphiti-memory
version: 0.2.1
requirements: graphiti-core
icon_url: data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzMiIgaGVpZ2h0PSIzMiIgdmlld0JveD0iMCAwIDMyIDMyIj4KICA8cGF0aCBkPSJNOCA5aDN2MTZIOHptNiAwaDN2MTZoLTN6bTYgMGgzdjE2aC0zeiIgZmlsbD0iIzRjNGM0YyIvPgogIDxyZWN0IHg9IjYiIHk9IjYiIHdpZHRoPSIyMCIgaGVpZ2h0PSIzIiByeD0iMSIgZmlsbD0iIzRjNGM0YyIvPgogIDxwYXRoIGQ9Ik0xMSA2VjRhMiAyIDAgMCAxIDItMmg2YTIgMiAwIDAgMSAyIDJ2MiIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjNGM0YzRjIiBzdHJva2Utd2lkdGg9IjEuNSIvPgogIDxwYXRoIGQ9Ik03IDloMTh2MThhMiAyIDAgMCAxLTIgMkg5YTIgMiAwIDAgMS0yLTJWOXoiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzRjNGM0YyIgc3Ryb2tlLXdpZHRoPSIxLjUiLz4KPC9zdmc+

Note on FalkorDB backend:
  FalkorDB requires additional setup due to Redis version conflicts.
  See the README for details: https://github.com/Skyzi000/open-webui-graphiti-memory#falkordb-alternative-not-recently-tested

Design:
- Main class: Action
- Searches for the episode matching the clicked message content
- Shows confirmation dialog with the found episode
- Deletes the episode after user confirmation
"""

import asyncio
import contextvars
import hashlib
import os
import re
import traceback
from typing import Optional
from urllib.parse import quote

from pydantic import BaseModel, Field

from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
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
from graphiti_core.search.search_config_recipes import (
    COMBINED_HYBRID_SEARCH_RRF,
    COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
)
from openai import AsyncOpenAI

# Context variable to store user-specific headers for each async request
user_headers_context = contextvars.ContextVar('user_headers', default={})


class MultiUserOpenAIClient(OpenAIClient):
    """
    Custom OpenAI LLM client that retrieves user-specific headers from context variables.
    """

    def __init__(self, config: LLMConfig | None = None, cache: bool = False, **kwargs):
        if config is None:
            config = LLMConfig()

        self._base_client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

        super().__init__(config, cache, self._base_client, **kwargs)

    @property
    def client(self) -> AsyncOpenAI:
        headers = user_headers_context.get()
        if headers:
            return self._base_client.with_options(default_headers=headers)
        return self._base_client

    @client.setter
    def client(self, value: AsyncOpenAI):
        self._base_client = value


class MultiUserOpenAIGenericClient(OpenAIGenericClient):
    """
    Custom OpenAI-compatible generic LLM client that retrieves user-specific headers from context variables.
    """

    def __init__(self, config: LLMConfig | None = None, cache: bool = False):
        if config is None:
            config = LLMConfig()

        self._base_client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

        super().__init__(config, cache, self._base_client)

    @property
    def client(self) -> AsyncOpenAI:
        headers = user_headers_context.get()
        if headers:
            return self._base_client.with_options(default_headers=headers)
        return self._base_client

    @client.setter
    def client(self, value: AsyncOpenAI):
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

        if client is not None:
            self._base_client = client
        else:
            self._base_client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)

        super().__init__(config, self._base_client)

    @property
    def client(self) -> AsyncOpenAI:
        headers = user_headers_context.get()
        if headers:
            return self._base_client.with_options(default_headers=headers)
        return self._base_client

    @client.setter
    def client(self, value: AsyncOpenAI):
        self._base_client = value


class Action:
    class Valves(BaseModel):
        llm_client_type: str = Field(
            default="openai",
            description="Type of LLM client to use: 'openai' for OpenAI client, 'generic' for OpenAI-compatible generic client.",
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

        semaphore_limit: int = Field(
            default=10,
            description="Maximum number of concurrent LLM operations in Graphiti.",
        )

        group_id_format: str = Field(
            default="{user_id}",
            description="Format string for group_id. Available placeholders: {user_id}, {user_email}, {user_name}.",
        )

        forward_user_info_headers: str = Field(
            default="default",
            description="Forward user information headers to OpenAI API. Options: 'default', 'true', 'false'.",
        )

        max_search_message_length: int = Field(
            default=512,
            description="Maximum length of message content to use for searching related episodes. Longer content is truncated.",
        )

        sanitize_search_query: bool = Field(
            default=True,
            description="Sanitize search queries to avoid FalkorDB/RediSearch syntax errors by removing special characters like @, :, \", (, ). Disable if you want to use raw queries or if using a different backend.",
        )

        search_strategy: str = Field(
            default="balanced",
            description="Search strategy: 'fast' (BM25 only, ~0.1s), 'balanced' (BM25+Cosine, ~0.5s), 'quality' (Cross-Encoder, ~1-5s)",
        )

        debug_print: bool = Field(
            default=False,
            description="Enable debug printing to console.",
        )

    class UserValves(BaseModel):
        show_status: bool = Field(
            default=True, description="Show status of the action."
        )
        search_candidates: int = Field(
            default=50,
            description="Number of candidate episodes to retrieve from semantic search before local content matching.",
        )
        confirmation_timeout: int = Field(
            default=60,
            description="Timeout in seconds for confirmation dialog.",
        )
        ui_language: str = Field(
            default="en",
            description="Language for UI labels and status messages: 'en' (English) or 'ja' (Japanese)",
        )
        pass

    def __init__(self):
        self.valves = self.Valves()
        self.graphiti = None
        self._indices_built = False
        self._last_config = None

    def _get_config_hash(self) -> str:
        config_str = f"{self.valves.llm_client_type}_{self.valves.openai_api_url}_{self.valves.model}_" \
                    f"{self.valves.embedding_model}_{self.valves.api_key}_{self.valves.graph_db_backend}_" \
                    f"{self.valves.neo4j_uri}_{self.valves.neo4j_user}_{self.valves.neo4j_password}_" \
                    f"{self.valves.falkordb_host}_{self.valves.falkordb_port}_{self.valves.falkordb_username}_" \
                    f"{self.valves.falkordb_password}"
        return hashlib.md5(config_str.encode()).hexdigest()

    def _config_changed(self) -> bool:
        current_hash = self._get_config_hash()
        return self._last_config != current_hash

    def _config_ready(self) -> tuple[bool, str]:
        if not (self.valves.api_key or "").strip():
            return False, "api_key is empty"

        backend = (self.valves.graph_db_backend or "").lower().strip()

        if backend == "neo4j":
            pass
        elif backend == "falkordb":
            pass
        else:
            return False, f"Unsupported backend '{self.valves.graph_db_backend}'"

        return True, ""

    def _initialize_graphiti(self) -> bool:
        try:
            os.environ['GRAPHITI_TELEMETRY_ENABLED'] = 'true' if self.valves.graphiti_telemetry_enabled else 'false'
            os.environ['SEMAPHORE_LIMIT'] = str(self.valves.semaphore_limit)

            llm_config = LLMConfig(
                api_key=self.valves.api_key,
                model=self.valves.model,
                small_model=self.valves.small_model,
                base_url=self.valves.openai_api_url,
            )

            if self.valves.llm_client_type.lower() == "openai":
                llm_client = MultiUserOpenAIClient(config=llm_config)
            elif self.valves.llm_client_type.lower() == "generic":
                llm_client = MultiUserOpenAIGenericClient(config=llm_config)
            else:
                llm_client = MultiUserOpenAIClient(config=llm_config)

            falkor_driver = None
            if self.valves.graph_db_backend.lower() == "falkordb":
                from graphiti_core.driver.falkordb_driver import FalkorDriver
                falkor_driver = FalkorDriver(
                    host=self.valves.falkordb_host,
                    port=self.valves.falkordb_port,
                    username=self.valves.falkordb_username,
                    password=self.valves.falkordb_password,
                )

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
                print("Graphiti initialized successfully for delete action.")
            return True

        except Exception as e:
            print(f"Graphiti initialization failed: {e}")
            return False

    async def _build_indices(self) -> bool:
        if self.graphiti is None:
            return False
        if self._indices_built:
            return True
        try:
            await self.graphiti.build_indices_and_constraints()
            self._indices_built = True
            return True
        except Exception as e:
            print(f"Failed to build Graphiti indices: {e}")
            return False

    async def _ensure_graphiti_initialized(self) -> bool:
        ready, reason = self._config_ready()
        if not ready:
            if self.valves.debug_print:
                print(f"Graphiti init skipped: {reason}")
            return False
        if self._config_changed():
            self.graphiti = None
            self._indices_built = False

        if self.graphiti is None:
            if not self._initialize_graphiti():
                return False

        if not self._indices_built:
            if not await self._build_indices():
                return False

        return True

    def _get_group_id(self, user: dict) -> Optional[str]:
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
        content = message.get("content")

        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    return item.get("text", "")
            return ""

        return content if isinstance(content, str) else ""

    def _calculate_content_similarity(self, query: str, episode_content: str) -> float:
        """
        Calculate similarity score between query and episode content.
        Uses substring matching and common word overlap.
        Returns a score between 0 and 1.
        """
        query_lower = query.lower()
        content_lower = episode_content.lower()

        # Check for exact substring match (highest priority)
        if query_lower in content_lower or content_lower in query_lower:
            # Calculate overlap ratio
            shorter = min(len(query_lower), len(content_lower))
            longer = max(len(query_lower), len(content_lower))
            return 0.8 + (0.2 * shorter / longer)

        # Word-based overlap
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())

        if not query_words or not content_words:
            return 0.0

        # Jaccard similarity
        intersection = query_words & content_words
        union = query_words | content_words

        return len(intersection) / len(union) if union else 0.0

    def _sanitize_search_query(self, query: str) -> str:
        """
        Sanitize search query to avoid FalkorDB/RediSearch syntax errors.

        Only removes the most problematic characters that cause RediSearch errors.
        Keeps most punctuation to preserve query meaning.
        """
        # Remove characters that cause RediSearch syntax errors:
        # @ - field prefix
        # : - field separator
        # " - quote operator
        # ( ) - grouping operators
        # | - OR operator
        sanitized = re.sub(r'[@:"()|]', ' ', query)

        # Replace multiple spaces with single space
        sanitized = re.sub(r'\s+', ' ', sanitized)

        # Trim whitespace
        sanitized = sanitized.strip()

        return sanitized

    def _get_search_config(self) -> SearchConfig:
        """Get search configuration based on the configured search strategy."""
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

        user_valves = self.UserValves.model_validate((__user__ or {}).get("valves", {}))

        is_ja = self._is_japanese_preferred(user_valves)

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

        if __event_call__ is None:
            if __event_emitter__ and user_valves.show_status:
                msg = "❌ 確認ダイアログが利用できません" if is_ja else "❌ Confirmation dialog not available"
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

        if not last_assistant_message and not last_user_message:
            if __event_emitter__ and user_valves.show_status:
                msg = "❌ 有効なメッセージが見つかりません" if is_ja else "❌ No valid message found"
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": msg, "done": True},
                    }
                )
            return None

        # Extract content from user and assistant messages
        user_content = self._get_content_from_message(last_user_message) if last_user_message else ""
        assistant_content = self._get_content_from_message(last_assistant_message) if last_assistant_message else ""

        if self.valves.debug_print:
            print(f"=== Delete Action: Message Analysis ===")
            print(f"Total messages in body: {len(messages)}")
            print(f"User content (first 100 chars): {user_content[:100] if user_content else 'N/A'}...")
            print(f"Assistant content (first 100 chars): {assistant_content[:100] if assistant_content else 'N/A'}...")

        if not user_content and not assistant_content:
            if __event_emitter__ and user_valves.show_status:
                msg = "❌ メッセージ内容が空です" if is_ja else "❌ Message content is empty"
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": msg, "done": True},
                    }
                )
            return None

        chat_id = body.get("chat_id", "unknown")

        # Set user headers in context variable
        headers = self._get_user_info_headers(__user__, chat_id)
        if headers:
            user_headers_context.set(headers)

        if __event_emitter__ and user_valves.show_status:
            msg = "🔍 関連エピソードを検索中..." if is_ja else "🔍 Searching for related episodes..."
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": msg, "done": False},
                }
            )

        try:
            # Search for episodes matching the message content
            group_id = self._get_group_id(__user__)
            search_config = self._get_search_config()
            max_len = self.valves.max_search_message_length

            # Search separately with user and assistant content to handle both filter settings
            # (save_user_message=True or save_assistant_response=True)
            all_candidates = {}  # uuid -> episode (dedupe)

            for content_type, content in [("user", user_content), ("assistant", assistant_content)]:
                if not content:
                    continue

                search_query = content
                if self.valves.sanitize_search_query:
                    search_query = self._sanitize_search_query(search_query)
                    if not search_query:
                        continue

                if len(search_query) > max_len:
                    search_query = search_query[:max_len]

                if self.valves.debug_print:
                    print(f"=== Searching with {content_type} content ===")
                    print(f"Query length: {len(search_query)} chars")
                    print(f"Query (first 150 chars): {search_query[:150]}...")

                if group_id is not None:
                    search_results = await self.graphiti.search_(
                        query=search_query,
                        group_ids=[group_id],
                        config=search_config,
                    )
                else:
                    search_results = await self.graphiti.search_(
                        query=search_query,
                        config=search_config,
                    )

                if search_results and search_results.episodes:
                    for ep in search_results.episodes[:user_valves.search_candidates]:
                        all_candidates[ep.uuid] = ep

            candidates = list(all_candidates.values())

            if self.valves.debug_print:
                print(f"=== Combined Search Results ===")
                print(f"Total unique candidates: {len(candidates)}")

            if not candidates:
                if __event_emitter__ and user_valves.show_status:
                    msg = "ℹ️ 関連エピソードが見つかりませんでした" if is_ja else "ℹ️ No related episodes found"
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": msg, "done": True},
                        }
                    )
                return None

            # Local content matching - find best match by string similarity
            # Try both user and assistant content separately, use the better score
            scored_episodes = []
            for ep in candidates:
                scores = []
                if user_content:
                    scores.append(self._calculate_content_similarity(user_content, ep.content))
                if assistant_content:
                    scores.append(self._calculate_content_similarity(assistant_content, ep.content))
                similarity = max(scores) if scores else 0.0
                scored_episodes.append((ep, similarity))

            # Sort by similarity score (highest first)
            scored_episodes.sort(key=lambda x: x[1], reverse=True)

            if self.valves.debug_print:
                print(f"=== Local Content Matching ===")
                for idx, (ep, score) in enumerate(scored_episodes[:5], 1):
                    print(f"  [{idx}] Score: {score:.3f} - {ep.name}")
                    print(f"      Content (first 150 chars): {ep.content[:150]}...")

            # Get the best match
            best_match, best_score = scored_episodes[0]

            if self.valves.debug_print:
                print(f"=== Best Match ===")
                print(f"Score: {best_score:.3f}")
                print(f"Name: {best_match.name}")
                print(f"Content: {best_match.content[:300]}...")

            # Use list for consistency with confirmation dialog
            episodes = [best_match]

            # Build confirmation message
            content_preview = best_match.content[:200] + "..." if len(best_match.content) > 200 else best_match.content

            if is_ja:
                confirmation_message = f"""エピソードが見つかりました (一致度: {best_score:.1%}):

{best_match.name}:
{content_preview}

---
この操作は取り消せません。"""
                confirmation_title = "🗑️ メモリからエピソードを削除"
            else:
                confirmation_message = f"""Found episode (similarity: {best_score:.1%}):

{best_match.name}:
{content_preview}

---
This operation cannot be undone."""
                confirmation_title = "🗑️ Delete Episode from Memory"

            if __event_emitter__ and user_valves.show_status:
                msg = f"⏳ エピソードが見つかりました (一致度: {best_score:.0%}). 確認待ち..." if is_ja else f"⏳ Found episode (similarity: {best_score:.0%}). Waiting for confirmation..."
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": msg, "done": False},
                    }
                )

            # Show confirmation dialog
            try:
                confirmation_task = __event_call__(
                    {
                        "type": "confirmation",
                        "data": {
                            "title": confirmation_title,
                            "message": confirmation_message,
                        },
                    }
                )

                result = await asyncio.wait_for(confirmation_task, timeout=user_valves.confirmation_timeout)

                if not result:
                    if self.valves.debug_print:
                        print("User cancelled deletion")
                    if __event_emitter__ and user_valves.show_status:
                        msg = "🚫 削除がキャンセルされました" if is_ja else "🚫 Deletion cancelled"
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {"description": msg, "done": True},
                            }
                        )
                    return None

            except asyncio.TimeoutError:
                if self.valves.debug_print:
                    print(f"Confirmation timed out after {user_valves.confirmation_timeout}s")
                if __event_emitter__ and user_valves.show_status:
                    msg = "⏰ 確認がタイムアウトしました" if is_ja else "⏰ Confirmation timed out"
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": msg, "done": True},
                        }
                    )
                return None

            # Delete the episode
            if self.valves.debug_print:
                print(f"=== Deleting episode ===")

            if __event_emitter__ and user_valves.show_status:
                msg = "🗑️ エピソードを削除中..." if is_ja else "🗑️ Deleting episode..."
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": msg, "done": False},
                    }
                )

            deleted_count = 0
            for ep in episodes:
                try:
                    await self.graphiti.remove_episode(ep.uuid)
                    deleted_count += 1
                    if self.valves.debug_print:
                        print(f"  Deleted episode: {ep.uuid} ({ep.name})")
                except Exception as e:
                    print(f"Failed to delete episode {ep.uuid}: {e}")
                    if self.valves.debug_print:
                        traceback.print_exc()

            if deleted_count > 0:
                if self.valves.debug_print:
                    print(f"=== Episode deleted ===")
                if __event_emitter__ and user_valves.show_status:
                    msg = "✅ エピソードを削除しました" if is_ja else "✅ Episode deleted"
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": msg, "done": True},
                        }
                    )
            else:
                if self.valves.debug_print:
                    print(f"=== Failed to delete episode ===")
                if __event_emitter__ and user_valves.show_status:
                    msg = "❌ エピソードの削除に失敗しました" if is_ja else "❌ Failed to delete episode"
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": msg, "done": True},
                        }
                    )

        except Exception as e:
            error_type = type(e).__name__
            print(f"Delete action error: {e}")
            if self.valves.debug_print:
                traceback.print_exc()

            if __event_emitter__ and user_valves.show_status:
                msg = f"❌ エラー: {error_type}" if is_ja else f"❌ Error: {error_type}"
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": msg, "done": True},
                    }
                )

        return None
