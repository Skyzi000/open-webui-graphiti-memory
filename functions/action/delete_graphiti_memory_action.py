"""
title: Delete Graphiti Memory Action Button
description: Action button to delete an episode from Graphiti knowledge graph memory by matching the clicked message's ID.
author: Skyzi000
author_url: https://github.com/Skyzi000
repository_url: https://github.com/Skyzi000/open-webui-graphiti-memory
version: 0.3.0
requirements: graphiti-core
icon_url: data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzMiIgaGVpZ2h0PSIzMiIgdmlld0JveD0iMCAwIDMyIDMyIj4KICA8cGF0aCBkPSJNOCA5aDN2MTZIOHptNiAwaDN2MTZoLTN6bTYgMGgzdjE2aC0zeiIgZmlsbD0iIzRjNGM0YyIvPgogIDxyZWN0IHg9IjYiIHk9IjYiIHdpZHRoPSIyMCIgaGVpZ2h0PSIzIiByeD0iMSIgZmlsbD0iIzRjNGM0YyIvPgogIDxwYXRoIGQ9Ik0xMSA2VjRhMiAyIDAgMCAxIDItMmg2YTIgMiAwIDAgMSAyIDJ2MiIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjNGM0YzRjIiBzdHJva2Utd2lkdGg9IjEuNSIvPgogIDxwYXRoIGQ9Ik03IDloMTh2MThhMiAyIDAgMCAxLTIgMkg5YTIgMiAwIDAgMS0yLTJWOXoiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzRjNGM0YyIgc3Ryb2tlLXdpZHRoPSIxLjUiLz4KPC9zdmc+

Note on FalkorDB backend:
  FalkorDB requires additional setup due to Redis version conflicts.
  See the README for details: https://github.com/Skyzi000/open-webui-graphiti-memory#falkordb-alternative-not-recently-tested

Design:
- Main class: Action
- Finds the episode matching the clicked message's ID via direct database query
- Shows confirmation dialog with the found episode
- Deletes the episode after user confirmation
"""

import asyncio
import contextvars
import hashlib
import os
import re
import traceback
from typing import Dict, Optional
from urllib.parse import quote

from pydantic import BaseModel, Field

from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.models.nodes.node_db_queries import EPISODIC_NODE_RETURN
from graphiti_core.nodes import EpisodicNode, get_episodic_node_from_record
from openai import AsyncOpenAI

# Context variable to store user-specific headers for each async request
# This ensures complete isolation between concurrent requests without locks
# Default is None, callers should always call .set() before use
user_headers_context: contextvars.ContextVar[Optional[Dict[str, str]]] = contextvars.ContextVar(
    'user_headers', default=None
)


async def find_episodes_by_message_id(
    driver, user_message_id: str, group_id: str | None,
) -> list[EpisodicNode]:
    """Find episodic nodes by message ID via direct Cypher query.

    Searches two storage formats:
    - Filter: source_description contains '_message_{user_message_id}'
    - Pipeline: name contains '_{user_message_id}'
    """
    sd_pattern = f"_message_{user_message_id}"
    name_pattern = f"_{user_message_id}"

    group_filter = "\nAND e.group_id = $group_id" if group_id is not None else ""

    query = (
        """
        MATCH (e:Episodic)
        WHERE (e.source_description CONTAINS $sd_pattern OR e.name CONTAINS $name_pattern)
        """
        + group_filter
        + "\nRETURN\n"
        + EPISODIC_NODE_RETURN
    )

    params: dict = {
        "sd_pattern": sd_pattern,
        "name_pattern": name_pattern,
    }
    if group_id is not None:
        params["group_id"] = group_id

    records, _, _ = await driver.execute_query(query, **params, routing_="r")

    return [get_episodic_node_from_record(record) for record in records]


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

        debug_print: bool = Field(
            default=False,
            description="Enable debug printing to console.",
        )

    class UserValves(BaseModel):
        show_status: bool = Field(
            default=True, description="Show status of the action."
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

        try:
            user_id = user.get('id', 'unknown')
            user_email = user.get('email') or user_id
            user_name = user.get('name') or user_id

            sanitized_email = str(user_email).replace('@', '_at_').replace('.', '_')
            sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '_', str(user_name))

            group_id = self.valves.group_id_format.format(
                user_id=user_id,
                user_email=sanitized_email,
                user_name=sanitized_name,
            )

            group_id = re.sub(r'[^a-zA-Z0-9_-]', '_', group_id)
            return group_id
        except (KeyError, ValueError, IndexError) as e:
            # Invalid template format - fall back to user_id
            if self.valves.debug_print:
                print(f"Warning: Invalid group_id_format template: {e}. Falling back to user_id.")
            user_id = user.get('id', 'unknown')
            return re.sub(r'[^a-zA-Z0-9_-]', '_', str(user_id))

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

        def _sanitize_header_value(value: str) -> str:
            """Remove control characters and newlines to prevent header injection."""
            return re.sub(r'[\x00-\x1f\x7f\r\n]', '', str(value))
        
        headers = {}
        if user:
            if user.get('name'):
                headers['X-OpenWebUI-User-Name'] = quote(_sanitize_header_value(user['name']), safe=" ")
            if user.get('id'):
                headers['X-OpenWebUI-User-Id'] = _sanitize_header_value(user['id'])
            if user.get('email'):
                headers['X-OpenWebUI-User-Email'] = _sanitize_header_value(user['email'])
            if user.get('role'):
                headers['X-OpenWebUI-User-Role'] = _sanitize_header_value(user['role'])
        
        if chat_id:
            headers['X-OpenWebUI-Chat-Id'] = _sanitize_header_value(chat_id)
        
        return headers

    def _is_japanese_preferred(self, user_valves: "Action.UserValves") -> bool:
        return user_valves.ui_language.lower() == 'ja'

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

        user_message_id = last_user_message.get("id") if last_user_message else None
        if not user_message_id:
            if __event_emitter__ and user_valves.show_status:
                msg = "❌ メッセージIDが見つかりません" if is_ja else "❌ Message ID not found"
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
        user_headers_context.set(headers)

        if __event_emitter__ and user_valves.show_status:
            msg = "🔍 エピソードを検索中..." if is_ja else "🔍 Searching for episodes..."
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": msg, "done": False},
                }
            )

        try:
            group_id = self._get_group_id(__user__)
            episodes = await find_episodes_by_message_id(
                self.graphiti.driver, user_message_id, group_id,
            )

            if self.valves.debug_print:
                print(f"=== Episode Search by Message ID ===")
                print(f"Message ID: {user_message_id}")
                print(f"Group ID: {group_id}")
                print(f"Found {len(episodes)} episode(s)")
                for ep in episodes:
                    content_snippet = ep.content[:150] + "..." if len(ep.content) > 150 else ep.content
                    print(f"  - {ep.name}: {content_snippet}")

            if not episodes:
                if __event_emitter__ and user_valves.show_status:
                    msg = "ℹ️ 対応するエピソードが見つかりませんでした" if is_ja else "ℹ️ No matching episodes found"
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": msg, "done": True},
                        }
                    )
                return None

            # Build confirmation message
            if len(episodes) == 1:
                ep = episodes[0]
                content_preview = ep.content[:200] + "..." if len(ep.content) > 200 else ep.content
                episodes_text = f"{ep.name}:\n{content_preview}"
            else:
                parts = []
                for i, ep in enumerate(episodes, 1):
                    content_preview = ep.content[:200] + "..." if len(ep.content) > 200 else ep.content
                    parts.append(f"{i}. {ep.name}:\n{content_preview}")
                episodes_text = "\n\n".join(parts)

            if is_ja:
                header = "エピソードが見つかりました:" if len(episodes) == 1 else f"{len(episodes)}件のエピソードが見つかりました:"
                confirmation_message = f"""{header}

{episodes_text}

---
この操作は取り消せません。"""
                confirmation_title = "🗑️ メモリからエピソードを削除"
            else:
                header = "Found episode:" if len(episodes) == 1 else f"Found {len(episodes)} episodes:"
                confirmation_message = f"""{header}

{episodes_text}

---
This operation cannot be undone."""
                confirmation_title = "🗑️ Delete Episode from Memory"

            if __event_emitter__ and user_valves.show_status:
                msg = f"⏳ {len(episodes)}件のエピソードが見つかりました。確認待ち..." if is_ja else f"⏳ Found {len(episodes)} episode(s). Waiting for confirmation..."
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
