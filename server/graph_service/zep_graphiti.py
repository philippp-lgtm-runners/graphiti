import logging
import os
from typing import Annotated

from fastapi import Depends, HTTPException
from graphiti_core import Graphiti  # type: ignore
from graphiti_core.edges import EntityEdge  # type: ignore
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig  # type: ignore
from graphiti_core.errors import EdgeNotFoundError, GroupsEdgesNotFoundError, NodeNotFoundError
from graphiti_core.llm_client import LLMClient  # type: ignore
from graphiti_core.nodes import EntityNode, EpisodicNode  # type: ignore

from graph_service.config import ZepEnvDep
from graph_service.dto import FactResult


def _build_independent_embedder():
    """Build an OpenAIEmbedder pointed at a separate (local) OpenAI-compatible
    endpoint so the LLM can live on a provider without embedding support
    (e.g. OpenRouter). Returns None if EMBEDDING_BASE_URL is unset.

    Must be passed to Graphiti.__init__ — patching self.embedder afterwards
    does not take effect because NodeNamespace/EdgeNamespace capture the
    original embedder reference at construction time.
    """
    emb_base = os.getenv('EMBEDDING_BASE_URL')
    if not emb_base:
        return None
    return OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key=os.getenv('EMBEDDING_API_KEY', 'not-needed'),
            base_url=emb_base,
            embedding_model=os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'),
        )
    )

logger = logging.getLogger(__name__)

# Module-level singleton for the FalkorDB driver.
# FalkorDriver.__init__ auto-schedules build_indices_and_constraints as a
# background task on every instantiation. Under FastAPI's per-request
# dependency (get_graphiti), this would fire dozens of parallel RediSearch
# index-creation queries that clobber each other on a single Redis connection
# ("Connection closed by server"). Caching one driver avoids that race.
_FALKOR_DRIVER = None


def _get_falkor_driver():
    global _FALKOR_DRIVER
    if _FALKOR_DRIVER is None:
        from graphiti_core.driver.falkordb_driver import FalkorDriver  # type: ignore
        _FALKOR_DRIVER = FalkorDriver(
            host=os.getenv('FALKORDB_HOST', 'falkordb'),
            port=int(os.getenv('FALKORDB_PORT', '6379')),
            username=os.getenv('FALKORDB_USER') or None,
            password=os.getenv('FALKORDB_PASSWORD') or None,
        )
    return _FALKOR_DRIVER


class ZepGraphiti(Graphiti):
    def __init__(self, uri: str, user: str, password: str, llm_client: LLMClient | None = None):
        graph_driver = None
        self._owns_driver = True
        if os.getenv('DB_BACKEND', '').lower() == 'falkordb':
            graph_driver = _get_falkor_driver()
            self._owns_driver = False
        embedder = _build_independent_embedder()
        super().__init__(
            uri, user, password, llm_client,
            graph_driver=graph_driver,
            embedder=embedder,
        )

    async def close(self):
        # Skip closing the shared singleton driver; it lives for the process
        # lifetime. Closing it on the first request would break subsequent ones.
        if self._owns_driver:
            await super().close()

    def _falkor_single_group_driver(self, group_ids, driver):
        # graphiti_core's handle_multiple_group_ids decorator only clones the
        # FalkorDB driver when len(group_ids) > 1. For a single group_id, it
        # relies on driver._database already pointing at that group's graph —
        # which is only true if add_episode ran for that group in the same
        # process. Clone explicitly so calls hit the right FalkorDB graph.
        if (
            driver is None
            and group_ids
            and len(group_ids) == 1
            and os.getenv('DB_BACKEND', '').lower() == 'falkordb'
        ):
            return self.driver.clone(database=group_ids[0])
        return driver

    async def search(self, query, center_node_uuid=None, group_ids=None, num_results=10,
                     search_filter=None, driver=None):
        driver = self._falkor_single_group_driver(group_ids, driver)
        return await super().search(
            query=query,
            center_node_uuid=center_node_uuid,
            group_ids=group_ids,
            num_results=num_results,
            search_filter=search_filter,
            driver=driver,
        )

    async def retrieve_episodes(self, reference_time, last_n=10, group_ids=None,
                                source=None, driver=None, saga=None):
        driver = self._falkor_single_group_driver(group_ids, driver)
        return await super().retrieve_episodes(
            reference_time=reference_time,
            last_n=last_n,
            group_ids=group_ids,
            source=source,
            driver=driver,
            saga=saga,
        )

    async def save_entity_node(self, name: str, uuid: str, group_id: str, summary: str = ''):
        new_node = EntityNode(
            name=name,
            uuid=uuid,
            group_id=group_id,
            summary=summary,
        )
        await new_node.generate_name_embedding(self.embedder)
        await new_node.save(self.driver)
        return new_node

    async def _find_entity_edge(self, uuid: str):
        # Locate an EntityEdge by uuid across FalkorDB graphs.
        #
        # Upstream `EntityEdge.get_by_uuid(self.driver, uuid)` queries whichever
        # graph the driver's `_database` currently points at, which for FalkorDB
        # is the last group_id that `add_episode` wrote to — or `default_db`
        # if nothing has been written in this process yet. Edges written for
        # other groups are therefore invisible to GET/DELETE /entity-edge/{uuid}.
        # We iterate all FalkorDB graphs and return the cloned driver that
        # contains the edge, so the caller can perform any follow-up ops
        # (e.g. delete) against the correct graph.
        if os.getenv('DB_BACKEND', '').lower() != 'falkordb':
            edge = await EntityEdge.get_by_uuid(self.driver, uuid)
            return edge, self.driver
        try:
            graphs = await self.driver.client.list_graphs()
        except Exception:
            graphs = []
        for g in graphs:
            if g == 'default_db':
                continue
            cloned = self.driver.clone(database=g)
            try:
                edge = await EntityEdge.get_by_uuid(cloned, uuid)
                return edge, cloned
            except EdgeNotFoundError:
                continue
        raise EdgeNotFoundError(uuid)

    async def get_entity_edge(self, uuid: str):
        try:
            edge, _ = await self._find_entity_edge(uuid)
            return edge
        except EdgeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e

    async def delete_group(self, group_id: str):
        try:
            edges = await EntityEdge.get_by_group_ids(self.driver, [group_id])
        except GroupsEdgesNotFoundError:
            logger.warning(f'No edges found for group {group_id}')
            edges = []

        nodes = await EntityNode.get_by_group_ids(self.driver, [group_id])

        episodes = await EpisodicNode.get_by_group_ids(self.driver, [group_id])

        for edge in edges:
            await edge.delete(self.driver)

        for node in nodes:
            await node.delete(self.driver)

        for episode in episodes:
            await episode.delete(self.driver)

    async def delete_entity_edge(self, uuid: str):
        try:
            edge, driver = await self._find_entity_edge(uuid)
            await edge.delete(driver)
        except EdgeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e

    async def delete_episodic_node(self, uuid: str):
        try:
            episode = await EpisodicNode.get_by_uuid(self.driver, uuid)
            await episode.delete(self.driver)
        except NodeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e


async def get_graphiti(settings: ZepEnvDep):
    client = ZepGraphiti(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    if settings.openai_base_url is not None:
        client.llm_client.config.base_url = settings.openai_base_url
    if settings.openai_api_key is not None:
        client.llm_client.config.api_key = settings.openai_api_key
    if settings.model_name is not None:
        client.llm_client.model = settings.model_name

    try:
        yield client
    finally:
        await client.close()


async def initialize_graphiti(settings: ZepEnvDep):
    client = ZepGraphiti(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    await client.build_indices_and_constraints()


def get_fact_result_from_edge(edge: EntityEdge):
    return FactResult(
        uuid=edge.uuid,
        name=edge.name,
        fact=edge.fact,
        valid_at=edge.valid_at,
        invalid_at=edge.invalid_at,
        created_at=edge.created_at,
        expired_at=edge.expired_at,
    )


ZepGraphitiDep = Annotated[ZepGraphiti, Depends(get_graphiti)]
