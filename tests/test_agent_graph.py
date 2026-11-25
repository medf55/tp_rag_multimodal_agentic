"""Tests déterministes du graphe LangGraph ReAct (sans appel réseau/API)."""

from __future__ import annotations

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from multimodal_rag.agent import AgenticMultimodalRAG
from multimodal_rag.config import Settings
from multimodal_rag.llm import MultimodalRAGPipeline
from multimodal_rag.models import RetrievedSource
from multimodal_rag.retriever import MultimodalRetriever
from multimodal_rag.vector_store import MultimodalVectorStore


class _FakeToolCallingChatModel(GenericFakeChatModel):
    """FakeChatModel supportant `bind_tools` tout en traçant le mode utilisé."""

    bound_calls: int = 0
    no_tool_calls: int = 0
    unbound_calls: int = 0

    def bind_tools(self, tools, **kwargs):  # type: ignore[override]
        parent = self
        tool_choice = kwargs.get("tool_choice")

        class _BoundWrapper:
            def invoke(self, messages, **inv_kwargs):
                if tool_choice == "none":
                    parent.no_tool_calls += 1
                else:
                    parent.bound_calls += 1
                return parent.invoke(messages, **inv_kwargs)

        return _BoundWrapper()

    def invoke(self, input, config=None, **kwargs):  # type: ignore[override]
        self.unbound_calls += 1
        return super().invoke(input, config=config, **kwargs)


class _StubStore(MultimodalVectorStore):
    def __init__(self) -> None:
        self._records = {
            "doc:texte:4:0": {
                "id": "doc:texte:4:0",
                "document": "Multi-head attention allows attending to different representation subspaces.",
                "metadata": {
                    "type": "texte",
                    "pdf": "attention is all you need.pdf",
                    "page": 4,
                    "source": "resultats_pdf/textes/page_004.txt",
                },
            }
        }

    def get_by_id(self, item_id: str):
        return self._records.get(item_id)


class _StubRetriever(MultimodalRetriever):
    def __init__(self) -> None:
        self.store = _StubStore()
        self.queries: list[str] = []

    def search(
        self,
        texte=None,
        image=None,
        type_resultat=None,
        k=5,
        image_quota=2,
    ):
        self.queries.append(str(texte))
        return [
            RetrievedSource(
                reference="S1",
                id="doc:texte:4:0",
                type="texte",
                pdf="attention is all you need.pdf",
                page=4,
                source="resultats_pdf/textes/page_004.txt",
                content="Multi-head attention allows attending to different representation subspaces.",
                similarity=0.812,
            )
        ]


def _make_agent(responses: list[AIMessage], max_tool_calls: int = 4):
    settings = Settings(max_tool_calls=max_tool_calls, recursion_limit=15)
    retriever = _StubRetriever()
    fake_llm = _FakeToolCallingChatModel(messages=iter(responses))
    rag_pipeline = MultimodalRAGPipeline(retriever=retriever, llm=fake_llm, max_images=3)
    agent = AgenticMultimodalRAG(
        settings=settings,
        retriever=retriever,
        rag_pipeline=rag_pipeline,
        llm=fake_llm,
    )
    return agent, retriever, fake_llm


def test_graph_direct_response_without_tool() -> None:
    agent, retriever, _ = _make_agent(
        [AIMessage(content="Bonjour ! Je peux interroger le PDF indexé.")]
    )
    resp = agent.ask("Bonjour !")

    assert resp.tool_calls == []
    assert retriever.queries == []
    assert "Bonjour" in resp.answer
    assert resp.steps_count == 2  # HumanMessage + AIMessage


def test_graph_react_loop_mind_body_mind() -> None:
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "retriever_tool",
                "args": {
                    "requete": "multi-head attention subspaces",
                    "type_resultat": "texte",
                    "k": 4,
                },
                "id": "call_1",
                "type": "tool_call",
            }
        ],
    )
    final_msg = AIMessage(
        content="L'attention multi-têtes observe plusieurs sous-espaces [attention is all you need.pdf, p. 4]."
    )
    agent, retriever, _ = _make_agent([tool_call_msg, final_msg], max_tool_calls=4)

    resp = agent.ask("Pourquoi utiliser plusieurs têtes d'attention ?", type_resultat="texte", k=4)

    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "retriever_tool"
    assert retriever.queries == ["multi-head attention subspaces"]
    assert len(resp.sources) == 1
    assert resp.sources[0].page == 4
    assert "sous-espaces" in resp.answer
    assert resp.steps_count == 4  # Human -> AI(tool_call) -> ToolMessage -> AI(final)


def test_graph_enforces_max_tool_calls_budget() -> None:
    # Avec max_tool_calls=1 : après 1 appel exécuté par body, mind doit être
    # rappelé avec tool_choice="none" (no_tool_calls) pour synthétiser au lieu de boucler.
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "retriever_tool",
                "args": {"requete": "transformer attention", "type_resultat": "texte", "k": 3},
                "id": "call_budget_1",
                "type": "tool_call",
            }
        ],
    )
    forced_final_msg = AIMessage(content="Synthèse finale sous contrainte de budget.")
    agent, _, fake_llm = _make_agent([tool_call_msg, forced_final_msg], max_tool_calls=1)

    resp = agent.ask("Explique l'attention.")

    assert len(resp.tool_calls) == 1
    assert fake_llm.bound_calls == 1
    assert fake_llm.no_tool_calls == 1
    assert resp.answer == "Synthèse finale sous contrainte de budget."
