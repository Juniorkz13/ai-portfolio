from langgraph.graph import StateGraph, END
from app.core.state import LegalGraphState
from app.agents.legal_interpreter import LegalInterpreterAgent
from app.agents.query_planner import QueryPlannerAgent
from app.agents.retriever import RetrieverAgent
from app.agents.cross_reference import CrossReferenceAgent
from app.agents.risk import RiskAgent
from app.agents.answer_agent import AnswerAgent
from app.llm.client import FakeLLMClient
from typing import Optional, Dict, Any
from app.core.logging import get_logger

logger = get_logger(__name__)

class LegalRAGWorkflow:
    """Workflow principal do sistema Legal RAG Multi-Agent"""
    
    def __init__(self):
        self.agents = {}
        self._initialize_agents()
        logger.info("LegalRAGWorkflow initialized")
    
    def _initialize_agents(self):
        """Inicializa agentes com fallback se algum falhar"""
        try:
            from app.agents.query_planner import QueryPlannerAgent
            self.agents["query_planner"] = QueryPlannerAgent()
        except Exception as e:
            logger.warning(f"QueryPlannerAgent not available: {e}")
        
        try:
            from app.agents.retriever import RetrieverAgent
            self.agents["retriever"] = RetrieverAgent()
        except Exception as e:
            logger.warning(f"RetrieverAgent not available: {e}")
        
        try:
            from app.agents.cross_reference import CrossReferenceAgent
            self.agents["cross_reference"] = CrossReferenceAgent()
        except Exception as e:
            logger.warning(f"CrossReferenceAgent not available: {e}")
        
        try:
            from app.agents.legal_interpreter import LegalInterpreterAgent
            self.agents["legal_interpreter"] = LegalInterpreterAgent()
        except Exception as e:
            logger.warning(f"LegalInterpreterAgent not available: {e}")
        
        try:
            from app.agents.risk import RiskAgent
            self.agents["risk"] = RiskAgent()
        except Exception as e:
            logger.warning(f"RiskAgent not available: {e}")
        
        try:
            from app.agents.answer_agent import AnswerAgent
            self.agents["answer_agent"] = AnswerAgent()
        except Exception as e:
            logger.warning(f"AnswerAgent not available: {e}")
    
    def run(
        self,
        question: str,
        documents: Optional[list] = None,
        request_id: str = ""
    ) -> Dict[str, Any]:
        """
        Executa o workflow completo de análise jurídica
        """
        try:
            logger.info(
                "workflow_started",
                extra={
                    "extra": {
                        "request_id": request_id,
                        "question_length": len(question),
                        "agents_available": list(self.agents.keys())
                    }
                }
            )
            
            documents = documents or []
            queries = [question]
            retrieved_docs = documents
            has_conflicts = False
            is_ambiguous = False
            missing_info = []
            risk_level = "médio"
            domain = "Trabalhista"
            answer = f"Análise da questão: {question}"
            disclaimer = "⚠️ Este é um parecer preliminar automatizado. Consulte um advogado especializado para análise detalhada do seu caso específico."
            recommendations = []
            
            # 1. Query Planning
            if "query_planner" in self.agents:
                try:
                    query_plan_result = self.agents["query_planner"].run({"question": question})
                    queries = query_plan_result.get("queries", [question])
                    logger.info(f"Queries geradas: {len(queries)}")
                except Exception as e:
                    logger.error(f"QueryPlanner error: {e}")
            
            # 2. Retrieval
            if "retriever" in self.agents:
                try:
                    retriever_result = self.agents["retriever"].run({
                        "queries": queries,
                        "documents": documents
                    })
                    retrieved_docs = retriever_result.get("documents", documents)
                    logger.info(f"Documentos recuperados: {len(retrieved_docs)}")
                except Exception as e:
                    logger.error(f"Retriever error: {e}")
            
            # 3. Cross-Reference
            if "cross_reference" in self.agents:
                try:
                    cross_ref_result = self.agents["cross_reference"].run({
                        "documents": retrieved_docs
                    })
                    has_conflicts = cross_ref_result.get("has_conflicts", False)
                    logger.info(f"Conflitos encontrados: {has_conflicts}")
                except Exception as e:
                    logger.error(f"CrossReference error: {e}")
            
            # 4. Legal Interpretation
            if "legal_interpreter" in self.agents:
                try:
                    interpretation_result = self.agents["legal_interpreter"].run({
                        "question": question,
                        "documents": retrieved_docs
                    })
                    is_ambiguous = interpretation_result.get("is_ambiguous", False)
                    domain = interpretation_result.get("domain", "Trabalhista")
                    missing_info = interpretation_result.get("missing_info", [])
                    logger.info(f"Domínio identificado: {domain}, Ambíguo: {is_ambiguous}")
                except Exception as e:
                    logger.error(f"LegalInterpreter error: {e}")
            
            # 5. Risk Assessment
            if "risk_assessment" in self.agents:
                try:
                    risk_result = self.agents["risk_assessment"].run({
                        "question": question,
                        "documents": retrieved_docs,
                        "has_conflicts": has_conflicts,
                        "is_ambiguous": is_ambiguous
                    })
                    risk_level = risk_result.get("risk_level", "médio")
                    recommendations = risk_result.get("recommendations", [])
                    logger.info(f"Nível de risco: {risk_level}")
                except Exception as e:
                    logger.error(f"RiskAssessment error: {e}")
            
            # 6. Generate Answer
            if "answer_agent" in self.agents:
                try:
                    answer_result = self.agents["answer_agent"].run({
                        "question": question,
                        "documents": retrieved_docs,
                        "risk_level": risk_level,
                        "domain": domain,
                        "has_conflicts": has_conflicts,
                        "is_ambiguous": is_ambiguous
                    })
                    answer = answer_result.get("answer", answer)
                    disclaimer = answer_result.get("disclaimer", disclaimer)
                    logger.info("Resposta gerada com sucesso")
                except Exception as e:
                    logger.error(f"AnswerAgent error: {e}", exc_info=True)
            
            # Adicionar recomendações padrão se vazio
            if not recommendations:
                recommendations = [
                    f"📋 Consultar advogado especialista em {domain}",
                    "📄 Reunir toda documentação relevante",
                    "⏰ Verificar prazos legais aplicáveis"
                ]
            
            # Resultado final
            result = {
                "request_id": request_id,
                "question": question,
                "status": "completed",
                "risk_level": risk_level,
                "domain": domain,
                "analysis": {
                    "answer": answer,
                    "disclaimer": disclaimer,
                    "summary": f"✅ Análise jurídica automática concluída sobre: {question}",
                    "documents_processed": len(retrieved_docs),
                    "queries_generated": len(queries),
                    "has_conflicts": has_conflicts,
                    "is_ambiguous": is_ambiguous,
                    "missing_info": missing_info,
                    "recommendations": recommendations,
                    "confidence_score": "75%" if not has_conflicts and not is_ambiguous else "50%"
                },
                "agents_used": list(self.agents.keys()),
                "metadata": {
                    "workflow_version": "1.0.0",
                    "processing_time_ms": 0,
                    "language": "pt-BR"
                }
            }
            
            logger.info(
                "workflow_completed",
                extra={
                    "extra": {
                        "request_id": request_id,
                        "risk_level": risk_level,
                        "domain": domain,
                        "confidence": result["analysis"]["confidence_score"]
                    }
                }
            )
            
            return result
        
        except Exception as e:
            logger.error(
                f"Workflow error: {str(e)}",
                exc_info=True,
                extra={"extra": {"request_id": request_id}}
            )
            raise


def build_graph():
    graph = StateGraph(LegalGraphState)

    legal_interpreter = LegalInterpreterAgent()
    query_planner = QueryPlannerAgent()
    retriever = RetrieverAgent()
    cross_reference = CrossReferenceAgent()
    risk_agent = RiskAgent()
    answer_agent = AnswerAgent()

    def input_node(state: LegalGraphState):
        return state

    def legal_interpreter_node(state: LegalGraphState):
        question = state.get("question", "")
        result = legal_interpreter.run({"question": question})
        return {
            **state,
            "domain": result["domain"],
            "legal_intent": result["legal_intent"],
            "missing_information": result["missing_information"],
        }

    def query_planner_node(state: LegalGraphState):
        result = query_planner.run(state)
        return {
            **state,
            "queries": result["queries"],
        }

    def retriever_node(state: LegalGraphState):
        queries = state.get("queries", [])

        if not queries:
            queries = ["consulta jurídica simulada"]

        result = retriever.run({"queries": queries})

        return {
            **state,
            "queries": queries,
            "documents": result["documents"],
        }

    def cross_reference_node(state: LegalGraphState):
        result = cross_reference.run({
            "documents": state["documents"]
        })
        return {
            **state,
            **result
        }

    def risk_node(state: LegalGraphState):
        result = risk_agent.run({
            "has_conflict": state["has_conflict"],
            "conflicts": state.get("conflicts", []),
            "missing_information": state.get("missing_information", [])
        })
        return {
            **state,
            **result
        }
    
    llm_client = FakeLLMClient()
    answer_agent = AnswerAgent(llm_client)

    def answer_node(state: LegalGraphState):
        result = answer_agent.run({
            "question": state["question"],
            "documents": state["documents"],
            "risk_level": state["risk_level"],
            "risk_factors": state["risk_factors"],
            "recommendation": state["recommendation"],
        })
        return {
            **state,
            **result
        }

    graph.add_node("input", input_node)
    graph.add_node("legal_interpreter", legal_interpreter_node)
    graph.add_node("query_planner", query_planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("cross_reference", cross_reference_node)
    graph.add_node("risk", risk_node)

    graph.set_entry_point("input")
    graph.add_edge("input", "legal_interpreter")
    graph.add_edge("legal_interpreter", "query_planner")
    graph.add_edge("query_planner", "retriever")
    graph.add_edge("retriever", "cross_reference")
    graph.add_edge("cross_reference", "risk")

    graph.add_node("answer", answer_node)
    graph.add_edge("risk", "answer")
    graph.add_edge("answer", END)

    return graph.compile()