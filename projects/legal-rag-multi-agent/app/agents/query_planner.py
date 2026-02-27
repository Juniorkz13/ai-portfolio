import time
from typing import Dict, Any, List
from app.core.logging import get_logger
from app.agents.base import BaseAgent

logger = get_logger(__name__)


class QueryPlannerAgent(BaseAgent):
    """Agente de planejamento de consultas."""

    def run(self, input_data: dict) -> dict:
        start_time = time.time()

        try:
            question = input_data.get("question", "")
            domain = input_data.get("domain", "")
            legal_intent = input_data.get("legal_intent", "")
            missing_information = input_data.get("missing_information", [])
            request_id = input_data.get("request_id")

            # Se houver os parâmetros específicos, usa _generate_queries
            if domain or legal_intent:
                queries = self._generate_queries(domain, legal_intent, missing_information)
            else:
                # Caso contrário, usa _create_query_plan
                query_plan = self._create_query_plan(question)
                queries = query_plan.get("search_queries", [])

            # Garante que queries não estão vazias
            queries = [q.strip() for q in queries if q.strip()]
            if not queries:
                queries = ["busca geral"]

            result = {
                "queries": queries,
                "num_queries": len(queries),
                "complexity": "medium",
            }

            duration_ms = int((time.time() - start_time) * 1000)

            logger.info(
                "query_plan_created",
                extra={
                    "extra": {
                        "request_id": request_id,
                        "agent": "QueryPlannerAgent",
                        "num_queries": result["num_queries"],
                        "complexity": result["complexity"],
                        "duration_ms": duration_ms,
                    }
                },
            )

            return result

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)

            logger.error(
                "query_planning_failed",
                extra={
                    "extra": {
                        "request_id": input_data.get("request_id"),
                        "agent": "QueryPlannerAgent",
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "duration_ms": duration_ms,
                    }
                },
                exc_info=True,
            )

            raise

    def _create_query_plan(self, question: str) -> Dict[str, Any]:
        """
        Cria um plano de consulta otimizado baseado na questão do usuário.
        
        O plano inclui:
        - Queries de busca decompostas
        - Nível de complexidade
        - Área jurídica sugerida
        - Estratégia de busca recomendada
        
        Args:
            question: Pergunta do usuário
            
        Returns:
            Dicionário com o plano de consulta estruturado
        """
        question_lower = question.lower()
        
        # 1. Identifica área(s) jurídica(s)
        legal_areas = self._identify_legal_areas(question_lower)
        
        # 2. Detecta complexidade da questão
        complexity = self._assess_complexity(question, question_lower)
        
        # 3. Decompõe a questão em queries específicas
        search_queries = self._decompose_question(question, question_lower, complexity)
        
        # 4. Define estratégia de busca
        search_strategy = self._define_search_strategy(complexity, legal_areas)
        
        # 5. Identifica termos-chave jurídicos
        key_terms = self._extract_legal_terms(question_lower)
        
        query_plan = {
            "original_question": question,
            "search_queries": search_queries,
            "legal_areas": legal_areas,
            "complexity": complexity,
            "search_strategy": search_strategy,
            "key_legal_terms": key_terms,
            "requires_multi_step": complexity in ["high", "very_high"],
            "estimated_documents_needed": self._estimate_doc_count(complexity)
        }
        
        return query_plan

    def _identify_legal_areas(self, question_lower: str) -> List[str]:
        """Identifica as áreas jurídicas relacionadas à questão."""
        areas = []
        
        area_keywords = {
            "trabalhista": ["trabalho", "emprego", "clt", "rescisão", "demissão", 
                           "férias", "salário", "justa causa", "fgts", "trabalhador"],
            "civil": ["contrato", "civil", "indenização", "responsabilidade", 
                     "obrigação", "patrimônio", "direito civil", "ato ilícito"],
            "penal": ["crime", "penal", "prisão", "pena", "delito", "código penal", 
                     "homicídio", "furto", "roubo"],
            "consumidor": ["consumidor", "cdc", "produto", "serviço", "compra", 
                          "fornecedor", "garantia", "defeito"],
            "constitucional": ["constituição", "constitucional", "direitos fundamentais", 
                              "supremo", "stf", "inconstitucional"],
            "tributário": ["imposto", "tributo", "fiscal", "icms", "iptu", "ir", 
                          "contribuição", "fazenda"],
            "processual": ["processo", "ação", "recurso", "apelação", "sentença", 
                          "cpc", "procedimento", "prazo processual"]
        }
        
        for area, keywords in area_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                areas.append(area)
        
        # Se não identificou nenhuma área, marca como "geral"
        if not areas:
            areas.append("geral")
        
        return areas

    def _assess_complexity(self, question: str, question_lower: str) -> str:
        """Avalia a complexidade da questão."""
        complexity_score = 0
        
        # 1. Tamanho da questão
        word_count = len(question.split())
        if word_count > 50:
            complexity_score += 3
        elif word_count > 30:
            complexity_score += 2
        elif word_count > 15:
            complexity_score += 1
        
        # 2. Presença de múltiplas questões
        multi_question_indicators = ["e também", "além disso", "outra questão", 
                                     "e ainda", "também gostaria", "?", "e se"]
        multi_count = sum(1 for indicator in multi_question_indicators 
                         if indicator in question_lower)
        complexity_score += multi_count
        
        # 3. Termos jurídicos complexos
        complex_terms = ["legislação", "jurisprudência", "precedente", "súmula", 
                        "doutrina", "hermenêutica", "analogia", "constitucionalidade"]
        complex_count = sum(1 for term in complex_terms if term in question_lower)
        complexity_score += complex_count * 2
        
        # 4. Indicadores de cenário complexo
        scenario_indicators = ["conflito", "divergência", "múltiplas", "vários", 
                              "diferentes posições", "interpretações"]
        if any(indicator in question_lower for indicator in scenario_indicators):
            complexity_score += 2
        
        # 5. Pedidos de análise comparativa
        comparative = ["comparar", "diferença entre", "versus", "ou", "qual melhor"]
        if any(term in question_lower for term in comparative):
            complexity_score += 1
        
        # Mapeia score para nível de complexidade
        if complexity_score >= 8:
            return "very_high"
        elif complexity_score >= 5:
            return "high"
        elif complexity_score >= 2:
            return "medium"
        else:
            return "low"

    def _decompose_question(self, question: str, question_lower: str, 
                           complexity: str) -> List[str]:
        """Decompõe a questão em queries de busca específicas."""
        queries = []
        
        # Sempre adiciona a query original se não estiver vazia
        if question.strip():
            queries.append(question.strip())
        
        # Para questões simples, apenas a query original é suficiente
        if complexity == "low":
            return queries if queries else ["busca geral"]
        
        # Detecta e extrai sub-questões
        sub_questions = []
        
        # 1. Separa por separadores comuns
        separators = [" e também ", " além disso, ", " e ainda ", " também "]
        temp_question = question_lower
        for sep in separators:
            if sep in temp_question:
                parts = temp_question.split(sep)
                sub_questions.extend([p.strip() for p in parts if p.strip() and len(p.strip()) > 5])
                break
        
        # Adiciona sub-questões encontradas
        if sub_questions:
            queries.extend(sub_questions)
        
        # 2. Extrai aspectos específicos mencionados
        aspect_indicators = {
            "prazo": ["prazo", "tempo", "dias", "meses", "quando"],
            "valor": ["valor", "quanto", "quantia", "indenização", "pagamento"],
            "procedimento": ["como", "procedimento", "processo", "etapas"],
            "direitos": ["direito", "pode", "permitido", "legal"],
            "consequências": ["consequência", "acontece", "penalidade", "sanção"]
        }
        
        for aspect, keywords in aspect_indicators.items():
            if any(keyword in question_lower for keyword in keywords):
                # Cria query focada nesse aspecto
                aspect_query = f"{aspect} em {question[:80]}"
                if aspect_query not in queries:
                    queries.append(aspect_query)
        
        # 3. Adiciona queries expandidas com sinônimos jurídicos
        legal_expansions = self._expand_legal_terms(question_lower)
        for expansion in legal_expansions[:2]:
            if expansion.strip() and expansion not in queries:
                queries.append(expansion)
        
        # Remove duplicatas e queries vazias
        queries = [q.strip() for q in queries if q.strip()]
        queries = list(dict.fromkeys(queries))  # Remove duplicatas mantendo ordem
        
        # Garante pelo menos 2 queries não-vazias
        if len(queries) < 2:
            # Adiciona queries genéricas baseadas na questão original
            first_words = " ".join(question.split()[:5]) if question.split() else "busca legal"
            queries.append(first_words)
            queries.append(f"legislação relacionada a {first_words[:50]}")
        
        max_queries = {
            "low": 1,
            "medium": 3,
            "high": 5,
            "very_high": 7
        }
        
        final_queries = queries[:max_queries.get(complexity, 3)]
        
        # Último resort - garante que nenhuma query está vazia
        return [q for q in final_queries if q.strip()] if final_queries else ["busca geral"]

    def _expand_legal_terms(self, question_lower: str) -> List[str]:
        """Expande termos jurídicos com sinônimos e variações."""
        expansions = []
        
        term_synonyms = {
            "demissão": ["rescisão contratual", "desligamento", "dispensa"],
            "contrato": ["acordo", "pacto", "instrumento contratual"],
            "indenização": ["reparação", "compensação", "ressarcimento"],
            "processo": ["ação judicial", "lide", "demanda"],
            "direito": ["prerrogativa", "faculdade legal", "garantia"]
        }
        
        for term, synonyms in term_synonyms.items():
            if term in question_lower:
                for synonym in synonyms[:1]:  # Pega apenas 1 sinônimo
                    expanded = question_lower.replace(term, synonym)
                    expansions.append(expanded)
        
        return expansions

    def _define_search_strategy(self, complexity: str, legal_areas: List[str]) -> str:
        """Define a estratégia de busca mais apropriada."""
        
        # Múltiplas áreas = busca ampla
        if len(legal_areas) > 2:
            return "broad_multi_area"
        
        # Área única + alta complexidade = busca profunda
        if len(legal_areas) == 1 and complexity in ["high", "very_high"]:
            return "deep_single_area"
        
        # Alta complexidade = busca multi-etapas
        if complexity in ["high", "very_high"]:
            return "multi_step_refinement"
        
        # Baixa complexidade = busca direta
        if complexity == "low":
            return "direct_retrieval"
        
        # Padrão = busca híbrida
        return "hybrid_search"

    def _extract_legal_terms(self, question_lower: str) -> List[str]:
        """Extrai termos jurídicos relevantes da questão."""
        legal_terms = []
        
        # Lista de termos jurídicos comuns
        common_legal_terms = [
            "art.", "artigo", "lei", "código", "clt", "cc", "cpc", "cf",
            "súmula", "jurisprudência", "acórdão", "ementa",
            "contrato", "acordo", "convenção", "tratado",
            "direito", "dever", "obrigação", "responsabilidade",
            "ação", "processo", "recurso", "sentença", "decisão",
            "crime", "delito", "contravenção", "pena",
            "trabalhista", "civil", "penal", "constitucional"
        ]
        
        for term in common_legal_terms:
            if term in question_lower:
                legal_terms.append(term)
        
        return legal_terms

    def _estimate_doc_count(self, complexity: str) -> int:
        """Estima quantos documentos serão necessários para responder."""
        doc_count_map = {
            "low": 3,
            "medium": 5,
            "high": 8,
            "very_high": 12
        }
        
        return doc_count_map.get(complexity, 5)

    def _generate_queries(self, domain: str, legal_intent: str, missing_information: List[str]) -> List[str]:
        """Gera múltiplas queries de busca baseadas no domínio e intenção."""
        queries = []
        
        # Query principal baseada na intenção legal
        if legal_intent:
            queries.append(legal_intent)
        
        # Queries adicionais baseadas no domínio
        domain_queries = {
            "Direito do Trabalho": [
                "CLT justa causa demissão",
                "Direitos trabalhistas e deveres",
                "Procedimento correto demissão"
            ],
            "Direito Civil": [
                "Responsabilidade civil e indenização",
                "Contrato e obrigações",
                "Danos morais e materiais"
            ],
            "Direito Penal": [
                "Crime e pena",
                "Procedimento criminal",
                "Direitos do acusado"
            ],
            "Direito do Consumidor": [
                "Direitos do consumidor",
                "Defeito em produto ou serviço",
                "Reclamação ao fornecedor"
            ]
        }
        
        # Adiciona queries específicas do domínio
        if domain in domain_queries:
            queries.extend(domain_queries[domain])
        
        # Adiciona queries baseadas em informações faltantes
        for missing_info in missing_information:
            if missing_info:
                queries.append(f"legislação sobre {missing_info}")
        
        # Remove duplicatas e queries vazias
        queries = [q.strip() for q in queries if q.strip()]
        queries = list(dict.fromkeys(queries))  # Remove duplicatas mantendo ordem
        
        # Garante pelo menos 2 queries
        if len(queries) < 2:
            queries.extend([
                f"legislação {domain}",
                f"procedimento legal {domain}"
            ])
        
        return queries[:5]  # Retorna no máximo 5 queries