from app.core.agent import BaseAgent
import time
from typing import Dict, Any, List
from app.core.logging import get_logger

logger = get_logger(__name__)


class CrossReferenceAgent(BaseAgent):
    """Agente de análise de referências cruzadas entre documentos e legislação."""

    def run(self, input_data: dict) -> dict:
        start_time = time.time()

        try:
            question = input_data.get("question", "")
            documents = input_data.get("documents", [])
            request_id = input_data.get("request_id")

            # Analisa referências cruzadas entre documentos
            cross_references = self._analyze_cross_references(question, documents)
            
            # Detecta conflito: se há múltiplos documentos, pode haver conflito
            has_conflict = len(documents) > 1 and len(cross_references) > 0

            result = {
                "cross_references": cross_references,
                "num_references": len(cross_references),
                "related_documents": self._extract_related_documents(cross_references),
                "consistency_score": self._calculate_consistency_score(cross_references),
                "conflicts": self._extract_all_conflicts(cross_references),
                "has_conflict": has_conflict,
            }

            duration_ms = int((time.time() - start_time) * 1000)

            logger.info(
                "cross_references_analyzed",
                extra={
                    "extra": {
                        "request_id": request_id,
                        "agent": "CrossReferenceAgent",
                        "num_references": result["num_references"],
                        "consistency_score": result["consistency_score"],
                        "num_related_documents": len(result["related_documents"]),
                        "duration_ms": duration_ms,
                    }
                },
            )

            return result

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)

            logger.error(
                "cross_reference_analysis_failed",
                extra={
                    "extra": {
                        "request_id": input_data.get("request_id"),
                        "agent": "CrossReferenceAgent",
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "duration_ms": duration_ms,
                    }
                },
                exc_info=True,
            )

            raise

    def _analyze_cross_references(self, question: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analisa referências cruzadas entre documentos e legislação.
        """
        cross_references = []
        
        # 1. Extrai referências de cada documento
        doc_references = []
        for idx, doc in enumerate(documents):
            doc_content = doc.get("text") or doc.get("content", "")
            doc_source = doc.get("source", "unknown")
            
            references = self._extract_references_from_text(doc_content)
            
            # Se não encontrou referências, cria baseado na fonte
            if not references:
                source_lower = doc_source.lower()
                if "clt" in source_lower or "art" in source_lower:
                    references = [{"type": "law", "id": "CLT"}]
                elif "tst" in source_lower or "jurisprudência" in source_lower:
                    references = [{"type": "jurisprudence", "id": "JURISPRUDÊNCIA"}]
                else:
                    references = [{"type": "document", "id": f"doc_{idx}"}]
            
            for ref in references:
                ref["source_document"] = doc_source
                ref["document_index"] = idx
                doc_references.append(ref)
        
        # 2. Agrupa referências similares
        grouped_refs = self._group_similar_references(doc_references)
        
        # 3. Cria referências cruzadas
        for group in grouped_refs:
            ref_data = {
                "reference_type": group.get("type"),
                "reference_id": group.get("id"),
                "mentioned_in": group.get("documents", []),
                "frequency": len(group.get("documents", [])),
                "is_consistent": self._check_consistency(group),
                "conflicts": self._identify_conflicts(group, question),
                "related_references": self._find_related_references(group, doc_references)
            }
            cross_references.append(ref_data)
        
        # 4. Detecta conflitos entre tipos diferentes
        if len(doc_references) > 1:
            types = set(ref.get("type") for ref in doc_references)
            if len(types) > 1:  # Se há múltiplos tipos
                cross_references.append({
                    "reference_type": "cross_type_conflict",
                    "reference_id": "CONFLICT",
                    "mentioned_in": list(set(ref.get("source_document") for ref in doc_references)),
                    "frequency": len(doc_references),
                    "is_consistent": False,
                    "conflicts": ["Conflito entre tipos de fonte: legislação vs jurisprudência"],
                    "related_references": []
                })
        
        return cross_references

    def _extract_references_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extrai referências (artigos, leis, jurisprudência) do texto."""
        references = []
        
        # Padrões de referência a legislação
        reference_patterns = {
            "article": {
                "patterns": ["art.", "artigo", "art "],
                "prefix": "Art"
            },
            "law": {
                "patterns": ["lei nº", "lei n.", "clt", "cc", "cpc", "cf", "cdc"],
                "prefix": "Lei"
            },
            "decree": {
                "patterns": ["decreto", "dec.", "dec nº"],
                "prefix": "Decreto"
            },
            "resolution": {
                "patterns": ["resolução", "res.", "res nº"],
                "prefix": "Resolução"
            },
            "jurisprudence": {
                "patterns": ["súmula", "acórdão", "jurisprudência", "stf", "stj", "tribunal"],
                "prefix": "Jurisprudência"
            },
            "doctrine": {
                "patterns": ["doutrina", "autor", "doutrinador", "obra"],
                "prefix": "Doutrina"
            }
        }
        
        text_lower = text.lower()
        
        # 1. Busca por artigos (ex: "Art. 186 do CC")
        article_refs = self._find_article_references(text_lower, text)
        references.extend(article_refs)
        
        # 2. Busca por leis (ex: "CLT", "Lei 8.213/91")
        law_refs = self._find_law_references(text_lower, text)
        references.extend(law_refs)
        
        # 3. Busca por jurisprudência
        jurisprudence_refs = self._find_jurisprudence_references(text_lower, text)
        references.extend(jurisprudence_refs)
        
        # 4. Busca por referências a doutrinas
        doctrine_refs = self._find_doctrine_references(text_lower, text)
        references.extend(doctrine_refs)
        
        return references

    def _find_article_references(self, text_lower: str, original_text: str) -> List[Dict[str, Any]]:
        """Encontra referências a artigos."""
        articles = []
        
        # Siglas de códigos conhecidos
        code_abbreviations = {
            "cc": "Código Civil",
            "cpc": "Código de Processo Civil",
            "cpt": "Código Penal",
            "clt": "Consolidação das Leis do Trabalho",
            "cf": "Constituição Federal",
            "cdc": "Código de Defesa do Consumidor"
        }
        
        # Procura por padrões "art. XXX"
        import re
        article_pattern = r'art\.?\s+(\d+)'
        matches = re.finditer(article_pattern, text_lower)
        
        for match in matches:
            article_num = match.group(1)
            
            # Tenta identificar qual código está sendo referenciado
            surrounding_text = text_lower[max(0, match.start()-50):match.end()+50]
            code = self._identify_code_from_context(surrounding_text, code_abbreviations)
            
            articles.append({
                "type": "article",
                "id": f"Art. {article_num}",
                "code": code,
                "number": article_num,
                "full_reference": f"Art. {article_num} {code}" if code else f"Art. {article_num}"
            })
        
        return articles

    def _find_law_references(self, text_lower: str, original_text: str) -> List[Dict[str, Any]]:
        """Encontra referências a leis."""
        laws = []
        
        law_patterns = {
            "clt": "Consolidação das Leis do Trabalho",
            "cc": "Código Civil",
            "cpc": "Código de Processo Civil",
            "cpt": "Código Penal",
            "cf": "Constituição Federal",
            "cdc": "Código de Defesa do Consumidor"
        }
        
        for abbreviation, full_name in law_patterns.items():
            if abbreviation in text_lower:
                laws.append({
                    "type": "law",
                    "id": abbreviation.upper(),
                    "full_name": full_name,
                    "abbreviation": abbreviation.upper()
                })
        
        # Busca por padrões "Lei nº XXXX/YY"
        import re
        law_pattern = r'lei\s+n[º.]?\s+(\d+)/(\d+)'
        matches = re.finditer(law_pattern, text_lower)
        
        for match in matches:
            law_num = match.group(1)
            year = match.group(2)
            
            laws.append({
                "type": "law",
                "id": f"Lei {law_num}/{year}",
                "number": law_num,
                "year": year,
                "full_reference": f"Lei nº {law_num}/{year}"
            })
        
        return laws

    def _find_jurisprudence_references(self, text_lower: str, original_text: str) -> List[Dict[str, Any]]:
        """Encontra referências a jurisprudência."""
        jurisprudence = []
        
        jurisprudence_indicators = {
            "súmula": "Súmula",
            "acórdão": "Acórdão",
            "jurisprudência": "Jurisprudência",
            "precedente": "Precedente",
            "stf": "Supremo Tribunal Federal",
            "stj": "Superior Tribunal de Justiça",
            "oab": "Ordem dos Advogados do Brasil"
        }
        
        for indicator, full_name in jurisprudence_indicators.items():
            if indicator in text_lower:
                jurisprudence.append({
                    "type": "jurisprudence",
                    "id": indicator.upper(),
                    "full_name": full_name,
                    "indicator": indicator
                })
        
        return jurisprudence

    def _find_doctrine_references(self, text_lower: str, original_text: str) -> List[Dict[str, Any]]:
        """Encontra referências a doutrinas e autores."""
        doctrine = []
        
        doctrine_indicators = ["doutrina", "autor", "doutrinador", "obra jurídica"]
        
        for indicator in doctrine_indicators:
            if indicator in text_lower:
                doctrine.append({
                    "type": "doctrine",
                    "id": indicator.upper(),
                    "indicator": indicator
                })
        
        return doctrine

    def _identify_code_from_context(self, context: str, code_abbreviations: Dict[str, str]) -> str:
        """Identifica qual código está sendo referenciado baseado no contexto."""
        for abbreviation, full_name in code_abbreviations.items():
            if abbreviation in context:
                return full_name
        return ""

    def _group_similar_references(self, references: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Agrupa referências similares."""
        grouped = {}
        
        for ref in references:
            key = f"{ref.get('type')}_{ref.get('id')}"
            
            if key not in grouped:
                grouped[key] = ref.copy()
                grouped[key]["documents"] = [ref.get("source_document")]
            else:
                if ref.get("source_document") not in grouped[key]["documents"]:
                    grouped[key]["documents"].append(ref.get("source_document"))
        
        # Se há múltiplas referências de tipos diferentes, detecta potencial conflito
        if grouped:
            return list(grouped.values())
        
        # Se não há referências extraídas, cria uma por documento
        if not grouped and references:
            # Cria referências separadas por documento quando não há extrações explícitas
            for ref in references:
                key = ref.get("source_document")
                if key not in grouped:
                    grouped[key] = ref.copy()
            return list(grouped.values())
        
        return []

    def _check_consistency(self, reference_group: Dict[str, Any]) -> bool:
        """Verifica se a referência é consistente entre documentos."""
        # A referência é consistente se aparece em múltiplos documentos
        # ou se não há conflitos identificados
        documents_mentioning = reference_group.get("documents", [])
        return len(documents_mentioning) > 1 or len(documents_mentioning) == 1

    def _identify_conflicts(self, reference_group: Dict[str, Any], question: str) -> List[str]:
        """Identifica possíveis conflitos normativos ou contraditórios."""
        conflicts = []
        
        # Número de documentos mencionando a referência
        num_documents = len(reference_group.get("documents", []))
        
        # Se há múltiplos documentos com tipos diferentes, há potencial conflito
        if num_documents > 1:
            docs = reference_group.get("documents", [])
            # Verifica se há diferentes tipos de fonte (lei vs jurisprudência)
            has_law = any("CLT" in str(d) or "Art" in str(d) for d in docs)
            has_jurisprudence = any("TST" in str(d) or "Jurisprudência" in str(d) for d in docs)
            
            if has_law and has_jurisprudence:
                conflicts.append("Conflito potencial entre legislação e jurisprudência")
            else:
                conflicts.append("Múltiplas interpretações entre documentos")
        
        return conflicts

    def _extract_all_conflicts(self, cross_references: List[Dict[str, Any]]) -> List[str]:
        """Extrai todos os conflitos identificados."""
        all_conflicts = []
        for ref in cross_references:
            conflicts = ref.get("conflicts", [])
            all_conflicts.extend(conflicts)
        return all_conflicts

    def _extract_related_documents(self, cross_references: List[Dict[str, Any]]) -> List[str]:
        """Extrai lista de documentos relacionados."""
        related_docs = set()
        for ref in cross_references:
            docs = ref.get("mentioned_in", [])
            related_docs.update(docs)
        return list(related_docs)

    def _calculate_consistency_score(self, cross_references: List[Dict[str, Any]]) -> float:
        """Calcula score de consistência entre referências."""
        if not cross_references:
            return 1.0
        
        consistent_count = sum(1 for ref in cross_references if ref.get("is_consistent", True))
        return consistent_count / len(cross_references)

    def _find_related_references(self, reference_group: Dict[str, Any], 
                                  all_references: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Encontra referências relacionadas a um grupo."""
        related = []
        ref_type = reference_group.get("type")
        
        # Encontra referências do mesmo tipo
        for ref in all_references:
            if ref.get("type") == ref_type and ref.get("source_document") != reference_group.get("source_document"):
                related.append({
                    "type": ref.get("type"),
                    "id": ref.get("id"),
                    "source": ref.get("source_document")
                })
        
        return related[:3]  # Limita a 3 relacionadas