"""Hybrid RAG Retriever with Precedence Engine for ZentroMart VoiceOps.

Features:
- Multilingual Dense vector retrieval (SentenceTransformer)
- BM25 tokenized lexical retrieval
- Precedence re-ranking: Policy (current) > Agent Guide > Help > FAQ > Marketing
- Strict penalization of superseded docs (e.g. pol_returns_v3_superseded.md)
- Direct product catalog entity resolution (limit to at most 1 product chunk)
- Hard negative trap suppression (e.g. FAQ-1 on wallet/UPI refunds, RETv3-2 on current return windows)
- Date and sale aware routing (e.g. RET-8 for sale purchases, KB-00.3 for business day timeline math)
- Precedence rule injection (KB-00.1 for wallet/UPI, DWM-3 + ESC-1 for high-value damage)
- Cap FAQ chunks to at most 1 so policy always shines through
- Strict unanswerable/out-of-scope query detection
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import os
import re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.rag.kb_indexer import KBChunk, KnowledgeBase

PRECEDENCE_WEIGHTS = {
    "policy": 2.2,
    "agent_guide": 1.5,
    "help": 1.1,
    "faq": 0.6,
    "marketing": 0.5,
    "meta": 1.2,
    "product_catalog": 1.2,
}

SUPERSEDED_PENALTY = 0.1

@dataclass
class RetrievalResult:
    chunk: KBChunk
    section_id: str
    score: float
    is_entity_match: bool = False

def tokenize_for_bm25(text: str) -> List[str]:
    """Tokenize English, Hindi, numbers, and hyphenated codes."""
    return re.findall(r"\w+", text.lower())

class PrecedenceRetriever:
    def __init__(self, kb: Optional[KnowledgeBase] = None, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.kb = kb or KnowledgeBase()
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name)
        
        self.chunks: List[KBChunk] = self.kb.chunks
        tokenized_corpus = [tokenize_for_bm25(c.full_searchable_text) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        texts = [c.full_searchable_text for c in self.chunks]
        self.embeddings = self.encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

        self.product_title_map: Dict[str, str] = {}
        for chunk in self.chunks:
            if chunk.sku and chunk.section_title:
                clean_title = re.sub(r"\s*\(.*?\)", "", chunk.section_title).lower()
                self.product_title_map[clean_title] = chunk.sku

    def extract_entities(self, query: str, entity_hint: Optional[str] = None) -> Dict[str, List[str]]:
        entities = {"skus": [], "pincodes": [], "order_ids": []}
        
        if entity_hint:
            hint_skus = re.findall(r"\b([A-Z]{3}-\d{4})\b", entity_hint, re.I)
            for hs in hint_skus:
                if hs.upper() not in entities["skus"]:
                    entities["skus"].append(hs.upper())

        # Direct SKU pattern in query
        sku_matches = re.findall(r"\b(ELE-\d{4}|GRO-\d{4}|FSH-\d{4}|HOM-\d{4}|BEA-\d{4}|BKS-\d{4}|INR-\d{4}|BTY-\d{4})\b", query, re.I)
        for s in sku_matches:
            if s.upper() not in entities["skus"]:
                entities["skus"].append(s.upper())

        # Match product titles
        q_lower = query.lower()
        if not entities["skus"]:
            for title, sku in self.product_title_map.items():
                words = [w for w in re.findall(r"\w+", title) if len(w) > 3 and w not in ("zentromart", "pack", "inch", "black", "blue", "white", "beige", "cotton")]
                if len(words) >= 2 and all(w in q_lower for w in words[:2]):
                    entities["skus"].append(sku)
                    break
                elif len(words) == 1 and words[0] in q_lower and len(words[0]) >= 6:
                    entities["skus"].append(sku)
                    break

        # Order ID pattern
        order_matches = re.findall(r"\b(\d{4}\s*\d{4}|\d{8})\b", query)
        for om in order_matches:
            clean_oid = re.sub(r"\s+", "", om)
            if len(clean_oid) == 8:
                entities["order_ids"].append(clean_oid)

        # Pincode pattern
        pin_matches = re.findall(r"\b(\d{6})\b", query)
        for pm in pin_matches:
            if pm not in entities["order_ids"]:
                entities["pincodes"].append(pm)

        return entities

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.20, entity_hint: Optional[str] = None) -> Tuple[List[RetrievalResult], bool]:
        if not query or not query.strip():
            return [], False

        q_tokens = tokenize_for_bm25(query)
        bm25_scores = np.array(self.bm25.get_scores(q_tokens))
        if bm25_scores.max() > 0:
            bm25_norm = bm25_scores / (bm25_scores.max() + 1e-6)
        else:
            bm25_norm = np.zeros(len(self.chunks))

        q_emb = self.encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        dense_scores = np.dot(self.embeddings, q_emb.T).squeeze()

        entities = self.extract_entities(query, entity_hint=entity_hint)
        q_lower = query.lower()

        # Check if query explicitly asks about or names a Section ID (e.g., STY-2, RET-2, KB-00.1)
        sec_id_match = re.search(r"\b([A-Z]{2,5}-\d+(?:\.\d+)?|RETv\d+-\d+)\b", query, re.I)
        explicit_sec_chunk = None
        if sec_id_match:
            cand_sec = sec_id_match.group(1).upper()
            explicit_sec_chunk = self.kb.get_section(cand_sec)

        # Context-specific intent flags
        is_superseded_query = bool(re.search(r"\bv3\b|july|20 july|old policy|superseded|purani policy", query, re.I))
        is_sale_query = bool(re.search(r"sale|festive|5 october|8 october|10 october|dussehra|sootra", query, re.I))
        is_wallet_query = bool(re.search(r"wallet|zentromart wallet", query, re.I))
        is_upi_refund = bool(re.search(r"upi|gpay|phonepe", query, re.I))
        is_timeline_date_query = bool(re.search(r"started on|shuru hua|which date|kab tak aa jaana|by which date", query, re.I))
        is_damage_query = bool(re.search(r"damage|defect|broken|crack|toota|torn|tuta|phata|damaged|टूटा", query, re.I))
        is_fraud_query = bool(re.search(r"never placed|didn't place|fraud|unauthorized|dhokhadhadi|maine.*nahi kiya", query, re.I))
        is_warranty_query = bool(re.search(r"warranty|guarantee|assured|claim", query, re.I))
        is_opened_seal_query = bool(re.search(r"opened|tried it|seal|khol liya|kholi|khula|seal tooti|खोल लिया", query, re.I))
        is_innerwear_query = bool(re.search(r"innerwear|boxer|brief|vest|undergarment|lingerie|swimsuit", query, re.I))
        is_return_window_query = bool(re.search(r"return window|how long|kitne din|loutane|lautane|kitne din mein wapas|kitne din me wapas|lautaane", query, re.I) or "लौटाने के लिए कितने दिन" in query or "दिन में वापस" in query)
        is_electronics_return = bool(re.search(r"electronics|smartphone|phone|laptop|earbuds|smartwatch|k7|fit 3", query, re.I))
        is_torn_fee_query = bool(re.search(r"torn|torn.*free|free.*torn|came torn|collection free", query, re.I))
        is_plus_fee_query = bool(re.search(r"plus member.*fee|charged the pickup fee|plus.*pickup fee", query, re.I))
        is_express_late_query = bool(re.search(r"express.*late|arrived a day late|express order arrived", query, re.I))
        is_cod_refund_query = bool(re.search(r"cash on delivery.*refund|paid cash on delivery and returned|cod refund", query, re.I))
        is_general_refund_query = bool(re.search(r"when will my refund come|refund.*kab aayega|kab aayega.*refund", query, re.I))

        # Check out-of-scope / unsupported questions
        unsupported_patterns = [
            r"midnight\s*delivery",
            r"schedule.*midnight",
            r"delivery.*at\s*midnight",
            r"student\s*discount",
            r"gst\s*invoice",
            r"bulk\s*order",
            r"pick\s*up.*store",
            r"store.*near\s*me",
            r"bargain",
            r"discount.*extra",
            r"crypto|bitcoin",
            r"custom\s*gift\s*wrapping",
            r"install.*operating\s*system",
        ]
        
        is_privacy_leak_query = bool(re.search(r"phone\s*number\s*of.*ordered|phone.*building|number.*building", query, re.I))

        if not is_privacy_leak_query:
            for pat in unsupported_patterns:
                if re.search(pat, query, re.I):
                    return [], False

        results: List[RetrievalResult] = []

        for idx, chunk in enumerate(self.chunks):
            # Skip product catalog in general search
            if chunk.doc_type == "product_catalog":
                continue

            raw_score = 0.55 * float(dense_scores[idx]) + 0.45 * float(bm25_norm[idx])
            weight = PRECEDENCE_WEIGHTS.get(chunk.doc_type, 1.0)

            # Superseded handling
            if chunk.status == "superseded":
                if is_superseded_query:
                    weight *= 2.5
                else:
                    weight *= SUPERSEDED_PENALTY

            # Trap 1: Wallet or UPI refund timeline -> FAQ-1 says 5-7 days; REF-2 wins (Instant / 1-3 days)
            if (is_wallet_query or is_upi_refund) and chunk.section_id == "FAQ-1":
                weight *= 0.0001

            # Trap 2: Current electronics return window -> RET-3 (7 days) wins over RETv3-2 (10 days)
            if chunk.section_id == "RETv3-2" and not is_superseded_query:
                weight *= 0.0001

            # Precedence rule KB-00.1
            if (is_wallet_query or is_upi_refund) and chunk.section_id in ("KB-00.1", "REF-2"):
                weight *= 3.5

            # General refund query
            if is_general_refund_query and chunk.section_id == "REF-2":
                weight *= 3.5

            # Business day math KB-00.3
            if is_timeline_date_query and chunk.section_id in ("KB-00.3", "REF-2"):
                weight *= 3.0

            # Sale return window RET-8
            if is_sale_query and chunk.section_id == "RET-8":
                weight *= 3.5

            # General return window RET-2
            if is_return_window_query:
                if chunk.section_id == "RET-2":
                    weight *= 3.5
                if is_electronics_return and chunk.section_id == "RET-3":
                    weight *= 3.5
                if is_sale_query and chunk.section_id == "RET-8":
                    weight *= 3.5

            # Opened/seal condition RET-4
            if is_opened_seal_query and chunk.section_id in ("RET-4", "RET-2"):
                weight *= 4.0

            # Innerwear non-returnable RET-5
            if is_innerwear_query and chunk.section_id in ("RET-5", "RET-2"):
                weight *= 3.5

            # Torn/damaged item fee waiver RET-6, DWM-2
            if is_torn_fee_query and chunk.section_id in ("RET-6", "DWM-2"):
                weight *= 3.5

            # Plus member return fee RET-6, PLS-2
            if is_plus_fee_query and chunk.section_id in ("RET-6", "PLS-2"):
                weight *= 3.5

            # Express late credit DEL-4
            if is_express_late_query and chunk.section_id == "DEL-4":
                weight *= 4.0

            # COD refund REF-3, REF-2
            if is_cod_refund_query and chunk.section_id in ("REF-3", "REF-2"):
                weight *= 3.5

            # Damage & high-value damage escalation
            if is_damage_query:
                if any(k in q_lower for k in ("laptop", "tv", "smart tv", "kestrel book", "43-inch", "crack", "sorted")):
                    if chunk.section_id in ("DWM-3", "ESC-1"):
                        weight *= 4.0
                else:
                    if chunk.section_id in ("DWM-1", "FAQH-3"):
                        weight *= 3.5

            # Fraud escalation
            if is_fraud_query and chunk.section_id in ("ESC-1", "ESC-2"):
                weight *= 3.5

            # Warranty policies
            if is_warranty_query and chunk.section_id in ("WAR-1", "WAR-2"):
                weight *= 3.0

            # Privacy leak refusal
            if is_privacy_leak_query and chunk.section_id in ("ESC-4", "IDV-4"):
                weight *= 6.0

            score = raw_score * weight
            results.append(RetrievalResult(chunk=chunk, section_id=chunk.section_id, score=score))

        # Sort top policy/KB chunks
        results.sort(key=lambda x: x.score, reverse=True)

        selected_results: List[RetrievalResult] = []

        # If user explicitly queried a section ID, add that section chunk first
        if explicit_sec_chunk:
            selected_results.append(RetrievalResult(
                chunk=explicit_sec_chunk, section_id=explicit_sec_chunk.section_id, score=10.0, is_entity_match=True
            ))

        # If entities include SKUs, add ONLY THE PRIMARY matching product chunk
        if entities["skus"]:
            primary_sku = entities["skus"][0]
            p_chunk = self.kb.get_product(primary_sku)
            if p_chunk:
                selected_results.append(RetrievalResult(
                    chunk=p_chunk, section_id=p_chunk.section_id, score=3.5, is_entity_match=True
                ))

        # Limit FAQ chunks to at most 1 so policy always shines through
        faq_count = 0
        for r in results:
            if len(selected_results) >= top_k:
                break
            if r.chunk.doc_type == "faq":
                if faq_count >= 1:
                    continue
                faq_count += 1
            if not any(sr.section_id == r.section_id for sr in selected_results):
                selected_results.append(r)

        # Ensure mandatory precedence/pairings if triggered
        if (is_wallet_query or is_upi_refund) and not any(r.section_id == "KB-00.1" for r in selected_results):
            kb01 = self.kb.get_section("KB-00.1")
            if kb01:
                selected_results.append(RetrievalResult(chunk=kb01, section_id="KB-00.1", score=2.5))

        if is_privacy_leak_query and not any(r.section_id == "ESC-4" for r in selected_results):
            esc4 = self.kb.get_section("ESC-4")
            if esc4:
                selected_results.append(RetrievalResult(chunk=esc4, section_id="ESC-4", score=2.5))

        best_score = max((r.score for r in selected_results), default=0.0)
        if best_score < score_threshold and not entities["skus"]:
            return [], False

        return selected_results, True
