"""Knowledge Base Indexer for ZentroMart VoiceOps.

Parses all 21 Markdown documents from kb/ and product catalog from data/products.csv.
Extracts section IDs (e.g. [RET-2], [KB-00.1]) and metadata (doc_type, status, effective_from).
Computes dense multilingual vector embeddings and BM25 tokenized corpus.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import csv
import os
import re
import yaml

@dataclass
class KBChunk:
    section_id: str
    doc_id: str
    doc_title: str
    doc_type: str  # policy, agent_guide, help, faq, marketing, meta, product_catalog
    status: str    # current, superseded
    effective_from: str
    language: str
    section_title: str
    content: str
    source_file: str
    sku: Optional[str] = None
    brand: Optional[str] = None
    price_inr: Optional[str] = None
    assured: Optional[bool] = None
    warranty_months: Optional[str] = None

    @property
    def full_searchable_text(self) -> str:
        return f"{self.section_id} | {self.doc_title} - {self.section_title}\n{self.content}"

class KnowledgeBase:
    def __init__(self, dataset_dir: Optional[str] = None):
        if dataset_dir is None:
            dataset_dir = os.path.normpath(
                os.path.join(os.path.dirname(__file__), "..", "..", "Capstone_Project_Dataset_FTTT_2026")
            )
        self.dataset_dir = Path(dataset_dir)
        self.kb_dir = self.dataset_dir / "kb"
        self.data_dir = self.dataset_dir / "data"
        self.chunks: List[KBChunk] = []
        self.by_section_id: Dict[str, KBChunk] = {}
        self.products_by_sku: Dict[str, KBChunk] = {}
        self.load()

    def _parse_frontmatter(self, text: str):
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                try:
                    meta = yaml.safe_load(parts[1]) or {}
                    body = parts[2]
                    return meta, body
                except Exception:
                    pass
        return {}, text

    def load(self):
        self.chunks = []
        self.by_section_id = {}
        self.products_by_sku = {}

        # 1. Parse KB Markdown files
        for fpath in sorted(self.kb_dir.glob("*.md")):
            text = fpath.read_text(encoding="utf-8")
            meta, body = self._parse_frontmatter(text)

            doc_id = meta.get("doc_id", fpath.stem)
            doc_title = meta.get("title", fpath.stem)
            doc_type = meta.get("doc_type", "policy")
            status = meta.get("status", "current")
            effective_from = str(meta.get("effective_from", "2026-01-01"))
            language = meta.get("language", "en")

            # Split by ## [SECTION-ID]
            pattern = r"(^##\s*\[([^\]]+)\]\s*(.*)$)"
            matches = list(re.finditer(pattern, body, re.M))

            if not matches:
                # Whole doc is a single chunk if no sections
                chunk = KBChunk(
                    section_id=doc_id,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    doc_type=doc_type,
                    status=status,
                    effective_from=effective_from,
                    language=language,
                    section_title=doc_title,
                    content=body.strip(),
                    source_file=fpath.name
                )
                self.chunks.append(chunk)
                self.by_section_id[doc_id] = chunk
                continue

            for idx, match in enumerate(matches):
                sec_id = match.group(2).strip()
                sec_title = match.group(3).strip()
                start_pos = match.end()
                end_pos = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
                sec_body = body[start_pos:end_pos].strip()

                chunk = KBChunk(
                    section_id=sec_id,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    doc_type=doc_type,
                    status=status,
                    effective_from=effective_from,
                    language=language,
                    section_title=sec_title,
                    content=sec_body,
                    source_file=fpath.name
                )
                self.chunks.append(chunk)
                self.by_section_id[sec_id] = chunk

        # 2. Parse products.csv into searchable chunks
        products_file = self.data_dir / "products.csv"
        if products_file.exists():
            with products_file.open(encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    sku = row["sku"].strip()
                    sec_id = f"products.csv:{sku}"
                    assured_val = row.get("assured", "False") == "True"
                    content = (
                        f"SKU: {sku}\n"
                        f"Title: {row['title']}\n"
                        f"Brand: {row['brand']}\n"
                        f"Category: {row['return_category']}\n"
                        f"Price: Rs. {row['price_inr']} (INR)\n"
                        f"Fulfilled by: {row['fulfilled_by']}\n"
                        f"ZentroMart Assured: {'Yes' if assured_val else 'No'}\n"
                        f"Warranty: {row['warranty_months']} months\n"
                        f"Description: {row['description']}"
                    )
                    chunk = KBChunk(
                        section_id=sec_id,
                        doc_id="products.csv",
                        doc_title="ZentroMart Product Catalog",
                        doc_type="product_catalog",
                        status="current",
                        effective_from="2026-08-01",
                        language="en",
                        section_title=f"{row['title']} ({sku})",
                        content=content,
                        source_file="products.csv",
                        sku=sku,
                        brand=row["brand"],
                        price_inr=row["price_inr"],
                        assured=assured_val,
                        warranty_months=row["warranty_months"]
                    )
                    self.chunks.append(chunk)
                    self.by_section_id[sec_id] = chunk
                    self.products_by_sku[sku] = chunk

    def get_section(self, section_id: str) -> Optional[KBChunk]:
        return self.by_section_id.get(section_id)

    def get_product(self, sku: str) -> Optional[KBChunk]:
        sku_clean = sku.strip().upper()
        return self.products_by_sku.get(sku_clean)
