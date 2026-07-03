from __future__ import annotations

import logging
import re
from typing import List

from .config import ENABLE_NER, MAX_LEMMAS_STORED, STORE_LEMMAS
from .models import NatashaAnnotation, PrimaryEntity
from .segmentation import Sentence, SentenceSegmenter

log = logging.getLogger(__name__)

_CONTENT_POS = {"NOUN", "PROPN", "ADJ", "VERB", "NUM", "X"}
_SENT_RE = re.compile(r"[^.!?\u2026]*[.!?\u2026]+|\S[^.!?\u2026]*$", re.S)


class NatashaPipeline:

    def __init__(self) -> None:
        try:
            from natasha import (
                Doc,
                MorphVocab,
                NewsEmbedding,
                NewsMorphTagger,
                NewsNERTagger,
                Segmenter,
            )
        except Exception as e:
            log.warning(
                "Natasha package unavailable (%s: %s) - segmentation-only mode",
                type(e).__name__,
                e,
            )
            self._Doc = None
            self._segmenter = None
            self._morph_vocab = None
            self._morph_tagger = None
            self._ner_tagger = None
            self._ner_ok = False
            return

        self._Doc = Doc
        self._segmenter = Segmenter()
        self._morph_vocab = MorphVocab()

        self._morph_tagger = None
        self._ner_tagger = None
        self._ner_ok = False
        try:
            emb = NewsEmbedding()
            self._morph_tagger = NewsMorphTagger(emb)
            if ENABLE_NER:
                self._ner_tagger = NewsNERTagger(emb)
            self._ner_ok = ENABLE_NER and self._ner_tagger is not None
        except Exception as e:
            log.warning(
                "Natasha morph/NER models unavailable (%s: %s) - segmentation-only mode",
                type(e).__name__,
                e,
            )

    def segment(self, text: str) -> List[Sentence]:
        if not text.strip():
            return []
        if self._Doc is None:
            return _regex_segment(text)
        doc = self._Doc(text)
        doc.segment(self._segmenter)
        return [Sentence(s.text, s.start, s.stop) for s in doc.sents]

    def annotate(self, text: str) -> NatashaAnnotation:
        if not text.strip():
            return NatashaAnnotation(n_sentences=0, n_tokens=0, ner_available=self._ner_ok)
        if self._Doc is None:
            return NatashaAnnotation(
                n_sentences=len(_regex_segment(text)),
                n_tokens=len(re.findall(r"\S+", text)),
                ner_available=False,
            )

        doc = self._Doc(text)
        doc.segment(self._segmenter)
        n_sentences = len(doc.sents)
        n_tokens = len(doc.tokens)

        lemmas: List[str] = []
        if self._morph_tagger is not None:
            try:
                doc.tag_morph(self._morph_tagger)
                if STORE_LEMMAS:
                    for tok in doc.tokens:
                        tok.lemmatize(self._morph_vocab)
                    lemmas = [
                        tok.lemma for tok in doc.tokens
                        if getattr(tok, "pos", None) in _CONTENT_POS and tok.lemma
                    ]
                    if MAX_LEMMAS_STORED > 0:
                        lemmas = lemmas[:MAX_LEMMAS_STORED]
            except Exception as e:
                log.warning("morph tagging failed on a chunk: %s", e)

        primary: List[PrimaryEntity] = []
        if self._ner_tagger is not None:
            try:
                doc.tag_ner(self._ner_tagger)
                for span in doc.spans:
                    span.normalize(self._morph_vocab)
                    primary.append(PrimaryEntity(
                        text=span.text,
                        normal=span.normal or span.text,
                        type=span.type,
                        start=span.start,
                        stop=span.stop,
                    ))
            except Exception as e:
                log.warning("NER failed on a chunk: %s", e)

        return NatashaAnnotation(
            n_sentences=n_sentences,
            n_tokens=n_tokens,
            ner_available=self._ner_ok,
            primary_entities=primary,
            lemmas=lemmas,
        )


_PIPELINE: NatashaPipeline | None = None


def _regex_segment(text: str) -> List[Sentence]:
    out: List[Sentence] = []
    for m in _SENT_RE.finditer(text):
        seg = m.group(0)
        core = seg.rstrip()
        if core.strip():
            out.append(Sentence(core, m.start(), m.start() + len(core)))
    return out


def get_pipeline() -> NatashaPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        log.info("loading Natasha models (once per process)...")
        _PIPELINE = NatashaPipeline()
    return _PIPELINE
