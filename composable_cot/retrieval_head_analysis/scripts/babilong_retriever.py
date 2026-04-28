"""
BABILongRetriever — subclass of QRHead's FullHeadRetriever that uses BABILong's
deployed prompt format (instead of QRHead's standard "Here are some paragraphs:
[1]... Query: {q}" wrapper).

Per prof guidance:
  1. Use BABILong's own prompt format. The exact constants come from
     `composable_cot/BABIlong/scripts/eval_babilong.py:62-105`.
  2. Each sentence (not 512-token paragraph) is one "document" for QR scoring.
     The build_detection_set.py script emits the JSON in this shape already;
     this class just consumes it correctly.

Implementation notes:
  - Inherits all model loading, attention extraction, FA2 + DynamicCacheWithQuery
    plumbing, calibration, and scoring math from FullHeadRetriever.
  - Overrides only get_prompt() and compose_scoring_prompt() — the two methods
    that produce / parse the prompt string.
  - The query_span covers only the question text. Everything before "Question:"
    in the prompt is identical between real and null queries (docs are the same;
    QA3_INSTRUCTION is constant), so the calibration assertion holds.
"""

from typing import Dict, List, Optional, Tuple, Union

from qrretriever.attn_retriever import FullHeadRetriever


# Exact BABILong QA3 prompt strings — copied verbatim from
# composable_cot/BABIlong/scripts/eval_babilong.py:62-76. Keep these in sync.
QA3_INSTRUCTION = (
    "I give you context with the facts about locations and actions of different persons "
    "hidden in some random text and a question. "
    "You need to answer the question based only on the information from the facts.\n"
    "If a person got an item in the first location and travelled to the second location "
    "the item is also in the second location. "
    "If a person dropped an item in the first location and moved to the second location "
    "the item remains in the first location."
)

QA3_POST_PROMPT = (
    "Your answer must be exactly one word — one of: "
    "bathroom, bedroom, garden, hallway, kitchen, office. "
    "Do not write anything else."
)


class BABILongRetriever(FullHeadRetriever):
    """FullHeadRetriever using BABILong's deployed prompt format."""

    def __init__(
        self,
        config_or_config_path: Optional[Union[Dict, str]] = None,
        model_name_or_path: Optional[str] = None,
        model_base_class: Optional[str] = None,
        attn_head_set: Optional[str] = None,
        device: Optional[str] = None,
    ):
        super().__init__(
            config_or_config_path=config_or_config_path,
            model_name_or_path=model_name_or_path,
            model_base_class=model_base_class,
            attn_head_set=attn_head_set,
            device=device,
        )

    # ----- prompt building ----------------------------------------------------

    def get_prompt(self, query: str, docs: List[Dict]) -> str:
        """Produce the BABILong-format prompt wrapped in Qwen's chat template.

        docs is a list of dicts with key 'paragraph_text' — each entry is one
        sentence (per prof guidance). We join them with a single space to
        reconstruct the original haystack flow.
        """
        if self.model_base_class.lower() != "qwen2.5-7b-instruct":
            raise NotImplementedError(
                f"BABILongRetriever currently only supports qwen2.5-7b-instruct "
                f"(got {self.model_base_class}). Llama prompt template would need "
                f"adjustment to the chat-template tokens."
            )

        haystack = " ".join(doc["paragraph_text"] for doc in docs)

        prompt = (
            "<|im_start|>user\n"
            f"{QA3_INSTRUCTION}\n\n"
            f"<context>\n{haystack}\n</context>\n\n"
            f"Question: {query}\n"
            f"{QA3_POST_PROMPT}"
            "<|im_end|>\n<|im_start|>assistant"
        )
        return prompt

    # ----- prompt-span parsing ------------------------------------------------

    def compose_scoring_prompt(
        self, query: str, docs: List[Dict]
    ) -> Tuple[str, List[int], Tuple[int, int], List[Tuple[int, int]]]:
        """Tokenize the BABILong prompt and locate query + per-sentence spans.

        Returns: (prompt_str, token_ids, query_span, document_span_intervals).
        """
        llm_prompt = self.get_prompt(query, docs)

        prompt_tok = self.tokenizer(llm_prompt, return_offsets_mapping=True)
        prompt_token_ids = prompt_tok["input_ids"]
        offset_mapping = prompt_tok["offset_mapping"]

        # Char-offset → token-index map (same trick as the original code).
        char_offset_to_token_idx = {}
        for i, (start, end) in enumerate(offset_mapping):
            for j in range(start, end):
                char_offset_to_token_idx[j] = i

        # Each sentence is its own "document". Find its span in the prompt by
        # exact substring match — sentences are unique enough in BABILong
        # haystacks that this is robust. (Falls back to first occurrence if a
        # PG19 sentence is somehow duplicated.)
        document_span_intervals = []
        missed = 0
        for doc in docs:
            sent = doc["paragraph_text"]
            try:
                start_idx, end_idx = self.get_content_span(
                    llm_prompt, char_offset_to_token_idx, sent
                )
            except ValueError:
                missed += 1
                # Push a degenerate span so indices line up with docs list. The
                # downstream code masks via doc_span ranges, and a 1-token span
                # with end<start will simply contribute nothing. Still log it.
                document_span_intervals.append((0, -1))
                continue
            document_span_intervals.append((start_idx, end_idx))

        if missed:
            print(
                f"[BABILongRetriever] warning: {missed}/{len(docs)} sentences "
                f"could not be located in the prompt (probably whitespace drift). "
                f"They will contribute zero attention mass.",
                flush=True,
            )

        # Query span = the actual question content. Everything before this is
        # identical between real and null queries (docs are the same, QA3_INSTRUCTION
        # is constant) so the calibration assertion in score_per_token_attention_to_query
        # holds.
        query_content = f"Question: {query}"
        q_start, q_end = self.get_content_span(
            llm_prompt, char_offset_to_token_idx, query_content
        )
        query_span = (q_start, q_end)

        return llm_prompt, prompt_token_ids, query_span, document_span_intervals
