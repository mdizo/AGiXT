import requests
from typing import Any, Dict, List, Optional


# API Documentation for KoboldCpp:
# https://lite.koboldai.net/koboldcpp_api

def clean_url(url: str) -> str:
    """Remove trailing slash and /api from url if present."""
    if url.endswith("/api"):
        return url[:-4]
    elif url.endswith("/"):
        return url[:-1]
    else:
        return url

def is_valid_float(value: float, no_range: bool = False) -> bool:
    """Check if a value is a float within 0 and 1"""
    if value in (0,1):
        return True
    if isinstance(value, float):
        if no_range:
            return value >= 0
        return 0 <= value <= 1
    return False

def is_valid_sample_order(value: list) -> bool:
    """Check if sample order is a list of integers ranging from 0 to 6 or 7"""
    if not isinstance(value, list):
        return False
    for num in value:
        if not isinstance(num, int):
            return False
    return set(value) == set(range(6)) or set(range(7))
    

class KoboldProvider:
    def __init__(
        self,
        AI_PROVIDER_URI: str = "",
        AI_MODEL: str = "default",
        PROMPT_PREFIX: str = "",
        PROMPT_SUFFIX: str = "",
        MAX_CONTEXT_LENGTH: int = 4096,
        MAX_LENGTH: int = 80,
        REP_PEN: float = 1.1,
        REP_PEN_RANGE: int = 320,
        SAMPLE_ORDER: List[int] = [6, 0, 1, 3, 4, 2, 5],
        SAMPLER_SEED: int = -1,
        STOP_SEQUENCE: Optional[List[str]] = None,
        TEMPERATURE: float = 0.7,
        TFS: float = 1,
        TOP_A: float = 0,
        TOP_K: int = 100,
        TOP_P: float = 0.92,
        MIN_P: float = 0,
        TYPICAL: float = 1,
        USE_DEFAULT_BADWORDSIDS: bool = False,
        MIROSTAT: int = 0.0,
        MIROSTAT_TAU: float = 5.0,
        MIROSTAT_ETA: float = 0.1,
        GRAMMAR: str = "",
        GRAMMAR_RETAIN_STATE: bool = False,
        MEMORY: str = "",
        TRIM_STOP: bool = False,
        **kwargs: Any,
    ):
        self.AI_PROVIDER_URI = (
            clean_url(AI_PROVIDER_URI) if AI_PROVIDER_URI else "http://host.docker.internal:5001"
        )
        self.AI_MODEL = AI_MODEL
        self.PROMPT_PREFIX = PROMPT_PREFIX if PROMPT_PREFIX else ""
        self.PROMPT_SUFFIX = PROMPT_SUFFIX if PROMPT_SUFFIX else ""
        self.MAX_CONTEXT_LENGTH = MAX_CONTEXT_LENGTH if MAX_CONTEXT_LENGTH else 4096
        self.MAX_LENGTH = MAX_LENGTH if MAX_LENGTH else 80
        self.REP_PEN = REP_PEN if isinstance(REP_PEN, float) and REP_PEN >= 1 else 1.1
        self.REP_PEN_RANGE = REP_PEN_RANGE if REP_PEN_RANGE >= 0 else 320
        self.SAMPLE_ORDER = SAMPLE_ORDER if is_valid_sample_order(SAMPLE_ORDER) else [6, 0, 1, 3, 4, 2, 5]
        self.SAMPLER_SEED = SAMPLER_SEED if -1 <= REP_PEN_RANGE <= 999999 else -1
        self.STOP_SEQUENCE = STOP_SEQUENCE
        self.TEMPERATURE = TEMPERATURE if is_valid_float(TEMPERATURE, False) else 0.7
        self.TFS = TFS if is_valid_float(TFS) else 1.0
        self.TOP_A = TOP_A if is_valid_float(TOP_A, False) else 0.0
        self.TOP_K = TOP_K if isinstance(TOP_K, int) and TOP_K >= 0 else 100
        self.TOP_P = TOP_P if is_valid_float(TOP_P) else 0.92
        self.MIN_P = MIN_P if is_valid_float(MIN_P) else 0.0
        self.TYPICAL = TYPICAL if is_valid_float(TYPICAL) else 1.0
        self.USE_DEFAULT_BADWORDSIDS = USE_DEFAULT_BADWORDSIDS if USE_DEFAULT_BADWORDSIDS else False
        self.MIROSTAT = MIROSTAT if MIROSTAT else 0
        self.MIROSTAT_TAU = MIROSTAT_TAU if MIROSTAT_TAU else 5.0
        self.MIROSTAT_ETA = MIROSTAT_ETA if MIROSTAT_ETA else 0.1
        self.GRAMMAR = GRAMMAR if GRAMMAR else ""
        self.GRAMMAR_RETAIN_STATE = GRAMMAR_RETAIN_STATE if GRAMMAR_RETAIN_STATE else False
        self.MEMORY = MEMORY if MEMORY else ""
        self.TRIM_STOP = TRIM_STOP if TRIM_STOP else False

    async def inference(self, prompt, tokens: int = 0):
        MAX_LENGTH = int(self.MAX_LENGTH) - tokens
        prompt = f"{self.PROMPT_PREFIX}{prompt}{self.PROMPT_SUFFIX}"
        params: Dict[str, Any] = {
            "prompt": prompt,
            "max_context_length": self.MAX_CONTEXT_LENGTH,
            "MAX_LENGTH": int(MAX_LENGTH),
            "rep_pen": float(self.REP_PEN),
            "rep_pen_range": int(self.REP_PEN_RANGE),
            "sampler_order": self.SAMPLE_ORDER,
            "temperature": float(self.TEMPERATURE),
            "tfs": float(self.TFS),
            "top_a": float(self.TOP_A),
            "top_k": int(self.TOP_K),
            "top_p": float(self.TOP_P),
            "min_p": float(self.MIN_P),
            "typical": float(self.TYPICAL),
        }

        if self.SAMPLER_SEED > 0:
            params["sampler_seed"] = self.SAMPLER_SEED

        if self.USE_DEFAULT_BADWORDSIDS:
            params["use_default_badwordsids"] = self.USE_DEFAULT_BADWORDSIDS

        if self.MEMORY:
            params["memory"] = self.MEMORY

        if self.GRAMMAR:
            params["grammar"] = self.GRAMMAR
            params["grammar_retain_state"] = self.GRAMMAR_RETAIN_STATE

        if self.MIROSTAT > 0:
            params["mirostat"] = self.MIROSTAT
            params["mirostat_tau"] = self.MIROSTAT_TAU
            params["mirostat_eta"] = self.MIROSTAT_ETA

        if self.STOP_SEQUENCE is not None:
            params["stop_sequence"] = self.STOP_SEQUENCE

            if self.TRIM_STOP:
                params["trim_stop"] = self.TRIM_STOP

        response = requests.post(f"{self.AI_PROVIDER_URI}/api/v1/generate", json=params)
        response.raise_for_status()
        json_response = response.json()

        if (
            "results" in json_response
            and len(json_response["results"]) > 0
            and "text" in json_response["results"][0]
        ):
            text = json_response["results"][0]["text"].strip()

            if self.STOP_SEQUENCE is not None:
                for sequence in self.STOP_SEQUENCE:
                    if text.endswith(sequence):
                        text = text[: -len(sequence)].rstrip()

            return text
        else:
            raise ValueError(
                f"Unexpected response format from Kobold API:  {response}"
            )