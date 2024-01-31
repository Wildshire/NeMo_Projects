from typing import List

from nemoguardrails import LLMRails
from nemoguardrails.embeddings.index import EmbeddingsIndex, IndexItem
import chromadb
from chromadb import HttpClient
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import chromadb.utils.embedding_functions as embedding_functions


class ChromaDBEmbeddingSearchProvider(EmbeddingsIndex):
    """A simple implementation of Qdrant as an embeddings search provider."""


    def __init__(self):
        self.collection_name = "collection_name"
        # self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        # self.size = self.encoder.get_sentence_embedding_dimension()
        
        
        self.chromaDB = chromadb.Client()
        self.chromaDB_collection = self.chromaDB.get_or_create_collection(self.collection_name)

        self.items: List[IndexItem] = []

    @property
    def embedding_size(self):
        return self.size

    async def add_item(self, item: IndexItem):
        """Adds a new item to the index."""
        self.add_items([item])
        
    async def add_items(self, items: List[IndexItem]):
        """Adds multiple items to the index."""
        for ids, doc in enumerate(items):
            self.chromaDB_collection.add(
                documents=[str(doc)],
                ids=[str(ids)]
            )
        
        
    async def search(self, text: str, max_results: int):
        """Searches the index for the closes matches to the provided text."""
        results = self.chromaDB_collection.query(query_texts=[text], n_results=2)
        final_results = []
        for doc in results["documents"]:
            final_results.append(eval(doc[0]))
        return final_results
            
    
import os
import os.path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.llm.helpers import get_llm_instance_wrapper
from nemoguardrails.llm.providers import (
    HuggingFacePipelineCompatible,
    register_llm_provider,
)


def _get_model_config(config: RailsConfig, type: str):
    """Quick helper to return the config for a specific model type."""
    for model_config in config.models:
        if model_config.type == type:
            return model_config


def _load_model(model_name_or_path, device, num_gpus, hf_auth_token=None, debug=False):
    """Load an HF locally saved checkpoint."""
    if device == "cpu":
        kwargs = {}
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if num_gpus == "auto":
            kwargs["device_map"] = "auto"
        else:
            num_gpus = int(num_gpus)
            if num_gpus != 1:
                kwargs.update(
                    {
                        "device_map": "auto",
                        "max_memory": {i: "13GiB" for i in range(num_gpus)},
                    }
                )
    elif device == "mps":
        kwargs = {"torch_dtype": torch.float16}
        # Avoid bugs in mps backend by not using in-place operations.
        print("mps not supported")
    else:
        raise ValueError(f"Invalid device: {device}")

    if hf_auth_token is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False,cache_dir="/.cache/huggingface/hub/", local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, low_cpu_mem_usage=True, cache_dir="/.cache/huggingface/hub/", local_files_only=True,**kwargs
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_auth_token=hf_auth_token, use_fast=False, cache_dir="/.cache/huggingface/hub/", local_files_only=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            low_cpu_mem_usage=True,
            use_auth_token=hf_auth_token,
            cache_dir="/.cache/huggingface/hub/", local_files_only=True,
            **kwargs,
        )

    if device == "cuda" and num_gpus == 1:
        model.to(device)

    if debug:
        print(model)

    return model, tokenizer


def init_main_llm(config: RailsConfig):
    """Initialize the main model from a locally saved path.

    The path is taken from the main model config.

    models:
      - type: main
        engine: hf_pipeline_bloke
        parameters:
          path: "<PATH TO THE LOCALLY SAVED CHECKPOINT>"
    """
    # loading custom llm  from disk with multiGPUs support
    # model_name = "< path_to_the_saved_custom_llm_checkpoints >"  # loading model ckpt from disk
    model_config = _get_model_config(config, "main")
    model_path = model_config.parameters.get("path")
    device = model_config.parameters.get("device", "cuda")
    num_gpus = model_config.parameters.get("num_gpus", 1)
    hf_token = os.environ[
        "HF_TOKEN"
    ]  # [TODO] to register this into the config.yaml as well
    model, tokenizer = _load_model(
        model_path, device, num_gpus, hf_auth_token=hf_token, debug=False
    )

    # repo_id="TheBloke/Wizard-Vicuna-13B-Uncensored-HF"
    # pipe = pipeline("text-generation", model=repo_id, device_map={"":"cuda:0"}, max_new_tokens=256, temperature=0.1, do_sample=True,use_cache=True)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.1,
        do_sample=True,
    )

    hf_llm = HuggingFacePipelineCompatible(pipeline=pipe)
    provider = get_llm_instance_wrapper(
        llm_instance=hf_llm, llm_type="hf_pipeline_llama2_7b"
    )
    register_llm_provider("hf_pipeline_llama2_7b", provider)


def init(app: LLMRails):
    app.register_embedding_search_provider("simple", ChromaDBEmbeddingSearchProvider)
    
    config = app.config

    # Initialize the various models
    init_main_llm(config)
    