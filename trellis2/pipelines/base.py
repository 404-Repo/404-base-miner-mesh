from typing import *
import torch
import torch.nn as nn
from .. import models


class Pipeline:
    """
    A base class for pipelines.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
    ):
        if models is None:
            return
        self.models = models
        for model in self.models.values():
            model.eval()

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        model_revisions: dict[str, str],
        config_file: str = "pipeline.json",
    ) -> "Pipeline":
        """
        Load a pretrained model.

        Args:
            path: The path to the model. Can be either local path or a Hugging Face repository.
            model_revisions: Dict mapping repo IDs to their revisions.
            config_file: The name of the config file.
        """
        import os
        import json
        from loguru import logger
        
        # Parse the main repo ID and get its revision from model_revisions
        main_repo_parts = path.split('/')
        main_repo_id = '/'.join(main_repo_parts[:2]) if len(main_repo_parts) >= 2 else path
        revision = model_revisions.get(main_repo_id)
        
        is_local = os.path.exists(f"{path}/{config_file}")

        if is_local:
            config_file = f"{path}/{config_file}"
        else:
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(path, config_file, revision=revision)

        with open(config_file, 'r') as f:
            args = json.load(f)['args']

        _models = {}
        for k, v in args['models'].items():
            if hasattr(cls, 'model_names_to_load') and k not in cls.model_names_to_load:
                continue
            
            # Try loading as a local/relative path first
            try:
                _models[k] = models.from_pretrained(f"{path}/{v}", revision=revision)
            except Exception as e:
                # External model path - check if it's a different repo
                v_parts = v.split('/')
                if len(v_parts) >= 2:
                    external_repo_id = '/'.join(v_parts[:2])
                else:
                    external_repo_id = v
                
                # Only use revision if it's the same repo, otherwise look up in model_revisions
                if external_repo_id == main_repo_id:
                    model_revision = revision
                else:
                    model_revision = model_revisions.get(external_repo_id)
                    if model_revision:
                        logger.info(f"Using pinned revision for external model {external_repo_id}: {model_revision}")
                    else:
                        logger.debug(f"No pinned revision for external model {external_repo_id}, using latest")
                
                _models[k] = models.from_pretrained(v, revision=model_revision)

        new_pipeline = cls(_models)
        new_pipeline._pretrained_args = args
        return new_pipeline

    @property
    def device(self) -> torch.device:
        if hasattr(self, '_device'):
            return self._device
        for model in self.models.values():
            if hasattr(model, 'device'):
                return model.device
        for model in self.models.values():
            if hasattr(model, 'parameters'):
                return next(model.parameters()).device
        raise RuntimeError("No device found.")

    def to(self, device: torch.device) -> None:
        for model in self.models.values():
            model.to(device)

    def cuda(self) -> None:
        self.to(torch.device("cuda"))

    def cpu(self) -> None:
        self.to(torch.device("cpu"))