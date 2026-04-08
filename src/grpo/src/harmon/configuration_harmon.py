from transformers import PretrainedConfig


class HarmonConfig(PretrainedConfig):
    model_type = "harmon"
    def __init__(self, llm=None, mar=None, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        self.mar = mar
