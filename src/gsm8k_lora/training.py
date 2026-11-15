"""GPU training components, imported only when a training job starts."""

from trl import SFTTrainer


class BudgetTrainer(SFTTrainer):
    """Count non-padding tokens and examples actually consumed by the trainer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exposure_tokens = 0
        self.answer_tokens = 0
        self.exposure_examples = 0

    def training_step(self, model, inputs, num_items_in_batch=None):
        self.exposure_tokens += int(inputs["attention_mask"].sum())
        self.answer_tokens += int((inputs["labels"][:, 1:] != -100).sum())
        self.exposure_examples += inputs["input_ids"].shape[0]
        return super().training_step(model, inputs, num_items_in_batch)
