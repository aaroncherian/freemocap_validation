# validation/pipeline/supports_variants.py
from typing import List
from validation.datatypes.data_component import DataComponent

class SupportsVariantsMixin:
    VARIANT_ENUM = None
    VARIANT_TO_COMPONENT = None
    variant_prefix: str = ""                     # set by make_variant

    # --- new: auto-clone PRODUCES once per subclass ---
    @classmethod
    def _clone_produces_with_prefix(cls, prefix: str) -> List[DataComponent]:
        return [c.clone_with_prefix(prefix) for c in cls.PRODUCES]

    # --- default store that respects variant prefix ---
    def store(self):
        for base_dc in self.PRODUCES:            # original list, NOT cloned
            prefixed_name = f"{self.variant_prefix}_{base_dc.name}"
            data = self.outputs.get(base_dc.name)          # calculate() used base name
            if data is None:
                continue
            clone_dc = base_dc.clone_with_prefix(self.variant_prefix)
            if clone_dc.saver:
                clone_dc.save(self.ctx.recording_dir, data)
            self.ctx.put(prefixed_name, data)
            self.logger.info(f"Saved {prefixed_name}")
