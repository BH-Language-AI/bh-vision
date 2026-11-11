"""BH-Vision Python API (samo interfejs Faze 1 — implementacija uz prototip)."""


class BHVision:
    """Vizuelni model: slike, dokumenti, OCR, video."""

    @classmethod
    def from_pretrained(cls, model_id):
        """Ucitaj model. Implementacija stize sa prototipom (Faza 6)."""
        raise NotImplementedError("prototip jos nije odabran (Faza 6)")

    def ask(self, image, question):
        """Pitanje o slici na bosanskom."""
        raise NotImplementedError("prototip jos nije odabran (Faza 6)")
