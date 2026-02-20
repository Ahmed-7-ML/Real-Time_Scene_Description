import re

class SmartSafetyClassifierV2:
    """
    Version 2 Smart Regex-Based Scene Safety Classifier.
    Implements multi-layer reasoning with severity tiers, context awareness, 
    and strict discourse handling.
    """

    def __init__(self):
        # ==========================================
        # 1. Regex Patterns & Definitions
        # ==========================================

        # --- A. Discourse Markers ---
        self.contrast_pattern = re.compile(
            r'\b(but|however|although|yet|though)\b',
            re.IGNORECASE
        )
        self.negation_pattern = re.compile(
            r'\b(no|not|without|never)\b',
            re.IGNORECASE
        )

        # --- B. Proximity Markers ---
        self.prox_near = re.compile(
            r'\b(near|close|beside|around|in front of)\b', re.IGNORECASE)
        self.prox_far = re.compile(r'\b(far|away|distant)\b', re.IGNORECASE)

        # --- C. Hazard Tiers ---

        # Tier 1: High-Risk Hazards (Always dangerous unless strictly negated)
        self.high_risk_hazards = re.compile(
            r'\b(fire|smoke|accident|crash|holes?|open hole|active construction)\b',
            re.IGNORECASE
        )

        # Train is special: High risk ONLY if near
        self.train_pattern = re.compile(
            r'\b(trains?|railway|tracks?)\b', re.IGNORECASE)

        # Tier 2: Conditional Hazards (Neutralized by 'Far' or benign context)
        self.conditional_hazards = re.compile(
            r'\b(construction|puddles?|scaffolding|tools?|debris|broken|damaged|crack|unpaved|rough|flooded|muddy|icy|snowy|blocked|obstacle|barrier|messy|cluttered|trash|garbage|litter)\b',
            re.IGNORECASE
        )

        # --- D. Context Filters (Whitelists) ---
        self.safe_context_patterns = [
            # Commercial
            re.compile(
                r'\b(store|shop|display|market|selling|shelf|shelves|rack)\b', re.IGNORECASE),
            re.compile(
                r'\b(toy|model|miniature|lego|plush|doll|teddy)\b', re.IGNORECASE),  # Toys
            # Representation
            re.compile(
                r'\b(picture|photo|painting|poster|sign|billboard)\b', re.IGNORECASE),
        ]

    def classify(self, text):
        """
        Main classification entry point.
        Applies the Multi-Layer Decision Logic.
        """
        if not text:
            return 'SAFE'
            
        # Layer 1: Normalization
        text = text.lower().strip()

        # Layer 2: Discourse Segmentation (Contrast Dominance)
        segments = self.contrast_pattern.split(text)
        dominant_segment = segments[-1].strip()

        # Layer 3: Evaluate the Dominant Segment
        return self._evaluate_segment(dominant_segment)

    def _evaluate_segment(self, text):
        """
        Evaluates a targeted text segment using the hierarchy of rules.
        """
        # 1. Check High-Risk Hazards
        high_risk_matches = list(self.high_risk_hazards.finditer(text))
        for match in high_risk_matches:
            if self._is_active_hazard(match, text, tier='HIGH_RISK'):
                return 'DANGEROUS'

        # 2. Check Trains
        train_matches = list(self.train_pattern.finditer(text))
        for match in train_matches:
            if self._is_active_hazard(match, text, tier='CONDITIONAL'):
                return 'DANGEROUS'

        # 3. Check Conditional Hazards
        conditional_matches = list(self.conditional_hazards.finditer(text))
        for match in conditional_matches:
            if self._is_active_hazard(match, text, tier='CONDITIONAL'):
                return 'DANGEROUS'

        return 'SAFE'

    def _is_active_hazard(self, match, text, tier):
        """
        Determines if a found hazard word is actually dangerous using context.
        """
        start, end = match.span()
        word = match.group(0)

        # Context Extraction
        window_size = 50
        window_start = max(0, start - window_size)
        window_end = min(len(text), end + window_size)

        preceding_context = text[window_start:start]
        following_context = text[end:window_end]
        full_context = preceding_context + " " + word + " " + following_context

        # 1. Context Whitelist
        for pattern in self.safe_context_patterns:
            if pattern.search(full_context):
                return False

        # 2. Negation
        immediate_preceding = preceding_context.split()[-4:]
        preceding_str = " ".join(immediate_preceding)
        if self.negation_pattern.search(preceding_str):
            return False

        # 3. Proximity
        if tier == 'CONDITIONAL':
            if self.prox_far.search(preceding_context) or self.prox_far.search(following_context):
                return False

        return True
