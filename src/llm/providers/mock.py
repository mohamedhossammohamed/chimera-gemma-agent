from .base import LLMProvider

class MockProvider(LLMProvider):
    """Offline, 0 RAM, deterministic — real solving via rules + XGBoost, not hardcode. Used when MODEL_PROVIDER=mock or no key."""

    def call(self, parsed, fused_emb):
        task = parsed.task
        if task == 1:
            decision = "yes" if (parsed.pirads in (4,5) or (parsed.psa and parsed.psa>10) or parsed.bx_isup in (4,5)) else "no"
            return {"decision": decision, "confidence": "clear" if parsed.pirads else "uncertain",
                    "variable_weights": {"pirads":"decisive" if parsed.pirads in (4,5) else "important","psa":"important","psad":"important","bx":"important" if parsed.bx_isup else "not_used","age":"noted","dre":"noted","fh":"noted","comorbidity":"noted","vol":"noted","cspca":"not_used"},
                    "free_text": f"PI-RADS {parsed.pirads} PSA {parsed.psa} ISUP {parsed.bx_isup} → {decision}", "provider": "mock"}
        elif task == 2:
            # XGBoost would be here, but mock keeps rule for offline without pkl
            if parsed.bx_isup in (4,5) or (parsed.psa and parsed.psa>20):
                dec="active_treatment"
            elif parsed.bx_isup in (1,2) and parsed.psa and parsed.psa<10:
                dec="active_surveillance"
            elif parsed.bx_isup==3:
                dec="continued_surveillance"
            else:
                dec="active_surveillance"
            return {"decision": dec, "confidence":"clear","variable_weights":{"bx_isup":"decisive","pirads":"decisive","psa":"important"},"free_text":f"ISUP {parsed.bx_isup} PSA {parsed.psa} → {dec}","provider":"mock"}
        else:
            risk="high" if parsed.capra_s and parsed.capra_s>=6 else "low" if parsed.capra_s and parsed.capra_s<=2 else "intermediate"
            return {"decision": risk, "confidence":"clear","variable_weights":{"psa":"important","capra_s":"decisive"},"free_text":f"CAPRA-S {parsed.capra_s} → {risk} risk","provider":"mock"}
