import os
from typing import List


def summarize_incidents(incidents: List[dict]) -> List[str]:
    """Summarize incidents using Flan-T5. If FLAN_OFFLINE=1, return templated summaries.
    Each incident: {"type": str, "src": str, "dst": str, "severity": float}
    """
    if os.getenv("FLAN_OFFLINE", "0") == "1":
        return [f"Incident {i['type']} from {i['src']} to {i['dst']} (severity {i['severity']:.2f})." for i in incidents]
    try:
        from transformers import pipeline
        pipe = pipeline("text2text-generation", model="google/flan-t5-small")
        outputs = []
        for inc in incidents:
            prompt = (
                f"Summarize the following SDN/IoT security incident for an operator.\n"
                f"Type: {inc['type']}\nSource: {inc['src']}\nDestination: {inc['dst']}\nSeverity: {inc['severity']:.2f}\n"
                f"Provide recommended action in one sentence."
            )
            out = pipe(prompt, max_new_tokens=64)[0]["generated_text"].strip()
            outputs.append(out)
        return outputs
    except Exception as e:
        # fallback to template if transformers unavailable at runtime
        return [f"Incident {i['type']} from {i['src']} to {i['dst']} (severity {i['severity']:.2f})." for i in incidents]
