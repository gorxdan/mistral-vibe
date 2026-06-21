---
name: deep-research
description: Fan out web searches across angles, fetch and cross-check sources, synthesize a cited report
---

import json

SEARCH_ANGLES = [
    "overview",
    "technical details",
    "criticism and limitations",
    "recent developments",
    "practical examples",
]

CLAIM_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "source": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                },
                "required": ["claim", "source"],
            },
        },
    },
    "required": ["claims"],
}

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verified": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["verified", "reason"],
}


async def main():
    question = args if args else "What are the latest developments in AI coding assistants?"

    phase("Search")
    log(f"Researching: {question}")

    search_results = await parallel(*[
        (lambda angle=angle: agent(
            f"Search the web for information about: {question}\n"
            f"Focus on: {angle}\n"
            f"Return a summary of what you found, with source URLs.",
            agent="research",
            label=f"search:{angle}",
            phase="Search",
        ))
        for angle in SEARCH_ANGLES
    ])

    phase("Extract")
    all_text = "\n\n---\n\n".join(r for r in search_results if r)

    claims_response = await agent(
        f"Extract factual claims from this research text about: {question}\n"
        f"Each claim should have a source. Return as structured data.\n\n"
        f"{all_text}",
        label="extract",
        phase="Extract",
        schema=CLAIM_SCHEMA,
    )

    claims = claims_response.get("claims", []) if isinstance(claims_response, dict) else []
    log(f"Extracted {len(claims)} claims")

    if not claims:
        return {"question": question, "report": "No claims could be extracted from the search results.", "claims": []}

    phase("Verify")

    async def verify_claim(claim):
        verdict = await agent(
            f"Verify this claim about: {question}\n"
            f"Claim: {claim.get('claim', '')}\n"
            f"Source: {claim.get('source', '')}\n"
            f"Search the web to check if this claim is accurate. "
            f"Return verified=true only if you can confirm it.",
            agent="research",
            label=f"verify:{claim.get('source', '?')[:20]}",
            phase="Verify",
            schema=VERDICT_SCHEMA,
        )
        return {**claim, **verdict} if verdict else {**claim, "verified": False, "reason": "verification failed"}

    verdicts = await pipeline(claims, verify_claim)
    verified = [v for v in verdicts if v.get("verified")]
    log(f"Verified {len(verified)}/{len(claims)} claims")

    phase("Synthesize")
    report = await agent(
        f"Synthesize a research report about: {question}\n"
        f"Use only these verified claims:\n{json.dumps(verified, indent=2)}\n"
        f"Write a clear, cited report. Mark unverified claims as unconfirmed.",
        label="synthesize",
        phase="Synthesize",
    )

    return {
        "question": question,
        "report": report,
        "total_claims": len(claims),
        "verified_claims": len(verified),
    }
