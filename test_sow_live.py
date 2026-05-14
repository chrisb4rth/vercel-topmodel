"""
Live integration test for the Source of Wealth agent using the Parallel API.

This test sends a real query to the Parallel API and prints the response.
Requires PARALLEL_API_KEY to be set in the environment.
"""

import asyncio
import os
import sys

# Set the API key for this test run.
os.environ["PARALLEL_API_KEY"] = "iTc0gyLS8dmktsVirXEUa1iQrQSMvJU_E-r7xMH7"

from agents import SourceOfWealthAgent
from models import ConversationContext


async def main():
    agent = SourceOfWealthAgent()
    ctx = ConversationContext(exchanges=[])

    query = (
        "Conduct a source of wealth research on Elon Musk. "
        "Name: Elon Musk. Date of Birth: 1971-06-28. "
        "Nationality: USA / South Africa. "
        "Known Business Affiliations: Tesla, SpaceX, X (formerly Twitter). "
        "Stated Wealth Source: Technology entrepreneurship and equity stakes."
    )

    print("=" * 60)
    print("LIVE TEST: Source of Wealth Agent via Parallel API")
    print("=" * 60)
    print(f"\nQuery:\n{query}\n")
    print("-" * 60)
    print("Calling Parallel API (this may take up to 60 seconds)...")
    print("-" * 60)

    resp = await agent.handle_query(query, ctx)

    print(f"\n{'=' * 60}")
    print("RESULT")
    print(f"{'=' * 60}")
    print(f"Domain ID:      {resp.domain_id}")
    print(f"Confidence:     {resp.confidence.value}")
    print(f"Out of Scope:   {resp.is_out_of_scope}")
    print(f"References:     {[(r.law_name, r.paragraph, r.section) for r in resp.references]}")
    print(f"Limitation:     {resp.limitation_note}")
    print(f"\n{'─' * 60}")
    print("ANSWER BODY (first 2000 chars):")
    print(f"{'─' * 60}")
    print(resp.answer_body[:2000])
    if len(resp.answer_body) > 2000:
        print(f"\n... [{len(resp.answer_body) - 2000} more characters]")
    print(f"\n{'=' * 60}")
    print("TEST COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
