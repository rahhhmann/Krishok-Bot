# KrishokBot Text/RAG/Tool Evaluation — Summary

Total queries: 2 | Completed: 2 | Runtime errors: 0

## Routing correctness (deterministic, always computed)
2/2 queries routed to the expected combination of RAG/weather/price (100.0%).

## Retrieval hit-rate (queries where expected_route includes 'rag')
2/2 returned at least one chunk (100.0%).

## Guardrail: prompt-injection probe (q29)
Probe query not present in this run.

## RAGAS metrics (faithfulness / answer_relevancy / context_precision / context_recall)
- faithfulness: nan
- answer_relevancy: nan

## Proxy faithfulness (fallback lexical overlap — NOT a RAGAS-equivalent metric)
Average token overlap between answer and retrieved chunks: 0.5551
This measures vocabulary grounding only, not semantic faithfulness. Treat as a weak sanity signal, not a substitute for RAGAS faithfulness.

## Per-query results
See rag_eval_results.json / rag_eval_results.csv for full detail.

## Runtime errors
None.