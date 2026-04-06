"""
Evaluation runner for the Agentic RAG AI Research Scientist.
Implements offline benchmark evaluation with retrieval and reasoning metrics.
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import Orchestrator
from utils.groq_client import groq_client
from config import config


@dataclass
class EvaluationResult:
    """Result for a single query evaluation."""
    query_id: str
    query: str
    success: bool
    papers_retrieved: List[str]
    expected_papers: List[str]
    recall_at_k: float
    mrr: float
    context_precision: float
    faithfulness_score: float
    unsupported_claims: int
    hallucination_flag: bool
    synthesis_score: float
    comparison_score: float
    limitation_score: float
    total_latency_ms: float
    tokens_used: int
    error: Optional[str] = None


class RetrievalMetrics:
    """Compute retrieval-based metrics."""
    
    @staticmethod
    def recall_at_k(retrieved: List[str], expected: List[str], k: int = None) -> float:
        """
        Compute Recall@K.
        
        Args:
            retrieved: List of retrieved paper IDs
            expected: List of expected (ground truth) paper IDs
            k: Limit to top K retrieved (optional)
            
        Returns:
            Recall score between 0 and 1
        """
        if not expected:
            return 0.0
        
        if k:
            retrieved = retrieved[:k]
        
        retrieved_set = set(retrieved)
        expected_set = set(expected)
        
        hits = len(retrieved_set.intersection(expected_set))
        return hits / len(expected_set)
    
    @staticmethod
    def mean_reciprocal_rank(retrieved: List[str], expected: List[str]) -> float:
        """
        Compute Mean Reciprocal Rank (MRR).
        
        Args:
            retrieved: List of retrieved paper IDs
            expected: List of expected paper IDs
            
        Returns:
            MRR score between 0 and 1
        """
        if not expected or not retrieved:
            return 0.0
        
        expected_set = set(expected)
        
        for i, paper_id in enumerate(retrieved, 1):
            if paper_id in expected_set:
                return 1.0 / i
        
        return 0.0
    
    @staticmethod
    def context_precision(retrieved: List[str], expected: List[str]) -> float:
        """
        Compute context precision (how many retrieved are relevant).
        
        Args:
            retrieved: List of retrieved paper IDs
            expected: List of expected paper IDs
            
        Returns:
            Precision score between 0 and 1
        """
        if not retrieved:
            return 0.0
        
        retrieved_set = set(retrieved)
        expected_set = set(expected)
        
        hits = len(retrieved_set.intersection(expected_set))
        return hits / len(retrieved_set)


class LLMJudge:
    """LLM-based evaluation using Groq."""
    
    FAITHFULNESS_PROMPT = """You are evaluating the faithfulness of a research synthesis. 
Compare the generated response against the retrieved evidence.

RESPONSE TO EVALUATE:
{response}

RETRIEVED EVIDENCE:
{evidence}

Evaluate:
1. Are all claims in the response supported by the evidence?
2. Count the number of unsupported claims
3. Is there any hallucination (information not in evidence)?

Respond in JSON format:
{{
    "faithfulness_score": 0.0-1.0,
    "unsupported_claims": [list of unsupported claim texts],
    "hallucination_detected": true/false,
    "reasoning": "brief explanation"
}}"""

    REASONING_DEPTH_PROMPT = """You are evaluating the reasoning depth of a research synthesis.

RESEARCH QUERY:
{query}

GENERATED RESPONSE:
{response}

Score each dimension from 1-5:

1. Cross-paper Synthesis (1-5): Does the response integrate information from multiple papers?
   - 1: Single source only
   - 3: Multiple sources, basic integration
   - 5: Deep integration with novel insights

2. Comparative Analysis (1-5): Does the response compare and contrast approaches?
   - 1: No comparison
   - 3: Basic pros/cons listed
   - 5: Nuanced comparison with tradeoffs

3. Limitation Awareness (1-5): Does the response acknowledge limitations?
   - 1: No limitations mentioned
   - 3: Some limitations listed
   - 5: Thoughtful limitation analysis with implications

Respond in JSON format:
{{
    "synthesis_score": 1-5,
    "comparison_score": 1-5,
    "limitation_score": 1-5,
    "reasoning": "brief explanation"
}}"""

    @staticmethod
    def evaluate_faithfulness(response: str, evidence: List[Dict]) -> Dict[str, Any]:
        """Evaluate faithfulness of response against evidence."""
        evidence_text = "\n\n".join([
            f"[{e.get('paper_title', 'Unknown')}]: {e.get('text', '')[:500]}"
            for e in evidence[:10]
        ])
        
        prompt = LLMJudge.FAITHFULNESS_PROMPT.format(
            response=response[:3000],
            evidence=evidence_text
        )
        
        result = groq_client.complete_json(
            prompt=prompt,
            model=config.FAST_MODEL,
            temperature=0.1,
            max_tokens=1024
        )
        
        if result.get("parsed"):
            parsed = result["parsed"]
            return {
                "faithfulness_score": parsed.get("faithfulness_score", 0.5),
                "unsupported_claims": len(parsed.get("unsupported_claims", [])),
                "hallucination_detected": parsed.get("hallucination_detected", False)
            }
        
        return {
            "faithfulness_score": 0.5,
            "unsupported_claims": 0,
            "hallucination_detected": False
        }
    
    @staticmethod
    def evaluate_reasoning_depth(query: str, response: str) -> Dict[str, Any]:
        """Evaluate reasoning depth of the response."""
        prompt = LLMJudge.REASONING_DEPTH_PROMPT.format(
            query=query,
            response=response[:3000]
        )
        
        result = groq_client.complete_json(
            prompt=prompt,
            model=config.FAST_MODEL,
            temperature=0.1,
            max_tokens=512
        )
        
        if result.get("parsed"):
            parsed = result["parsed"]
            return {
                "synthesis_score": parsed.get("synthesis_score", 3),
                "comparison_score": parsed.get("comparison_score", 3),
                "limitation_score": parsed.get("limitation_score", 3)
            }
        
        return {
            "synthesis_score": 3,
            "comparison_score": 3,
            "limitation_score": 3
        }


class EvaluationRunner:
    """Run evaluation on test queries."""
    
    def __init__(self):
        """Initialize the evaluation runner."""
        self.orchestrator = Orchestrator()
        self.metrics = RetrievalMetrics()
        self.judge = LLMJudge()
        
        # Load test data
        eval_dir = os.path.dirname(os.path.abspath(__file__))
        
        with open(os.path.join(eval_dir, "test_queries.json"), "r") as f:
            self.test_queries = json.load(f)
        
        with open(os.path.join(eval_dir, "expected_papers.json"), "r") as f:
            self.expected_papers = json.load(f)
    
    def run_single_query(self, query_data: Dict) -> EvaluationResult:
        """
        Run evaluation for a single query.
        
        Args:
            query_data: Dict with query_id and query text
            
        Returns:
            EvaluationResult for this query
        """
        query_id = query_data["query_id"]
        query = query_data["query"]
        expected = self.expected_papers.get(query_id, {})
        expected_paper_ids = expected.get("expected_paper_ids", [])
        
        print(f"Evaluating {query_id}: {query[:50]}...")
        
        try:
            # Run the RAG pipeline
            result = self.orchestrator.run(
                query=query,
                papers_k=5,
                chunks_top_n=10,
                include_safety_check=True
            )
            
            if not result.get("success"):
                return EvaluationResult(
                    query_id=query_id,
                    query=query,
                    success=False,
                    papers_retrieved=[],
                    expected_papers=expected_paper_ids,
                    recall_at_k=0.0,
                    mrr=0.0,
                    context_precision=0.0,
                    faithfulness_score=0.0,
                    unsupported_claims=0,
                    hallucination_flag=False,
                    synthesis_score=0,
                    comparison_score=0,
                    limitation_score=0,
                    total_latency_ms=result.get("metrics", {}).get("total_latency_ms", 0),
                    tokens_used=result.get("metrics", {}).get("tokens_used", 0),
                    error=result.get("error")
                )
            
            # Extract retrieved paper IDs
            papers = result.get("papers", [])
            retrieved_paper_ids = [p.get("paper_id", "") for p in papers]
            
            # Compute retrieval metrics
            recall = self.metrics.recall_at_k(retrieved_paper_ids, expected_paper_ids)
            mrr = self.metrics.mean_reciprocal_rank(retrieved_paper_ids, expected_paper_ids)
            precision = self.metrics.context_precision(retrieved_paper_ids, expected_paper_ids)
            
            # LLM-based evaluation (simulated evidence from response)
            response = result.get("response", "")
            
            # Faithfulness evaluation
            faith_result = self.judge.evaluate_faithfulness(response, [])
            
            # Reasoning depth evaluation
            depth_result = self.judge.evaluate_reasoning_depth(query, response)
            
            return EvaluationResult(
                query_id=query_id,
                query=query,
                success=True,
                papers_retrieved=retrieved_paper_ids,
                expected_papers=expected_paper_ids,
                recall_at_k=recall,
                mrr=mrr,
                context_precision=precision,
                faithfulness_score=faith_result["faithfulness_score"],
                unsupported_claims=faith_result["unsupported_claims"],
                hallucination_flag=faith_result["hallucination_detected"],
                synthesis_score=depth_result["synthesis_score"],
                comparison_score=depth_result["comparison_score"],
                limitation_score=depth_result["limitation_score"],
                total_latency_ms=result.get("metrics", {}).get("total_latency_ms", 0),
                tokens_used=result.get("metrics", {}).get("tokens_used", 0)
            )
            
        except Exception as e:
            return EvaluationResult(
                query_id=query_id,
                query=query,
                success=False,
                papers_retrieved=[],
                expected_papers=expected_paper_ids,
                recall_at_k=0.0,
                mrr=0.0,
                context_precision=0.0,
                faithfulness_score=0.0,
                unsupported_claims=0,
                hallucination_flag=False,
                synthesis_score=0,
                comparison_score=0,
                limitation_score=0,
                total_latency_ms=0,
                tokens_used=0,
                error=str(e)
            )
    
    def run_all(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Run evaluation on all test queries.
        
        Args:
            limit: Optional limit on number of queries to evaluate
            
        Returns:
            Dict with results and aggregate metrics
        """
        queries = self.test_queries[:limit] if limit else self.test_queries
        results = []
        
        print(f"\nRunning evaluation on {len(queries)} queries...\n")
        
        for query_data in queries:
            result = self.run_single_query(query_data)
            results.append(result)
            print(f"  {result.query_id}: Recall={result.recall_at_k:.2f}, MRR={result.mrr:.2f}, Faithfulness={result.faithfulness_score:.2f}")
        
        # Compute aggregate metrics
        successful = [r for r in results if r.success]
        
        aggregate = {
            "total_queries": len(results),
            "successful_queries": len(successful),
            "success_rate": len(successful) / len(results) if results else 0,
            "avg_recall_at_k": sum(r.recall_at_k for r in successful) / len(successful) if successful else 0,
            "avg_mrr": sum(r.mrr for r in successful) / len(successful) if successful else 0,
            "avg_context_precision": sum(r.context_precision for r in successful) / len(successful) if successful else 0,
            "avg_faithfulness": sum(r.faithfulness_score for r in successful) / len(successful) if successful else 0,
            "total_unsupported_claims": sum(r.unsupported_claims for r in successful),
            "hallucination_rate": sum(1 for r in successful if r.hallucination_flag) / len(successful) if successful else 0,
            "avg_synthesis_score": sum(r.synthesis_score for r in successful) / len(successful) if successful else 0,
            "avg_comparison_score": sum(r.comparison_score for r in successful) / len(successful) if successful else 0,
            "avg_limitation_score": sum(r.limitation_score for r in successful) / len(successful) if successful else 0,
            "avg_latency_ms": sum(r.total_latency_ms for r in successful) / len(successful) if successful else 0,
            "total_tokens_used": sum(r.tokens_used for r in results)
        }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "aggregate_metrics": aggregate,
            "individual_results": [asdict(r) for r in results]
        }
    
    def save_results(self, results: Dict[str, Any], output_path: str = None):
        """Save evaluation results to JSON file."""
        if output_path is None:
            output_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        return output_path


def main():
    """Run evaluation from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Agentic RAG evaluation")
    parser.add_argument("--limit", type=int, help="Limit number of queries to evaluate")
    parser.add_argument("--output", type=str, help="Output file path for results")
    args = parser.parse_args()
    
    # Check API key
    if not config.GROQ_API_KEY:
        print("Error: GROQ_API_KEY is not set. Please configure it in .env")
        sys.exit(1)
    
    runner = EvaluationRunner()
    results = runner.run_all(limit=args.limit)
    
    # Print summary
    agg = results["aggregate_metrics"]
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total Queries:        {agg['total_queries']}")
    print(f"Successful:           {agg['successful_queries']} ({agg['success_rate']*100:.1f}%)")
    print(f"Avg Recall@K:         {agg['avg_recall_at_k']:.3f}")
    print(f"Avg MRR:              {agg['avg_mrr']:.3f}")
    print(f"Avg Context Precision: {agg['avg_context_precision']:.3f}")
    print(f"Avg Faithfulness:     {agg['avg_faithfulness']:.3f}")
    print(f"Hallucination Rate:   {agg['hallucination_rate']*100:.1f}%")
    print(f"Avg Synthesis Score:  {agg['avg_synthesis_score']:.1f}/5")
    print(f"Avg Comparison Score: {agg['avg_comparison_score']:.1f}/5")
    print(f"Avg Limitation Score: {agg['avg_limitation_score']:.1f}/5")
    print(f"Avg Latency:          {agg['avg_latency_ms']/1000:.1f}s")
    print(f"Total Tokens Used:    {agg['total_tokens_used']:,}")
    print("="*60)
    
    # Save results
    runner.save_results(results, args.output)


if __name__ == "__main__":
    main()
