import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import asyncio
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv()
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness,
    ContextRelevance,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from Evaluation.evaluation_dataset import HR_POLICY_EVAL_DATASET
from ReRanker.reRanker import ReRanker
from consts import Consts

class Evaluate():
    def __init__(self):
        nvidia_client = AsyncOpenAI(
            base_url='https://integrate.api.nvidia.com/v1',
            api_key=os.environ.get('NVIDIA_API_KEY'),
            timeout=30
        )
        ragas_llm = llm_factory('meta/llama-3.1-8b-instruct', client=nvidia_client, adapter='instructor')

        self.context_precision = ContextPrecision(llm=ragas_llm)
        self.context_recall = ContextRecall(llm=ragas_llm)
        self.context_relevance = ContextRelevance(llm=ragas_llm)
        self.faithfulness = Faithfulness(llm=ragas_llm)


    def format_docs(self, docs: list[Document]) -> str:
        if not docs:
            return "No relevant documents found."
        return "\n\n".join(doc.page_content.strip() for doc in docs)


    def get_chat_chain(self):
        llm = Consts()._get_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("system",
            "You are an HR policy assistant for Resilience X.\n"
            "ONLY answer based on the provided context from documents.\n"
            "Do NOT use any knowledge outside the provided documents.\n"
            "If the answer is not in the documents, respond that you do not have enough information to answer.\n"
            "Keep answers concise and professional."),
            ("human",
            "Context from Documents:\n{context}\n\n"
            "User Query: {input}\n\n"
            "Final Answer:")
        ])
        return prompt | llm | StrOutputParser()


    async def run_retriever_evaluation(self, samples: List[Any] = None) -> Dict[str, float]:
        if samples is None:
            samples = HR_POLICY_EVAL_DATASET
        
        print(f"\n--- Running Retriever Evaluation on {len(samples)} samples ---")
        retriever = ReRanker()
        
        precision_scores = []
        recall_scores = []
        relevance_scores = []
        
        for i, sample in enumerate(samples):
            print(f"Sample {i+1}/{len(samples)}: {sample.user_input[:50]}...")
            try:
                retrieved_docs = retriever.invoke(sample.user_input)
                contexts = [doc.page_content for doc in retrieved_docs]
                
                p_res = await self.context_precision.ascore(
                    user_input=sample.user_input,
                    retrieved_contexts=contexts,
                    reference=sample.reference
                )
                precision_scores.append(p_res.value)
                
                r_res = await self.context_recall.ascore(
                    user_input=sample.user_input,
                    retrieved_contexts=contexts,
                    reference=sample.reference
                )
                recall_scores.append(r_res.value)
                
                rel_res = await self.context_relevance.ascore(
                    user_input=sample.user_input,
                    retrieved_contexts=contexts
                )
                relevance_scores.append(rel_res.value)
                
                print(f"  Prescision: {p_res.value:.2f}, Recall: {r_res.value:.2f}, context_Relevance: {rel_res.value:.2f}")
            except Exception as e:
                print(f"  Error evaluating sample {i+1}: {e}")
                continue

        if not precision_scores:
            return {"context_precision": 0.0, "context_recall": 0.0, "context_relevance": 0.0}

        return {
            "context_precision": sum(precision_scores) / len(precision_scores),
            "context_recall": sum(recall_scores) / len(recall_scores),
            "context_relevance": sum(relevance_scores) / len(relevance_scores),
        }


    async def run_generation_evaluation(self, samples: List[Any] = None) -> Dict[str, float]:
        """Evaluate generation using faithfulness, AnswerRelevancy, AnswerCorrectness."""
        if samples is None:
            samples = HR_POLICY_EVAL_DATASET
            
        print(f"\n--- Running Generation Evaluation on {len(samples)} samples ---")
        retriever = ReRanker()
        chat_chain = self.get_chat_chain()
        
        faithfulness_scores = []
        
        for i, sample in enumerate(samples):
            print(f"Sample {i+1}/{len(samples)}: {sample.user_input[:50]}...")
            try:
                retrieved_docs = retriever.invoke(sample.user_input)
                contexts = [doc.page_content for doc in retrieved_docs]
                context_str = self.format_docs(retrieved_docs)
                
                answer = chat_chain.invoke({"context": context_str, "input": sample.user_input})
                print(answer)
                
                f_res = await self.faithfulness.ascore(
                    user_input=sample.user_input,
                    response=answer,
                    retrieved_contexts=contexts
                )
                faithfulness_scores.append(f_res.value)
                
                
                print(f"  F: {f_res.value:.2f}")
            except Exception as e:
                print(f"  Error evaluating sample {i+1}: {e}")
                continue

        if not faithfulness_scores:
            return {"faithfulness": 0.0, "answer_relevancy": 0.0, "answer_correctness": 0.0}

        return {
            "faithfulness": sum(faithfulness_scores) / len(faithfulness_scores),
        }


    async def run_full_evaluation(self):
        """Run complete evaluation pipeline."""
        samples = HR_POLICY_EVAL_DATASET
        
        retriever_results = await self.run_retriever_evaluation(samples)
        print(f"\nRetriever results: {retriever_results}")
        
        generation_results = await self.run_generation_evaluation(samples)
        print(f"\nGeneration results: {generation_results}")
        
        results = {
            "retriever": retriever_results,
            "generation": generation_results,
            "num_samples": len(samples),
        }
        
        output_path = "Evaluation/results.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_path}")
        return results


if __name__ == "__main__":
    asyncio.run(Evaluate().run_full_evaluation())
