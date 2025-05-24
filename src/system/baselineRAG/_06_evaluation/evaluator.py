"""
This module helps check if the RAG system is working well.
It can measure how good the answers are and find problems.
"""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Default evaluation prompt
DEFAULT_EVALUATION_PROMPT = """You are an expert evaluator of RAG systems.
Check if the answer is correct based on the context.
Give a score from 0 to 100.
Explain your score.
Find any problems or missing information."""

class RAGEvaluator:
    """
    A class that helps check if the RAG system is working well.
    
    This class can:
    - Check if answers are correct
    - Find missing information
    - Measure answer quality
    - Find system problems
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        evaluation_prompt: str = DEFAULT_EVALUATION_PROMPT
    ):
        """
        Start the RAGEvaluator with optional model settings.
        
        Args:
            model_name: Name of the AI model to use
            temperature: How creative the evaluations should be (0.0 to 1.0)
            evaluation_prompt: Prompt to guide the evaluation process
            
        Example:
            >>> evaluator = RAGEvaluator(model_name="gpt-4")
            >>> score = evaluator.evaluate_answer("What is RAG?", answer, documents)
        """
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature
        )
        
        # Create evaluation prompt
        self.eval_prompt = ChatPromptTemplate.from_messages([
            ("system", evaluation_prompt),
            ("human", """Context: {context}
            
            Question: {question}
            
            Answer: {answer}
            
            Evaluation:""")
        ])
        
        # Create the chain
        self.chain = (
            {
                "context": RunnablePassthrough(),
                "question": RunnablePassthrough(),
                "answer": RunnablePassthrough()
            }
            | self.eval_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def evaluate_answer(
        self,
        question: str,
        answer: str,
        documents: List[Document]
    ) -> Dict[str, Any]:
        """
        Check if an answer is good.
        
        Args:
            question: The original question
            answer: The generated answer
            documents: List of relevant documents
            
        Returns:
            Dictionary with evaluation results
            
        Example:
            >>> result = evaluator.evaluate_answer("What is RAG?", answer, documents)
            >>> print(f"Score: {result['score']}")
            >>> print(f"Feedback: {result['feedback']}")
        """
        context = "\n\n".join(doc.page_content for doc in documents)
        
        evaluation = self.chain.invoke({
            "context": context,
            "question": question,
            "answer": answer
        })
        
        # Try to extract score from evaluation text
        score = 0
        for line in evaluation.split("\n"):
            if "score" in line.lower():
                try:
                    score = int(line.split(":")[-1].strip())
                    break
                except:
                    continue
                    
        return {
            "score": score,
            "feedback": evaluation,
            "context": context
        }
    
    def find_missing_information(
        self,
        question: str,
        answer: str,
        documents: List[Document]
    ) -> List[str]:
        """
        Find information that should be in the answer but is missing.
        
        Args:
            question: The original question
            answer: The generated answer
            documents: List of relevant documents
            
        Returns:
            List of missing information points
            
        Example:
            >>> missing = evaluator.find_missing_information("What is RAG?", answer, documents)
            >>> print(f"Missing information: {missing}")
        """
        context = "\n\n".join(doc.page_content for doc in documents)
        
        # Create prompt for finding missing information
        missing_prompt = ChatPromptTemplate.from_messages([
            ("system", """Find information in the context that should be in the answer but is missing.
            List each missing point on a new line.
            Be specific and clear."""),
            ("human", """Context: {context}
            
            Question: {question}
            
            Answer: {answer}
            
            Missing information:""")
        ])
        
        chain = (
            {
                "context": RunnablePassthrough(),
                "question": RunnablePassthrough(),
                "answer": RunnablePassthrough()
            }
            | missing_prompt
            | self.llm
            | StrOutputParser()
        )
        
        missing_info = chain.invoke({
            "context": context,
            "question": question,
            "answer": answer
        })
        
        return [line.strip() for line in missing_info.split("\n") if line.strip()]
    
    def evaluate_retrieval(
        self,
        question: str,
        documents: List[Document]
    ) -> Dict[str, Any]:
        """
        Check if the right documents were found.
        
        Args:
            question: The original question
            documents: List of retrieved documents
            
        Returns:
            Dictionary with retrieval evaluation results
            
        Example:
            >>> result = evaluator.evaluate_retrieval("What is RAG?", documents)
            >>> print(f"Relevance score: {result['relevance_score']}")
            >>> print(f"Feedback: {result['feedback']}")
        """
        context = "\n\n".join(doc.page_content for doc in documents)
        
        # Create prompt for evaluating retrieval
        retrieval_prompt = ChatPromptTemplate.from_messages([
            ("system", """Check if these documents are relevant to the question.
            Give a score from 0 to 100.
            Explain your score.
            Find any irrelevant or missing documents."""),
            ("human", """Question: {question}
            
            Retrieved documents: {context}
            
            Evaluation:""")
        ])
        
        chain = (
            {
                "context": RunnablePassthrough(),
                "question": RunnablePassthrough()
            }
            | retrieval_prompt
            | self.llm
            | StrOutputParser()
        )
        
        evaluation = chain.invoke({
            "context": context,
            "question": question
        })
        
        # Try to extract score from evaluation text
        score = 0
        for line in evaluation.split("\n"):
            if "score" in line.lower():
                try:
                    score = int(line.split(":")[-1].strip())
                    break
                except:
                    continue
                    
        return {
            "relevance_score": score,
            "feedback": evaluation,
            "retrieved_docs": len(documents)
        }

if __name__ == "__main__":
    """
    This part runs when you run this file directly.
    It shows examples of how to use the RAGEvaluator class.
    """
    from langchain_core.documents import Document
    
    # Create sample documents
    sample_docs = [
        Document(
            page_content="RAG stands for Retrieval-Augmented Generation. It combines retrieval of relevant documents with language model generation.",
            metadata={"source": "test1"}
        ),
        Document(
            page_content="RAG helps language models provide more accurate and up-to-date answers by using external knowledge.",
            metadata={"source": "test2"}
        )
    ]
    
    # Create sample answer
    sample_answer = "RAG is a system that helps AI models give better answers by using information from documents."
    
    # Test answer evaluation
    print("\nTesting answer evaluation...")
    try:
        evaluator = RAGEvaluator()
        result = evaluator.evaluate_answer("What is RAG?", sample_answer, sample_docs)
        print(f"Score: {result['score']}")
        print(f"Feedback: {result['feedback']}")
    except Exception as e:
        print(f"Answer evaluation test failed: {e}")
    
    # Test missing information
    print("\nTesting missing information...")
    try:
        missing = evaluator.find_missing_information("What is RAG?", sample_answer, sample_docs)
        print("Missing information:")
        for item in missing:
            print(f"- {item}")
    except Exception as e:
        print(f"Missing information test failed: {e}")
    
    # Test retrieval evaluation
    print("\nTesting retrieval evaluation...")
    try:
        result = evaluator.evaluate_retrieval("What is RAG?", sample_docs)
        print(f"Relevance score: {result['relevance_score']}")
        print(f"Feedback: {result['feedback']}")
    except Exception as e:
        print(f"Retrieval evaluation test failed: {e}")
